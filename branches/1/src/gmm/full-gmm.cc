// gmm/full-gmm.cc

// Copyright 2009-2011  Jan Silovsky;  Saarland University;
//                      Microsoft Corporation;  Georg Stemmer

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <limits>

#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "util/stl-utils.h"

namespace kaldi {

void FullGmm::Resize(int32 nmix, int32 dim) {
  KALDI_ASSERT(nmix > 0 && dim > 0);
  if (gconsts_.Dim() != nmix) gconsts_.Resize(nmix);
  if (weights_.Dim() != nmix) weights_.Resize(nmix);
  if (means_invcovars_.NumRows() != nmix
      || means_invcovars_.NumCols() != dim)
    means_invcovars_.Resize(nmix, dim);
  ResizeInvCovars(nmix, dim);
}

void FullGmm::ResizeInvCovars(int32 nmix, int32 dim) {
  KALDI_ASSERT(nmix > 0 && dim > 0);
  if (inv_covars_.size() != static_cast<size_t>(nmix))
    inv_covars_.resize(nmix);
  for (int32 i = 0; i < nmix; ++i) {
    if (inv_covars_[i].NumRows() != dim) {
      inv_covars_[i].Resize(dim);
      inv_covars_[i].SetUnit();
      // must be initialized to unit for case of calling SetMeans while having
      // covars/invcovars that are not set yet (i.e. zero)
    }
  }
}

void FullGmm::CopyFromFullGmm(const FullGmm &fullgmm) {
  Resize(fullgmm.NumGauss(), fullgmm.Dim());
  gconsts_.CopyFromVec(fullgmm.gconsts_);
  weights_.CopyFromVec(fullgmm.weights_);
  means_invcovars_.CopyFromMat(fullgmm.means_invcovars_);
  int32 ncomp = NumGauss();
  for (int32 mix = 0; mix < ncomp; ++mix) {
    inv_covars_[mix].CopyFromSp(fullgmm.inv_covars_[mix]);
  }
  valid_gconsts_ = fullgmm.valid_gconsts_;
}

void FullGmm::CopyFromDiagGmm(const DiagGmm &diaggmm) {
  Resize(diaggmm.NumGauss(), diaggmm.Dim());
  gconsts_.CopyFromVec(diaggmm.gconsts());
  weights_.CopyFromVec(diaggmm.weights());
  means_invcovars_.CopyFromMat(diaggmm.means_invvars());
  int32 ncomp = NumGauss(), dim = Dim();
  for (int32 mix = 0; mix < ncomp; ++mix) {
    inv_covars_[mix].SetZero();
    for (int32 d = 0; d < dim; ++d) {
      inv_covars_[mix](d, d) = diaggmm.inv_vars()(mix, d);
    }
  }
  ComputeGconsts();
}

int32 FullGmm::ComputeGconsts() {
  int32 num_mix = NumGauss(),
         dim = Dim();
  KALDI_ASSERT(num_mix > 0 && dim > 0);
  BaseFloat offset = -0.5 * M_LOG_2PI * dim;  // constant term in gconst.
  int32 num_bad = 0;

  // Resize if Gaussians have been removed during Update()
  if (num_mix != gconsts_.Dim()) gconsts_.Resize(num_mix);

  for (int32 mix = 0; mix < num_mix; mix++) {
    KALDI_ASSERT(weights_(mix) >= 0);  // Cannot have negative weights.
    BaseFloat gc = log(weights_(mix)) + offset;  // May be -inf if weights == 0
    SpMatrix<BaseFloat> covar(inv_covars_[mix]);
    covar.InvertDouble();
    BaseFloat logdet = covar.LogPosDefDet();
    gc -= 0.5 * (logdet + VecSpVec(means_invcovars_.Row(mix),
                                   covar, means_invcovars_.Row(mix)));
    // Note that mean_invcovars(mix)' * covar(mix) * mean_invcovars(mix, d) is
    // really mean' * inv(covar) * mean, since mean_invcovars(mix, d) contains
    // the inverse covariance times mean.
    // So gc is the likelihood at zero feature value.

    if (KALDI_ISNAN(gc)) {  // negative infinity is OK but NaN is not acceptable
      KALDI_ERR << "At component" << mix
                << ", not a number in gconst computation";
    }
    if (KALDI_ISINF(gc)) {
      num_bad++;
      // If positive infinity, make it negative infinity.
      // Want to make sure the answer becomes -inf in the end, not NaN.
      if (gc > 0) gc = -gc;
    }
    gconsts_(mix) = gc;
  }

  valid_gconsts_ = true;
  return num_bad;
}

void FullGmm::Split(int32 target_components, float perturb_factor) {
  if (target_components <= NumGauss() || NumGauss() == 0) {
    KALDI_ERR << "Cannot split from " << NumGauss() <<  " to "
              << target_components << " components";
  }
  int32 current_components = NumGauss(), dim = Dim();
  FullGmm *tmp = new FullGmm();
  tmp->CopyFromFullGmm(*this);  // so we have copies of matrices...
  // First do the resize:
  weights_.Resize(target_components);
  weights_.Range(0, current_components).CopyFromVec(tmp->weights_);
  means_invcovars_.Resize(target_components, dim);
  means_invcovars_.Range(0, current_components, 0,
      dim).CopyFromMat(tmp->means_invcovars_);
  ResizeInvCovars(target_components, dim);
  for (int32 mix = 0; mix < current_components; ++mix) {
    inv_covars_[mix].CopyFromSp(tmp->inv_covars_[mix]);
  }
  for (int32 mix = current_components; mix < target_components; ++mix) {
    inv_covars_[mix].SetZero();
  }
  gconsts_.Resize(target_components);

  delete tmp;

  // future work(arnab): Use a priority queue instead?
  while (current_components < target_components) {
    BaseFloat max_weight = weights_(0);
    int32 max_idx = 0;
    for (int32 i = 1; i < current_components; i++) {
      if (weights_(i) > max_weight) {
        max_weight = weights_(i);
        max_idx = i;
      }
    }
    weights_(max_idx) /= 2;
    weights_(current_components) = weights_(max_idx);
    Vector<BaseFloat> rand_vec(dim);
    rand_vec.SetRandn();
    TpMatrix<BaseFloat> invcovar_l(dim);
    invcovar_l.Cholesky(inv_covars_[max_idx]);
    rand_vec.MulTp(invcovar_l, kTrans);
    inv_covars_[current_components].CopyFromSp(inv_covars_[max_idx]);
    means_invcovars_.Row(current_components).CopyFromVec(means_invcovars_.Row(
        max_idx));
    means_invcovars_.Row(current_components).AddVec(perturb_factor, rand_vec);
    means_invcovars_.Row(max_idx).AddVec(-perturb_factor, rand_vec);
    current_components++;
  }
  ComputeGconsts();
}

void FullGmm::Merge(int32 target_components) {
  if (target_components <= 0 || NumGauss() < target_components) {
    KALDI_ERR << "Invalid argument for target number of Gaussians (="
        << target_components << ")";
  }
  if (NumGauss() == target_components) {
    KALDI_WARN << "No components merged, as target = total.";
    return;
  }

  int32 num_comp = NumGauss(), dim = Dim();

  if (target_components == 1) {  // global mean and variance
    Vector<BaseFloat> weights(weights_);
    // Undo variance inversion and multiplication of mean by this
    std::vector<SpMatrix<BaseFloat> > covars(num_comp);
    Matrix<BaseFloat> means(num_comp, dim);
    for (int32 i = 0; i < num_comp; ++i) {
      covars[i].Resize(dim);
      covars[i].CopyFromSp(inv_covars_[i]);
      covars[i].InvertDouble();
      means.Row(i).AddSpVec(1.0, covars[i], means_invcovars_.Row(i), 0.0);
      covars[i].AddVec2(1.0, means.Row(i));
    }

    // Slightly more efficient than calling this->Resize(1, dim)
    gconsts_.Resize(1);
    weights_.Resize(1);
    means_invcovars_.Resize(1, dim);
    inv_covars_.resize(1);
    inv_covars_[0].Resize(dim);
    Vector<BaseFloat> tmp_mean(dim);

    for (int32 i = 0; i < num_comp; ++i) {
      weights_(0) += weights(i);
      tmp_mean.AddVec(weights(i), means.Row(i));
      inv_covars_[0].AddSp(weights(i), covars[i]);
    }
    if (!ApproxEqual(weights_(0), 1.0, 1e-6)) {
      KALDI_WARN << "Weights sum to " << weights_(0) << ": rescaling.";
      tmp_mean.Scale(weights_(0));
      inv_covars_[0].Scale(weights_(0));
      weights_(0) = 1.0;
    }
    inv_covars_[0].AddVec2(-1.0, tmp_mean);
    inv_covars_[0].InvertDouble();
    means_invcovars_.Row(0).AddSpVec(1.0, inv_covars_[0], tmp_mean, 0.0);
    ComputeGconsts();
    return;
  }

  // If more than 1 merged component is required, use the hierarchical
  // clustering of components that lead to the smallest decrease in likelihood.
  std::vector<bool> discarded_component(num_comp);
  Vector<BaseFloat> logdet(num_comp);   // logdet for each component
  logdet.SetZero();
  for (int32 i = 0; i < num_comp; ++i) {
    discarded_component[i] = false;
    logdet(i) += 0.5 * inv_covars_[i].LogPosDefDet();
    // +0.5 because var is inverted
  }

  // Undo variance inversion and multiplication of mean by this
  // Makes copy of means and vars for all components - memory inefficient?
  std::vector<SpMatrix<BaseFloat> > vars(num_comp);
  Matrix<BaseFloat> means(num_comp, dim);
  for (int32 i = 0; i < num_comp; ++i) {
    vars[i].Resize(dim);
    vars[i].CopyFromSp(inv_covars_[i]);
    vars[i].InvertDouble();
    means.Row(i).AddSpVec(1.0, vars[i], means_invcovars_.Row(i), 0.0);

    // add means square to variances; get second-order stats
    // (normalized by zero-order stats)
    vars[i].AddVec2(1.0, means.Row(i));
  }

  // compute change of likelihood for all combinations of components
  SpMatrix<BaseFloat> delta_like(num_comp);
  for (int32 i = 0; i < num_comp; ++i) {
    for (int32 j = 0; j < i; ++j) {
      BaseFloat w1 = weights_(i), w2 = weights_(j), w_sum = w1 + w2;
      BaseFloat merged_logdet = merged_components_logdet(w1, w2,
        means.Row(i), means.Row(j), vars[i], vars[j]);
      delta_like(i, j) = w_sum * merged_logdet
        - w1 * logdet(i) - w2 * logdet(j);
    }
  }

  // Merge components with smallest impact on the loglike
  for (int32 removed = 0; removed < num_comp - target_components; ++removed) {
    // Search for the least significant change in likelihood
    // (maximum of negative delta_likes)
    BaseFloat max_delta_like = -std::numeric_limits<BaseFloat>::max();
    int32 max_i = 0, max_j = 0;
    for (int32 i = 0; i < NumGauss(); ++i) {
      if (discarded_component[i]) continue;
      for (int32 j = 0; j < i; ++j) {
        if (discarded_component[j]) continue;
        if (delta_like(i, j) > max_delta_like) {
          max_delta_like = delta_like(i, j);
          max_i = i;
          max_j = j;
        }
      }
    }

    // make sure that different components will be merged
    assert(max_i != max_j);

    // Merge components
    BaseFloat w1 = weights_(max_i), w2 = weights_(max_j);
    BaseFloat w_sum = w1 + w2;
    // merge means
    means.Row(max_i).AddVec(w2/w1, means.Row(max_j));
    means.Row(max_i).Scale(w1/w_sum);
    // merge vars
    vars[max_i].AddSp(w2/w1, vars[max_j]);
    vars[max_i].Scale(w1/w_sum);
    // merge weights
    weights_(max_i) = w_sum;

    // Update gmm for merged component
    // copy second-order stats (normalized by zero-order stats)
    inv_covars_[max_i].CopyFromSp(vars[max_i]);
    // centralize
    inv_covars_[max_i].AddVec2(-1.0, means.Row(max_i));
    // invert
    inv_covars_[max_i].InvertDouble();
    // copy first-order stats (normalized by zero-order stats)
    // and multiply by inv_vars
    means_invcovars_.Row(max_i).AddSpVec(1.0, inv_covars_[max_i],
      means.Row(max_i), 0.0);

    // Update logdet for merged component
    logdet(max_i) += 0.5 * inv_covars_[max_i].LogPosDefDet();
    // +0.5 because var is inverted

    // Label the removed component as discarded
    discarded_component[max_j] = true;

    // Update delta_like for merged component
    for (int32 j = 0; j < num_comp; ++j) {
      if ((j == max_i) || (discarded_component[j])) continue;
      BaseFloat w1 = weights_(max_i), w2 = weights_(j), w_sum = w1 + w2;
      BaseFloat merged_logdet = merged_components_logdet(w1, w2,
        means.Row(max_i), means.Row(j), vars[max_i], vars[j]);
      delta_like(max_i, j) = w_sum * merged_logdet
        - w1 * logdet(max_i) - w2 * logdet(j);
      // doesn't respect lower triangular indeces,
      // relies on implicitly performed swap of coordinates if necessary
    }
  }

  // Remove the consumed components
  int32 m = 0;
  for (int32 i = 0; i < num_comp; ++i) {
    if (discarded_component[i]) {
      weights_.RemoveElement(m);
      means_invcovars_.RemoveRow(m);
      inv_covars_.erase(inv_covars_.begin() + m);
    } else {
      ++m;
    }
  }

  ComputeGconsts();
}

BaseFloat FullGmm::merged_components_logdet(BaseFloat w1, BaseFloat w2,
                                            const VectorBase<BaseFloat> &f1,
                                            const VectorBase<BaseFloat> &f2,
                                            const SpMatrix<BaseFloat> &s1,
                                            const SpMatrix<BaseFloat> &s2)
                                            const {
  int32 dim = f1.Dim();
  Vector<BaseFloat> tmp_mean(dim);
  SpMatrix<BaseFloat> tmp_var(dim);
  BaseFloat merged_logdet = 0.0;

  BaseFloat w_sum = w1 + w2;
  tmp_mean.CopyFromVec(f1);
  tmp_mean.AddVec(w2/w1, f2);
  tmp_mean.Scale(w1/w_sum);
  tmp_var.CopyFromSp(s1);
  tmp_var.AddSp(w2/w1, s2);
  tmp_var.Scale(w1/w_sum);
  tmp_var.AddVec2(-1.0, tmp_mean);
  merged_logdet -= 0.5 * tmp_var.LogPosDefDet();
  // -0.5 because var is not inverted
  return merged_logdet;
}

BaseFloat FullGmm::ComponentLogLikelihood(const VectorBase<BaseFloat> &data,
                                          int32 comp_id) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";
  if (data.Dim() != Dim()) {
    KALDI_ERR << "DiagGmm::ComponentLogLikelihood, dimension "
        << "mismatch" << (data.Dim()) << "vs. "<< (Dim());
  }
  BaseFloat loglike;

  // loglike =  means * inv(vars) * data.
  loglike = VecVec(means_invcovars_.Row(comp_id), data);
  // loglike += -0.5 * tr(data*data'*inv(covar))
  loglike -= 0.5 * VecSpVec(data, inv_covars_[comp_id], data);
  return loglike + gconsts_(comp_id);
}



// Gets likelihood of data given this.
BaseFloat FullGmm::LogLikelihood(const VectorBase<BaseFloat> &data) const {
  Vector<BaseFloat> loglikes;
  LogLikelihoods(data, &loglikes);
  BaseFloat log_sum = loglikes.LogSumExp();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  return log_sum;
}

void FullGmm::LogLikelihoods(const VectorBase<BaseFloat> &data,
                             Vector<BaseFloat> *loglikes) const {
  loglikes->Resize(gconsts_.Dim(), kUndefined);
  loglikes->CopyFromVec(gconsts_);
  int32 dim = Dim();
  if (data.Dim() != dim) {
    KALDI_ERR << "DiagGmm::ComponentLogLikelihood, dimension "
        << "mismatch" << (data.Dim()) << "vs. "<< (Dim());
  }
  SpMatrix<BaseFloat> data_sq(dim);  // Initialize and make zero
  data_sq.AddVec2(1.0, data);
  // The following enables an optimization below: TraceSpSpLower, which is
  // just like a dot product internally.
  data_sq.ScaleDiag(0.5);

  // loglikes += mean' * inv(covar) * data.
  loglikes->AddMatVec(1.0, means_invcovars_, kNoTrans, data, 1.0);
  // loglikes -= 0.5 * data'*inv(covar)*data = 0.5 * tr(data*data'*inv(covar))
  int32 num_comp = NumGauss();
  for (int32 mix = 0; mix < num_comp; ++mix) {
    // was: (*loglikes)(mix) -= 0.5 * TraceSpSp(data_sq, inv_covars_[mix]);
    (*loglikes)(mix) -= TraceSpSpLower(data_sq, inv_covars_[mix]);
  }
}

// Gets likelihood of data given this. Also provides per-Gaussian posteriors.
BaseFloat FullGmm::ComponentPosteriors(const VectorBase<BaseFloat> &data,
                                       Vector<BaseFloat> *posterior) const {
  if (posterior == NULL) KALDI_ERR << "NULL pointer passed as return argument.";
  Vector<BaseFloat> loglikes;
  LogLikelihoods(data, &loglikes);
  BaseFloat log_sum = loglikes.ApplySoftMax();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  posterior->CopyFromVec(loglikes);
  return log_sum;
}

void FullGmm::RemoveComponent(int32 gauss) {
  KALDI_ASSERT(gauss < NumGauss());

  weights_.RemoveElement(gauss);
  gconsts_.RemoveElement(gauss);
  means_invcovars_.RemoveRow(gauss);
  inv_covars_.erase(inv_covars_.begin() + gauss);
  BaseFloat sum_weights = weights_.Sum();
  weights_.Scale(1/sum_weights);
  valid_gconsts_ = false;
}

void FullGmm::RemoveComponents(const std::vector<int32> &gauss_in) {
  std::vector<int32> gauss(gauss_in);
  std::sort(gauss.begin(), gauss.end());
  KALDI_ASSERT(IsSortedAndUniq(gauss));
  // If efficiency is later an issue, will code this specially (unlikely,
  // except for quite large GMMs).
  for (size_t i = 0; i < gauss.size(); i++) {
    RemoveComponent(gauss[i]);
    for (size_t j = i + 1; j < gauss.size(); j++)
      gauss[j]--;
  }
}

void FullGmm::Write(std::ostream &out_stream, bool binary) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before writing the model.";
  WriteMarker(out_stream, binary, "<FullGMM>");
  if (!binary) out_stream << "\n";
  WriteMarker(out_stream, binary, "<GCONSTS>");
  gconsts_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<WEIGHTS>");
  weights_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<MEANS_INVCOVARS>");
  means_invcovars_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<INV_COVARS>");
  for (int32 i = 0; i < NumGauss(); ++i) {
    inv_covars_[i].Write(out_stream, binary);
  }
  WriteMarker(out_stream, binary, "</FullGMM>");
  if (!binary) out_stream << "\n";
}

std::ostream & operator <<(std::ostream & out_stream,
                           const kaldi::FullGmm &gmm) {
  gmm.Write(out_stream, false);
  return out_stream;
}

void FullGmm::Read(std::istream &in_stream, bool binary) {
//  ExpectMarker(in_stream, binary, "<FullGMMBegin>");
  std::string marker;
  ReadMarker(in_stream, binary, &marker);
  // <FullGMMBegin> is for compatibility. Will be deleted later
  if (marker != "<FullGMMBegin>" && marker != "<FullGMM>")
    KALDI_ERR << "Expected <FullGMM>, got " << marker;
//  ExpectMarker(in_stream, binary, "<GCONSTS>");
  ReadMarker(in_stream, binary, &marker);
  if (marker == "<GCONSTS>") {  // The gconsts are optional.
    gconsts_.Read(in_stream, binary);
    ExpectMarker(in_stream, binary, "<WEIGHTS>");
  } else {
    if (marker != "<WEIGHTS>")
      KALDI_ERR << "DiagGmm::Read, expected <WEIGHTS> or <GCONSTS>, got "
                << marker;
  }
  weights_.Read(in_stream, binary);
  ExpectMarker(in_stream, binary, "<MEANS_INVCOVARS>");
  means_invcovars_.Read(in_stream, binary);
  ExpectMarker(in_stream, binary, "<INV_COVARS>");
  int32 ncomp = weights_.Dim(), dim = means_invcovars_.NumCols();
  ResizeInvCovars(ncomp, dim);
  for (int32 i = 0; i < ncomp; ++i) {
    inv_covars_[i].Read(in_stream, binary);
  }
//  ExpectMarker(in_stream, binary, "<FullGMMEnd>");
  ReadMarker(in_stream, binary, &marker);
  // <FullGMMEnd> is for compatibility. Will be deleted later
  if (marker != "<FullGMMEnd>" && marker != "</FullGMM>")
    KALDI_ERR << "Expected </FullGMM>, got " << marker;

  ComputeGconsts();  // safer option than trusting the read gconsts
}

std::istream & operator >>(std::istream & in_stream, kaldi::FullGmm &gmm) {
  gmm.Read(in_stream, false);  // false == non-binary.
  return in_stream;
}

}  // End namespace kaldi
