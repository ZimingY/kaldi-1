// nnet-dp/am-nnet1.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet-dp/am-nnet1.h"
#include "thread/kaldi-thread.h"

namespace kaldi {


/*
  At input, leaf_mapping maps pdf-ids to indices that correspond to leaves
  in a coarser tree.  These indices should be zero-based and dense.  This
  function computes a map, "l1_to_category", that maps from these indexes
  to a 1-based category index; this mapping isn't just an "add-one" mapping,
  it's also a reordering that ensures that singleton categories are placed
  last.
 */
void AmNnet1::ComputeCategoryInfo(const std::vector<int32> &leaf_mapping) {
  leaf_mapping_ = leaf_mapping;
  l1_to_pdfs_.clear();
  pdf_to_sub_index_.resize(leaf_mapping.size());
  for (int32 i = 0; i < leaf_mapping.size(); i++) {
    int32 idx = leaf_mapping[i];
    KALDI_ASSERT(idx >= 0);
    if (idx >= l1_to_pdfs_.size()) l1_to_pdfs_.resize(idx + 1);
    pdf_to_sub_index_[i] = l1_to_pdfs_[idx].size();
    l1_to_pdfs_[idx].push_back(i);
  }
  for (int32 i = 0; i < l1_to_pdfs_.size(); i++)
    if (l1_to_pdfs_[i].empty())
      KALDI_ERR << "leaf_mapping has empty index " << i << " in coarse tree. "
                << "Input to this program is invalid.";
  KALDI_ASSERT(l1_to_pdfs_.size() > 1); // Don't allow just a single leaf
  // in level-1 tree.

  int32 cur_category_index = 1; // We'll now be computing the mapping from level-one
  // index to category index.  The category indexes start from one, not zero,
  // and don't include "singleton categories".
  l1_to_category_.resize(l1_to_pdfs_.size());
  for (int32 l1 = 0; l1 < l1_to_pdfs_.size(); l1++) {
    if (l1_to_pdfs_[l1].size() == 1 )
      l1_to_category_[l1] = -1; // singleton category; no category index.
    else
      l1_to_category_[l1] = cur_category_index++;
  }

  category_sizes_.resize(cur_category_index);
  category_sizes_[0] = l1_to_category_.size(); // number of l1 indices.
  for (int32 i = 0; i < l1_to_category_.size(); i++) {
    int32 category = l1_to_category_[i];
    if (category != -1)
      category_sizes_[category] = l1_to_pdfs_[i].size();
  }
  KALDI_ASSERT(*std::min_element(category_sizes_.begin(),
                                 category_sizes_.end()) > 0);
}

AmNnet1::AmNnet1(const Nnet1InitConfig &config,
                 const std::vector<int32> &leaf_mapping) {
  ComputeCategoryInfo(leaf_mapping);
  Nnet1InitInfo init_info(config, category_sizes_);
  nnet_.Init(init_info);
}

void AmNnet1::Write(std::ostream &os, bool binary) const {
  // Note: all the mappings are derived from leaf_mapping_, so we don't really
  // have to write them, but it might be easier for debugging purposes to have
  // them written out.
  WriteToken(os, binary, "<AmNnet1>");
  WriteToken(os, binary, "<LeafMapping>");
  WriteIntegerVector(os, binary, leaf_mapping_);
  WriteToken(os, binary, "<L1ToCategory>");
  WriteIntegerVector(os, binary, l1_to_category_);
  // Don't include l1_to_pdfs_, as it's more complex to write; we'll
  // work it out from other data.
  WriteToken(os, binary, "<PdfToSubIndex>");
  WriteIntegerVector(os, binary, pdf_to_sub_index_);
  WriteToken(os, binary, "<CategorySize>");
  WriteIntegerVector(os, binary, category_sizes_);
  nnet_.Write(os, binary);
  WriteToken(os, binary, "</AmNnet1>");  
}


void AmNnet1::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<AmNnet1>");
  ExpectToken(is, binary, "<LeafMapping>");
  ReadIntegerVector(is, binary, &leaf_mapping_);
  ExpectToken(is, binary, "<L1ToCategory>");
  ReadIntegerVector(is, binary, &l1_to_category_);
  // Don't include l1_to_pdfs_, as it's more complex to write; we'll
  // work it out from other data.
  ExpectToken(is, binary, "<PdfToSubIndex>");
  ReadIntegerVector(is, binary, &pdf_to_sub_index_);
  ExpectToken(is, binary, "<CategorySize>");
  ReadIntegerVector(is, binary, &category_sizes_);
  ComputeCategoryInfo(leaf_mapping_); // This will compute various quantities,
  // including l1_to_pdfs_ which we didn't read in.
  nnet_.Read(is, binary);
  ExpectToken(is, binary, "</AmNnet1>");  
}

void AmNnet1::GetCategoryInfo(int32 pdf_id,
                              std::vector<std::pair<int32, int32> > *pairs) const {
  pairs->clear();
  KALDI_ASSERT(pdf_id >= 0 && pdf_id < leaf_mapping_.size());
  int32 l1 = leaf_mapping_[pdf_id];
  pairs->push_back(std::make_pair(0, l1)); // index within category zero.
  KALDI_ASSERT(l1 >= 0 && l1 < l1_to_category_.size());
  int32 category = l1_to_category_[l1];
  if (category != -1)
    pairs->push_back(std::make_pair(category, pdf_to_sub_index_[pdf_id]));
}

void AmNnet1::GetFinalLayerSets(std::vector<std::vector<int32> > *sets) {
  // one set containing category zero and another set containing
  // all the rest.
  sets->clear();
  sets->resize(2);
  (*sets)[0].push_back(0);
  for (int32 i = 1; i < category_sizes_.size(); i++)
    (*sets)[1].push_back(i);
}

// Get a vector of priors indexed by pdf-id; this is derived from
// the "occupancy" data members stored in the softmax layers.
void AmNnet1::GetPriors(Vector<BaseFloat> *priors) const {
  if (priors_.Dim() != 0) { // Someone has called FixPriors() before now.
    // E.g. we've started doing MMI training.
    *priors = priors_;
    return;
  }
  int32 num_pdfs = leaf_mapping_.size(),
      num_categories = nnet_.NumCategories(); // <= number of
  // l1 indexes + 1.
  std::vector<Vector<BaseFloat> > priors_for_category(num_categories); // zeroth
  // element will be empty.
  for (int32 category = 0; category < num_categories; category++)
    nnet_.GetPriorsForCategory(category, &priors_for_category[category]);
  priors->Resize(num_pdfs);
  Vector<BaseFloat> &l1_prior = priors_for_category[0];
  for (int32 pdf_id = 0; pdf_id < num_pdfs; pdf_id++) {
    int32 l1 = leaf_mapping_[pdf_id],
        category = l1_to_category_[l1],
        sub_index = pdf_to_sub_index_[pdf_id];
    BaseFloat prior_in_sub_tree;
    if (category == -1) prior_in_sub_tree = 1.0; // singleton sub-tree.
    else prior_in_sub_tree = priors_for_category[category](sub_index);
    (*priors)(pdf_id) = l1_prior(l1) * prior_in_sub_tree;
  }
  KALDI_ASSERT(fabs(1.0 - priors->Sum()) < 0.01);
}


} // namespace kaldi
