#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "gmm/decodable-am-diag-gmm.h"

#include "fb/simple-forward.h"
#include "fb/simple-backward.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::VectorFst;
    using fst::StdArc;
    typedef fst::StdArc::Label Label;

    const char *usage =
        "fb-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.fb\n";

    ParseOptions po(usage);
    BaseFloat beam = std::numeric_limits<BaseFloat>::infinity();
    BaseFloat delta = 1E-6;

    po.Register("beam", &beam, "Beam prunning threshold");
    po.Register("delta", &delta, "Comparison delta (see fstshortestdistance)");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        alignment_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }


    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

    int num_done = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    std::unordered_map<Label, double> occupancy;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string utt = fst_reader.Key();
      if (!feature_reader.HasKey(utt)) {
        num_err++;
        KALDI_WARN << "No features for utterance " << utt;
      } else {
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        VectorFst<StdArc> fst(fst_reader.Value());
        SimpleForward forwarder(fst, beam, delta);
        SimpleBackward backwarder(fst, beam, delta);
        DecodableAmDiagGmm gmm_decodable(am_gmm, trans_model, features);

        if (!forwarder.Forward(&gmm_decodable)) {
          KALDI_WARN << "Forward did not reach any final state for utt "
                     << fst_reader.Key();
          ++num_err;
          continue;
        }
        if (!backwarder.Backward(&gmm_decodable)) {
          KALDI_WARN << "Backward did not reach the start state for utt "
                     << fst_reader.Key();
          ++num_err;
          continue;
        }
        const double lkh = -forwarder.TotalCost();
        const int64 nfrm = forwarder.NumFramesDecoded();
#ifdef KALDI_PARANOID
        KALDI_ASSERT(forwarder.NumFramesDecoded() ==
                     backwarder.NumFramesDecoded());
        KALDI_ASSERT(forwarder.GetTable().size() ==
                     backwarder.GetTable().size());
        const double lkh_back = -backwarder.TotalCost();
        kaldi::AssertEqual(lkh, lkh_back);
#endif
        KALDI_LOG << "Processing Data: " << fst_reader.Key();
        KALDI_LOG << "Utterance prob per frame = " << (lkh / nfrm);
        tot_like += lkh;
        frame_count += nfrm;
        ++num_done;

        std::vector<std::unordered_map<Label, double> > pst =
            forwarder.GetTable();
        for (size_t t = 0; t < pst.size(); ++t) {
          // forward * backward
          for (std::unordered_map<Label,double>::const_iterator bi =
                   backwarder.GetTable()[t].begin();
               bi != backwarder.GetTable()[t].end(); ++bi) {
            double& ctl = pst[t].insert(make_pair(
                bi->first, -kaldi::kLogZeroDouble)).first->second;
            ctl += bi->second;
          }
          double sum_t = -kaldi::kLogZeroDouble;
          for (std::unordered_map<Label,double>::const_iterator it =
                   pst[t].begin(); it != pst[t].end(); ++it) {
            sum_t = -kaldi::LogAdd(-sum_t, -it->second);
          }
          KALDI_LOG << "sum(" << t << ") = " << sum_t;
          // normalize for time t
          for (std::unordered_map<Label,double>::iterator it = pst[t].begin();
               it != pst[t].end(); ++it) {
            it->second -= sum_t;
            KALDI_LOG << "N(" << fst_reader.Key() << "," << t << ","
                      << it->first << ") -> " << it->second;
            double& occ = occupancy.insert(make_pair(
                it->first, -kaldi::kLogZeroDouble)).first->second;
            occ = -kaldi::LogAdd(-occ, -it->second);
          }
        }

      }
    }
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like / frame_count) << " over " << frame_count
              << " frames";
    KALDI_LOG << "Done " << num_done << ", errors on " << num_err;

    for(std::unordered_map<Label,double>::const_iterator it = occupancy.begin();
        it != occupancy.end(); ++it) {
      KALDI_LOG << "Occupacy " << it->first << " = " << exp(-it->second);
    }

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }

}
