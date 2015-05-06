#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "gmm/decodable-am-diag-gmm.h"

#include "fb/simple-common.h"
#include "fb/simple-forward.h"
#include "fb/simple-backward.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::VectorFst;
    using fst::StdArc;
    typedef fst::StdArc::Label Label;

    const char *usage =
        "gmm-fb-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.psts.ark\n";

    ParseOptions po(usage);
    BaseFloat beam = std::numeric_limits<BaseFloat>::infinity();
    BaseFloat delta = 0.000976562; // same delta as in fstshortestdistance
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;

    po.Register("beam", &beam, "Beam prunning threshold");
    po.Register("delta", &delta, "Comparison delta (see fstshortestdistance)");
    po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("self-loop-scale", &self_loop_scale,
                "Scale of self-loop versus non-self-loop log probs "
                "[relative to acoustics]");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        posterior_wspecifier = po.GetArg(4);

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
    PosteriorWriter posterior_writer(posterior_wspecifier);

    int num_done = 0, num_err = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string utt = fst_reader.Key();
      if (!feature_reader.HasKey(utt)) {
        num_err++;
        KALDI_WARN << "No features for utterance " << utt;
      } else {
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        VectorFst<StdArc> fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             &fst);
        }

        SimpleForward forwarder(fst, beam, delta);
        SimpleBackward backwarder(fst, beam, delta);
        DecodableAmDiagGmm gmm_decodable(am_gmm, trans_model, features);

        if (!forwarder.Forward(&gmm_decodable)) {
          KALDI_WARN << "Forward did not reach any final state for utt "
                     << utt;
          ++num_err;
          continue;
        }
        if (!backwarder.Backward(&gmm_decodable)) {
          KALDI_WARN << "Backward did not reach the start state for utt "
                     << utt;
          ++num_err;
          continue;
        }
        const double lkh = ComputeLikelihood(
            forwarder.GetTable()[0], backwarder.GetTable()[0]);
        const int64 nfrm = forwarder.NumFramesDecoded();
#ifdef KALDI_PARANOID
        KALDI_ASSERT(forwarder.NumFramesDecoded() ==
                     backwarder.NumFramesDecoded());
        KALDI_ASSERT(forwarder.GetTable().size() ==
                     backwarder.GetTable().size());
        const double lkh_back = -backwarder.TotalCost();
        kaldi::AssertEqual(lkh, lkh_back);
#endif
        KALDI_LOG << "Processing Data: " << utt;
        KALDI_LOG << "Utterance prob per frame = " << (lkh / nfrm);
        tot_like += lkh;
        frame_count += nfrm;
        ++num_done;

        std::vector<LabelMap> pst_map;
        ComputeLabelsPosterior(
            fst, forwarder.GetTable(), backwarder.GetTable(), &gmm_decodable,
            &pst_map);
        Posterior pst(pst_map.size());
        for (size_t t = 0; t < pst.size(); ++t) {
          for(LabelMap::const_iterator l = pst_map[t].begin();
              l != pst_map[t].end(); ++l) {
            pst[t].push_back(*l);
            pst[t].back().second = exp(pst[t].back().second);
          }
        }
        posterior_writer.Write(utt, pst);
      }
    }
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like / frame_count) << " over " << frame_count
              << " frames";
    KALDI_LOG << "Done " << num_done << ", errors on " << num_err;


  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }

}
