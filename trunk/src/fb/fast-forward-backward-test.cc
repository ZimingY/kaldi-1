#include "fb/fast-forward-backward.h"
#include "fb/test-common.h"
#include "util/kaldi-io.h"

int main(int argc, char** argv) {
  fst::VectorFst<fst::StdArc> fst_fwd;
  fst::VectorFst<fst::StdArc> fst_bkw;
  kaldi::unittest::DummyDecodable dec;
  kaldi::FastForwardBackward fb(fst_fwd, fst_bkw, 1E20, 1E20, 1E-9);

  // Arbitrary WFST. Arbitrary input sequence.
  kaldi::unittest::CreateWFST_EpsilonBucle(&fst_fwd);
  // Reverse forward WFST to obtain backward's
  fst::Reverse(fst_fwd, &fst_bkw);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(fb.ForwardBackward(&dec));

  const vector<kaldi::LabelMap>& pst = fb.LabelPosteriors();
  const double pst_ref_1[] = {
    log(0.0), log(0.5), log(0.5),
    log(0.0), log(0.0), log(1.0),
    log(0.0), log(1.0), log(0.0),
    log(0.0), log(0.65), log(0.35)
  };
  kaldi::unittest::CheckLabelPosteriors(pst, pst_ref_1, 4, 3, FB_EQ_EPS);

  std::cout << "Test OK.\n";
  return 0;
}
