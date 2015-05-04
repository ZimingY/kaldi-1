#include "fb/simple-backward.h"
#include "fb/simple-forward.h"
#include "fb/test-utils.h"
#include "util/kaldi-io.h"

#define EQ_EPS 1E-6

int main(int argc, char** argv) {
  fst::VectorFst<fst::StdArc> fst;
  kaldi::unittest::DummyDecodable dec;
  kaldi::SimpleForward forward(fst, 1E20, 1E-9);
  kaldi::SimpleBackward backward(fst, 1E20, 1E-9);

  // Arbitrary WFST. Arbitrary input sequence.
  kaldi::unittest::CreateWFST_Arbitrary(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), backward.TotalCost(), EQ_EPS);

  std::cout << "Test OK.\n";
  return 0;
}
