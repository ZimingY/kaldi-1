#include "fb/simple-backward.h"
#include "fb/test-utils.h"
#include "util/kaldi-io.h"

#define EQ_EPS 1E-7

int main(int argc, char** argv) {
  fst::VectorFst<fst::StdArc> fst;
  kaldi::unittest::DummyDecodable dec;
  kaldi::SimpleBackward backward(fst, 1E9, 1E-9);

  // Dummy WFST with a single non-final state. Empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, false);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(backward.Backward(&dec) == false);
  kaldi::AssertEqual(backward.TotalCost(), -kaldi::kLogZeroDouble, EQ_EPS);

  // Dummy WFST with a single final state. Empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, true);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), 0.0, EQ_EPS);

  // Dummy WFST with a single final state. Non-empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, true);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(backward.Backward(&dec) == false);
  kaldi::AssertEqual(backward.TotalCost(), -kaldi::kLogZeroDouble, EQ_EPS);

  // Arbitrary WFST. Empty input sequence.
  kaldi::unittest::CreateWFST_Arbitrary(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), -log(1.75), EQ_EPS);

  // Arbitrary WFST. Arbitrary input sequence.
  kaldi::unittest::CreateWFST_Arbitrary(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), -log(145.9125), EQ_EPS);

  KALDI_ASSERT(backward.GetTable().size() == 4);

  KALDI_ASSERT(backward.GetTable()[3].count(1));
  KALDI_ASSERT(backward.GetTable()[3].count(2));
  kaldi::AssertEqual(
      backward.GetTable()[3].find(1)->second, -log(11.5375), EQ_EPS);
  kaldi::AssertEqual(
      backward.GetTable()[3].find(2)->second, -log(6.5625), EQ_EPS);

  KALDI_ASSERT(backward.GetTable()[2].count(1));
  KALDI_ASSERT(backward.GetTable()[2].count(2) == 0);
  kaldi::AssertEqual(
      backward.GetTable()[2].find(1)->second, -log(89.98125), EQ_EPS);

  KALDI_ASSERT(backward.GetTable()[1].count(1) == 0);
  KALDI_ASSERT(backward.GetTable()[1].count(2));
  kaldi::AssertEqual(
      backward.GetTable()[1].find(2)->second, -log(135.028125), EQ_EPS);

  KALDI_ASSERT(backward.GetTable()[0].count(1));
  KALDI_ASSERT(backward.GetTable()[0].count(2));
  kaldi::AssertEqual(
      backward.GetTable()[0].find(1)->second, -log(97.97109375), EQ_EPS);
  kaldi::AssertEqual(
      backward.GetTable()[0].find(2)->second, -log(94.39921875), EQ_EPS);


  /////////////////////////////////////////
  // A WFST with an epsilon loop.
  /////////////////////////////////////////

  // Empty input sequence
  kaldi::unittest::CreateWFST_EpsilonLoop(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  // kaldi::AssertEqual will fail, since log(1.0) is exactly 0.0, but
  // the total cost is something like 6.4E-8. kaldi::AssertEqual checks
  // relative error, and it will fail in this case, since one term is 0.0
  KALDI_ASSERT(fabs(backward.TotalCost() - log(1.0)) < EQ_EPS);

  // Arbitrary input sequence
  kaldi::unittest::CreateWFST_EpsilonLoop(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), -log(0.375), EQ_EPS);

  KALDI_ASSERT(backward.GetTable().size() == 4);

  KALDI_ASSERT(backward.GetTable()[3].count(1));
  KALDI_ASSERT(backward.GetTable()[3].count(2));
  kaldi::AssertEqual(
      backward.GetTable()[3].find(1)->second, -log(0.65), EQ_EPS);
  kaldi::AssertEqual(
      backward.GetTable()[3].find(2)->second, -log(0.35), EQ_EPS);

  KALDI_ASSERT(backward.GetTable()[2].count(1));
  KALDI_ASSERT(backward.GetTable()[2].count(2) == 0);
  kaldi::AssertEqual(
      backward.GetTable()[2].find(1)->second, -log(1.5), EQ_EPS);

  KALDI_ASSERT(backward.GetTable()[1].count(1) == 0);
  KALDI_ASSERT(backward.GetTable()[1].count(2));
  kaldi::AssertEqual(
      backward.GetTable()[1].find(2)->second, -log(0.75), EQ_EPS);

  KALDI_ASSERT(backward.GetTable()[0].count(1));
  KALDI_ASSERT(backward.GetTable()[0].count(2));
  kaldi::AssertEqual(
      backward.GetTable()[0].find(1)->second, -log(0.1875), EQ_EPS);
  kaldi::AssertEqual(
      backward.GetTable()[0].find(2)->second, -log(0.1875), EQ_EPS);

  std::cout << "Test OK.\n";
  return 0;
}
