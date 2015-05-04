#include "fb/simple-forward.h"
#include "fb/test-utils.h"
#include "util/kaldi-io.h"

#define EQ_EPS 1E-6

int main(int argc, char** argv) {
  fst::VectorFst<fst::StdArc> fst;
  kaldi::unittest::DummyDecodable dec;
  kaldi::SimpleForward forward(fst, 1E9, 1E-9);

  // Dummy WFST with a single non-final state. Empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, false);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -kaldi::kLogZeroDouble, EQ_EPS);

  // Dummy WFST with a single final state. Empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, true);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), 0.0, EQ_EPS);

  // Dummy WFST with a single final state. Non-empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, true);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(forward.Forward(&dec) == false);
  kaldi::AssertEqual(forward.TotalCost(), -kaldi::kLogZeroDouble, EQ_EPS);

  // Arbitrary WFST. Empty input sequence.
  kaldi::unittest::CreateWFST_Arbitrary(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -log(1.75), EQ_EPS);

  // Arbitrary WFST. Arbitrary input sequence.
  kaldi::unittest::CreateWFST_Arbitrary(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -log(145.9125), EQ_EPS);

  KALDI_ASSERT(forward.GetTable().size() == 4);

  KALDI_ASSERT(forward.GetTable()[0].count(1));
  KALDI_ASSERT(forward.GetTable()[0].count(2));
  kaldi::AssertEqual(
      forward.GetTable()[0].find(1)->second, -log(5.9375), EQ_EPS);
  kaldi::AssertEqual(
      forward.GetTable()[0].find(2)->second, -log(5.5625), EQ_EPS);

  KALDI_ASSERT(forward.GetTable()[1].count(1) == 0);
  KALDI_ASSERT(forward.GetTable()[1].count(2));
  kaldi::AssertEqual(
      forward.GetTable()[1].find(2)->second, -log(19.5), EQ_EPS);

  KALDI_ASSERT(forward.GetTable()[2].count(1));
  KALDI_ASSERT(forward.GetTable()[2].count(2) == 0.0);
  kaldi::AssertEqual(
      forward.GetTable()[2].find(1)->second, -log(101.25), EQ_EPS);

  KALDI_ASSERT(forward.GetTable()[3].count(1));
  KALDI_ASSERT(forward.GetTable()[3].count(2));
  kaldi::AssertEqual(
      forward.GetTable()[3].find(1)->second, -log(196.4625), EQ_EPS);
  kaldi::AssertEqual(
      forward.GetTable()[3].find(2)->second, -log(95.550), EQ_EPS);


  /////////////////////////////////////////
  // A WFST with an epsilon loop.
  /////////////////////////////////////////

  // Empty input sequence
  kaldi::unittest::CreateWFST_EpsilonLoop(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  // kaldi::AssertEqual will fail, since log(1.0) is exactly 0.0, but
  // the total cost is something like 2E-7. kaldi::AssertEqual checks
  // relative error, and it will fail in this case, since one term is 0.0
  KALDI_ASSERT(fabs(forward.TotalCost() - log(1.0)) < EQ_EPS);

  // Arbitrary input sequence
  kaldi::unittest::CreateWFST_EpsilonLoop(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -log(0.375), EQ_EPS);

  KALDI_ASSERT(forward.GetTable().size() == 4);

  KALDI_ASSERT(forward.GetTable()[0].count(1));
  KALDI_ASSERT(forward.GetTable()[0].count(2));
  kaldi::AssertEqual(
      forward.GetTable()[0].find(1)->second, -log(0.5), EQ_EPS);
  kaldi::AssertEqual(
      forward.GetTable()[0].find(2)->second, -log(0.5), EQ_EPS);

  KALDI_ASSERT(forward.GetTable()[1].count(1) == 0.0);
  KALDI_ASSERT(forward.GetTable()[1].count(2));
  kaldi::AssertEqual(
      forward.GetTable()[1].find(2)->second, -log(0.5), EQ_EPS);

  KALDI_ASSERT(forward.GetTable()[2].count(1));
  KALDI_ASSERT(forward.GetTable()[2].count(2) == 0);
  kaldi::AssertEqual(
      forward.GetTable()[2].find(1)->second, -log(0.75), EQ_EPS);

  KALDI_ASSERT(forward.GetTable()[3].count(1));
  KALDI_ASSERT(forward.GetTable()[3].count(2));
  kaldi::AssertEqual(
      forward.GetTable()[3].find(1)->second, -log(0.4875), EQ_EPS);
  kaldi::AssertEqual(
      forward.GetTable()[3].find(2)->second, -log(0.2625), EQ_EPS);

  std::cout << "Test OK.\n";
  return 0;
}
