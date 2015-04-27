#include "fb/simple-backward.h"
#include "fb/test-utils.h"
#include "util/kaldi-io.h"

using fst::VectorFst;
using fst::StdArc;
typedef fst::StdArc::Label Label;

namespace kaldi {
namespace unittest {

void PrintBackwardTable(const SimpleBackward& backward) {
  typedef unordered_map<Label, double> label_weight_map;
  typedef vector<label_weight_map> fwd_table_type;
  fwd_table_type table = backward.GetTable();
  for (size_t t = 0; t < table.size(); ++t) {
    for (label_weight_map::const_iterator e = table[t].begin();
         e != table[t].end(); ++e) {
      std::cout << "F[" << t << "," << e->first << "] = "
                << e->second << std::endl;
    }
  }
}

void CreateWFST_SingleState(VectorFst<StdArc>* fst) {
  fst->DeleteStates();

  fst->AddState();  // State 0

  fst->SetStart(0);
  fst->SetFinal(0, -log(0.5));

  fst->AddArc(0, StdArc(1, 1, -log(0.5), 0));  // self-loop in state 0, label 1
  fst->AddArc(0, StdArc(2, 2, -log(0.5), 0));  // self-loop in state 0, label 2
}

void CreateWFST_TwoStatesWithEpsilonTransition(VectorFst<StdArc>* fst) {
  fst->DeleteStates();

  fst->AddState();  // State 0
  fst->AddState();  // State 1

  fst->SetStart(0);
  fst->SetFinal(0, -log(1.0));
  fst->SetFinal(1, -log(0.5));

  fst->AddArc(0, StdArc(0, 0, -log(0.5), 1));  // epsilon transition
  fst->AddArc(0, StdArc(1, 1, -log(0.3), 0));  // self-loop in state 0, label 1
  fst->AddArc(1, StdArc(2, 2, -log(0.5), 1));  // self-loop in state 1, label 2
}

void CreateWFST_LinearHMM(VectorFst<StdArc>* fst) {
  fst->DeleteStates();

  fst->AddState();  // State 0
  fst->AddState();  // State 1
  fst->AddState();  // State 2

  fst->SetStart(0);
  fst->SetFinal(2, 0.0);  // Only final state is 2, with prob = 1.0

  fst->AddArc(0, StdArc(1, 1, -log(0.3), 0));
  fst->AddArc(0, StdArc(2, 2, -log(0.1), 0));
  fst->AddArc(0, StdArc(1, 1, -log(0.2), 1));
  fst->AddArc(0, StdArc(2, 2, -log(0.4), 1));

  fst->AddArc(1, StdArc(2, 2, -log(0.2), 1));
  fst->AddArc(1, StdArc(3, 3, -log(0.3), 1));
  fst->AddArc(1, StdArc(2, 2, -log(0.3), 2));
  fst->AddArc(1, StdArc(3, 3, -log(0.2), 2));
}

void CreateWFST_Arbitrary(VectorFst<StdArc>* fst) {
  fst->DeleteStates();

  fst->AddState();
  fst->AddState();
  fst->AddState();
  fst->AddState();

  fst->SetStart(0);
  fst->SetFinal(0, -log(1.0));
  fst->SetFinal(3, -log(0.5));

  fst->AddArc(0, StdArc(1, 1, -log(1.0), 0));
  fst->AddArc(0, StdArc(2, 2, -log(1.2), 0));
  fst->AddArc(0, StdArc(1, 1, -log(0.5), 1));
  fst->AddArc(0, StdArc(0, 0, -log(1.0), 1));

  fst->AddArc(1, StdArc(1, 1, -log(0.5), 1));
  fst->AddArc(1, StdArc(0, 0, -log(1.0), 2));
  fst->AddArc(1, StdArc(1, 1, -log(0.5), 3));
  fst->AddArc(1, StdArc(2, 2, -log(0.2), 3));

  fst->AddArc(2, StdArc(1, 1, -log(1.0), 0));
  fst->AddArc(2, StdArc(2, 2, -log(0.1), 3));

  fst->AddArc(3, StdArc(2, 2, -log(0.5), 2));
}

void CreateObservation_Empty(DummyDecodable* decodable) {
  decodable->Init(3, 0, vector<BaseFloat>());
}

void CreateObservation_Short(DummyDecodable* decodable) {
  vector<BaseFloat> observations(3 * 1);
  observations[0] = log(0.6);  // input label = 1
  observations[1] = log(0.4);  // input label = 2
  observations[2] = log(0.0);  // input label = 3, (cannot produce this)

  decodable->Init(3, 1, observations);
}

void CreateObservation_Large(DummyDecodable* decodable) {
  vector<BaseFloat> observations(3 * 3);
  // frame = 0
  observations[0] = log(0.1);
  observations[1] = log(0.9);
  observations[2] = log(0.0);  // input label = 3, (cannot produce this)
  // frame = 1
  observations[3] = log(0.5);
  observations[4] = log(0.5);
  observations[5] = log(0.0);  // input label = 3, (cannot produce this)
  // frame = 2
  observations[6] = log(0.2);
  observations[7] = log(0.5);
  observations[8] = log(0.3);

  decodable->Init(3, 3, observations);
}

void CreateObservation_LargeNoStochastic(DummyDecodable* decodable) {
  vector<BaseFloat> observations(3 * 4);
  // frame = 0
  observations[0] = log(0.2);
  observations[1] = log(0.5);
  observations[2] = log(1.0);
  // frame = 1
  observations[3] = log(1.0);
  observations[4] = log(1.0);
  observations[5] = log(1.0);
  // frame = 2
  observations[6] = log(0.0);
  observations[7] = log(0.5);
  observations[8] = log(0.3);
  // frame = 3
  observations[9]  = log(0.3);
  observations[10] = log(0.7);
  observations[11] = log(0.3);

  decodable->Init(3, 4, observations);
}

}  // namespace unittest
}  // namespace kaldi

int main(int argc, char** argv) {
  VectorFst<StdArc> fst;
  kaldi::unittest::DummyDecodable decodable;
  kaldi::SimpleBackward backward(fst, 1E+9, 1E-9);

  void (*fsts[])(VectorFst<StdArc>*) = {
    kaldi::unittest::CreateWFST_SingleState,
    kaldi::unittest::CreateWFST_TwoStatesWithEpsilonTransition,
    kaldi::unittest::CreateWFST_Arbitrary
  };

  void (*deco[])(kaldi::unittest::DummyDecodable*) = {
    kaldi::unittest::CreateObservation_Empty,
    kaldi::unittest::CreateObservation_Short,
    kaldi::unittest::CreateObservation_Large,
    kaldi::unittest::CreateObservation_LargeNoStochastic
  };

  const double expected_total_cost[] = {
    -log(0.5), -log(0.25), -log(0.04375), -log(0.021875),
    -log(1.25), -log(0.275), -log(0.00811875), -log(0.003784375),
    -log(1.0), -log(1.89)
  };

  const int NF = 3;
  const int ND = 4;

  for (int f = 0; f < NF; ++f) {
    fsts[f](&fst);
    for (int d = 0; d < 2; ++d) {
      deco[d](&decodable);
      KALDI_ASSERT(backward.Backward(&decodable));
      KALDI_LOG << f << " " << d << " "
                << backward.TotalCost() << " " << expected_total_cost[f * ND + d];
      kaldi::AssertEqual(
          backward.TotalCost(), expected_total_cost[f * ND + d], 1E-9);
    }
  }

  std::cout << "Test OK.\n";
  return 0;
}
