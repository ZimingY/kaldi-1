#include "decoder/simple-forward.h"
#include "util/kaldi-io.h"

using fst::VectorFst;
using fst::StdArc;
typedef fst::StdArc::Label Label;

namespace kaldi {
namespace unittest {

class DummyDecodable : public DecodableInterface {
 private:
  int32 num_states_;
  int32 num_frames_;
  vector<BaseFloat> observations_;

 public:
  DummyDecodable() : DecodableInterface(), num_states_(0), num_frames_(-1) { }

  void Init(int32 num_states, int32 num_frames,
            vector<BaseFloat>& observations) {
    KALDI_ASSERT(observations.size() == num_states * num_frames);
    num_states_ = num_states;
    num_frames_ = num_frames;
    observations_ = observations;
  }

  virtual BaseFloat LogLikelihood(int32 frame, int32 state_index) {
    KALDI_ASSERT(frame >= 0 && frame < NumFramesReady());
    KALDI_ASSERT(state_index > 0 && state_index <= NumIndices());
    return observations_[frame * num_states_ + state_index - 1];
  }

  virtual int32 NumFramesReady() const { return num_frames_; }

  virtual int32 NumIndices() const { return num_states_; }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }
};

void PrintForwardTable(const SimpleForward& forward) {
  typedef unordered_map<Label, double> label_weight_map;
  typedef vector<label_weight_map> fwd_table_type;
  fwd_table_type table = forward.GetTable();
  for (size_t t = 0; t < table.size(); ++t) {
    for (label_weight_map::const_iterator e = table[t].begin();
         e != table[t].end(); ++e) {
      std::cout << "F[" << t << "," << e->first << "] = "
                << e->second << std::endl;
    }
  }
}


void ForwardSingleStateNoEpsUnitTest() {
  // Fst with a single node, and emitting two symbols
  VectorFst<StdArc> fst;
  fst.AddState();
  fst.SetStart(0);
  fst.AddArc(0, StdArc(1, 1, -log(0.5), 0));
  fst.AddArc(0, StdArc(2, 2, -log(0.5), 0));
  fst.SetFinal(0, -log(0.5));

  DummyDecodable decodable;
  SimpleForward forward(fst, 1E+30, 1E-9);

  // A single observation
  {
    vector<BaseFloat> observations(2);
    observations[0] = log(0.6);
    observations[1] = log(0.4);
    decodable.Init(2, 1, observations);

    // Check simple decode
    KALDI_ASSERT(forward.Decode(&decodable));
    KALDI_ASSERT(forward.ReachedFinal());
    kaldi::AssertEqual(forward.FinalCost(), -log(0.25), 1E-9);

    // Check advanced decoding
    forward.InitDecoding();
    forward.AdvanceDecoding(&decodable);
    KALDI_ASSERT(forward.ReachedFinal());
    kaldi::AssertEqual(forward.FinalCost(), -log(0.25), 1E-9);

    // Check forward table
    KALDI_ASSERT(forward.GetTable().size() == 1);
    // frame = 0
    KALDI_ASSERT(forward.GetTable()[0].count(1) == 1);
    KALDI_ASSERT(forward.GetTable()[0].count(2) == 1);
    kaldi::AssertEqual(forward.GetTable()[0].find(1)->second, -log(0.3), 1E-9);
    kaldi::AssertEqual(forward.GetTable()[0].find(2)->second, -log(0.2), 1E-9);
    // end
    //KALDI_ASSERT(forward.GetTable()[1].count(1) == 1);
    //KALDI_ASSERT(forward.GetTable()[1].count(2) == 2);
  }

  {
    vector<BaseFloat> observations(3 * 2);
    // frame = 0
    observations[0] = log(0.1);
    observations[1] = log(0.9);
    // frame = 1
    observations[2] = log(0.5);
    observations[3] = log(0.5);
    // frame = 2
    observations[4] = log(0.4);
    observations[5] = log(0.6);
    decodable.Init(2, 3, observations);

    // Check simple decode
    KALDI_ASSERT(forward.Decode(&decodable));
    KALDI_ASSERT(forward.ReachedFinal());
    kaldi::AssertEqual(forward.FinalCost(), -log(0.0625), 1E-9);

    // Check advanced decoding
    forward.InitDecoding();
    forward.AdvanceDecoding(&decodable);
    KALDI_ASSERT(forward.ReachedFinal());
    kaldi::AssertEqual(forward.FinalCost(), -log(0.0625), 1E-9);

    // Check forward table
    KALDI_ASSERT(forward.GetTable().size() == 3);
    // frame = 0
    KALDI_ASSERT(forward.GetTable()[0].count(1) == 1);
    KALDI_ASSERT(forward.GetTable()[0].count(2) == 1);
    kaldi::AssertEqual(forward.GetTable()[0].find(1)->second, -log(0.05), 1E-9);
    kaldi::AssertEqual(forward.GetTable()[0].find(2)->second, -log(0.45), 1E-9);
    // frame = 1
    KALDI_ASSERT(forward.GetTable()[1].count(1) == 1);
    KALDI_ASSERT(forward.GetTable()[1].count(2) == 1);
    kaldi::AssertEqual(forward.GetTable()[1].find(1)->second, -log(0.125), 1E-9);
    kaldi::AssertEqual(forward.GetTable()[1].find(2)->second, -log(0.125), 1E-9);
    // frame = 2
    KALDI_ASSERT(forward.GetTable()[2].count(1) == 1);
    KALDI_ASSERT(forward.GetTable()[2].count(2) == 1);
    kaldi::AssertEqual(forward.GetTable()[2].find(1)->second, -log(0.050), 1E-9);
    kaldi::AssertEqual(forward.GetTable()[2].find(2)->second, -log(0.075), 1E-9);
    // end
    //KALDI_ASSERT(forward.GetTable()[3].count(1) == 1);
    //KALDI_ASSERT(forward.GetTable()[3].count(2) == 1);
  }
}

void ForwardEpsilonTransitionUnitTest() {
  // Fst with a single node, and emitting two symbols
  VectorFst<StdArc> fst;
  fst.AddState(); // State 0
  fst.AddState(); // State 1

  fst.AddArc(0, StdArc(0, 0, -log(0.5), 1)); // epsilon transition
  fst.AddArc(0, StdArc(1, 1, -log(0.3), 0));
  fst.AddArc(1, StdArc(2, 2, -log(0.5), 1));

  fst.SetStart(0);
  fst.SetFinal(0, 0.0);
  fst.SetFinal(1, 0.0);

  DummyDecodable decodable;
  SimpleForward forward(fst, 1E+30, 1E-9);

  // A single observation
  {
    vector<BaseFloat> observations(1 * 2);
    observations[0] = log(0.7);
    observations[1] = log(0.3);
    decodable.Init(2, 1, observations);

    // Check simple decode
    KALDI_ASSERT(forward.Decode(&decodable));
    KALDI_ASSERT(forward.ReachedFinal());
    kaldi::AssertEqual(forward.FinalCost(), -log(0.39), 1E-9);

    // Check forward table
    KALDI_ASSERT(forward.GetTable().size() == 1);
    KALDI_ASSERT(forward.GetTable()[0].count(1) == 1);
    KALDI_ASSERT(forward.GetTable()[0].count(2) == 1);
  }

  // Two observations
  {
    vector<BaseFloat> observations(2 * 2);
    // frame = 0
    observations[0] = log(0.6);
    observations[1] = log(0.4);
    // frame = 1
    observations[2] = log(0.3);
    observations[3] = log(0.7);
    decodable.Init(2, 2, observations);

    // Check simple decode
    KALDI_ASSERT(forward.Decode(&decodable));
    KALDI_ASSERT(forward.ReachedFinal());
    kaldi::AssertEqual(forward.FinalCost(), -log(0.0908), 1E-9);
  }
}

}  // namespace unittest
}  // namespace kaldi

int main(int argc, char** argv) {
  kaldi::unittest::ForwardSingleStateNoEpsUnitTest();
  kaldi::unittest::ForwardEpsilonTransitionUnitTest();
  std::cout << "Test OK.\n";
  return 0;
}
