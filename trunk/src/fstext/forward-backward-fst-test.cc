#include "fstext/forward-backward-fst.h"
#include "fst/float-weight.h"
#include "util/kaldi-io.h"

void TestForwardBackwardArc() {
  // Test empty constructor
  {
    fst::ForwardBackwardArc<fst::LogArc> arc;
    arc.weight = fst::LogWeight::One();
    arc.ilabel = 1;
    arc.olabel = 2;
    arc.nextstate = 3;
    arc.prevstate = 4;
  }
  // Test constructor from parent class object
  {
    fst::StdArc std_arc(1, 2, fst::TropicalWeight::One(), 3);
    fst::ForwardBackwardArc<fst::StdArc> fb_arc(std_arc);
    fb_arc.prevstate = 4;
    KALDI_ASSERT(arc.weight == fst::TropicalWeight::One());
    KALDI_ASSERT(arc.ilabel == 1);
    KALDI_ASSERT(arc.olabel == 2);
    KALDI_ASSERT(arc.nextstate == 3);
    KALDI_ASSERT(arc.prevstate == 4);
  }
  // Test constructor receiveing parent class object + prevstate
  {
    fst::StdArc std_arc(1, 2, fst::TropicalWeight::One(), 3);
    fst::ForwardBackwardArc<fst::StdArc> fb_arc(std_arc, 4);
    KALDI_ASSERT(arc.weight == fst::TropicalWeight::One());
    KALDI_ASSERT(arc.ilabel == 1);
    KALDI_ASSERT(arc.olabel == 2);
    KALDI_ASSERT(arc.nextstate == 3);
    KALDI_ASSERT(arc.prevstate == 4);
  }

}

void TestForwardBackwardFstImpl() {
  fst::ForwardBackwardFstImpl<fst::StdArc, size_t> impl;

  fst::ForwardBackwardFst<fst::StdArc> fst;
}

int main(int argc, char** argv) {
  TestForwardBackwardArc();
  TestForwardBackwardFstImpl();
  return 0;
}
