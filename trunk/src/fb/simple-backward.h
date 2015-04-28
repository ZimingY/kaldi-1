// fb/simple-backward.h

// Copyright 2015  Joan Puigcerver

// See ../../COPYING for clarification regarding multiple authors
//
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

#ifndef KALDI_FB_SIMPLE_BACKWARD_H_
#define KALDI_FB_SIMPLE_BACKWARD_H_

#include "fst/fstlib.h"
#include "hmm/posterior.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h"
#include "util/stl-utils.h"
#include "fb/simple-common.h"

namespace kaldi {

class SimpleBackward {
 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef fst::Fst<StdArc> Fst;
  typedef fst::ArcIterator<Fst> ArcIterator;
  typedef fst::StateIterator<Fst> StateIterator;

  SimpleBackward(const Fst &fst, BaseFloat beam, BaseFloat loop_epsilon) :
      fst_(fst), beam_(beam), delta_(loop_epsilon) { }

  ~SimpleBackward();

  /// Decode this utterance.
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether we reached a final state.
  bool Backward(DecodableInterface *decodable);

  /// *** The next functions are from the "new interface". ***

  /// TotalCost() returns the total cost of reaching any of the final states.
  /// This is useful to obtain the total likelihood of the complete
  /// decoding input. It will usually be nonnegative.
  double TotalCost() const;

  void InitBackward(DecodableInterface *decodable);

  /// Returns the number of frames already decoded.
  int32 NumFramesDecoded() const { return num_frames_decoded_; }

  /// Returns the backward table of the in the input labels of the WFST
  /// This typically is the backward table of the transition-ids
  const vector<unordered_map<Label, double> >& GetTable() const {
    return backward_;
  }

  void FillPosterior(Posterior* posterior) const;

 private:

  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  void ProcessEmitting(DecodableInterface *decodable);
  void ProcessNonemitting();

  vector<LabelMap> backward_;
  vector<double> scale_factor_;

  typedef unordered_map<StateId, Token> TokenMap;
  TokenMap curr_toks_;
  TokenMap prev_toks_;
  const Fst &fst_;
  BaseFloat beam_;
  BaseFloat delta_;
  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(SimpleBackward);
};


} // end namespace kaldi.


#endif
