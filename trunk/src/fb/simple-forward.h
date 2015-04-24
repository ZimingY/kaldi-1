// fb/simple-forward.h

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

#ifndef KALDI_FB_SIMPLE_FORWARD_H_
#define KALDI_FB_SIMPLE_FORWARD_H_


#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"

namespace kaldi {

class SimpleForward {
 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef fst::Fst<StdArc> Fst;
  typedef fst::ArcIterator<Fst> ArcIterator;

  SimpleForward(const Fst &fst, BaseFloat beam, BaseFloat loop_epsilon) :
      fst_(fst), beam_(beam), loop_epsilon_(loop_epsilon) { }

  ~SimpleForward();

  /// Decode this utterance.
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether we reached a final state.
  bool Forward(DecodableInterface *decodable);

  /// *** The next functions are from the "new interface". ***

  /// FinalCost() serves the same function as ReachedFinal(), but gives
  /// more information.  It returns the total cost of reaching all final
  /// states. This is useful to obtain the total likelihood of the complete
  /// decoding input. It will usually be nonnegative.
  double TotalCost() const;

  void InitForward();

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.
  void AdvanceForward(DecodableInterface *decodable,
                      int32 max_num_frames = -1);

  /// Returns the number of frames already decoded.
  int32 NumFramesDecoded() const { return num_frames_decoded_; }

  /// Returns the forward table of the in the input labels of the WFST
  /// This typically is the forward table of the transition-ids
  const std::vector<unordered_map<Label, double> >& GetTable() const {
    return forward_;
  }

 private:

  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  void ProcessEmitting(DecodableInterface *decodable);

  void ProcessNonemitting();

  // Add a new row to the table, with the cost of each label (i.e.
  // transition-id) reaching the end of the input.
  void UpdateForwardTableFinal();
  void UpdateForwardTable();

  typedef unordered_map<Label, double>  LabelMap;
  std::vector<LabelMap> forward_;

  struct Token {
    double cost; // total cost of the token
    double last_cost;    // shortest-distance algorithm
    LabelMap ilabels; // cost split for each input label
    LabelMap last_ilabels;

    Token(double c) : cost(c), last_cost(-kaldi::kLogZeroDouble) { }

    bool Update(const Token& parent, Label label, const double inc_cost,
                double threshold) {
      const double new_cost = -kaldi::LogAdd(-cost, -inc_cost);
      if (kaldi::ApproxEqual(cost, new_cost, threshold))
        return false;
      cost = new_cost;
      last_cost = -kaldi::LogAdd(-last_cost, -inc_cost);

      // Update weights comming from each input label
      if (label == 0) {
        for (unordered_map<Label, double>::const_iterator pi =
                 parent.last_ilabels.begin();
             pi != parent.last_ilabels.end(); ++pi) {
          // Total weight
          LabelMap::iterator l = ilabels.insert(
              make_pair(pi->first, -kaldi::kLogZeroDouble)).first;
          l->second = -kaldi::LogAdd(-l->second, -inc_cost);
          // Weight increment since last extraction
          l = last_ilabels.insert(
              make_pair(pi->first, -kaldi::kLogZeroDouble)).first;
          l->second = -kaldi::LogAdd(-l->second, -inc_cost);
        }
      } else {
        LabelMap::iterator l = ilabels.insert(
            make_pair(label, -kaldi::kLogZeroDouble)).first;
        l->second = -kaldi::LogAdd(-l->second, -inc_cost);
      }
      return true;
    }
  };

  typedef unordered_map<StateId, Token> TokenMap;
  TokenMap curr_toks_;
  TokenMap prev_toks_;
  const Fst &fst_;
  BaseFloat beam_;
  BaseFloat loop_epsilon_;
  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  static void PruneToks(BaseFloat beam, TokenMap *toks);

  KALDI_DISALLOW_COPY_AND_ASSIGN(SimpleForward);
};


} // end namespace kaldi.


#endif
