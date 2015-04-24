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


#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"

namespace kaldi {

class TransposeEpsilonWFST {
 private:
  typedef fst::StdArc StdArc;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef fst::Fst<StdArc> Fst;
  typedef fst::StateIterator<Fst> StateIterator;
  typedef fst::ArcIterator<Fst> ArcIterator;
  std::unordered_map<StateId, std::list<StdArc> > fst_;
  std::vector<StateId> states_;

 public:
  TransposeEpsilonWFST(const Fst& fst) {
    for (StateIterator siter(fst); !siter.Done(); siter.Next()) {
      for (ArcIterator aiter(fst, siter.Value()); !aiter.Done(); aiter.Next()) {
        const StdArc& arc = aiter.Value();
        if (arc.ilabel != 0) continue;
        fst_.insert(make_pair(arc.nextstate, std::list<StdArc>())).first->second.push_back(
            StdArc(arc.ilabel, arc.olabel, arc.weight, siter.Value()));
      }
    }
    for (std::unordered_map<StateId, std::list<StdArc> >::const_iterator it = fst_.begin();
         it != fst_.end(); ++it) {
      states_.push_back(it->first);
    }
  }

  const std::vector<StateId>& States() const {
    return states_;
  }

  const std::list<StdArc>& NodeArcs(StateId s) const {
    const std::unordered_map<StateId, std::list<StdArc> >::const_iterator it = fst_.find(s);
    KALDI_ASSERT(it != fst_.end());
    return it->second;
  }
};

class SimpleBackward {
 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef fst::Fst<StdArc> Fst;
  typedef fst::StateIterator<Fst> StateIterator;
  typedef fst::ArcIterator<Fst> ArcIterator;

  SimpleBackward(const Fst &fst, BaseFloat beam, BaseFloat loop_epsilon) :
      fst_(fst), eps_fst_(fst), beam_(beam), loop_epsilon_(loop_epsilon) { }

  ~SimpleBackward();

  bool Backward(DecodableInterface *decodable);

  double TotalCost() const;

  /// Returns the backward table of the in the input labels of the WFST
  /// This typically is the backward table of the transition-ids
  const std::vector<unordered_map<Label, double> >& GetTable() const {
    return backward_;
  }

 private:
  void InitBackward(DecodableInterface *decode);
  void ProcessEmitting(DecodableInterface *decodable);
  void ProcessNonemitting();
  void UpdateBackwardTable();

  typedef unordered_map<Label, double>  LabelMap;
  std::vector<LabelMap> backward_;

  struct Token {
    double cost; // total cost of the token
    LabelMap ilabels; // cost split for each input label

    Token(double c) : cost(c) {}

    // Update token with a path coming from `parent' through the given arc
    // and with the given acoustic cost.
    void Update(const Token& parent, const StdArc& arc, double acoustic) {
      const double inc_cost = parent.cost + arc.weight.Value() + acoustic;
      cost = -kaldi::LogAdd(-cost, -inc_cost);
      LabelMap::iterator lab = ilabels.insert(
          make_pair(arc.ilabel, -kaldi::kLogZeroDouble)).first;
      lab->second = -kaldi::LogAdd(-lab->second, -inc_cost);
    }

    // Update a token with a path coming from `parent' through an epsilon
    // arc with the given cost.
    void Update(const Token& parent, double epsilon_cost) {
      const double inc_cost = parent.cost + epsilon_cost;
      cost = -kaldi::LogAdd(-cost, -inc_cost);
      for (unordered_map<Label, double>::const_iterator pi =
               parent.ilabels.begin(); pi != parent.ilabels.end(); ++pi) {
        const double inc_lab_cost = pi->second + epsilon_cost;
        LabelMap::iterator lab = ilabels.insert(
            make_pair(pi->first, -kaldi::kLogZeroDouble)).first;
        lab->second = -kaldi::LogAdd(-lab->second, -inc_lab_cost);
      }
    }
  };

  typedef unordered_map<StateId, Token> TokenMap;
  TokenMap curr_toks_;
  TokenMap next_toks_;
  const Fst &fst_;
  TransposeEpsilonWFST eps_fst_;
  BaseFloat beam_;
  BaseFloat loop_epsilon_;
  // Keep track of the number of frames decoded in the current file.
  int32 remaining_frames_;

  static void PruneToks(BaseFloat beam, TokenMap *toks);

  KALDI_DISALLOW_COPY_AND_ASSIGN(SimpleBackward);
};

} // end namespace kaldi.

#endif
