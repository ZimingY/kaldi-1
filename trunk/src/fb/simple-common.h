// fb/simple-common.h

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

#ifndef KALDI_FB_SIMPLE_COMMON_H_
#define KALDI_FB_SIMPLE_COMMON_H_

#include "fst/fstlib.h"
#include "util/stl-utils.h"

namespace kaldi {

typedef fst::StdArc::StateId StateId;
typedef fst::StdArc::Label Label;
typedef unordered_map<Label, BaseFloat>  LabelMap;

struct Token {
  BaseFloat cost;         // total cost to the state
  BaseFloat last_cost;    // cost to the state, since the last extraction from
                          // the shortest-distance algorithm queue (see [1]).
  LabelMap ilabels;       // total cost to the state, for each input symbol.
  LabelMap last_ilabels;  //  cost to the state, for each input symbol,
                          // since the last extraction from the
                          // shortest-distance algorithm queue (see [1]).

  Token(BaseFloat c) : cost(c), last_cost(-kaldi::kLogZeroBaseFloat) { }

  // Update token when processing non-epsilon edges
  void UpdateEmitting(
      const Label label, const BaseFloat prev_cost, const BaseFloat edge_cost,
      const BaseFloat acoustic_cost);

  // Update token when processing epsilon edges
  bool UpdateNonEmitting(
      const LabelMap& parent_ilabels, const BaseFloat prev_cost,
      const BaseFloat edge_cost, const BaseFloat threshold);
};
typedef unordered_map<StateId, Token> TokenMap;

void AccumulateToks(const TokenMap& toks, LabelMap *acc);
void PruneToks(BaseFloat beam, TokenMap *toks);
BaseFloat RescaleToks(TokenMap* toks);

void ComputePosteriorgram(
    const std::vector<LabelMap>& fwd, const std::vector<LabelMap>& bkw,
    std::vector<LabelMap>* pst);

}  // namespace kaldi

#endif  // KALDI_FB_SIMPLE_COMMON_H_
