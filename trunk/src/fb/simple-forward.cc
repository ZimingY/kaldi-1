// fb/simple-forward.cc

// Copyright 2015 Joan Puigcerver

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

#include "fb/simple-forward.h"
#include "fb/queue-set.h"
#include <algorithm>

namespace kaldi {

SimpleForward::~SimpleForward() {
  curr_toks_.clear();
  prev_toks_.clear();
}


void SimpleForward::InitForward() {
  // clean up from last time:
  curr_toks_.clear();
  prev_toks_.clear();
  forward_.clear();
  // initialize decoding:
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  curr_toks_.insert(make_pair(start_state, 0.0));
  num_frames_decoded_ = 0;
  ProcessNonemitting();
}


bool SimpleForward::Forward(DecodableInterface *decodable) {
  InitForward();
  while(!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    prev_toks_.clear();
    curr_toks_.swap(prev_toks_);
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &curr_toks_);
    forward_.push_back(unordered_map<Label, double>());
    AccumulateToks(curr_toks_, &forward_.back());
  }
  return (!curr_toks_.empty());
}


void SimpleForward::ProcessEmitting(DecodableInterface *decodable) {
  // Processes emitting arcs for one frame.  Propagates from prev_toks_ to
  // curr_toks_.
  const int32 frame = num_frames_decoded_;
  for (TokenMap::const_iterator ptok = prev_toks_.begin();
       ptok != prev_toks_.end(); ++ptok) {
    const StateId state = ptok->first;
    for (ArcIterator aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const StdArc& arc = aiter.Value();
      if (arc.ilabel != 0) {  // propagate emitting only...
        const double acoustic_cost =
            -decodable->LogLikelihood(frame, arc.ilabel);
        Token& ctok = curr_toks_.insert(make_pair(
            arc.nextstate, Token(-kaldi::kLogZeroDouble))).first->second;
        ctok.UpdateEmitting(
            arc.ilabel, ptok->second.cost, arc.weight.Value(), acoustic_cost);
      }
    }
  }
  num_frames_decoded_++;
}


void SimpleForward::ProcessNonemitting() {
  // Processes nonemitting arcs for one frame.  Propagates within
  // curr_toks_.
  QueueSet<StateId> queue_set;
  for (TokenMap::iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    queue_set.push(tok->first);
    tok->second.last_cost = tok->second.cost;
    tok->second.last_ilabels = tok->second.ilabels;
  }

  while (!queue_set.empty()) {
    const StateId state = queue_set.front();
    queue_set.pop();

    Token& ptok = curr_toks_.find(state)->second;
    const double last_cost = ptok.last_cost;
    const LabelMap last_cost_labels = ptok.last_ilabels;
    ptok.last_cost = -kaldi::kLogZeroDouble;
    ptok.last_ilabels.clear();

    for (ArcIterator aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const StdArc& arc = aiter.Value();
      if (arc.ilabel != 0) continue;
      Token& ctok = curr_toks_.insert(make_pair(
          arc.nextstate, Token(-kaldi::kLogZeroDouble))).first->second;
      if (ctok.UpdateNonEmitting(
              last_cost_labels, last_cost, arc.weight.Value(), delta_)) {
        queue_set.push(arc.nextstate);
      }
    }
  }
}


double SimpleForward::TotalCost() const {
  double total_cost = -kaldi::kLogZeroDouble;
  for (TokenMap::const_iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    total_cost = -kaldi::LogAdd(
        -total_cost, -(tok->second.cost + fst_.Final(tok->first).Value()));
  }
  if (total_cost != total_cost) { // NaN. This shouldn't happen; it indicates
                                  // some kind of error, most likely.
    KALDI_WARN << "Found NaN (likely failure in decoding)";
    return -kaldi::kLogZeroDouble;
  }
  return total_cost;
}

} // end namespace kaldi.
