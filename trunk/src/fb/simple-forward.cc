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
    UpdateForwardTable();
  }
  return (!curr_toks_.empty());
}


void SimpleForward::AdvanceForward(DecodableInterface *decodable,
                                  int32 max_num_frames) {
  KALDI_ASSERT(num_frames_decoded_ >= 0 &&
               "You must call InitForward() before AdvanceForward()");
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= num_frames_decoded_);
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     num_frames_decoded_ + max_num_frames);
  while (num_frames_decoded_ < target_frames_decoded) {
    // note: ProcessEmitting() increments num_frames_decoded_
    prev_toks_.clear();
    curr_toks_.swap(prev_toks_);
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &curr_toks_);
    UpdateForwardTable();
  }
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
      if (ctok.UpdateNonEmitting(last_cost_labels, last_cost,
                                 arc.weight.Value(), loop_epsilon_)) {
        queue_set.push(arc.nextstate);
      }
    }
  }
}


// static
void SimpleForward::PruneToks(BaseFloat beam, TokenMap *toks) {
  if (toks->empty()) {
    KALDI_VLOG(2) <<  "No tokens to prune.\n";
    return;
  }
  TokenMap::const_iterator tok = toks->begin();
  // Get best cost
  double best_cost = tok->second.cost;
  for (++tok; tok != toks->end(); ++tok) {
    best_cost = std::min(best_cost, tok->second.cost);
  }
  // Mark all tokens with cost greater than the cutoff
  std::vector<TokenMap::const_iterator> remove_toks;
  const double cutoff = best_cost + beam;
  for (tok = toks->begin(); tok != toks->end(); ++tok) {
    if (tok->second.cost > cutoff) remove_toks.push_back(tok);
  }
  // Prune tokens
  for (size_t i = 0; i < remove_toks.size(); ++i) {
    toks->erase(remove_toks[i]);
  }
  KALDI_VLOG(2) <<  "Pruned " << remove_toks.size() << " to "
                << toks->size() << " toks.\n";
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


void SimpleForward::UpdateForwardTable() {
  // Add costs of each symbol across all active states.
  forward_.push_back(unordered_map<Label, double>());
  for (TokenMap::const_iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    for (LabelMap::const_iterator ti = tok->second.ilabels.begin();
         ti != tok->second.ilabels.end(); ++ti) {
      LabelMap::iterator fi = forward_.back().insert(
          make_pair(ti->first, -kaldi::kLogZeroDouble)).first;
      fi->second = -kaldi::LogAdd(-fi->second, -ti->second);
    }
  }
}


// Update token when processing non-epsilon edges
void SimpleForward::Token::UpdateEmitting(
    const Label label, const double prev_cost, const double edge_cost,
    const double acoustic_cost) {
  const double inc_cost = prev_cost + edge_cost + acoustic_cost;
  // Update total cost to the state, using input symbol `label'
  LabelMap::iterator l = ilabels.insert(make_pair(
      label, -kaldi::kLogZeroDouble)).first;
  l->second = -kaldi::LogAdd(-l->second, -inc_cost);
  // Update total cost to the state
  cost = -kaldi::LogAdd(-cost, -inc_cost);
}


// Update token when processing epsilon edges
bool SimpleForward::Token::UpdateNonEmitting(
    const LabelMap& parent_ilabels, const double prev_cost,
    const double edge_cost, const double threshold) {
  const double old_cost = cost;
  // Propagate all the parent input symbols to this state, since we are
  // using a epsilon-transition
  for (unordered_map<Label, double>::const_iterator pl =
           parent_ilabels.begin(); pl != parent_ilabels.end(); ++pl) {
    const double inc_cost = pl->second + edge_cost;
    // Total cost using symbol `pl->first' to this state
    LabelMap::iterator l =
        ilabels.insert(make_pair(pl->first, -kaldi::kLogZeroDouble)).first;
    l->second = -kaldi::LogAdd(-l->second, -inc_cost);
    // Cost since the last time this state was extracted from the search
    // queue (see [1]).
    l = last_ilabels.insert(make_pair(pl->first, -kaldi::kLogZeroDouble)).first;
    l->second = -kaldi::LogAdd(-l->second, -inc_cost);
  }
  // Total cost to this state, using any symbol
  cost = -kaldi::LogAdd(-cost, - (prev_cost + edge_cost));
  last_cost = -kaldi::LogAdd(-last_cost, -(prev_cost + edge_cost));
  return !kaldi::ApproxEqual(cost, old_cost, threshold);
}

} // end namespace kaldi.
