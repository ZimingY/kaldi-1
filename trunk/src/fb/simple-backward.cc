// fb/simple-backward.cc

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

#include "fb/simple-backward.h"
#include "fb/queue-set.h"
#include <algorithm>

namespace kaldi {

SimpleBackward::~SimpleBackward() {
  curr_toks_.clear();
  prev_toks_.clear();
}


void SimpleBackward::InitBackward(DecodableInterface *decodable) {
  // clean up from last time:
  curr_toks_.clear();
  prev_toks_.clear();
  backward_.clear();
  // initialize decoding:
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    if (fst_.Final(siter.Value()) != -kaldi::kLogZeroDouble) {
      curr_toks_.insert(make_pair(
          siter.Value(), fst_.Final(siter.Value()).Value()));
    }
  }
  // We need this asserts because the backward pass cannot be done in a
  // sequence processed online, that is, we need the whole sequence to be
  // available from the beginning
  KALDI_ASSERT(decodable->NumFramesReady() >= 0);
  KALDI_ASSERT(decodable->IsLastFrame(decodable->NumFramesReady() - 1));
  backward_.resize(decodable->NumFramesReady());
  num_frames_decoded_ = 0;
  ProcessNonemitting();
}


bool SimpleBackward::Backward(DecodableInterface *decodable) {
  InitBackward(decodable);
  while(num_frames_decoded_ < decodable->NumFramesReady()) {
    prev_toks_.clear();
    curr_toks_.swap(prev_toks_);
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &curr_toks_);
    // Here num_frames_decoded_ has already been updated
    AccumulateToks(curr_toks_, &backward_[decodable->NumFramesReady() -
                                          num_frames_decoded_]);
  }
  return (!curr_toks_.empty());
}


void SimpleBackward::ProcessEmitting(DecodableInterface *decodable) {
  // Processes emitting arcs for one frame.  Propagates from prev_toks_ to
  // curr_toks_.
  const int32 frame = decodable->NumFramesReady() - num_frames_decoded_ - 1;
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    const StateId state = siter.Value();
    for (ArcIterator aiter(fst_, state); !aiter.Done(); aiter.Next()) {
      const StdArc& arc = aiter.Value();
      TokenMap::const_iterator ptok = prev_toks_.find(arc.nextstate);
      if (arc.ilabel == 0 || ptok == prev_toks_.end())
        continue;
      const double acoustic_cost =
          -decodable->LogLikelihood(frame, arc.ilabel);
      Token& ctok = curr_toks_.insert(make_pair(
          state, Token(-kaldi::kLogZeroDouble))).first->second;
      ctok.UpdateEmitting(
          arc.ilabel, ptok->second.cost, arc.weight.Value(), acoustic_cost);
    }
  }
  num_frames_decoded_++;
}


void SimpleBackward::ProcessNonemitting() {
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

    for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
      for (ArcIterator aiter(fst_, siter.Value()); !aiter.Done();
           aiter.Next()) {
        const StdArc& arc = aiter.Value();
        if (arc.ilabel != 0 || arc.nextstate != state) continue;
        Token& ctok = curr_toks_.insert(make_pair(
            siter.Value(), Token(-kaldi::kLogZeroDouble))).first->second;
        if (ctok.UpdateNonEmitting(
                last_cost_labels, last_cost, arc.weight.Value(), delta_)) {
          queue_set.push(siter.Value());
        }
      }
    }
  }
}


double SimpleBackward::TotalCost() const {
  TokenMap::const_iterator tok = curr_toks_.find(fst_.Start());
  const double total_cost =
      tok == curr_toks_.end() ? -kaldi::kLogZeroDouble : tok->second.cost;
  if (total_cost != total_cost) { // NaN. This shouldn't happen; it indicates
                                  // some kind of error, most likely.
    KALDI_WARN << "Found NaN (likely failure in decoding)";
    return -kaldi::kLogZeroDouble;
  }
  return total_cost;
}




} // end namespace kaldi.
