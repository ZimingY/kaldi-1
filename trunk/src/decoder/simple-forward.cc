// decoder/simple-forward.cc

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

#include "decoder/simple-forward.h"
#include <algorithm>

namespace kaldi {

SimpleForward::~SimpleForward() {
  curr_toks_.clear();
  prev_toks_.clear();
}


bool SimpleForward::Decode(DecodableInterface *decodable) {
  InitDecoding();
  while(!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    prev_toks_.clear();
    curr_toks_.swap(prev_toks_);
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &curr_toks_);
  }
  return (!curr_toks_.empty());
}


void SimpleForward::InitDecoding() {
  // clean up from last time:
  curr_toks_.clear();
  prev_toks_.clear();
  // initialize decoding:
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  curr_toks_[start_state] = Weight::One();
  num_frames_decoded_ = 0;
  ProcessNonemitting();
}

void SimpleForward::AdvanceDecoding(DecodableInterface *decodable,
                                    int32 max_num_frames) {
  KALDI_ASSERT(num_frames_decoded_ >= 0 &&
               "You must call InitDecoding() before AdvanceDecoding()");
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
  }
}


bool SimpleForward::ReachedFinal() const {
  for (unordered_map<StateId, Weight>::const_iterator iter = curr_toks_.begin();
       iter != curr_toks_.end(); ++iter) {
    if (iter->second != Weight::Zero() &&
        fst_.Final(iter->first) != Weight::Zero())
      return true;
  }
  return false;
}


double SimpleForward::FinalCost() const {
  Weight total_cost = Weight::Zero();
  for (unordered_map<StateId, Weight>::const_iterator iter = curr_toks_.begin();
       iter != curr_toks_.end(); ++iter) {
    total_cost = fst::Plus(
        fst::Times(iter->second, fst_.Final(iter->first)),
        total_cost);
  }
  if (total_cost != total_cost) { // NaN. This shouldn't happen; it indicates
                                  // some kind of error, most likely.
    KALDI_WARN << "Found NaN (likely failure in decoding)";
    return Weight::Zero().Value();
  }
  return total_cost.Value();
}


void SimpleForward::ProcessEmitting(DecodableInterface *decodable) {
  // Processes emitting arcs for one frame.  Propagates from prev_toks_ to
  // curr_toks_.
  int32 frame = num_frames_decoded_;
  for (unordered_map<StateId, Weight>::iterator iter = prev_toks_.begin();
       iter != prev_toks_.end(); ++iter) {
    StateId state = iter->first;
    const Weight cost_to_state = iter->second;
    for (fst::ArcIterator<Fst> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const LogArc& arc = aiter.Value();
      if (arc.ilabel != 0) {
        const double acoustic_cost =
            -decodable->LogLikelihood(frame, arc.ilabel);
        const Weight path_cost = fst::Times(
            cost_to_state,
            fst::Times(arc.weight, acoustic_cost));

        unordered_map<StateId, Weight>::iterator find_iter =
            curr_toks_.find(arc.nextstate);
        if (find_iter == curr_toks_.end()) {
          curr_toks_[arc.nextstate] = path_cost;
        } else {
          find_iter->second = fst::Plus(find_iter->second, path_cost);
        }
      }
    }
  }
  num_frames_decoded_++;
}

void SimpleForward::ProcessNonemitting() {
  // Processes nonemitting arcs for one frame.  Propagates within
  // curr_toks_.
  std::vector<StateId> queue_;
  for (unordered_map<StateId, Weight>::iterator iter = curr_toks_.begin();
       iter != curr_toks_.end(); ++iter) {
    queue_.push_back(iter->first);
  }

  while (!queue_.empty()) {
    StateId state = queue_.back();
    const Weight cost_to_state = curr_toks_[state];
    queue_.pop_back();
    for (fst::ArcIterator<Fst> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const LogArc& arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        const Weight path_cost = fst::Times(cost_to_state, arc.weight);
        unordered_map<StateId, Weight>::iterator find_iter =
            curr_toks_.find(arc.nextstate);
        if (find_iter == curr_toks_.end()) {
          curr_toks_[arc.nextstate] = path_cost;
          queue_.push_back(arc.nextstate);
        } else {
          const Weight old_cost = find_iter->second;
          find_iter->second = fst::Plus(find_iter->second, path_cost);
          // This is to prevent the algorithm to hang with loops made by
          // empty transitions. (As long as the serie induced is convergent,
          // the FST introduces a valid-but-unormalized distribution).
          if (!fst::ApproxEqual(find_iter->second, old_cost, loop_epsilon_))
            queue_.push_back(arc.nextstate);
          // TODO: If a non-emitting loop has cost >= 0, then the serie is
          // divergent, and it cannot define a valid distribution. I am not
          // checking this because Viterbi decoders do not check that either.
        }
      }
    }
  }
}


// static
void SimpleForward::PruneToks(BaseFloat beam, unordered_map<StateId, Weight> *toks) {
  if (toks->empty()) {
    KALDI_VLOG(2) <<  "No tokens to prune.\n";
    return;
  }
  Weight best_cost = Weight::Zero();
  for (unordered_map<StateId, Weight>::iterator iter = toks->begin();
       iter != toks->end(); ++iter)
    best_cost = std::min(best_cost.Value(), iter->second.Value());
  std::vector<StateId> removed;
  const double cutoff = best_cost.Value() + beam;
  for (unordered_map<StateId, Weight>::const_iterator iter = toks->begin();
       iter != toks->end(); ++iter) {
    if (iter->second.Value() <= cutoff) continue;
    removed.push_back(iter->first);
  }
  for (size_t i = 0; i < removed.size(); ++i) {
    toks->erase(removed[i]);
  }
  KALDI_VLOG(2) <<  "Pruned to " << toks->size() << " toks.\n";
}

} // end namespace kaldi.
