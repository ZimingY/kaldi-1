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

#define COST_ADD(x, y) (-kaldi::LogAdd(-(x), -(y)))
#define COST_MAX       (-kaldi::kLogZeroDouble)

SimpleForward::~SimpleForward() {
  curr_toks_.clear();
  prev_toks_.clear();
}


bool SimpleForward::Decode(DecodableInterface *decodable) {
  InitDecoding();
  while(!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    prev_toks_.clear();
    curr_toks_.swap(prev_toks_);
    accessible_from_.clear();
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &curr_toks_);
  }
  //UpdateForwardTableFinal();
  return (!curr_toks_.empty());
}


void SimpleForward::InitDecoding() {
  // clean up from last time:
  curr_toks_.clear();
  prev_toks_.clear();
  forward_.clear();
  accessible_from_.clear();
  // initialize decoding:
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  curr_toks_[start_state] = 0.0;
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
  //UpdateForwardTableFinal();
}


bool SimpleForward::ReachedFinal() const {
  for (unordered_map<StateId, double>::const_iterator iter = curr_toks_.begin();
       iter != curr_toks_.end(); ++iter) {
    if (iter->second != COST_MAX &&
        fst_.Final(iter->first) != fst::StdArc::Weight::Zero())
      return true;
  }
  return false;
}


double SimpleForward::FinalCost() const {
  double total_cost = COST_MAX;
  for (unordered_map<StateId, double>::const_iterator iter = curr_toks_.begin();
       iter != curr_toks_.end(); ++iter) {
    total_cost = COST_ADD(total_cost, iter->second + fst_.Final(iter->first).Value());
  }
  if (total_cost != total_cost) { // NaN. This shouldn't happen; it indicates
                                  // some kind of error, most likely.
    KALDI_WARN << "Found NaN (likely failure in decoding)";
    return COST_MAX;
  }
  return total_cost;
}


void SimpleForward::ProcessEmitting(DecodableInterface *decodable) {
  // Processes emitting arcs for one frame.  Propagates from prev_toks_ to
  // curr_toks_.
  int32 frame = num_frames_decoded_;
  forward_.push_back(unordered_map<Label, double>());
  for (unordered_map<StateId, double>::iterator iter = prev_toks_.begin();
       iter != prev_toks_.end(); ++iter) {
    StateId state = iter->first;
    const double cost_to_state = iter->second;
    for (fst::ArcIterator<Fst> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      if (arc.ilabel != 0) {
        const double acoustic_cost =
            -decodable->LogLikelihood(frame, arc.ilabel);
        const double path_cost =
            cost_to_state + acoustic_cost + arc.weight.Value();

        unordered_map<StateId, double>::iterator find_iter =
            curr_toks_.find(arc.nextstate);
        if (find_iter == curr_toks_.end()) {
          curr_toks_[arc.nextstate] = path_cost;
          unordered_set<Label> lbl_set; lbl_set.insert(arc.ilabel);
          accessible_from_[arc.nextstate] = lbl_set;
        } else {
          find_iter->second = COST_ADD(find_iter->second, path_cost);
          accessible_from_[arc.nextstate].insert(arc.ilabel);
        }

        unordered_map<Label, double>::iterator label_iter =
            forward_.back().find(arc.ilabel);
        if (label_iter == forward_.back().end()) {
          forward_.back()[arc.ilabel] = path_cost;
        } else {
          label_iter->second = COST_ADD(label_iter->second, path_cost);
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
  for (unordered_map<StateId, double>::iterator iter = curr_toks_.begin();
       iter != curr_toks_.end(); ++iter) {
    queue_.push_back(iter->first);
  }

  while (!queue_.empty()) {
    StateId state = queue_.back();
    const double cost_to_state = curr_toks_[state];
    queue_.pop_back();
    for (fst::ArcIterator<Fst> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        const double path_cost = cost_to_state + arc.weight.Value();
        unordered_map<StateId, double>::iterator find_iter =
            curr_toks_.find(arc.nextstate);
        if (find_iter == curr_toks_.end()) {
          curr_toks_[arc.nextstate] = path_cost;
          accessible_from_[arc.nextstate] = accessible_from_[state];
          queue_.push_back(arc.nextstate);
        } else {
          const double old_cost = find_iter->second;
          find_iter->second = COST_ADD(find_iter->second, path_cost);
          accessible_from_[arc.nextstate].insert(
              accessible_from_[state].begin(), accessible_from_[state].end());

          // TODO(jpuigcerver): This tries to prevent the algorithm to hang
          // with epsilon-loops. This should be able to detect convergence
          // in the cost of the state, however it will still hang if the
          // cost is divergent. Anyway, this solution will be slow in the
          // case of an WFST with epsilon-cycles. Viterbi algorithm does not
          // have this problem, as long as the cost of the epsilon-cycle is
          // positive. Otherwise, the current implementation of the Decoder
          // will hang anyway.
          if (!kaldi::ApproxEqual(find_iter->second, old_cost, loop_epsilon_)) {
            queue_.push_back(arc.nextstate);
          }
        }
      }
    }
  }
}


void SimpleForward::UpdateForwardTableFinal() {
  if (forward_.size() == 0) {
    forward_.push_back(unordered_map<Label, double>());
    return;
  }
  const unordered_map<Label, double>& prev_fwd = forward_.back();
  forward_.push_back(unordered_map<Label, double>());
  for (unordered_map<StateId, unordered_set<Label> >::const_iterator st_it =
           accessible_from_.begin(); st_it != accessible_from_.end(); ++st_it) {
    const StateId state = st_it->first;
    const fst::StdArc::Weight final_state_cost = fst_.Final(state);
    if (final_state_cost != fst::StdArc::Weight::Zero()) {
      for (unordered_set<Label>::const_iterator la_it = st_it->second.begin();
           la_it != st_it->second.end(); ++la_it) {
        const Label label = *la_it;
        const double final_label_cost =
            final_state_cost.Value() + prev_fwd.find(label)->second;
        unordered_map<Label, double>::iterator find_iter =
            forward_.back().find(label);
        if (find_iter == forward_.back().end()) {
          forward_.back()[label] = final_label_cost;
        } else {
          find_iter->second = COST_ADD(find_iter->second, final_label_cost);
        }
      }
    }
  }
}


// static
void SimpleForward::PruneToks(BaseFloat beam, unordered_map<StateId, double> *toks) {
  if (toks->empty()) {
    KALDI_VLOG(2) <<  "No tokens to prune.\n";
    return;
  }
  double best_cost = COST_MAX;
  for (unordered_map<StateId, double>::iterator iter = toks->begin();
       iter != toks->end(); ++iter)
    best_cost = std::min(best_cost, iter->second);
  std::vector<StateId> removed;
  const double cutoff = best_cost + beam;
  for (unordered_map<StateId, double>::const_iterator iter = toks->begin();
       iter != toks->end(); ++iter) {
    if (iter->second <= cutoff) continue;
    removed.push_back(iter->first);
  }
  for (size_t i = 0; i < removed.size(); ++i) {
    toks->erase(removed[i]);
  }
  KALDI_VLOG(2) <<  "Pruned " << removed.size() << " to " << toks->size() << " toks.\n";
}

} // end namespace kaldi.
