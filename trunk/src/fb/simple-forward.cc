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
#include <algorithm>

namespace kaldi {

SimpleForward::~SimpleForward() {
  curr_toks_.clear();
  prev_toks_.clear();
}


bool SimpleForward::Forward(DecodableInterface *decodable) {
  InitForward();

  for (TokenMap::const_iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    KALDI_LOG << "F[0, " << tok->first << "] = " << tok->second.cost;
  }

  while(!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    prev_toks_.clear();
    curr_toks_.swap(prev_toks_);
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &curr_toks_);
    UpdateForwardTable();
    for (TokenMap::const_iterator tok = curr_toks_.begin();
         tok != curr_toks_.end(); ++tok) {
      KALDI_LOG << "F[" << num_frames_decoded_ << ", " << tok->first << "] = " << tok->second.cost;
    }
  }
  //UpdateForwardTableFinal();
  return (!curr_toks_.empty());
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
  //UpdateForwardTableFinal();
}

double SimpleForward::TotalCost() const {
  if (curr_toks_.empty()) {
    return -kaldi::kLogZeroDouble;
  }
  TokenMap::const_iterator tok = curr_toks_.begin();
  double total_cost = tok->second.cost + fst_.Final(tok->first).Value();
  for (++tok; tok != curr_toks_.end(); ++tok) {
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
        ctok.Update(
            ptok->second, arc.ilabel,
            ptok->second.cost + arc.weight.Value() + acoustic_cost,
            loop_epsilon_);
      }
    }
  }
  num_frames_decoded_++;
}

template <typename T>
class QueueSet {
 private:
  std::queue<T> queue_;
  std::set<T> set_;

 public:
  bool empty() const {
    return queue_.empty();
  }
  size_t size() const {
    return queue_.size();
  }
  void push(const T& n) {
    if (set_.insert(n).second)
      queue_.push(n);
  }
  const T& front() const {
    return queue_.front();
  }
  T& front() {
    return queue_.front();
  }
  void pop() {
    set_.erase(queue_.front());
    queue_.pop();
  }
};

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
    const StateId q = queue_set.front();
    queue_set.pop();

    Token& ptok = curr_toks_.find(q)->second;
    const double r_q = ptok.last_cost;
    ptok.last_cost = -kaldi::kLogZeroDouble;

    for (ArcIterator aiter(fst_, q); !aiter.Done();
         aiter.Next()) {
      const StdArc& arc = aiter.Value();
      if (arc.ilabel != 0) continue;
      Token& ctok = curr_toks_.insert(make_pair(
          arc.nextstate, Token(-kaldi::kLogZeroDouble))).first->second;
      if (ctok.Update(ptok, arc.ilabel, r_q + arc.weight.Value(),
                      loop_epsilon_)) {
        queue_set.push(arc.nextstate);
      }
    }
  }
}


void SimpleForward::UpdateForwardTable() {
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


void SimpleForward::UpdateForwardTableFinal() {
  forward_.push_back(LabelMap());
  for (TokenMap::const_iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    const double final_cost = fst_.Final(tok->first).Value();
    for (LabelMap::const_iterator ti = tok->second.ilabels.begin();
         ti != tok->second.ilabels.end(); ++ti) {
      LabelMap::iterator fi = forward_.back().insert(
              make_pair(ti->first, -kaldi::kLogZeroDouble)).first;
      fi->second = -kaldi::LogAdd(-fi->second, -(ti->second + final_cost));
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

} // end namespace kaldi.
