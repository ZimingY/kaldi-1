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
    UpdateForwardTable();
  }
  UpdateForwardTableFinal();
  return (!curr_toks_.empty());
}


void SimpleForward::InitDecoding() {
  // clean up from last time:
  curr_toks_.clear();
  prev_toks_.clear();
  forward_.clear();
  // initialize decoding:
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  curr_toks_[start_state] = Token();
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
    UpdateForwardTable();
  }
  UpdateForwardTableFinal();
}


bool SimpleForward::ReachedFinal() const {
  for (TokenMap::const_iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    if (tok->second.cost != -kaldi::kLogZeroDouble &&
        fst_.Final(tok->first) != fst::StdArc::Weight::Zero())
      return true;
  }
  return false;
}


double SimpleForward::FinalCost() const {
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
    for (fst::ArcIterator<Fst> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      if (arc.ilabel != 0) {  // propagate emitting only...
        const double acoustic_cost =
            -decodable->LogLikelihood(frame, arc.ilabel);
        TokenMap::iterator ctok = curr_toks_.insert(
            make_pair(arc.nextstate, Token(-kaldi::kLogZeroDouble))).first;
        ctok->second.Update(ptok->second, arc, acoustic_cost);
      }
    }
  }
  num_frames_decoded_++;
}

void SimpleForward::ProcessNonemitting() {
  // Processes nonemitting arcs for one frame.  Propagates within
  // curr_toks_.
  std::vector<StateId> queue_;
  for (TokenMap::const_iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    queue_.push_back(tok->first);
  }

  while (!queue_.empty()) {
    const StateId state = queue_.back();
    TokenMap::const_iterator ptok = curr_toks_.find(state);
#ifdef KALDI_PARANOID
    KALDI_ASSERT(ptok != curr_toks_.end());
#endif
    queue_.pop_back();
    for (fst::ArcIterator<Fst> aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        TokenMap::iterator ctok = curr_toks_.insert(
            make_pair(arc.nextstate, Token(-kaldi::kLogZeroDouble))).first;
        const double old_cost_to_ctok = ctok->second.cost;
        ctok->second.Update(ptok->second, arc.weight.Value());
        // TODO(jpuigcerver): This tries to prevent the algorithm to hang
        // with epsilon-loops. This should be able to detect convergence
        // in the cost of the state, however it will still hang if the
        // cost is divergent. Anyway, this solution will be slow in the
        // case of an WFST with epsilon-cycles. Viterbi algorithm does not
        // have this problem, as long as the cost of the epsilon-cycle is
        // positive. Otherwise, the current implementation of the Decoder
        // will hang as well.
        if (!kaldi::ApproxEqual(
                ctok->second.cost, old_cost_to_ctok, loop_epsilon_)) {
          queue_.push_back(arc.nextstate);
        }
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
