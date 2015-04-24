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
#include <algorithm>
#include <functional>

namespace kaldi {

SimpleBackward::~SimpleBackward() {
  curr_toks_.clear();
  next_toks_.clear();
}

bool SimpleBackward::Backward(DecodableInterface *decodable) {
  InitBackward(decodable);
  while (remaining_frames_ > 0) {
    next_toks_.clear();
    curr_toks_.swap(next_toks_);
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &curr_toks_);
    UpdateBackwardTable();
  }
  return (!curr_toks_.empty());
}


void SimpleBackward::InitBackward(DecodableInterface *decodable) {
  curr_toks_.clear();
  next_toks_.clear();
  backward_.clear();
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    const double final_cost = fst_.Final(siter.Value()).Value();
    if (final_cost < fst::StdArc::Weight::Zero().Value()) {
      curr_toks_.insert(make_pair(siter.Value(), Token(final_cost)));
    }
  }
  remaining_frames_ = decodable->NumFramesReady();
  KALDI_ASSERT(decodable->IsLastFrame(remaining_frames_ - 1) &&
               "Backward cannot work with an online DecodableInterface");
  ProcessNonemitting();
}


double SimpleBackward::TotalCost() const {
  TokenMap::const_iterator tok = curr_toks_.find(fst_.Start());
  const double total_cost = tok != curr_toks_.end() ?
      tok->second.cost : -kaldi::kLogZeroDouble;
  if (total_cost != total_cost) { // NaN. This shouldn't happen; it indicates
                                  // some kind of error, most likely.
    KALDI_WARN << "Found NaN (likely failure in decoding)";
    return -kaldi::kLogZeroDouble;
  }
  return total_cost;
}


void SimpleBackward::ProcessEmitting(DecodableInterface *decodable) {
  // Processes emitting arcs for one frame.  Propagates from next_toks_ to
  // curr_toks_.
  const int32 frame = --remaining_frames_;
  // TODO(jpuigcerver): This does not take advantage of prunning!
  // Running time is O(|V| + |E|), where |V| is the number of states in
  // the FST and |E| the number of arcs.
  // This should be reduced to O(|K| + |E|), where |K| is the number of
  // active states after prunning.
  // However, I do not know how to iterate through the input arcs of a
  // given state using OpenFST. So, I cannot simply iterate through the
  // alive states in next_toks_, as I do in the Forward algorithm.
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    for (ArcIterator aiter(fst_, siter.Value()); !aiter.Done();
         aiter.Next()) {
      const StdArc& arc = aiter.Value();
      if (arc.ilabel == 0) continue;
      TokenMap::const_iterator ntok = next_toks_.find(arc.nextstate);
      if (ntok == next_toks_.end()) continue;
      const double acoustic_cost = -decodable->LogLikelihood(frame, arc.ilabel);
      TokenMap::iterator ctok = curr_toks_.insert(
          make_pair(siter.Value(), Token(-kaldi::kLogZeroDouble))).first;
      ctok->second.Update(ntok->second, arc, acoustic_cost);
    }
  }
}

void SimpleBackward::ProcessNonemitting() {
  /// WTF!!!
}


void SimpleBackward::UpdateBackwardTable() {
  /*backward_.push_back(unordered_map<Label, double>());
  for (TokenMap::const_iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    for (LabelMap::const_iterator ti = tok->second.ilabels.begin();
         ti != tok->second.ilabels.end(); ++ti) {
      LabelMap::iterator fi = backward_.back().insert(
          make_pair(ti->first, -kaldi::kLogZeroDouble)).first;
      fi->second = -kaldi::LogAdd(-fi->second, -ti->second);
    }
  }*/
}


// static
void SimpleBackward::PruneToks(BaseFloat beam, TokenMap *toks) {
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
