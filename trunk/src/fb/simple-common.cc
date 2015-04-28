// fb/simple-common.cc

#include "fb/simple-common.h"

namespace kaldi {


void Token::UpdateEmitting(
    const Label label, const double prev_cost, const double edge_cost,
    const double acoustic_cost) {
  const double inc_cost = prev_cost + edge_cost + acoustic_cost;
  if (inc_cost == -kaldi::kLogZeroDouble) return;
  // Update total cost to the state, using input symbol `label'
  LabelMap::iterator l = ilabels.insert(make_pair(
      label, -kaldi::kLogZeroDouble)).first;
  l->second = -kaldi::LogAdd(-l->second, -inc_cost);
  // Update total cost to the state
  cost = -kaldi::LogAdd(-cost, -inc_cost);
}


bool Token::UpdateNonEmitting(
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


void AccumulateToks(const TokenMap& toks, LabelMap *acc) {
  KALDI_ASSERT(acc);
  for (TokenMap::const_iterator t = toks.begin(); t != toks.end(); ++t) {
    const LabelMap& lbls = t->second.ilabels;
    for (LabelMap::const_iterator l = lbls.begin(); l != lbls.end(); ++l) {
      if (l->second == -kaldi::kLogZeroDouble) continue;
      double& x = acc->insert(make_pair(
          l->first, -kaldi::kLogZeroDouble)).first->second;
      x = -kaldi::LogAdd(-x, -l->second);
    }
  }
}


void PruneToks(BaseFloat beam, TokenMap *toks) {
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
    if (tok->second.cost > cutoff) {
      remove_toks.push_back(tok);
    }
  }
  // Prune tokens
  for (size_t i = 0; i < remove_toks.size(); ++i) {
    toks->erase(remove_toks[i]);
  }
  KALDI_VLOG(2) <<  "Pruned " << remove_toks.size() << " to "
                << toks->size() << " toks.\n";
}


double RescaleToks(TokenMap* toks) {
  // Compute scale constant
  double scale = -kaldi::kLogZeroDouble;
  for (TokenMap::iterator t = toks->begin(); t != toks->end(); ++t) {
    scale = -kaldi::LogAdd(-scale, -t->second.cost);
  }
  for (TokenMap::iterator t = toks->begin(); t != toks->end(); ++t) {
    t->second.cost -= scale;
    LabelMap* lbls = &t->second.ilabels;
    for (LabelMap::iterator l = lbls->begin(); l != lbls->end(); ++l) {
      l->second -= scale;
    }
  }
  return scale;
}


}  // namespace kaldi
