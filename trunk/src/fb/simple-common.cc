// fb/simple-common.cc

#include "fb/simple-common.h"

namespace kaldi {

void Token::UpdateEmitting(
    const Label label, const BaseFloat prev_cost, const BaseFloat edge_cost,
    const BaseFloat acoustic_cost) {
  const BaseFloat inc_cost = prev_cost + edge_cost + acoustic_cost;
  if (inc_cost == -kaldi::kLogZeroBaseFloat) return;
  // Update total cost to the state, using input symbol `label'
  LabelMap::iterator l = ilabels.insert(make_pair(
      label, -kaldi::kLogZeroBaseFloat)).first;
  l->second = -kaldi::LogAdd(-l->second, -inc_cost);
  // Update total cost to the state
  cost = -kaldi::LogAdd(-cost, -inc_cost);
}


bool Token::UpdateNonEmitting(
    const LabelMap& parent_ilabels, const BaseFloat prev_cost,
    const BaseFloat edge_cost, const BaseFloat threshold) {
  const BaseFloat old_cost = cost;
  // Propagate all the parent input symbols to this state, since we are
  // using a epsilon-transition
  for (unordered_map<Label, BaseFloat>::const_iterator pl =
           parent_ilabels.begin(); pl != parent_ilabels.end(); ++pl) {
    const BaseFloat inc_cost = pl->second + edge_cost;
    // Total cost using symbol `pl->first' to this state
    LabelMap::iterator l =
        ilabels.insert(make_pair(pl->first, -kaldi::kLogZeroBaseFloat)).first;
    l->second = -kaldi::LogAdd(-l->second, -inc_cost);
    // Cost since the last time this state was extracted from the search
    // queue (see [1]).
    l = last_ilabels.insert(make_pair(pl->first, -kaldi::kLogZeroBaseFloat)).first;
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
      if (l->second == -kaldi::kLogZeroBaseFloat) continue;
      BaseFloat& x = acc->insert(make_pair(
          l->first, -kaldi::kLogZeroBaseFloat)).first->second;
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
  BaseFloat best_cost = tok->second.cost;
  for (++tok; tok != toks->end(); ++tok) {
    best_cost = std::min(best_cost, tok->second.cost);
  }
  // Mark all tokens with cost greater than the cutoff
  std::vector<TokenMap::const_iterator> remove_toks;
  const BaseFloat cutoff = best_cost + beam;
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


BaseFloat RescaleToks(TokenMap* toks) {
  // Compute scale constant
  BaseFloat scale = -kaldi::kLogZeroBaseFloat;
  for (TokenMap::iterator t = toks->begin(); t != toks->end(); ++t) {
    scale = -kaldi::LogAdd(-scale, -t->second.cost);
  }
  // Rescale tokens and labels
  for (TokenMap::iterator t = toks->begin(); t != toks->end(); ++t) {
    t->second.cost -= scale;
    LabelMap* lbls = &t->second.ilabels;
    for (LabelMap::iterator l = lbls->begin(); l != lbls->end(); ++l) {
      l->second -= scale;
    }
  }
  return scale;
}

void ComputePosteriorgram(
    const std::vector<LabelMap>& fwd, const std::vector<LabelMap>& bkw,
    std::vector<LabelMap>* pst) {
  KALDI_ASSERT(fwd.size() == bkw.size());
  pst->clear();
  pst->resize(fwd.size());
  for (size_t t = 0; t < pst->size(); ++t) {
    const LabelMap& f_t = fwd[t];
    const LabelMap& b_t = bkw[t];
    BaseFloat sum_t = kaldi::kLogZeroBaseFloat;
    for (LabelMap::const_iterator f_l = f_t.begin(); f_l != f_t.end(); ++f_l) {
      LabelMap::const_iterator b_l = b_t.find(f_l->first);
      // Skip products that will result into 0
      if (b_l == b_t.end() || f_l->second == -kaldi::kLogZeroBaseFloat ||
          b_l->second == -kaldi::kLogZeroBaseFloat)
        continue;
      // Compute total cost and change the sign to transform it into a
      // probability
      const BaseFloat p = -(f_l->second + b_l->second);
      (*pst)[t].insert(make_pair(f_l->first, p));
      sum_t = kaldi::LogAdd(sum_t, p);
    }
    for (LabelMap::iterator p_l = (*pst)[t].begin(); p_l != (*pst)[t].end();
         ++p_l) {
      p_l->second -= sum_t;
    }
  }
}


}  // namespace kaldi
