// fstext/deterministic-fst.h

// Copyright 2011-2012 Gilles Boulianne  Johns Hopkins University (author: Daniel Povey)

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
//
// This file includes material from the OpenFST Library v1.2.7 available at
// http://www.openfst.org and released under the Apache License Version 2.0.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2010 Google, Inc.
// Author: riley@google.com (Michael Riley)

#ifndef KALDI_FSTEXT_DETERMINISTIC_FST_H_
#define KALDI_FSTEXT_DETERMINISTIC_FST_H_

/* This header defines the DeterministicOnDemand interface,
   which is an FST with a special interface that allows
   only a single arc with a non-epsilon input symbol
   out of each state.
*/

#include <algorithm>
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
using std::tr1::unordered_map;

#include <string>
#include <utility>
#include <vector>

#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <fst/slist.h>

#include "util/stl-utils.h"

namespace fst {

/// \addtogroup deterministic_fst_group "Classes and functions related to on-demand deterministic FST's"
/// @{


/// class DeterministicOnDemandFst is an "FST-like" base-class.
/// It does not actually inherit from any Fst class because its
/// interface is not exactly the same (it doesn't have the
/// GetArc function).
/// It assumes that the FST can have only one arc for any
/// given input symbol, which makes the GetArc function below
/// possible.
/// Note: we don't use "const" in this interface, because
/// it creates problems when we do things like caching,
template<class Arc>
class DeterministicOnDemandFst {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;
  
  virtual StateId Start() = 0;

  virtual Weight Final(StateId s) = 0;

  /// Note: ilabel must not be epsilon.
  virtual bool GetArc(StateId s, Label ilabel, Arc *oarc) = 0;

  virtual ~DeterministicOnDemandFst() { }
};

/**
   This class wraps a conventional Fst, representing a
   language model, in the interface for "BackoffDeterministicOnDemandFst".
   We expect that backoff arcs in the language model will have the
   epsilon label (label 0) on the arcs, and that there will be
   no other epsilons in the language model.
   We follow the epsilon arcs if a particular arc (or a final-prob)
   is not found at the current state.
 */
template<class Arc>
class BackoffDeterministicOnDemandFst: public DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  
  BackoffDeterministicOnDemandFst(const Fst<Arc> &fst_);
  
  StateId Start() { return fst_.Start(); }

  Weight Final(StateId s);

  bool GetArc(StateId s, Label ilabel, Arc *oarc);
  
 private:
  inline StateId GetBackoffState(StateId s, Weight *w);
  
  const Fst<Arc> &fst_;
};

template<class Arc>
class ComposeDeterministicOnDemandFst: public DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;

  /// Note: constructor does not "take ownership" of the input fst's.  The input
  /// fst's should be treated as const, in that their contents, do not change,
  /// but they are not const as the DeterministicOnDemandFst's data-access
  /// functions are not const, for reasons relating to caching.
  ComposeDeterministicOnDemandFst(DeterministicOnDemandFst<Arc> *fst1,
                                  DeterministicOnDemandFst<Arc> *fst2);

  virtual StateId Start() { return start_state_; }

  virtual Weight Final(StateId s);
  
  virtual bool GetArc(StateId s, Label ilabel, Arc *oarc);

 private:
  DeterministicOnDemandFst<Arc> *fst1_;
  DeterministicOnDemandFst<Arc> *fst2_;
  typedef unordered_map<std::pair<StateId, StateId>, StateId, kaldi::PairHasher<StateId> > MapType;
  MapType state_map_;
  std::vector<std::pair<StateId, StateId> > state_vec_; // maps from
  // StateId to pair.
  StateId next_state_;
  StateId start_state_;
};
    
template<class Arc>
class CacheDeterministicOnDemandFst: public DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;
  
  /// We don't take ownership of this pointer.  The argument is "really" const.
  CacheDeterministicOnDemandFst(DeterministicOnDemandFst<Arc> *fst,
                                StateId num_cached_arcs = 100000);

  virtual StateId Start() { return fst_->Start(); }

  /// We don't bother caching the final-probs, just the arcs.
  virtual Weight Final(StateId s) { return fst_->Final(s); }
  
  virtual bool GetArc(StateId s, Label ilabel, Arc *oarc);
  
 private:
  // Get index for cached arc.
  inline size_t GetIndex(StateId src_state, Label ilabel);
  
  DeterministicOnDemandFst<Arc> *fst_;
  StateId num_cached_arcs_;  
  std::vector<std::pair<StateId, Arc> > cached_arcs_;
};
  

/// @}

}  // namespace fst

#include "deterministic-fst-inl.h"

#endif
