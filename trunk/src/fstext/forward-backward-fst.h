// fb/forward-backward-fst.h

// Copyright 2015  Joan Puigcerver

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

#ifndef KALDI_FSTEXT_FORWARD_BACKWARD_FST_H_
#define KALDI_FSTEXT_FORWARD_BACKWARD_FST_H_

#include <string>
#include <vector>
using std::vector;

#include "fst/mutable-fst.h"
#include "fst/test-properties.h"

namespace fst {

template <class A, class U> class ForwardBackwardFst;
template <class F, class G> void Cast(const F &, G *);


/// Forward-Backward FST needs a special arc type that keeps track of both
/// input (nextstate) and output (prevstate) of each arc.
/// This arc has exactly the same attributes than other arcs, plus includes
/// the prevstate attribute.
template <class A>
struct ForwardBackwardArc : public A {
  typedef typename A::Weight Weight;
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;

  ForwardBackwardArc() : A() {}
  explicit ForwardBackwardArc(const A& arc) : A(arc) {}
  ForwardBackwardArc(const A& arc, StateId p) : A(arc), prevstate(p) {}
  ForwardBackwardArc(Label i, Label o, const Weight& w, StateId n, StateId p)
      : A(i, o, w, n), prevstate(p) {}

  StateId prevstate;
};

/// Implementation of the Forward-Backward FST. Notice that it can be
/// instantiated with any arc type (A), but internally it will use the
/// ForwardBackwardArc inherited from A.
template <class A, class U>
class ForwardBackwardFstImpl :
      public FstImpl< ForwardBackwardArc<A> > {
 public:
  typedef typename A::Weight Weight;
  typedef typename A::StateId StateId;
  typedef ForwardBackwardArc<A> Arc;

  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;
  using FstImpl<Arc>::Properties;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::SetType;

  friend class MutableArcIterator< ForwardBackwardFst<A, U> >;

 private:
  struct State {
    vector<Arc*> iarcs_;
    vector<Arc*> oarcs_;
    Weight final_;
    U niepsilons_;
    U noepsilons_;
  };

 public:

  ForwardBackwardFstImpl() : start_(kNoStateId) {
    SetType("forwardbackward");
    SetProperties(kNullProperties | kStaticProperties);
  }

  explicit ForwardBackwardFstImpl(const Fst<A>& fst);

  static ForwardBackwardFstImpl<A, U>* Read(
      istream& strm, const FstReadOptions& opts);

  size_t NumInputEpsilons(StateId s) const { return states_[s]->niepsilons_; }

  size_t NumOutputEpsilons(StateId s) const { return states_[s]->noepsilons_; }

  StateId Start() const { return start_; }

  Weight Final(StateId s) const { return states_[s]->final_; }

  StateId NumStates() const { return states_.size(); }

  template <bool input_arcs = false>
  size_t NumArcs(StateId s) {
    return input_arcs ? states_[s]->iarcs_.size() : states_[s]->oarcs_.size();
  }

  void SetStart(StateId s) {
    start_ = s;
    SetProperties(SetStartProperties(Properties()));
  }

  void SetFinal(StateId s, Weight w) {
    const Weight& ow = states_[s]->final_;
    states_[s]->final_ = w;
    SetProperties(SetFinalProperties(Properties(), ow, w));
  }

  StateId AddState() {
    states_.push_back(new State);
    SetProperties(AddStateProperties(Properties()));
    return states_.size() - 1;
  }

  // Add arc using the
  void AddArc(StateId s, const A& arc) {
    const StateId n = arc.nextstate;
    Arc* new_arc = new Arc(arc, s);
    const Arc *prev_arc =
        states_[s]->oarcs_.empty() ? 0 : states_[s]->oarcs_.back();
    states_[s]->oarcs_.push_back(new_arc);
    states_[s]->iarcs_.push_back(new_arc);
    SetProperties(AddArcProperties(Properties(), s, *new_arc, prev_arc));
  }

  void DeleteStates(const vector<StateId>& dstates) {
    // Get new id for all states (deleted states will have kNoStateId)
    vector<StateId> newid(states_.size(), 0);
    for (size_t i = 0; i < dstates.size(); ++i) {
      newid[dstates[i]] = kNoStateId;
    }
    // Delete states (and its input/output arcs)
    StateId nstates = 0;
    for (size_t s = 0; s < states_.size(); ++s) {
      if (newid[s] != kNoStateId) {
        // change state id
        newid[s] = nstates;
        if (s != nstates)
          states_[nstates] = states_[s];
        ++nstates;
      } else {
        // Delete input/output arcs to/from s
        DeleteArcs<true>(s);
        DeleteArcs<false>(s);
        // Now, delete state
        delete states_[s];
      }
    }
    states_.resize(nstates);
    // We need to fix the nextnode/prevnode from the arcs
    for (size_t s = 0; s < states_.size(); ++s) {
      vector<Arc*>& arcs = states_[s]->oarcs_;
      for (size_t a = 0; a < arcs.size(); a++) {
        arcs[a]->prevstate = s;
        arcs[a]->nextstate = newid[arcs[a]->nextstate];
      }
    }
    // Fix start state
    if (Start() != kNoStateId)
      SetStart(newid[Start()]);
    SetProperties(DeleteStatesProperties(Properties()));
  }

  void DeleteStates() {
    for (StateId s = 0; s < states_.size(); ++s) {
      for (size_t a = 0; a < states_[s]->oarcs_.size(); ++a)
        delete states_[s]->oarcs_[a];
      delete states_[s];
    }
    states_.clear();
    SetStart(kNoStateId);
    SetProperties(DeleteAllStatesProperties(Properties(), kStaticProperties));
  }

  /// Delete last n input/output arcs to/from state s
  template <bool input_arcs = false>
  void DeleteArcs(StateId s, size_t n) {
    // Delete input or output arcs from state s?
    vector<Arc*>& arcs = input_arcs ? states_[s]->iarcs_ : states_[s]->oarcs_;
    // Make sure n <= arcs.size()
    if (n > arcs.size()) n = arcs.size();
    // Traverse all the arcs that we want to delete
    for (size_t a = arcs.size() - n; a < arcs.size(); ++a) {
      // Arc to delete
      Arc* arc = arcs[a];
      // Before deleting the arc, we need to remove it from the
      // outputs/inputs of the previus/next state
      vector<Arc*>& arcs2 = input_arcs ?
          states_[arc->prevstate]->oarcs_ :
          states_[arc->nextstate]->iarcs_;
      arcs2.erase(find(arcs2.begin(), arcs2.end(), arc));
      // Now, we can free the memory
      delete arc;
    }
    arcs.resize(arcs.size() - n);
    SetProperties(DeleteArcsProperties(Properties()));
  }

  // Delete all input/output arcs to/from state s
  template <bool input_arcs = false>
  void DeleteArcs(StateId s) {
    DeleteArcs<input_arcs>(s, std::numeric_limits<size_t>::max());
  }

  State* GetState(StateId s) { return states_[s]; }

  const State* GetState(StateId s) const { return states_[s]; }

  void ReserveStates(StateId n) { states_.reserve(n); }

  template <bool input_arcs = false>
  void ReserveArcs(StateId s, size_t n) {
    if (input_arcs)
      states_[s]->iarcs_.reserve(n);
    else
      states_[s]->oarcs_.reserve(n);
  }

  ///  Provide information needed for generic state iterator
  void InitStateIterator(StateIteratorData<Arc> *data) const {
    data->base = 0;
    data->nstates = states_.size();
  }

  ///  Provide information needed for generic arc iterator
  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const {
    data->base = 0;
    data->narcs = states_[s]->oarcs_.size();
    data->arcs = data->narcs > 0 ? &states_[s]->oarcs_[0] : 0;
    data->ref_count = 0;
  }

  static const uint64 kStaticProperties = kExpanded | kMutable;

 private:
  ///  Current file format version
  static const int kFileVersion = 1;
  ///  Minimum file format version supported
  static const int kMinFileVersion = 1;

  vector<Arc> arcs_;
  vector<State*> states_;
  StateId start_;

  DISALLOW_COPY_AND_ASSIGN(ForwardBackwardFstImpl);
};

template <class A, class U>
const uint64 ForwardBackwardFstImpl<A, U>::kStaticProperties;
template <class A, class U>
const int ForwardBackwardFstImpl<A, U>::kFileVersion;
template <class A, class U>
const int ForwardBackwardFstImpl<A, U>::kMinFileVersion;

template <class A, class U>
ForwardBackwardFstImpl<A, U>::ForwardBackwardFstImpl(const Fst<A>& fst) {
  SetType("forwardbackward");
  SetInputSymbols(fst.InputSymbols());
  SetOutputSymbols(fst.OutputSymbols());
  SetStart(fst.Start());
  if (fst.Properties(kExpanded, false))
    ReserveStates(CountStates(fst));

  for (StateIterator< Fst<A> > siter(fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    assert(AddState() == s);
    SetFinal(s, fst.Final(s));
    ReserveArcs(s, fst.NumArcs(s));
    for (ArcIterator< Fst<A> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const A& arc = aiter.Value();
      AddArc(s, arc);
    }
  }

  SetProperties(fst.Properties(kCopyProperties, false) | kStaticProperties);
}

// static
template <class A, class U>
ForwardBackwardFstImpl<A, U>* ForwardBackwardFstImpl<A, U>::Read(
    istream& strm, const FstReadOptions& opts) {
  ForwardBackwardFstImpl<A, U> *impl = new ForwardBackwardFstImpl<A, U>;
  FstHeader hdr;
  if (!impl->ReadHeader(strm, opts, kMinFileVersion, &hdr)) {
    delete impl;
    return 0;
  }
  impl->SetStart(hdr.Start());
  if (hdr.NumStates() != kNoStateId)
    impl->ReserveStates(hdr.NumStates());
  StateId s = 0;
  for (; hdr.NumStates() == kNoStateId || s < hdr.NumStates(); ++s) {
    // Read final weight
    typename A::Weight final;
    if (!final.Read(strm)) break;
    assert(impl->AddState() == s);
    impl->states_[s]->final = final;
    // Read state arcs
    int64 narcs;
    ReadType(strm, &narcs);
    if (!strm) {
      LOG(ERROR) << "ForwardBackwardFst::Read: read failed: " << opts.source;
      delete impl;
      return 0;
    }
    impl->ReserveArcs(s, narcs);
    for (size_t j = 0; j < narcs; ++j) {
      A arc;
      ReadType(strm, &arc.ilabel);
      ReadType(strm, &arc.olabel);
      arc.weight.Read(strm);
      ReadType(strm, &arc.nextstate);
      if (!strm) {
        LOG(ERROR) << "ForwardBackwardFst::Read: read failed: " << opts.source;
        delete impl;
        return 0;
      }
      impl->AddArc(s, arc);
    }
  }
  if (hdr.NumStates() != kNoStateId && s != hdr.NumStates()) {
    LOG(ERROR) << "ForwardBackwardFst::Read: unexpected end of file: "
               << opts.source;
    delete impl;
    return 0;
  }
  return impl;
}


template <class A, class U = size_t>
class ForwardBackwardFst :
      public ImplToMutableFst< ForwardBackwardFstImpl<A, U> > {
 public:
  friend class StateIterator< ForwardBackwardFst<A, U> >;
  friend class ArcIterator< ForwardBackwardFst<A, U> >;
  friend class MutableArcIterator< ForwardBackwardFst<A, U> >;

  typedef typename ForwardBackwardFstImpl<A, U>::Arc Arc;
  typedef typename A::StateId StateId;
  typedef ForwardBackwardFstImpl<A, U> Impl;

  ForwardBackwardFst() : ImplToMutableFst<Impl>(new Impl) {}

  explicit ForwardBackwardFst(const Fst<A>& fst)
      : ImplToMutableFst<Impl>(new Impl(fst)) {}

  ForwardBackwardFst(const ForwardBackwardFst<A, U>& fst)
      : ImplToMutableFst<Impl>(fst) {}

  ///  Get a copy of this ForwardBackwardFst
  virtual ForwardBackwardFst<A, U>* Copy(bool safe = false) const {
    return new ForwardBackwardFst<A, U>(*this);
  }

  ForwardBackwardFst<A, U>& operator=(const ForwardBackwardFst<A, U>& fst) {
    SetImpl(fst.GetImpl(), false);
    return *this;
  }

  virtual ForwardBackwardFst<A, U>& operator=(const Fst<A>& fst) {
    if (this != &fst) SetImpl(new Impl(fst));
    return *this;
  }

  ///  Read a ForwardBackwardFst from an input stream; return NULL on error
  static ForwardBackwardFst<A, U>* Read(
      istream &strm, const FstReadOptions &opts) {
    Impl* impl = Impl::Read(strm, opts);
    return impl ? new ForwardBackwardFst<A, U>(impl) : 0;
  }

  ///  Read a ForwardBackwardFst from a file; return NULL on error
  ///  Empty filename reads from standard input
  static ForwardBackwardFst<A, U> *Read(const string &filename) {
    Impl* impl = ImplToExpandedFst<Impl, MutableFst<A> >::Read(filename);
    return impl ? new ForwardBackwardFst<A, U>(impl) : 0;
  }

  virtual bool Write(ostream& strm, const FstWriteOptions& opts) const {
    return WriteFst(*this, strm, opts);
  }

  virtual bool Write(const string& filename) const {
    //return Fst<A>::WriteFile(filename);
    return false;
  }

  template <class F>
  static bool WriteFst(
      const F& fst, ostream& strm, const FstWriteOptions& opts) {
    return false;
  }

  void ReserveStates(StateId n) {
    MutateCheck();
    GetImpl()->ReserveStates(n);
  }

   void ReserveArcs(StateId s, size_t n) {
     MutateCheck();
     GetImpl()->ReserveArcs(s, n);
   }

   virtual void InitStateIterator(StateIteratorData<Arc> *data) const {
     GetImpl()->InitStateIterator(data);
   }

   virtual void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const {
     GetImpl()->InitArcIterator(s, data);
   }

  virtual inline
  void InitMutableArcIterator(StateId s, MutableArcIteratorData<Arc>* data) {
    //data->base = new MutableArcIterator< ForwardBackwardFst<A, U> >(this, s);
  }

 private:
  explicit ForwardBackwardFst(Impl *impl) : ImplToMutableFst<Impl>(impl) {}

  ///  Makes visible to friends.
  Impl *GetImpl() const {
    return ImplToFst< Impl, MutableFst< ForwardBackwardArc<A> > >::GetImpl();
  }

  void SetImpl(Impl *impl, bool own_impl = true) {
    ImplToFst< Impl, MutableFst<A> >::SetImpl(impl, own_impl);
  }

  void MutateCheck() { return ImplToMutableFst<Impl>::MutateCheck(); }
};

///  Specialization for ForwadBackwardFst; see generic version in fst.h
///  for sample usage (but use the ForwardBackwardFst type!). This version
///  should inline.
template <class A, class U>
class StateIterator< ForwardBackwardFst<A, U> > {
 public:
  typedef typename A::StateId StateId;

  explicit StateIterator(const ForwardBackwardFst<A, U> &fst)
      : nstates_(fst.GetImpl()->NumStates()), s_(0) {}

  bool Done() const { return s_ >= nstates_; }

  StateId Value() const { return s_; }

  void Next() { ++s_; }

  void Reset() { s_ = 0; }

 private:
  StateId nstates_;
  StateId s_;

  DISALLOW_COPY_AND_ASSIGN(StateIterator);
};

///  Specialization for ForwardBackwardFst; see generic version in fst.h
///  for sample usage (but use the ForwardBackwardFst type!). This version
///  should inline.
template <class A, class U>
class ArcIterator< ForwardBackwardFst<A, U> > {
 public:
  typedef typename ForwardBackwardFst<A, U>::Arc Arc;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;

  ArcIterator(const ForwardBackwardFst<A, U> &fst, StateId s)
      : arcs_(fst.GetImpl()->states_[s]->oarcs_), i_(0) {}

  bool Done() const { return i_ >= arcs_.size(); }

  const Arc& Value() const { return *arcs_[i_]; }

  void Next() { ++i_; }

  void Reset() { i_ = 0; }

  void Seek(size_t a) { i_ = a; }

  size_t Position() const { return i_; }

  uint32 Flags() const { return kArcValueFlags; }

  void SetFlags(uint32 f, uint32 m) {}

 private:
  const vector<Arc*>& arcs_;
  size_t i_;

  DISALLOW_COPY_AND_ASSIGN(ArcIterator);
};

template <class A, class U>
using OutputArcIterator = ArcIterator<A>;

/// InputArcIterator
template <class A, class U>
class InputArcIterator {
 public:
  typedef typename ForwardBackwardFst<A, U>::Arc Arc;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;

  InputArcIterator(const ForwardBackwardFst<A, U> &fst, StateId s)
      : arcs_(fst.GetImpl()->states_[s]->iarcs_), i_(0) {}

  bool Done() const { return i_ >= arcs_.size(); }

  const Arc& Value() const { return *arcs_[i_]; }

  void Next() { ++i_; }

  void Reset() { i_ = 0; }

  void Seek(size_t a) { i_ = a; }

  size_t Position() const { return i_; }

  uint32 Flags() const { return kArcValueFlags; }

  void SetFlags(uint32 f, uint32 m) {}

 private:
  const vector<Arc*>& arcs_;
  size_t i_;

  DISALLOW_COPY_AND_ASSIGN(InputArcIterator);
};

// REGISTER_FST(ForwardBackwardFst, StdArc);

typedef ForwardBackwardFst<StdArc, size_t> StdForwardBackwardFst;
typedef ForwardBackwardFst<LogArc, size_t> LogForwardBackwardFst;

}  // namespace fst


#endif  // KALDI_FSTEXT_FORWARD_BACKWARD_FST_H_
