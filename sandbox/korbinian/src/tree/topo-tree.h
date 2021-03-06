// tree/topo-tree.h

// Copyright 2013  Korbinian Riedhammer

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

#ifndef KALDI_TREE_TOPO_TREE_H_
#define KALDI_TREE_TOPO_TREE_H_

#include "itf/context-dep-itf.h"
#include "matrix/matrix-lib.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "tree/cluster-utils.h"
#include "tree/event-map.h"

#include <algorithm>
#include <utility>
#include <map>
#include <string>
#include <sstream>

/*
  This header provides the declarations for the class TopoTree, which inherits
  from the interface class "ContextDependencyInterface" in itf/context-dep-itf.h.
  In contrast to the ContextDependency class, the TopoTree is a polyphone
  topology organized as a multi-tree, where the context grows with each step
  down the tree, e.g., /a/ -> b/a/ -> cb/a/ -> dcb/a/.  This is different from
  a conventional tree, as there are no splits in a traditional sense, but the
  tree returns the most detailed polyphone model given a phone sequence.
  For example,
    /h e L o/ would be translated to
    /h/eLo h/e/Lo he/L/o heL/o/
  This type of tree is analoguous to the phonetic network of polyphones as in
  Schukat-Tallamazini '95 "Automatische Spracherkennung".
*/


namespace kaldi {

/**
 * Generate a string representation of an EventType, used for debugging.
 */
std::string EventTypeToString(const EventType &e, int32 P);

/**
 * Result of the comparison of two EventTypes;  note that this comparison is not
 * symmetric or transitive.
 */
class EventTypeComparison {
 public:
  EventTypeComparison():
    num_generalizations_(0),
    num_specializations_(0),
    num_incompatibilities_(0),
    left_generalization_(false),
    left_specialization_(false),
    right_generalization_(false),
    right_specialization_(false),
    phone_ref_(0),
    phone_chk_(0),
    pdf_class_ref_(kNoPdf),
    pdf_class_chk_(kNoPdf) {

  }

  EventTypeComparison(const EventType &ref, const EventType &chk, int32 P) {
    compare(ref, chk, P);
  }

  EventTypeComparison(const EventTypeComparison &copy):
    num_generalizations_(copy.num_generalizations_),
    num_specializations_(copy.num_specializations_),
    num_incompatibilities_(copy.num_incompatibilities_),
    left_generalization_(copy.left_generalization_),
    left_specialization_(copy.left_specialization_),
    right_generalization_(copy.right_generalization_),
    right_specialization_(copy.right_specialization_),
    phone_ref_(copy.phone_ref_),
    phone_chk_(copy.phone_chk_),
    pdf_class_ref_(copy.pdf_class_ref_),
    pdf_class_chk_(copy.pdf_class_chk_) {

  }

  /**
   * Compare two EventType instances and internally store the results.
   */
  void compare(const EventType &ref, const EventType &chk, int32 P);

  inline bool PhoneMatch() {
    return phone_ref_ == phone_chk_;
  }

  inline bool PdfClassMatch() {
    return pdf_class_ref_ == pdf_class_chk_;
  }

  bool StrictPhoneMatch() {
    return PhoneMatch() && PdfClassMatch();
  }

  /**
   * Returns true if EventType chk is a generalization of ref ("chk generalizes ref"),
   * e.g.: EventTypeComparison("b/a/", "/a/").IsGeneralization(): true
   * Note that generalization and specialization are not the inverse.
   */
  bool IsGeneralization() {
    if (!PhoneMatch() || num_incompatibilities_ > 0 || num_specializations_ > 0)
      return false;

    // chk is a generalization of ref, if there is at least one generalization
    // or no generalization but chk is the general phone w/out pdf class
    return num_generalizations_ > 0;
  }

  /**
   * Returns true if EventType chk is a specialization of ref ("chk specializes ref"),
   * e.g.: EventTypeComparison("/b/", "a/b/").IsSpecialization(): true
   * Note that generalization and specialization are not the inverse.
   */
  bool IsSpecialization() {
    if (!PhoneMatch() || num_incompatibilities_ > 0 || num_generalizations_ > 0)
      return false;

    // chk is a specialization of ref, if there is at least one specialization
    // or no specialization but ref is the general phone w/out pdf class
    return num_specializations_ > 0;
  }

  /**
   * Returns zero if undefined, -1 for left and +1 for right context
   * specialization
   */
  int32 IsPartialGeneralization() {
    if (left_generalization_)
      return -1;
    else if (right_generalization_)
      return  1;
    else
      return 0;
  }

  /**
   * Returns zero if undefined, -1 for left and +1 for right context
   * specialization
   */
  int32 IsPartialSpecialization() {
    if (left_specialization_)
      return -1;
    else if (right_specialization_)
      return  1;
    else
      return 0;
  }

  /**
   * Returns true if EventType chk "fits" the EventType ref, i.e., if the phone
   * and pdf class, as well as the context left and right match or generalize.
   * For example, b/c/d fits b/c/d, ab/b/cd, etc.
   */
  bool Fits() {
    return
        (StrictPhoneMatch() || PhoneMatch() && pdf_class_chk_ == kNoPdf) &&
        num_incompatibilities_ == 0 &&
        num_specializations_ == 0 &&
        num_generalizations_ >= 0;
  }

  inline EventValueType PhoneRef() { return phone_ref_; }
  inline EventValueType PhoneChk() { return phone_chk_; }
  inline EventValueType PdfClassRef() { return pdf_class_ref_; }
  inline EventValueType PdfClassChk() { return pdf_class_chk_; }

  inline int32 NumGeneralizations() { return num_generalizations_; }
  inline int32 NumSpecializations() { return num_specializations_; }
  inline int32 NumIncompatibilities() { return num_incompatibilities_; }

 private:
  int32 num_generalizations_;
  int32 num_specializations_;
  int32 num_incompatibilities_;

  bool left_generalization_;
  bool left_specialization_;

  bool right_generalization_;
  bool right_specialization_;


  EventValueType phone_ref_;
  EventValueType phone_chk_;

  EventValueType pdf_class_ref_;
  EventValueType pdf_class_chk_;


};

/**
 * Sort the specializations by ascending context size (first) and descending
 * size of left context (second).  This is implemented via the "balance" of
 * the poly-phone, i.e. len(ctx-right) - len(ctx-left).
 * Forward declaration.
 */
class TopoNodeComparison;

/**
 * A TopoNode consists of a corresponding EventType and PdfId, along with
 * pointers to its generalization and specializations (if any).
 */
class TopoNode {
 public:
  TopoNode(const EventType &event_type, TopoNode *generalization):
    event_type_(event_type), pdf_id_(kNoPdf), generalization_(generalization) {

  }


  TopoNode(const EventType &event_type):
    event_type_(event_type), pdf_id_(kNoPdf), generalization_(NULL) {

  }

  TopoNode(TopoNode *generalization):
    pdf_id_(kNoPdf), generalization_(generalization) {

  }

  inline int32 PdfId() {
    return pdf_id_;
  }

  inline bool IsRoot() {
    return event_type_[0].second == kNoPdf;
  }

  inline bool IsLeaf() {
    return specializations_.size() == 0;
  }

  inline bool IsVirtual() {
    return pdf_id_ == kNoPdf;
  }

  inline void SetPdfId(int32 pdf_id) {
    pdf_id_ = pdf_id;
  }

  /// Read TopoTree object from disk;  throws on error
  void Read (std::istream &is, bool binary);

  void ReadCompressed(std::istream &is, bool binary, int32 N);

  /// Write TopoTree object to disk;  throws error
  void Write (std::ostream &os, bool binary) const;

  void WriteCompressed(std::ostream &os, bool binary) const;

 protected:
  /// Corresponding EventType, may be RootEventType.
  EventType event_type_;

  /// Corresponding PdfId, or kNoPdf for virtual nodes.
  int32 pdf_id_;

  TopoNode *generalization_;
  std::vector<TopoNode *> specializations_;

  /**
   * Get a list of all specializations below this node.  Iterative traversal.
   */
  int32 TraverseSpecializations(std::vector<TopoNode *> *node_list);

  /**
   * Clear and delete all specializations;  recursive call.
   */
  void Clear();

  /**
   * Set all pointers to NULL (no delete!)
   */
  void ClearPointers() {
    generalization_ = NULL;
    specializations_.clear();
  }


 protected:
  TopoNode(): pdf_id_(kNoPdf) { }

  // make sure the TopoTree can assign pdf ids etc.
  friend class TopoTree;
  friend class TopoNodeComparison;
  friend bool operator==(const TopoNode &lhs, const TopoNode& rhs);
};

inline bool operator==(const TopoNode& lhs, const TopoNode& rhs) {
  return lhs.event_type_ == rhs.event_type_;
}


/**
 * The TopoTree holds a topological n-ary tree of EventTypes and their
 * associated PdfIds.  It is used to map phones in context to their acoustic
 * model correlates.
 */
class TopoTree : public ContextDependencyInterface {
 public:
  virtual int32 ContextWidth() const { return N_; }
  virtual int32 CentralPosition() const { return P_; }

  /**
   * Query the tree for a pdf id for the given phone sequence and pdf class.
   * @return true on success, false if phone sequence is not mappable.
   */
  virtual bool Compute(const std::vector<int32> &phoneseq,
                       int32 pdf_class, int32 *pdf_id) const;

  /**
   * Query the tree for a TopoNode that matches the given EventType.
   * @return Matching node, or NULL on failure.
   */
  virtual TopoNode *Compute(const EventType &event_type) const;

  /**
   * Get the number of non-virtual tree nodes, which is resembles the actual
   * states.
   */
  virtual int32 NumPdfs() const {
    return num_pdfs_;
  }

  /**
   * GetPdfInfo returns a vector indexed by pdf-id, saying for each pdf which
   * pairs of (phone, pdf-class) it can correspond to.  (Usually just one).
   * c.f. hmm/hmm-topology.h for meaning of pdf-class.
   */
  void GetPdfInfo(const std::vector<int32> &phones,  // list of phones
                  const std::vector<int32> &num_pdf_classes,  // indexed by phone,
                  std::vector<std::vector<std::pair<int32, int32> > > *pdf_info) const;

  // Constructor takes ownership of pointers.
  TopoTree(int32 N, int32 P, std::map<EventValueType, TopoNode *> roots):
    N_(N), P_(P), num_pdfs_(0), roots_(roots) {

  }

  TopoTree(int32 N, int32 P):
    N_(N), P_(P), num_pdfs_(0) {

  }

  // Constructor with no arguments; will normally be called
  // prior to Read()
  TopoTree(): N_(0), P_(0), num_pdfs_(0) {

  }

  virtual ContextDependencyInterface *Copy() const {
    return new TopoTree(N_, P_, roots_);
  }

  virtual ~TopoTree() {
    Clear();
  }

  /// Read TopoTree object from disk;  throws on error
  void Read (std::istream &is, bool binary);

  static ContextDependencyInterface *ReadInstance(std::istream &is, bool binary);

  /// Write TopoTree object to disk;  throws error
  void Write (std::ostream &os, bool binary) const;

  /**
   * Find the specialization that fits the referenced EventType, or return NULL.
   */
  TopoNode *FindSpecialization(const TopoNode *node, const EventType &event_type) const;

  /**
   * Insert a new EventType to the tree.  Returns false if the node already existed.
   */
  bool Insert(const EventType &event_type);

  /**
   * Remove a certain EventType from the tree.  This is not equal to Virtualize.
   */
  bool Remove(const EventType &event_type);

  /**
   * Make a certain EventType virtual.  This is not equal to Remove.
   */
  bool Virtualize(const EventType &event_type);

  /**
   * Print the TopoTree;  this is a bit more readable than Write(std::cout, false)
   */
  void Print(std::ostream &out);

  /**
   * Populate all non-virtual (pdf_id_ != kNoPdf) with PdfIds.
   */
  int32 Populate();

  /**
   * Fill the tree with all intermediate phones in context given the leaves, e.g.
   * /a/ -> xx/a/xx will be filled with /a/ -> x/a/ -> x/a/x -> ...
   */
  void Fill();

  /**
   * Clear all elements of the tree.
   */
  void Clear() {
    for (std::map<EventValueType, TopoNode *>::iterator it = roots_.begin();
        it != roots_.end(); it++) {
      it->second->Clear();
      delete it->second;
    }
  }

  void VirtualizeRootNodes() {
    for (std::map<EventValueType, TopoNode *>::iterator it = roots_.begin();
        it != roots_.end(); it++) {
      it->second->SetPdfId(kNoPdf);
      delete it->second;
    }
  }

 private:
  /**
   * Insert a TopoNode to the given TopoNode.  Returns false if the node already existed.
   */
  bool Insert(TopoNode *target, TopoNode *node);

  /**
   * Query the given node for a TopoNode that matches the given EventType.
   * @return Matching node, or NULL on failure.
   */
  virtual TopoNode *Compute(TopoNode *node, const EventType &event_type) const;

  /// Acoustic context size
  int32 N_;

  /// Center of phonetic context window
  int32 P_;

  /// Number of Pdfs in this tree (0 if not populated).
  int32 num_pdfs_;

  /// Map of RootEventType to TopoNode
  std::map<EventValueType, TopoNode *> roots_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(TopoTree);
};


/**
 * Generate the root EventType, i.e. a copy of the EventType, where all but the
 * context center phone are the boundary (0) phone.
 */
void RootEventType(const EventType &event_type_in, EventType *event_type_out, int32 P);


/**
 * Generalize an EventType from the left or right by setting the first non-zero
 * phone to zero and thus making it a boundary phone.  If the input event type
 * is fully generalized from both left and right, the root event type is returned.
 */
bool GeneralizeEventType(const EventType &event_type_in, EventType *event_type_out, int32 P, bool left = true);


/**
 * Compute the balance value of the EventType as len(ctx-right) - len(ctx-left)
 */
int32 EventTypeBalance(const EventType &event_type, int32 P);


/**
 * Compute the EventType actual context size
 */
int32 EventTypeContextSize(const EventType &event_type, int32 P);


/**
 * Compress an EventType to only contain non-boundary phones and pdf classes
 * different from kNoPdf
 */
EventType CompressEventType(const EventType &event_type);

/**
 * Inflate a compressed EventType by adding the pdf_class=kNoPdf (if applicable)
 * and the boundary phones.
 */
EventType InflateEventType(const EventType &event_type, int32 N);

/**
 * Sort the specializations by "balance" (len(ctx_right) - len(ctx_left)) and
 * total context length
 */
class TopoNodeComparison {
 public:
  TopoNodeComparison(int32 P): P_(P) { }
  bool operator() (const TopoNode *a, const TopoNode *b) const;

 private:
  int32 P_;
};

} // end namespace kaldi.


#endif /* KALDI_TREE_TOPO_TREE_H_ */
