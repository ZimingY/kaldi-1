// nnet-dp/nnet1.h

// Copyright 2012  Daniel Povey

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

#ifndef KALDI_NNET_DP_AM_NNET1_H_
#define KALDI_NNET_DP_AM_NNET1_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "nnet-dp/nnet1.h"

namespace kaldi {

/*
  The class AmNnet1 has the job of taking the "Nnet1" class, which has a relatively
  simple interface, and giving it an interface that's suitable for acoustic modeling,
  dealing with 2-level trees, and so on.  Basically, this class handles various
  integer mappings that relate to the 2-level trees.
  
  Note: this class deals with setting up, and with storing, the neural net, but
  the likelihood computation is done by a separate class.
*/


class AmNnet1 {
 public:
  AmNnet1() { }
  
  // The vector "leaf_mapping" is an output from the program build-tree-two-level.
  // This maps from the leaves of the tree ("level-2 leaves") to the coarser "level-1"
  // leaves.  I.e. it's a fine-to-coarse mapping.
  AmNnet1(const Nnet1InitConfig &config,
          const std::vector<int32> &leaf_mapping);
  
  int32 NumPdfs() const { return leaf_mapping_.size(); }
  
  void Write(std::ostream &os, bool binary) const;
  
  void Read(std::istream &is, bool binary);

  // This turns a pdf_id into a pair of one, possibly two pairs
  // (category; label within that category).  There will always
  // be one pair for category zero (top-level tree), and usually
  // one more for the finer-level tree.
  void GetCategoryInfo(int32 pdf_id,
                       std::vector<std::pair<int32, int32> > *pairs) const;

  const Nnet1 &Nnet() const { return nnet_; }
  Nnet1 &Nnet() { return nnet_; }

  // This function returns a set of sets of the final layer nodes,
  // which for this type of model are just the set { 0 } and the rest of them;
  // we use this to group them for purposes of reporting auxf
  // improvements and adjusting learning rates.  [especially, this
  // makes the adjustment of learning rates more robust.]
  void GetFinalLayerSets(std::vector<std::vector<int32> > *sets);

  // GetPriors() gets the prior for each pdf-id; this is derived
  // from the occupancy information stored with the neural network.
  void GetPriors(Vector<BaseFloat> *priors) const;

  // FixPriors() will cause all future calls to GetPriors() to
  // return the same value of priors_ as if it had been called now.
  // This is useful prior to MMI training, as it makes sense
  // to fix the priors that we divide by, before discriminatively
  // traning.
  void FixPriors() { GetPriors(&priors_); }
 private:
  // called from constructor and Read function; see .cc file for comments:
  void ComputeCategoryInfo(const std::vector<int32> &leaf_mapping);

  
  std::vector<int32> leaf_mapping_; // Maps from pdf-ids (zero-based) to
  // a "level-one tree index", also zero-based.  Could also be called
  // pdf_to_l1_.

  std::vector<int32> l1_to_category_; // Maps from level-one tree index
  // to a "category-index", or -1 if it would be a singleton category.
  // the category-index starts from 1, because zero is reserved for the
  // level-one tree.

  // For each l1 index starting from zero, a list of the pdfs that
  // that l1 index covers.  Will include all pdfs, each once.
  std::vector<std::vector<int32> > l1_to_pdfs_;

  // Map from the pdf-id to the index within the category it's part of, which is
  // the same as the index in l1_to_pdfs_. (this will be 0 if it's part of a
  // singleton category)
  std::vector<int32> pdf_to_sub_index_;

  // category_sizes_ is the size of each category, including category zero.
  std::vector<int32> category_sizes_;

  // Note: during ML training this vector will be empty.  It's used to store
  // the priors extracted from the model, during MMI training, so they don't
  // change after that.
  Vector<BaseFloat> priors_;

  Nnet1 nnet_;
};



} // namespace

#endif // KALDI_NNET_DP_AM_NNET1_H_
