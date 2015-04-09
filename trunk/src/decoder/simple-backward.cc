// decoder/simple-backward.cc

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

#include "decoder/simple-backward.h"
#include <algorithm>

namespace kaldi {

SimpleBackward::~SimpleBackward() {

}


bool SimpleBackward::Decode(DecodableInterface *decodable) {
  return false;
}


void SimpleBackward::InitDecoding() {

}

void SimpleBackward::AdvanceDecoding(DecodableInterface *decodable,
                                    int32 max_num_frames) {
}


bool SimpleBackward::ReachedFinal() const {
  return false;
}


double SimpleBackward::FinalCost() const {
  return Weight::Zero().Value();
}


void SimpleBackward::ProcessEmitting(DecodableInterface *decodable) {

}

void SimpleBackward::ProcessNonemitting() {

}


// static
void SimpleBackward::PruneToks(BaseFloat beam, unordered_map<StateId, Weight> *toks) {

}

} // end namespace kaldi.
