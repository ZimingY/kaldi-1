// idlaktxp/mod-cex.h

// Copyright 2012 CereProc Ltd.  (Author: Matthew Aylett)

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

#ifndef KALDI_IDLAKTXP_MOD_CEX_H_
#define KALDI_IDLAKTXP_MOD_CEX_H_

// This file defines the basic txp module which incrementally parses
// either text, tox (token oriented xml) tokens, or spurts (phrases)
// containing tox tokens.

#include <string>
#include "idlaktxp/txpmodule.h"
#include "idlaktxp/txpcexspec.h"

namespace kaldi {

/// Linguistic context extraction: Converts output from text normalisation
/// into full context model names
class TxpCex : public TxpModule {
 public:
  explicit TxpCex();
  ~TxpCex();
  bool Init(const TxpParseOptions &opts);
  bool Process(pugi::xml_document* input);
  /// Returns true if the system is splitting mid phrase pauses to allow
  /// spurt processing of model names
  bool IsSptPauseHandling();

 private:
  /// Object containing specification for feature extraction
  TxpCexspec cexspec_;
  TxpCexspecModels models_;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_MOD_CEX_H_
