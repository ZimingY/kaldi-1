// idlaktxp/fexfunctions.h

// Copyright 2013 CereProc Ltd.  (Author: Matthew Aylett)

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

// Automatically generated: Fri Jun 28 14:06:33 2013

#ifndef SRC_IDLAKTXP_FEXFUNCTIONS_H
#define SRC_IDLAKTXP_FEXFUNCTIONS_H

// This file autogenerated by running create_catalog.py
// Do not edit manually

#include "./txpfexspec.h"

namespace kaldi {

#define FEX_NO_FEATURES 5

bool FexFuncPRESTRbbp(const TxpFexspec * fex,
                      const TxpFexspecFeat * feat,
                      const TxpFexspecContext * context,
                      char * buffer);

bool FexFuncPRESTRbp(const TxpFexspec * fex,
                     const TxpFexspecFeat * feat,
                     const TxpFexspecContext * context,
                     char * buffer);

bool FexFuncCURSTRp(const TxpFexspec * fex,
                    const TxpFexspecFeat * feat,
                    const TxpFexspecContext * context,
                    char * buffer);

bool FexFuncPSTSTRfp(const TxpFexspec * fex,
                     const TxpFexspecFeat * feat,
                     const TxpFexspecContext * context,
                     char * buffer);

bool FexFuncPSTSTRffp(const TxpFexspec * fex,
                     const TxpFexspecFeat * feat,
                     const TxpFexspecContext * context,
                     char * buffer);



}  // namespace kaldi

#endif  // SRC_IDLAKTXP_FEXFUNCTIONS_H_
