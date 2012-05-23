// featbin/fmpe-copy.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)  Yanmin Qian

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "transform/fmpe.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Copy fMPE transform\n"
        "Usage: fmpe-init [options...] <fmpe-in> <fmpe-out>\n"
        "E.g. fmpe-copy --binary=false 1.fmpe text.fmpe\n";

    ParseOptions po(usage);
    FmpeOptions opts;
    bool binary = true;
    po.Register("binary", &binary, "If true, output fMPE object in binary mode.");
    opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fmpe_rxfilename = po.GetArg(1),
        fmpe_wxfilename = po.GetArg(2);

    Fmpe fmpe;
    ReadKaldiObject(fmpe_rxfilename, &fmpe);
    

    Output ko(fmpe_wxfilename, binary);
    fmpe.Write(ko.Stream(), binary);

    KALDI_LOG << "Copyied fMPE object to " << fmpe_wxfilename;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
