// kwsbin/kws-index-union.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
//                 Lucas Ondel

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
#include "fstext/fstext-utils.h"
#include "lat/kaldi-kws.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;

    const char *usage =
        "Take a union of the indexed lattices. The input index is in the T*T*T semiring and\n"
        "the output index is also in the T*T*T semiring. At the end of this program, encoded\n"
        "epsilon removal, determinization and minimization will be applied.\n"
        "\n"
        "Usage: kws-index-union [options]  index-rspecifier index-wspecifier\n"
        " e.g.: kws-index-union ark:input.idx ark:global.idx\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string index_rspecifier = po.GetArg(1),
        index_wspecifier = po.GetOptArg(2);

    SequentialTableReader< VectorFstTplHolder<KwsLexicographicArc> > index_reader(index_rspecifier);
    TableWriter< VectorFstTplHolder<KwsLexicographicArc> > index_writer(index_wspecifier);

    int32 n_done = 0;
    KwsLexicographicFst global_index;
    for (; !index_reader.Done(); index_reader.Next()) {
      std::string key = index_reader.Key();
      KwsLexicographicFst index = index_reader.Value();
      index_reader.FreeCurrent();

      Union(&global_index, index);

      n_done++;
    }

    // Do the encoded epsilon removal, determinization and minimization
    KwsLexicographicFst ifst = global_index;
    EncodeMapper<KwsLexicographicArc> encoder(kEncodeLabels, ENCODE);
    Encode(&ifst, &encoder);
    DeterminizeStar(ifst, &global_index);
    Minimize(&global_index);
    Decode(&global_index, encoder);

    // Write the result
    index_writer.Write("global", global_index);

    KALDI_LOG << "Done " << n_done << " indices";
    return (n_done != 0 ? 0 : 1);    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
