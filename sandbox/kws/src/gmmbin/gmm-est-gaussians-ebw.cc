// gmmbin/gmm-est-gaussians-ebw.cc

// Copyright 2009-2011  Petr Motlicek  Chao Weng

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
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "gmm/ebw-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Do EBW update for MMI, MPE or MCE discriminative training.\n"
        "Numerator stats should already be I-smoothed (e.g. use gmm-ismooth-stats)\n"
        "Usage:  gmm-est-gaussians-ebw [options] <model-in> <stats-num-in> <stats-den-in> <model-out>\n"
        "e.g.: gmm-est-gaussians-ebw 1.mdl num.acc den.acc 2.mdl\n";

    bool binary_write = false;
    std::string update_flags_str = "mv";

    EbwOptions ebw_opts;
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters to "
                "update: e.g. m or mv (w, t ignored).");
    
    ebw_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    kaldi::GmmFlagsType update_flags =
        StringToGmmFlags(update_flags_str);    

    std::string model_in_filename = po.GetArg(1),
        num_stats_filename = po.GetArg(2),
        den_stats_filename = po.GetArg(3),
        model_out_filename = po.GetArg(4);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    Vector<double> num_transition_accs; // won't be used.
    Vector<double> den_transition_accs; // won't be used.

    AccumAmDiagGmm num_stats;
    AccumAmDiagGmm den_stats;
    {
      bool binary;
      Input ki(num_stats_filename, &binary);
      num_transition_accs.Read(ki.Stream(), binary);
      num_stats.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }
    
    {
      bool binary;
      Input ki(den_stats_filename, &binary);
      num_transition_accs.Read(ki.Stream(), binary);
      den_stats.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }
      
 
    {  // Update GMMs.
      BaseFloat auxf_impr, count;
      int32 num_floored;
      UpdateEbwAmDiagGmm(num_stats, den_stats, update_flags, ebw_opts, &am_gmm,
                          &auxf_impr, &count, &num_floored);
      KALDI_LOG << "Num count " << num_stats.TotStatsCount() << ", den count "
                << den_stats.TotStatsCount();
      KALDI_LOG << "Overall auxf impr/frame from Gaussian update is " << (auxf_impr/count)
                << " over " << count << " frames; floored D for "
                << num_floored << " Gaussians.";
    }

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;

  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
