// feat/feature-fbank.h

// Copyright 2009-2012  Karel Vesely

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

#ifndef KALDI_FEAT_FEATURE_FBANK_H_
#define KALDI_FEAT_FEATURE_FBANK_H_


#include <string>

#include "feat/feature-functions.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


/// FbankOptions contains basic options for computing FBANK features
/// It only includes things that can be done in a "stateless" way, i.e.
/// it does not include energy max-normalization.
/// It does not include delta computation.
struct FbankOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;
  bool use_energy;  // use energy; else C0
  BaseFloat energy_floor;
  bool raw_energy;  // compute energy before preemphasis and hamming window (else after)
  bool htk_compat;  // if true, put energy/C0 last and introduce a factor of sqrt(2)
  // on C0 to be the same as HTK.

  FbankOptions(): mel_opts(23),  // defaults the #mel-banks to 23 for the FBANK computations.
                 // this seems to be common for 16khz-sampled data, but for 8khz-sampled
                 // data, 15 may be better.
                 use_energy(false),
                 energy_floor(0.0),  // not in log scale: a small value e.g. 1.0e-10
                 raw_energy(true),
                 htk_compat(false) { }
  void Register(ParseOptions *po) {
    frame_opts.Register(po);
    mel_opts.Register(po);
    po->Register("use-energy", &use_energy, "Use energy (not C0) in FBANK computation");
    po->Register("energy-floor", &energy_floor, "Floor on energy (absolute, not relative) in FBANK computation");
    po->Register("raw-energy", &raw_energy, "If true, compute energy (if using energy) before Hamming window and preemphasis");
    po->Register("htk-compat", &htk_compat, "If true, put energy or C0 last and put factor of sqrt(2) on C0.  Warning: not sufficient to get HTK compatible features (need to change other parameters).");
  }

};

class MelBanks;


/// Class for computing FBANK features; see \ref feat_mfcc for more information.
class Fbank {
 public:
  Fbank(const FbankOptions &opts);
  ~Fbank();

  /// Will throw exception on failure (e.g. if file too short for
  /// even one frame).
  void Compute(const VectorBase<BaseFloat> &wave,
               BaseFloat vtln_warp,
               Matrix<BaseFloat> *output,
               Vector<BaseFloat> *wave_remainder = NULL);

 private:
  const MelBanks *GetMelBanks(BaseFloat vtln_warp);
  FbankOptions opts_;
  BaseFloat log_energy_floor_;
  std::map<BaseFloat, MelBanks*> mel_banks_;  // BaseFloat is VTLN coefficient.
  FeatureWindowFunction feature_window_function_;
  SplitRadixRealFft<BaseFloat> *srfft_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Fbank);
};


/// @} End of "addtogroup feat"
}// namespace kaldi


#endif
