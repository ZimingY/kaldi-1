#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

orig_args=
for x in "$@"; do orig_args="$orig_args '$x'"; done

# begin configuration section.  we include all the options that score_sclite.sh or
# score_basic.sh might need, or parse_options.sh will die.
cmd=run.pl
stage=0
min_lmwt=7
max_lmwt=17
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [options] <data-dir> <lang-dir|graph-dir> <decode-dir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
graph=$2
decode=$3

if [ $stage -le 0 ] ; then
  eval local/lattice_to_ctm.sh $orig_args
fi

if [ -f $data/stm ]; then # use sclite scoring.
  echo "$data/stm exists: using local/score_sclite.sh"

  [ -f $data/glm ] && glm=" --glm $data/glm "
  eval local/score_stm.sh --cer 1 $glm  $orig_args
  
  if [ -f $data/stm.utf8 ] ; then
    for seqid in `seq $min_lmwt $max_lmwt` ; do
      ./IBM/scoring/score.py  $decode/score_$seqid/`basename $data`.ctm $data/stm.utf8 > $decode/log/ibm.$seqid.log
    done
  fi

else
  echo "$data/stm does not exist: using local/score_basic.sh"
  eval local/score_basic.sh $orig_args
fi
