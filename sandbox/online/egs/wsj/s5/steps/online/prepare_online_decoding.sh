#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration.
stage=0 # This allows restarting after partway, when something when wrong.
online_cmvn_config=conf/online_cmvn.conf
feature_type=mfcc
per_utt_basis=true # If true, then treat each utterance as a separate speaker
                   # for purposes of basis training... this is recommended if
                   # the number of actual speakers in your training set is less
                   # than (feature-dim) * (feature-dim+1).
per_utt_cmvn=false # If true, apply online CMVN normalization per utterance rather
                   # than per speaker.
silence_weight=0.01
cmd=run.pl
cleanup=true
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
   echo "Usage: $0 [options] <data-dir> <lang-dir> <input-sat-dir> <input-MMI-model> <output-dir>"
   echo "e.g.: $0 data/train data/lang exp/tri3b exp/tri3b_mmi/final.mdl exp/tri3b_online"
   echo "main options (for others, see top of script file)"
   echo "  --online-cmvn-config <config>                    # config for online cmvn,"
   echo "                                                   # default conf/online_cmvn.conf"
   echo "  --feature-type <mfcc|plp>                        # Type of the base features; "
   echo "                                                   # important to generate the correct"
   echo "                                                   # configs in <output-dir>/conf/"
   echo "  --per-utt-cmvn <true|false>                      # Apply online CMVN per utt, not"
   echo "                                                   # per speaker (default: false)"
   echo "  --per-utt-basis <true|false>                     # Do basis computation per utterance"
   echo "                                                   # (default: true)"
   echo "  --silence-weight <weight>                        # Weight on silence for basis fMLLR;"
   echo "                                                   # default 0.01."
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi


data=$1
lang=$2
srcdir=$3
mmi_model=$4
dir=$5


for f in $srcdir/final.mdl $srcdir/ali.1.gz $data/feats.scp $lang/phones.txt \
    $srcdir/trans.1 $mmi_model $online_cmvn_config; do
  [ ! -f $f ] && echo "train_deltas.sh: no such file $f" && exit 1;
done

nj=`cat $srcdir/num_jobs` || exit 1;
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

mkdir -p $dir/log
echo $nj >$dir/num_jobs || exit 1;

splice_opts=`cat $srcdir/splice_opts 2>/dev/null` 
norm_vars=`cat $srcdir/norm_vars 2>/dev/null` || norm_vars=false 
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
cp $srcdir/splice_opts $srcdir/norm_vars $srcdir/final.mat $srcdir/final.mdl $dir/ 2>/dev/null

cp $mmi_model $dir/final.rescore_mdl

# Set up the unadapted features "$sifeats".
if [ -f $dir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
if ! $per_utt_cmvn; then
  online_cmvn_spk2utt_opt=
else
  online_cmvn_spk2utt_opt="--spk2utt=ark:$sdata/JOB/spk2utt"
fi


# create global_cmvn.stats
if ! matrix-sum --binary=false scp:$data/cmvn.scp - >$dir/global_cmvn.stats 2>/dev/null; then
  echo "$0: Error summing cmvn stats"
  exit 1
fi

echo "$0: feature type is $feat_type";
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
        online_sifeats="ark,s,cs:apply-cmvn-online --config=$online_cmvn_config $dir/global_cmvn.stats $online_cmvn_spk2utt_opt scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
       online_sifeats="ark,s,cs:apply-cmvn-online --config=$online_cmvn_config $online_cmvn_spk2utt_opt $dir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

# Set up the adapted features "$feats" for training set.
if [ -f $srcdir/trans.1 ]; then 
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$srcdir/trans.JOB ark:- ark:- |";
else
  feats="$sifeats";
fi


if $per_utt_basis; then
  spk2utt_opt=  # treat each utterance as separate speaker when computing basis.
  echo "Doing per-utterance adaptation for purposes of computing the basis."
else
  echo "Doing per-speaker adaptation for purposes of computing the basis."
  [ `cat $sdata/spk2utt | wc -l` -lt $[41*40] ] && \
    echo "Warning: number of speakers is small, might be better to use --per-utt=true."
  spk2utt_opt="--spk2utt=ark:$sdata/JOB/spk2utt"
fi

if [ $stage -le 0 ]; then
  echo "$0: Accumulating statistics for basis-fMLLR computation"
# Note: we get Gaussian level alignments with the "final.mdl" and the
# speaker adapted features. 
  $cmd JOB=1:$nj $dir/log/basis_acc.JOB.log \
    ali-to-post "ark:gunzip -c $srcdir/ali.JOB.gz|" ark:- \| \
    weight-silence-post $silence_weight $silphonelist $dir/final.mdl ark:- ark:- \| \
    gmm-post-to-gpost $dir/final.mdl "$feats" ark:- ark:- \| \
    gmm-basis-fmllr-accs-gpost $spk2utt_opt \
    $dir/final.mdl "$sifeats" ark,s,cs:- $dir/basis.acc.JOB || exit 1; 
fi

if [ $stage -le 1 ]; then
  echo "$0: computing the basis matrices."
  $cmd $dir/log/basis_training.log \
    gmm-basis-fmllr-training $dir/final.mdl $dir/fmllr.basis $dir/basis.acc.* || exit 1;
  if $cleanup; then
    rm $dir/basis.acc.* 2>/dev/null
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: accumulating stats for online alignment model."

  # Accumulate stats for "online alignment model"-- this model is computed with
  # the speaker-independent features and online CMVN, but matches
  # Gaussian-for-Gaussian with the final speaker-adapted model.

  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $srcdir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/final.mdl "$feats" "$online_sifeats" \
    ark,s,cs:- $dir/final.JOB.acc || exit 1;
  [ `ls $dir/final.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
  # Update model.
  $cmd $dir/log/est_online_alimdl.log \
    gmm-est --remove-low-count-gaussians=false $dir/final.mdl \
    "gmm-sum-accs - $dir/final.*.acc|" $dir/final.oalimdl  || exit 1;
  if $cleanup; then
    rm $dir/final.*.acc
  fi
fi

if [ $stage -le 3 ]; then
  mkdir -p $dir/conf
  rm $dir/{plp,mfcc}.conf 2>/dev/null
  echo "$0: preparing configuration files in $dir/conf"
  if [ -f $dir/conf/config ]; then
    echo "$0: moving $dir/conf/config to $dir/conf/config.bak"
    mv $dir/conf/config $dir/conf/config.bak
  fi
  conf=$dir/conf/config
  echo -n >$conf
  case "$feature_type" in
    mfcc)
      echo "--mfcc-config=$dir/conf/mfcc.conf" >>$conf
      cp conf/mfcc.conf $dir/conf/ ;;
    plp)
      echo "--plp-config=$dir/conf/plp.conf" >>$conf
      cp conf/plp.conf $dir/conf/ ;;
    *)
      echo "Unknown feature type $feature_type"
  esac
  if ! cp $online_cmvn_config $dir/conf/online_cmvn.conf; then 
    echo "$0: error copying online cmvn config to $dir/conf/"
    exit 1;
  fi
  echo "--cmvn-config=$dir/conf/online_cmvn.conf" >>$conf
  if [ -f $dir/final.mat ]; then
    echo "$0: creating $dir/splice.conf"
    for x in $(cat $dir/splice_opts); do echo $x; done > $dir/conf/splice.conf
    echo "--splice-config=$dir/conf/splice.conf" >>$conf
    echo "--lda-matrix=$dir/final.mat" >>$conf
  else
    echo "$0: creating $dir/delta.conf"
    echo -n >$dir/delta.conf # no non-default options currently supported.  It
                             # needs it to know we're applying delta features,
                             # though.
    echo "--delta-config=$dir/conf/delta.conf" >>$conf
  fi
  # Later we'll add support for pitch, but first we have to implement the online
  # version of the pitch extractor.
  
  echo "--fmllr-basis=$dir/fmllr.basis" >>$conf
  echo "--online-alignment-model=$dir/final.oalimdl" >>$conf
  echo "--model=$dir/final.mdl" >>$conf
  if ! cmp --quiet $dir/final.mdl $dir/final.rescore_mdl; then
    echo "--rescore-model=$dir/final.rescore_mdl" >>$conf
  fi
  echo "--silence-phones=$silphonelist" >>$conf
  echo "--global-cmvn-stats=$dir/global_cmvn.stats" >>$conf
  echo "$0: created config file $conf"
fi