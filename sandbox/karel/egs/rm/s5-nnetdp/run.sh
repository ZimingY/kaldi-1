#!/bin/bash

# CAUTION: I changed e.g. 1.trans to trans.1 in the scripts.  If you ran it
# part-way through prior to this, to convert to the new naming
# convention, run:
# for x in `find . -name '*.trans'`; do mv $x `echo $x | perl -ane 's/(\d+)\.trans/trans.$1/;print;'`; done
# but be careful as this will not follow soft links.

. cmd.sh

# call the next line with the directory where the RM data is
# (the argument below is just an example).  This should contain
# subdirectories named as follows:
#    rm1_audio1  rm1_audio2	rm2_audio

#local/rm_data_prep.sh /mnt/matylda2/data/RM || exit 1;
local/rm_data_prep.sh /export/corpora5/LDC/LDC93S3A/rm_comp || exit 1;

utils/prepare_lang.sh data/local/dict '!SIL' data/local/lang data/lang || exit 1;

local/rm_prepare_grammar.sh || exit 1;

# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
featdir=feats
mkdir data-fbank
for x in test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92 train; do
  steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/$x exp/make_mfcc/$x $featdir  || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $featdir  || exit 1;

  cp -r data/$x data-fbank/$x
  steps/make_fbank.sh --nj 8 --cmd "run.pl" data-fbank/$x exp/make_fbank/$x $featdir  || exit 1;
  steps/compute_cmvn_stats.sh data-fbank/$x exp/make_fbank/$x $featdir  || exit 1;
done

# Make a combined data dir where the data from all the test sets goes-- we do
# all our testing on this averaged set.  This is just less hassle.  We
# regenerate the CMVN stats as one of the speakers appears in two of the 
# test sets; otherwise tools complain as the archive has 2 entries.
for data in data data-fbank; do
  utils/combine_data.sh $data/test $data/test_{mar87,oct87,feb89,oct89,feb91,sep92}
  steps/compute_cmvn_stats.sh $data/test exp/make_mfcc/test $featdir  
done


utils/subset_data_dir.sh data/train 1000 data/train.1k  || exit 1;

steps/train_mono.sh --nj 4 --cmd "$train_cmd" data/train.1k data/lang exp/mono  || exit 1;

#show-transitions data/lang/phones.txt exp/tri2a/final.mdl  exp/tri2a/final.occs | perl -e 'while(<>) { if (m/ sil /) { $l = <>; $l =~ m/pdf = (\d+)/|| die "bad line $l";  $tot += $1; }} print "Total silence count $tot\n";'


utils/mkgraph.sh --mono data/lang exp/mono exp/mono/graph

steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/mono/graph data/test exp/mono/decode


# Get alignments from monophone system.
steps/align_si.sh --nj 8 --cmd "$train_cmd" \
  data/train data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh --cmd "$train_cmd" \
 1800 9000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

# decode tri1
utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph || exit 1;
steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri1/graph data/test exp/tri1/decode

#draw-tree data/lang/phones.txt exp/tri1/tree | dot -Tps -Gsize=8,10.5 | ps2pdf - tree.pdf

# align tri1
steps/align_si.sh --nj 8 --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# train and decode tri2b [LDA+MLLT]
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" \
 1800 9000 data/train data/lang exp/tri1_ali exp/tri2b || exit 1;
utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b/decode

# Align all data with LDA+MLLT system (tri2b)
steps/align_si.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
   data/train data/lang exp/tri2b exp/tri2b_ali || exit 1;


(
  ## Build a LDA+MLLT system just for decorrelation, on the raw filterbank
  ## features, but using the tree from the tri2b system. This is just to get the
  ## decorrelating transform and will also allow us to get SAT on the
  ## filterbank features.
  featdim=`feat-to-dim scp:data-fbank/train/feats.scp -` 

  local/train_lda_mllt_notree.sh --cmd "$train_cmd" --dim $featdim \
    --splice-opts "--left-context=0 --right-context=0" \
    --realign-iters "" 9000 data-fbank/train data/lang exp/tri2b_ali exp/tri3b

  steps/decode.sh --nj 20 --config conf/decode.config --cmd "$decode_cmd" \
    exp/tri2b/graph data-fbank/test exp/tri3b/decode

  ## Train fMLLR on top of fbank features.  Note: we give it directory
  ## tri3b instead of an alignment directory like tri3b_ali, because we want
  ## to use the alignments in tri3b which were just copied from the conventional
  ## system.
  local/train_sat_notree.sh --cmd "$train_cmd" \
    --realign-iters "" 9000 data-fbank/train data/lang exp/tri3b exp/tri4b

  ## Decode the test data with this system (will need it for nnet testing,
  ## for the transforms.)
  ## Use transcripts from the 2b system which had frame splicing -> better 
  ## supervision.
  steps/decode_fmllr.sh --si-dir exp/tri2b/decode \
    --nj 20 --config conf/decode.config --cmd "$decode_cmd" \
    exp/tri2b/graph data-fbank/test exp/tri4b/decode
)

# Do SAT on top of the standard splice+LDA+MLLT system,
# because we want good alignments to build the 2-level tree on top of,
# for the neural nset system.
steps/train_sat.sh 1800 9000 data/train data/lang exp/tri2b_ali exp/tri3c

# Align all data with LDA+MLLT+SAT system (tri3c)
steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
  data/train data/lang exp/tri3c exp/tri3c_ali || exit 1;

## First we just build a two-level tree, using the LDA+MLLT+SAT system's
## alignments and features, since that system is pretty good.

local/train_two_level_tree.sh 150 5000 data/train data/lang exp/tri3c_ali exp/tri4a_tree

# smaller version of the tree.
local/train_two_level_tree.sh 150 2000 data/train data/lang exp/tri3c_ali exp/tri4b_tree

# Train version of the tree with really just one level
local/train_two_level_tree.sh 5000 5000 data/train data/lang exp/tri3c_ali exp/tri4c_tree

## Now train the neural net itself.
local/train_nnet1.sh 10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5a_nnet

utils/mkgraph.sh data/lang exp/tri5a_nnet exp/tri5a_nnet/graph

local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
  --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5a_nnet/decode

# add --max-iter-inc 10 to the below, when ready.
local/train_nnet1.sh --add-layer-iters "5 10" --num-iters 15 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5b_nnet

local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
   --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5b_nnet/decode

local/train_nnet1.sh --add-layer-iters "3 5" --num-iters 9 \
  --max-iter-inc 6 --left-context 0 --right-context 0 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5c_nnet

local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
   --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5c_nnet/decode

local/train_nnet1.sh --initial-layer-context "4,4" --add-layer-iters "3 5" --num-iters 9 \
  --max-iter-inc 6 --left-context 0 --right-context 0 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5d_nnet

local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
  --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5d_nnet/decode

local/train_nnet1.sh --initial-layer-context "4,4" --add-layer-iters "3 5" --num-iters 9 \
  --max-iter-inc 6 --left-context 0 --right-context 0 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5e_nnet

 # Caution: d and e seem to be the same.
local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
  --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5e_nnet/decode

local/train_nnet1.sh --initial-layer-context "4,4" --add-layer-iters "3 5" --num-iters 9 \
  --max-iter-inc 6 --hidden-layer-size 350 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5f_nnet

local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
  --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5f_nnet/decode

local/train_nnet1.sh --initial-layer-context "8,8" --add-layer-iters "3 5" --num-iters 9 \
  --max-iter-inc 6 --hidden-layer-size 350 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5g_nnet

local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
  --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5g_nnet/decode

local/train_nnet1.sh --initial-layer-context "4,4"  --num-iters 9 \
  --chunk-size 1 --num-chunks 1000 --num-minibatches 500 --num-phases 10 \
  --max-iter-inc 6 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5h_nnet

local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
  --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5h_nnet/decode


# 5h2 is as 5h but newer code, with shrinkage.
local/train_nnet1.sh --initial-layer-context "4,4"  --num-iters 9 \
  --chunk-size 1 --num-chunks 1000 --num-minibatches 500 --num-phases 10 \
  --max-iter-inc 6 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5h2_nnet

iter=7
local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
  --iter $iter --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5h2_nnet/decode_it$iter

# 5i is as 5h2, but using a smaller tree and also fewer mixture components.
local/train_nnet1.sh --initial-layer-context "4,4"  --num-iters 9 \
  --chunk-size 1 --num-chunks 1000 --num-minibatches 500 --num-phases 10 \
  --max-iter-inc 6 \
  5000 data-fbank/train data/lang exp/tri4b exp/tri4b_tree exp/tri5i_nnet

utils/mkgraph.sh data/lang exp/tri5i_nnet exp/tri5i_nnet/graph

iter=7
local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
  --iter $iter --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5i_nnet/graph data-fbank/test exp/tri5i_nnet/decode_it$iter

# 5j is as 5h, but adding another layer (with no context)
local/train_nnet1.sh --cmd "queue.pl -q all.q@a* -l ram_free=1200M,mem_free=1200M -pe smp 4" \
  --initial-layer-context "4,4"  --num-iters 9 \
  --chunk-size 1 --num-chunks 1000 --num-minibatches 500 --num-phases 10 \
  --max-iter-inc 6 \
  --add-layer-iters "4" --left-context 0 --right-context 0 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5j_nnet

for iter in 9 8 6 4 2; do 
  local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
    --iter $iter --config conf/decode.config --nj 20 \
   --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5j_nnet/decode_it$iter
done


#5k is as 5h2 but more context (6 not 4 frames on either side.
local/train_nnet1.sh \
  --cmd "queue.pl -q all.q@a* -l ram_free=1200M,mem_free=1200M -pe smp 4" \
  --initial-layer-context "6,6"  --num-iters 16 \
  --chunk-size 1 --num-chunks 1000 --num-minibatches 500 --num-phases 10 \
  --max-iter-inc 6 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5k_nnet

for iter in 2 4 6 8 9; do
  while [ ! -f exp/tri5k_nnet/$iter.mdl ]; do sleep 120; done
 local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
   --iter $iter --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5k_nnet/decode_it$iter &
done


#5l_nnet is as 5k_nnet but using 4c for the tree, which  is effectively
# a single-level tree.  Note: it crashed.
local/train_nnet1.sh \
  --cmd "queue.pl -q all.q@a* -l ram_free=1200M,mem_free=1200M -pe smp 4" \
  --initial-layer-context "6,6"  --num-iters 16 \
  --chunk-size 1 --num-chunks 1000 --num-minibatches 500 --num-phases 10 \
  --max-iter-inc 6 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4c_tree exp/tri5l_nnet

utils/mkgraph.sh data/lang exp/tri5l_nnet exp/tri5l_nnet/graph
for iter in 2 4 6 8 9; do
  while [ ! -f exp/tri5l_nnet/$iter.mdl ]; do sleep 120; done
 local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
   --iter $iter --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5l_nnet/graph data-fbank/test exp/tri5l_nnet/decode_it$iter &
done


# 5m is as 5k, but one layer of 1000 then one of 250.
local/train_nnet1b.sh \
  --hidden-layer-sizes "1000:250" --context "6,6:0,0" \
  --cmd "queue.pl -q all.q@a* -l ram_free=1200M,mem_free=1200M -pe smp 4" \
  --num-iters 16 --chunk-size 1 --num-chunks 1000 --num-minibatches 500 --num-phases 10 \
  --max-iter-inc 6 \
  10000 data-fbank/train data/lang exp/tri4b exp/tri4a_tree exp/tri5m_nnet

for iter in 2 4 6 8 9; do
  while [ ! -f exp/tri5k_nnet/$iter.mdl ]; do sleep 120; done
 local/decode_nnet1.sh --transform-dir exp/tri4b/decode \
   --iter $iter --config conf/decode.config --nj 20 \
  --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5k_nnet/decode_it$iter &
done



 mkdir exp/tri4b_nofmllr
 cp exp/tri4b/final.mat exp/tri4b_nofmllr # Give this dir to the script, it will train
 # without SAT.

 # Train like 5d but without SAT.
 local/train_nnet1.sh --initial-layer-context "4,4" --add-layer-iters "3 5" --num-iters 8 \
   --max-iter-inc 6 --left-context 0 --right-context 0 \
   10000 data-fbank/train data/lang exp/tri4b_nofmllr exp/tri4a_tree exp/tri5d_nnet_nofmllr

  local/decode_nnet1.sh \
    --config conf/decode.config --nj 20 \
    --cmd "$decode_cmd" exp/tri5a_nnet/graph data-fbank/test exp/tri5d_nnet_nofmllr/decode
# I AM HERE.
exit 0;


#  Do MMI on top of LDA+MLLT.
steps/make_denlats.sh --nj 8 --cmd "$train_cmd" \
  data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1;
steps/train_mmi.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it4
steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it3

# Do the same with boosting.
steps/train_mmi.sh --boost 0.05 data/train data/lang \
   exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi_b0.05 || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it4 || exit 1;
steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it3 || exit 1;

<<<<<<< .working
# Do MPE.
steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4 || exit 1;
steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it3 || exit 1;
=======
# Do MPE.
steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe || exit 1;
steps/decode.sh --config conf/decode.config --iter 4 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4 || exit 1;
steps/decode.sh --config conf/decode.config --iter 3 --nj 20 --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it3 || exit 1;


## Do LDA+MLLT+SAT, and decode.
steps/train_sat.sh 1800 9000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph || exit 1;
steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri3b/graph data/test exp/tri3b/decode || exit 1;
>>>>>>> .merge-right.r1321


<<<<<<< .working



=======

>>>>>>> .merge-right.r1321
# Align all data with LDA+MLLT+SAT system (tri3b)
steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
  data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;

## MMI on top of tri3b (i.e. LDA+MLLT+SAT+MMI)
steps/make_denlats.sh --config conf/decode.config \
   --nj 8 --cmd "$train_cmd" --transform-dir exp/tri3b_ali \
  data/train data/lang exp/tri3b exp/tri3b_denlats || exit 1;
steps/train_mmi.sh data/train data/lang exp/tri3b_ali exp/tri3b_denlats exp/tri3b_mmi || exit 1;

steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  --alignment-model exp/tri3b/final.alimdl --adapt-model exp/tri3b/final.mdl \
   exp/tri3b/graph data/test exp/tri3b_mmi/decode || exit 1;

# Do a decoding that uses the exp/tri3b/decode directory to get transforms from.
steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_mmi/decode2 || exit 1;


#first, train UBM for fMMI experiments.
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 8 --cmd "$train_cmd" \
  250 data/train data/lang exp/tri3b_ali exp/dubm3b

# Next, various fMMI+MMI configurations.
steps/train_mmi_fmmi.sh --learning-rate 0.0025 \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_b || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 20 --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_b/decode_it$iter &
done

steps/train_mmi_fmmi.sh --learning-rate 0.001 \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_c || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 20 --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_c/decode_it$iter &
done

# for indirect one, use twice the learning rate.
steps/train_mmi_fmmi_indirect.sh --learning-rate 0.01 --schedule "fmmi fmmi fmmi fmmi mmi mmi mmi mmi" \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/dubm3b exp/tri3b_denlats \
  exp/tri3b_fmmi_d || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 20 --config conf/decode.config --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_fmmi_d/decode_it$iter &
done

# You don't have to run all 3 of the below, e.g. you can just run the run_sgmm2x.sh
local/run_sgmm.sh
local/run_sgmm2.sh
local/run_sgmm2x.sh

