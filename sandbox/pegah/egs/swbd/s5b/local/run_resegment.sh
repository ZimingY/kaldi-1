#!/bin/bash

. cmd.sh

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_30k_nodup data/lang exp/tri3b exp/tri3b_ali_30k_nodup || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" --realign-iters "" \
  1000 10000 data/train_30k_nodup data/lang exp/tri3b_ali_30k_nodup exp/tri4b_seg || exit 1;

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri3b exp/tri3b_ali_all || exit 1;

# Make the phone decoding-graph.
steps/make_phone_graph.sh data/lang exp/tri3b_ali_all exp/tri4b_seg || exit 1;

mkdir -p data_reseg

for data in train eval2000; do
  cp -rT data/${data} data_reseg/${data}_orig; rm -r data_reseg/${data}_orig/split*
  for f in text utt2spk spk2utt feats.scp cmvn.scp segments; do rm data_reseg/${data}_orig/$f; done
  cat data_reseg/${data}_orig/wav.scp  | awk '{print $1, $1;}' | \
    tee data_reseg/${data}_orig/spk2utt > data_reseg/${data}_orig/utt2spk
  mfccdir=mfcc_reseg # don't use mfcc because of the way names are assigned within that
                     # dir, we'll overwrite the old data.
  mkdir -p mfcc_reseg
  steps/make_mfcc.sh --compress true --nj 20 --cmd "$train_cmd" data_reseg/${data}_orig exp/make_mfcc/${data}_orig $mfccdir 
  # caution: the new speakers don't correspond to the old ones, since they now have "sw0" at the start..
  steps/compute_cmvn_stats.sh --two-channel data_reseg/${data}_orig exp/make_mfcc/${data}_orig $mfccdir 
done


steps/decode_nolats.sh --write-words false --write-alignments true \
   --cmd "$decode_cmd" --nj 60 --beam 7.0 --max-active 1000 \
  exp/tri4b_seg/phone_graph data_reseg/train_orig exp/tri4b_seg/decode_train_orig

steps/decode_nolats.sh --write-words false --write-alignments true \
   --cmd "$decode_cmd" --nj 10 --beam 7.0 --max-active 1000 \
  exp/tri4b_seg/phone_graph data_reseg/eval2000_orig exp/tri4b_seg/decode_eval2000_orig


# Here: resegment.
# Note: it would be perfectly possible to use exp/tri3b_ali_train here instead
# of exp/tri4b_seg/decode_train_orig.  In this case we'd be relying on the transcripts.
# I chose not to do this for more consistency with what happens in test time.

steps/resegment_data.sh --cmd "$train_cmd" data_reseg/train_orig data/lang \
  exp/tri4b_seg/decode_train_orig data_reseg/train exp/tri4b_resegment_train

steps/resegment_data.sh --cmd "$train_cmd" data_reseg/eval2000_orig data/lang \
  exp/tri4b_seg/decode_eval2000_orig data_reseg/eval2000 exp/tri4b_resegment_eval2000

# We need all the training data to be aligned (not just "train_nodup"), in order
# to get the resegmented "text".
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri3b exp/tri3b_ali_train || exit 1;

# Get the file data_reseg/train/text
steps/resegment_text.sh --cmd "$train_cmd" data/train data/lang \
  exp/tri3b_ali_train data_reseg/train exp/tri4b_resegment_train


for data in train eval2000; do
  utils/fix_data_dir.sh data_reseg/${data}
  utils/validate_data_dir.sh --no-feats --no-text data_reseg/${data}
  mfccdir=mfcc_reseg # don't use mfcc because of the way names are assigned within that
                     # dir, we'll overwrite the old data.
  steps/make_mfcc.sh --compress true --nj 40 --cmd "$train_cmd" data_reseg/${data} \
    exp/make_mfcc/${data}_reseg $mfccdir  || exit 1;
  steps/compute_cmvn_stats.sh data_reseg/${data} exp/make_mfcc/${data}_reseg $mfccdir  || exit 1;
  utils/fix_data_dir.sh data_reseg/${data} || exit 1;
done


# Note: we'll be comparing tri4b, which was trained on train_nodup, with tri4c_reseg, which
# was trained on *all* the resegmented data.  However, it's comparable because the actual hours
# of data is less in tri4c_reseg: 265h, versus 284 in the nodup data.
# cat data/train_nodup/segments | awk '{nf += $4 - $3; } END{print nf /3600;}'
# 284.433
# cat data_reseg/train/segments | awk '{nf += $4 - $3; } END{print nf /3600;}'
# 265.154

steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data_reseg/train data/lang exp/tri3b exp/tri3b_ali_reseg || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  11500 200000 data_reseg/train data/lang exp/tri3b_ali_reseg exp/tri4c_reseg || exit 1;


for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/tri4c_reseg/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri4c_reseg $graph_dir
    steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
       $graph_dir data_reseg/eval2000 exp/tri4c_reseg/decode_eval2000_sw1_${lm_suffix}
  ) &
done


exit 0;

# Below is experimental.
# I'm figuring out whether we should keep the segments the the 1st pass designated as noise.
steps/resegment_data.sh --cmd "$train_cmd" \
   --segmentation-opts "--remove-noise-only-segments false" \
  data_reseg/eval2000_orig data/lang \
  exp/tri4b_seg/decode_eval2000_orig data_reseg/eval2000_with_noise exp/tri4b_resegment_eval2000_with_noise

for data in eval2000_with_noise; do
  utils/fix_data_dir.sh data_reseg/${data}
  utils/validate_data_dir.sh --no-feats --no-text data_reseg/${data}
  mfccdir=mfcc_reseg # don't use mfcc because of the way names are assigned within that
                     # dir, we'll overwrite the old data.
  steps/make_mfcc.sh --compress true --nj 40 --cmd "$train_cmd" data_reseg/${data} \
    exp/make_mfcc/${data}_reseg $mfccdir  || exit 1;
  steps/compute_cmvn_stats.sh data_reseg/${data} exp/make_mfcc/${data}_reseg $mfccdir  || exit 1;
  utils/fix_data_dir.sh data_reseg/${data} || exit 1;
done

for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/tri4c_reseg/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri4c_reseg $graph_dir
    steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
       $graph_dir data_reseg/eval2000_with_noise exp/tri4c_reseg/decode_eval2000_with_noise_sw1_${lm_suffix}
  ) &
done
