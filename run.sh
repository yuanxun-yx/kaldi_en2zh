#!/usr/bin/env bash

# modified from egs/aishell/s5/run.sh
# transfer learning from TED-LIUM to AISHELL

. ./cmd.sh
. ./path.sh


set -e -o pipefail -u

nj=35
decode_nj=38   # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.
stage=0
train_rnnlm=false
train_lm=false

. utils/parse_options.sh # accept options

# genetate dict data from AISHELL
if [ $stage -le 0 ]; then
  local/prepare_dict.sh
fi

if [ $stage -le 1 ]; then
  utils/prepare_lang.sh data/local/dict \
    "<unk>" data/local/lang data/lang
fi

# G compilation, check LG composition
if [ $stage -le 2 ]; then
  utils/format_lm.sh data/lang $AISHELL_ROOT/data/local/lm/3gram-mincount/lm_unpruned.gz \
      data/local/dict/lexicon.txt data/lang_test || exit 1;
fi

# Train a monophone model on delta features.
if [ $stage -le 3 ]; then
  steps/train_mono.sh --cmd "$train_cmd" --nj 10 \
    $AISHELL_ROOT/data/train data/lang exp/mono || exit 1;
fi

# Decode with the monophone model.
if [ $stage -le 4 ]; then
  # utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
    exp/mono/graph $AISHELL_ROOT/data/dev exp/mono/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
    exp/mono/graph $AISHELL_ROOT/data/test exp/mono/decode_test
fi

# Get alignments from monophone system.
if [ $stage -le 5 ]; then
  steps/align_si.sh --cmd "$train_cmd" --nj 10 \
    $AISHELL_ROOT/data/train data/lang exp/mono exp/mono_ali || exit 1;
fi

# Train the first triphone pass model tri1 on delta + delta-delta features.
if [ $stage -le 6 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
  2500 20000 $AISHELL_ROOT/data/train data/lang exp/mono_ali exp/tri1 || exit 1;
fi

# decode tri1
if [ $stage -le 7 ]; then
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
    exp/tri1/graph $AISHELL_ROOT/data/dev exp/tri1/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
    exp/tri1/graph $AISHELL_ROOT/data/test exp/tri1/decode_test
fi

# align tri1
if [ $stage -le 8 ]; then
  steps/align_si.sh --cmd "$train_cmd" --nj 10 \
    $AISHELL_ROOT/data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
fi

# train tri2 [delta+delta-deltas]
if [ $stage -le 9 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
  2500 20000 $AISHELL_ROOT/data/train data/lang exp/tri1_ali exp/tri2 || exit 1;
fi

# decode tri2
if [ $stage -le 10 ]; then
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
    exp/tri2/graph $AISHELL_ROOT/data/dev exp/tri2/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
    exp/tri2/graph $AISHELL_ROOT/data/test exp/tri2/decode_test
fi

# Align training data with the tri2 model.
if [ $stage -le 11 ]; then
  steps/align_si.sh --cmd "$train_cmd" --nj 10 \
    $AISHELL_ROOT/data/train data/lang exp/tri2 exp/tri2_ali || exit 1;
fi

# Train the second triphone pass model tri3a on LDA+MLLT features.
if [ $stage -le 12 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
  2500 20000 $AISHELL_ROOT/data/train data/lang exp/tri2_ali exp/tri3a || exit 1;
fi
 
# Run a test decode with the tri3a model.
if [ $stage -le 13 ]; then
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
    exp/tri3a/graph $AISHELL_ROOT/data/dev exp/tri3a/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
    exp/tri3a/graph $AISHELL_ROOT/data/test exp/tri3a/decode_test
fi

# align tri3a with fMLLR
if [ $stage -le 14 ]; then
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
    $AISHELL_ROOT/data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;
fi

# Train the third triphone pass model tri4a on LDA+MLLT+SAT features.
# From now on, we start building a more serious system with Speaker
# Adaptive Training (SAT).
if [ $stage -le 15 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    2500 20000 $AISHELL_ROOT/data/train data/lang exp/tri3a_ali exp/tri4a || exit 1;
fi

# decode tri4a
if [ $stage -le 16 ]; then
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
    exp/tri4a/graph $AISHELL_ROOT/data/dev exp/tri4a/decode_dev
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
    exp/tri4a/graph $AISHELL_ROOT/data/test exp/tri4a/decode_test
fi
  
# align tri4a with fMLLR
if [ $stage -le 17 ]; then
  steps/align_fmllr.sh  --cmd "$train_cmd" --nj 10 \
    $AISHELL_ROOT/data/train data/lang exp/tri4a exp/tri4a_ali
fi

# Train tri5a, which is LDA+MLLT+SAT
# Building a larger SAT system. You can see the num-leaves is 3500 and tot-gauss is 100000
if [ $stage -le 18 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    3500 100000 $AISHELL_ROOT/data/train data/lang exp/tri4a_ali exp/tri5a || exit 1;
fi
  
# decode tri5a
if [ $stage -le 19 ]; then
  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
    exp/tri5a/graph $AISHELL_ROOT/data/dev exp/tri5a/decode_dev || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
    exp/tri5a/graph $AISHELL_ROOT/data/test exp/tri5a/decode_test || exit 1;
fi
   
# align tri5a with fMLLR
if [ $stage -le 20 ]; then
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
    $AISHELL_ROOT/data/train data/lang exp/tri5a exp/tri5a_ali || exit 1;
fi

if [ $stage -le 21 ]; then
  # This will only work if you have GPUs on your system (and note that it requires
  # you to have the queue set up the right way... see kaldi-asr.org/doc/queue.html)
  local/chain/run_tdnn.sh --stage 7
fi

if [ $stage -le 22 ]; then
  # getting results (see RESULTS file)
  for x in exp/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
  for x in exp/*/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
fi

echo "$0: success."
exit 0
