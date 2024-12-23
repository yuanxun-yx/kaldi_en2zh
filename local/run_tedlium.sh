#!/usr/bin/env bash
#
# Based mostly on the Switchboard recipe. The training database is TED-LIUM,
# it consists of TED talks with cleaned automatic transcripts:
#
# https://lium.univ-lemans.fr/ted-lium3/
# http://www.openslr.org/resources (Mirror).
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  Vincent Nguyen
#            2016  Johns Hopkins University (Author: Daniel Povey)
#            2018  François Hernandez
#
# Apache 2.0
#

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

add_pitch=true

. utils/parse_options.sh # accept options

pitch_affix=
if [ $add_pitch = true ]; then
  pitch_affix=_pitch
fi

# Data preparation
if [ $stage -le 0 ]; then
  local/download_data.sh
fi

if [ $stage -le 1 ]; then
  local/prepare_data.sh
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  # [we chose 3 minutes because that gives us 38 speakers for the dev data, which is
  #  more than our normal 30 jobs.]
  for dset in dev test train; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
  done
fi


if [ $stage -le 2 ]; then
  local/prepare_dict.sh
fi

if [ $stage -le 3 ]; then
  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp
fi

if [ $stage -le 4 ]; then
  # later on we'll change this script so you have the option to
  # download the pre-built LMs from openslr.org instead of building them
  # locally.
  if $train_lm; then
    local/ted_train_lm.sh
  else
    local/ted_download_lm.sh
  fi
fi

if [ $stage -le 5 ]; then
  local/format_lms.sh
fi

# Feature extraction
if [ $stage -le 6 ]; then
  for set in test dev train; do
    dir=data/$set
    steps/make_mfcc$pitch_affix.sh --nj 30 --cmd "$train_cmd" $dir
    steps/compute_cmvn_stats.sh $dir
  done
fi

# Now we have 452 hours of training data.
# Well create a subset with 10k short segments to make flat-start training easier:
if [ $stage -le 7 ]; then
  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort
  utils/data/remove_dup_utts.sh 10 data/train_10kshort data/train_10kshort_nodup
fi

# Train
if [ $stage -le 8 ]; then
  steps/train_mono.sh --nj 20 --cmd "$train_cmd" \
    data/train_10kshort_nodup data/lang_nosp exp/mono$pitch_affix
fi

if [ $stage -le 9 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/mono$pitch_affix exp/mono${pitch_affix}_ali
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train data/lang_nosp exp/mono${pitch_affix}_ali exp/tri1$pitch_affix
fi

if [ $stage -le 10 ]; then
  utils/mkgraph.sh data/lang_nosp exp/tri1$pitch_affix exp/tri1$pitch_affix/graph_nosp

  # The slowest part about this decoding is the scoring, which we can't really
  # control as the bottleneck is the NIST tools.
  for dset in dev test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri1$pitch_affix/graph_nosp data/${dset} exp/tri1$pitch_affix/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
       data/${dset} exp/tri1$pitch_affix/decode_nosp_${dset} exp/tri1$pitch_affix/decode_nosp_${dset}_rescore
  done
fi

if [ $stage -le 11 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri1$pitch_affix exp/tri1${pitch_affix}_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/train data/lang_nosp exp/tri1${pitch_affix}_ali exp/tri2$pitch_affix
fi

if [ $stage -le 12 ]; then
  utils/mkgraph.sh data/lang_nosp exp/tri2${pitch_affix} exp/tri2${pitch_affix}/graph_nosp
  for dset in dev test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri2${pitch_affix}/graph_nosp data/${dset} exp/tri2${pitch_affix}/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
       data/${dset} exp/tri2${pitch_affix}/decode_nosp_${dset} exp/tri2${pitch_affix}/decode_nosp_${dset}_rescore
  done
fi

if [ $stage -le 13 ]; then
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_nosp exp/tri2${pitch_affix}
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp exp/tri2${pitch_affix}/pron_counts_nowb.txt \
    exp/tri2${pitch_affix}/sil_counts_nowb.txt \
    exp/tri2${pitch_affix}/pron_bigram_counts_nowb.txt data/local/dict
fi

if [ $stage -le 14 ]; then
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  cp -rT data/lang data/lang_rescore
  cp data/lang_nosp/G.fst data/lang/
  cp data/lang_nosp_rescore/G.carpa data/lang_rescore/

  utils/mkgraph.sh data/lang exp/tri2${pitch_affix} exp/tri2${pitch_affix}/graph

  for dset in dev test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri2${pitch_affix}/graph data/${dset} exp/tri2${pitch_affix}/decode_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
       data/${dset} exp/tri2${pitch_affix}/decode_${dset} exp/tri2${pitch_affix}/decode_${dset}_rescore
  done
fi

if [ $stage -le 15 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2${pitch_affix} exp/tri2${pitch_affix}_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train data/lang exp/tri2${pitch_affix}_ali exp/tri3${pitch_affix}

  utils/mkgraph.sh data/lang exp/tri3${pitch_affix} exp/tri3${pitch_affix}/graph

  for dset in dev test; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri3${pitch_affix}/graph data/${dset} exp/tri3${pitch_affix}/decode_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
       data/${dset} exp/tri3${pitch_affix}/decode_${dset} exp/tri3${pitch_affix}/decode_${dset}_rescore
  done
fi

if [ $stage -le 16 ]; then
  # this does some data-cleaning.  It actually degrades the GMM-level results
  # slightly, but the cleaned data should be useful when we add the neural net and chain
  # systems.  If not we'll remove this stage.
  local/run_cleanup_segmentation.sh --srcdir exp/tri3${pitch_affix}
fi

if [ $stage -le 17 ]; then
  # This will only work if you have GPUs on your system (and note that it requires
  # you to have the queue set up the right way... see kaldi-asr.org/doc/queue.html)
  local/chain/run_tdnn.sh --stage 20
fi

if [ $stage -le 18 ]; then
  # You can either train your own rnnlm or download a pre-trained one
  if $train_rnnlm; then
    local/rnnlm/tuning/run_lstm_tdnn_a.sh
    local/rnnlm/average_rnnlm.sh
  else
    local/ted_download_rnnlm.sh
  fi
fi

if [ $stage -le 19 ]; then
  # Here we rescore the lattices generated at stage 17
  rnnlm_dir=exp/rnnlm_lstm_tdnn_a_averaged
  lang_dir=data/lang_chain
  ngram_order=4

  for dset in dev test; do
    data_dir=data/${dset}_hires
    decoding_dir=exp/chain_cleaned_1e/tdnn1e_sp/decode_${dset}
    suffix=$(basename $rnnlm_dir)
    output_dir=${decoding_dir}_$suffix

    rnnlm/lmrescore_pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      $lang_dir $rnnlm_dir \
      $data_dir $decoding_dir \
      $output_dir
  done
fi


echo "$0: success."
exit 0
