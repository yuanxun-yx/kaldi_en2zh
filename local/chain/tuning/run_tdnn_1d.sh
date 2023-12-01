#!/usr/bin/env bash

# 1b is the same as 1b, except that 1b adds an additional attention layer.

# modified from egs/rm/s5/local/chain/tuning/run_tdnn_wsj_rm_1a.sh

set -e

. ./path.sh

# configs for 'chain'
stage=0
train_set=train_sp
test_sets="dev test"
get_egs_stage=-10
dir=exp/chain/tdnn_tedlium_aishell_1d
# LSTM/chain options
train_stage=-10
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'
remove_egs=false

# Setting 'online_cmvn' to true replaces 'apply-cmvn' by
# 'apply-cmvn-online' both for i-vector extraction and TDNN input.
# The i-vector extractor uses the config 'conf/online_cmvn.conf' for
# both the UBM and the i-extractor. The TDNN input is configured via
# '--feat.cmvn-opts' that is set to the same config, so we use the
# same cmvn for i-extractor and the TDNN input.
online_cmvn=true

# configs for transfer learning

common_egs_dir=
primary_lr_factor=0.25 # learning-rate factor for all except last layer in transferred source model
nnet_affix=_online_tedlium

# model and dirs for source model used for transfer learning
src_mdl=$TEDLIUM_ROOT/exp/chain_cleaned_1e/tdnn1e_sp/final.mdl # input chain model trained on source dataset (tedlium) and this model is transfered to the target domain.
src_mfcc_config=$TEDLIUM_ROOT/conf/mfcc_hires.conf # mfcc config used to extract higher dim mfcc features used for ivector training in source domain.

# dirs for src-to-tgt transfer experiment
gmm=tri5a
gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_sp_ali
tree_dir=exp/chain/tedlium_tree_sp
lang=data/lang_chain_tedlium
lat_dir=exp/tri5a_lats

# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

required_files="$src_mfcc_config $src_mdl"

use_ivector=false
ivector_dim=$(nnet3-am-info --print-args=false $src_mdl | grep "ivector-dim" | cut -d" " -f2)
if [ -z $ivector_dim ]; then ivector_dim=0 ; fi

for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f" && exit 1;
  fi
done

ivector_dir=$AISHELL_ROOT/exp/nnet3/ivectors_${train_set}

if [ $stage -le 0 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    $AISHELL_ROOT/data/${train_set} data/lang $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 1 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" $AISHELL_ROOT/data/${train_set} \
    data/lang exp/tri5a $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz 2>/dev/null || true # save space
fi

if [ $stage -le 2 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -r $lang 2>/dev/null || true
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 3 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 5000 $AISHELL_ROOT/data/${train_set} $lang $ali_dir $tree_dir || exit 1;
fi

if [ $stage -le 4 ]; then
  mkdir -p $dir

  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  # relu-batchnorm-layer name=tdnn-target dim=1024 input=Append(tdnnf13.batchnorm@-3)
  tdnnf-layer name=tdnn-target $tdnn_opts dim=1024 bottleneck-dim=128 time-stride=3 input=tdnnf12.noop
  attention-relu-renorm-layer name=attention time-stride=2 num-heads=10 value-dim=60 key-dim=40 num-left-inputs=4 num-right-inputs=4
  linear-component name=prefinal-l dim=256 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs

  # Set the learning-rate-factor to be primary_lr_factor for transferred layers "
  # and adding new layers to them.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor" $src_mdl - \| \
      nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0: generate egs for chain to train new model on rm dataset."
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/rm-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  
 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir "$ivector_dir" \
    --feat.cmvn-opts="--config=conf/online_cmvn.conf" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false --online-cmvn $online_cmvn" \
    --egs.chunk-width 150,110,100 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 4 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $AISHELL_ROOT/data/train_sp_hires \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir || exit 1;
fi

if [ $stage -le 6 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 7 ]; then
  for test_set in dev test; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 10 --cmd "$decode_cmd" \
      --online-ivector-dir $AISHELL_ROOT/exp/nnet3/ivectors_$test_set \
      $graph_dir $AISHELL_ROOT/data/${test_set}_hires $dir/decode_${test_set} || exit 1;
  done
fi
exit 0