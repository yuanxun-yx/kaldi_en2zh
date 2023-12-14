# Transfer TED-LIUM to AISHELL

## Where to Place This Repository

By default, this folder should be placed as `egs/<project-name>/<version>`. Otherwise, please specify the path for kaldi's root diretory in `path.sh` and update the links.

## Paper Summary

### Title

Transfer Learning for Automatic Speech Recognition from English to Mandarin

### Author

Xun Yuan (xy2569@columbia.edu)

### Abstract

Most languages do not have sufficient data to train a automatic speech recognition (ASR) system of high performance.
In this work, we explore transfer learning from English to Mandarin on Kaldi to tackle this problem.
Several experiments were conducted to find the best setting for transfer learning.
We choose only part of the model to be transferred, with different learning rates.
Our results prove the effectiveness of transfer learning, and show that converting phonemes to source model and using i-vector from target model could improve the performance. 

### Results

CER for all experiments are listed in `RESULTS`.

## Run From Scratch

Replace `run.sh` of `tedlium/s5_r3` with `local/run_tedlium.sh`. Execute `run.sh` of `tedlium/s5_r3` and `aishell/s5` before executing `run.sh` of current directory. Finally use `result.sh` to see the CER results.

## Run Only Decode 

Add `--stage 7` argument to `local/chain/run_tdnn.sh` in `run.sh` and execute:

```
./run.sh --stage 21
```

## Contributions

1. Pinyin to CMU format conversion in `local/prepare_dict.sh`.

2. Transfer learning network configuration `network.xconfig` in `local/chain/tuning/run_tdnn_*.sh`.

3. Path modification (for transfer learning) in `local/chain/tuning/run_tdnn_*.sh` and `run.sh`.