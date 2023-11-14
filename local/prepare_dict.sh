#!/usr/bin/env bash

# prepare dictionary for AISHELL with CMU phonemes
# modified from $KALDI_ROOT/egs/aidatatang_200zh/s5/local/prepare_dict.sh

. ./path.sh

[ $# != 0 ] && echo "Usage: $0" && exit 1;

train_dir=data/local/train
dev_dir=data/local/dev
test_dir=data/local/test
dict_dir=data/local/dict
mkdir -p $dict_dir

aishell_dict=$AISHELL_ROOT/data/local/dict
tedlium_dict=$TEDLIUM_ROOT/data/local/dict

# convert Chinese pinyin to CMU format
# 1. remove silence lexicons
# 2. remove space between consonant and vowel
# 3. change to correct pinyin format
# 4. upppercase
cat $aishell_dict/lexicon.txt | sed -e '/^SIL/d' -e '/^<SPOKEN_NOISE>/d' | sed -e 's/\([a-z]\) \([a-z]\)/\1\2/g' | sed -e 's/\([a-z]\)\1\1/\1/g' -e 's/iuo/o/g' -e 's/ in/ yin/g' -e 's/ i/ y/g' -e 's/i[xyz]/i/g' -e 's/ y\([0-9]\)/ yi\1/g' -e 's/ ui/ wei/g' -e 's/ u/ w/g' -e 's/ w\([0-9]\)/ wu\1/g' -e 's/ v/ yu/g' -e 's/\([xqj]\)van/\1uan/g' | sed -e 's/\([a-z]\)/\U\1/g' | utils/pinyin_map.pl conf/pinyin2cmu > $dict_dir/lexicon_words.txt || exit 1;

# extract nonsilence phones
cat $dict_dir/lexicon_words.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}'| \
  sort -u |\
  perl -e '
  my %ph_cl;
  while (<STDIN>) {
    $phone = $_;
    chomp($phone);
    chomp($_);
    $phone =~ s:([A-Z]+)[0-9]:$1:;
    if (exists $ph_cl{$phone}) { push(@{$ph_cl{$phone}}, $_)  }
    else { $ph_cl{$phone} = [$_]; }
  }
  foreach $key ( keys %ph_cl ) {
     print "@{ $ph_cl{$key} }\n"
  }
  ' | sort -k1 > $dict_dir/nonsilence_phones.txt  || exit 1;

# copy files from tedlium
cp $tedlium_dict/silence_phones.txt $dict_dir/silence_phones.txt
cp $tedlium_dict/optional_silence.txt $dict_dir/optional_silence.txt

# generate extra questions for tones in pinyin
cat $dict_dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dict_dir/extra_questions.txt || exit 1;
cat $dict_dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dict_dir/extra_questions.txt || exit 1;

# Add to the lexicon the silences, noises etc.
# Typically, you would use "<UNK> NSN" here, but the Cantab Research language models
# use <unk> instead of <UNK> to represent out of vocabulary words.
echo '<unk> NSN' | cat - $dict_dir/lexicon_words.txt | sort | uniq > $dict_dir/lexicon.txt

echo "$0: dict preparation succeeded"
exit 0;
