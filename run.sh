#!/usr/bin/env bash

data="./data"
checkpoints="checkpoints"
vocab_dir=${data}/vocab
json_poem=${data}/json_poem
corpus_dir=${data}/corpus
utils="./utils"

#cat $txt | python clean.py > $json_poem
#make_vocab
cat $json_poem | python ${utils}/make_vocab.py \
	--dir $vocab_dir 
#make_corpus
cat $json_poem | \
	python ${utils}/make_corpus.py \
		--corpus_file ${corpus_dir}/corpus.title \
		--vocab_file ${vocab_dir}/vocab.title

#wgan model train
python wgan.py \
	--type "train" \
	--batchsize 1 \
	--input ${corpus_dir}/corpus.title \
	--vocab ${vocab_dir}/vocab.title \
	--epochs 30 \
	--learning_rate_G 1e-6 \
	--learning_rate_D 1e-3 \
    --save_dir ./save 

#wgan model predict
