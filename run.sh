#!/usr/bin/env bash

source ~/bashrc_pytorch
txt="./tangshi.gb"
vocab_dir="./vocab"
json_poem="./json_poem"
corpus_dir="./corpus"
#cat $txt | python clean.py > $json_poem
#make_vocab
#cat $json_poem | python make_vocab.py \
#	--dir $vocab_dir 
#make_corpus
#cat $json_poem | \
#	python make_corpus.py \
#		--corpus_file ${corpus_dir}/corpus.title \
#		--vocab_file ${vocab_dir}/vocab.title

#python gen_onehot.py \
#    --input ${corpus_dir}/corpus.title \
#    --output ${corpus_dir}/corpus.title.onehot \
 #   --vocab ${vocab_dir}/vocab.title

#wgan model train
python wgan.py \
	--type "train" \
	--batchsize 1 \
	--input ${corpus_dir}/corpus.title \
	--vocab ${vocab_dir}/vocab.title \
	--epochs 30 \
	--learning_rate_G 1e-6 \
	--learning_rate_D 1e-3 \
    --save_dir ./save \
    --netG_param ./save/netG1.pth \
    --netD_param ./save/netD1.pth

#wgan model predict
