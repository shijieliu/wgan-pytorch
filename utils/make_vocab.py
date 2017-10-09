#!/usr/bin/env python
#encoding=utf-8

from argparse import ArgumentParser
import os
import sys
import json

def build_vocab(FLAGS):
    FLAGS.vocab_tt = sub_build_vocab(FLAGS.title,FLAGS.vocab_tt)
    FLAGS.vocab_at = sub_build_vocab(FLAGS.author,FLAGS.vocab_at)
    FLAGS.vocab_pm = sub_build_vocab(FLAGS.poem,FLAGS.vocab_pm)

def sub_build_vocab(string,vocab):
    for i in string:
        if i == u'，' or i == u'。':
            continue
        if i in vocab:
            continue
        vocab.add(i)
    return vocab

def print_sequence(FLAGS,name,vocab):
    file_name = os.path.join(FLAGS.dir,name)
    f = open(file_name,"w")
    for index,token in enumerate(vocab):
        print >> f,"\t".join([token,str(index)]).encode('utf-8')
    f.close()

def main(FLAGS):
    #title
    #author
    #poem
    vocab_tt = set()
    #FLAGS.vocab_at = set()
    #FLAGS.vocab_pm = set()
    for lines in sys.stdin:
        lines = lines.strip()
        jsonobj = json.loads(lines.decode('utf-8'))
        try:
            FLAGS.title = jsonobj["title"]
            FLAGS.author = jsonobj["author"]
            poem = jsonobj["poem"]
        except:
            continue
        #build_vocab(FLAGS)
        for i in poem:
            if i in vocab_tt:
                #print i
                continue
            vocab_tt.add(i)
    print_sequence(FLAGS,"vocab.title",vocab_tt)
    #print_sequence(FLAGS,"vocab.author",FLAGS.vocab_at)
    #print_sequence(FLAGS,"vocab.poem",FLAGS.vocab_pm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir",type = str,help = "corpus dir")
    FLAGS = parser.parse_args()
    print FLAGS

    #if os.path.exists(FLAGS.dir):
    #    os.system('rm -rf '+FLAGS.dir)
    #os.mkdir(FLAGS.dir)

    main(FLAGS)
