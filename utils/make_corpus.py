#!/usr/bin/env python
#encoding=utf-8

from argparse import ArgumentParser
import os
import sys
import json
from vocab import vocab

def main(FLAGS):
    #tt_vocab = vocab(os.path.join(FLAGS.vocab_dir,"vocab.title"))
    #at_vocab = vocab(os.path.join(FLAGS.vocab_dir,"vocab.author"))
    pm_vocab = vocab(FLAGS.vocab)
    print len(pm_vocab)

    pm_corpus_name = FLAGS.corpus
    f = open(pm_corpus_name,'w')
    for lines in sys.stdin:
        lines = lines.strip()
        jsonobj = json.loads(lines)
        output_dict = {}
        try:
            title = jsonobj["title"]
            author = jsonobj["author"]
            poem = jsonobj["poem"]
        except:
            continue
        total_poem = []
        one_sentence = []
        for token in poem:
            #if token == u'，' or token == u'。':
            #    total_poem.append(one_sentence)
            #    one_sentence = []
            #   continue
            idx = pm_vocab[token]
            total_poem.append(idx)
        output_dict['poem'] = total_poem
        output_dict['ori'] = jsonobj["poem"]
        print >> f, json.dumps(output_dict,ensure_ascii = False).encode('utf-8')
        #print json.dumps(output_dict,ensure_ascii = False).encode('utf-8')
    f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus_file",type = str,dest = "corpus")
    parser.add_argument("--vocab_file",type = str,dest = "vocab")
    FLAGS = parser.parse_args()
    print FLAGS

    main(FLAGS)

