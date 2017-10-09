#!/usr/bin/env python
#encoding=utf-8

from argparse import ArgumentParser
import sys
import json
from vocab import vocab

parser = ArgumentParser()
parser.add_argument('--input',type = str)
parser.add_argument('--output',type = str)
parser.add_argument('--vocab',type = str)
FLAGS = parser.parse_args()

FLAGS.vocab_size = len(vocab(FLAGS.vocab))

fo = open(FLAGS.output,'w')
with open(FLAGS.input,'r') as f:
    for lines in f:
        try:
            lines = lines.strip().decode('utf-8')
            jsonobj = json.loads(lines)
        except:
            continue
        newjson = {}
        poem = jsonobj["poem"]
        list_ = []
        for i in poem:
            oneline = [0 for i in range(FLAGS.vocab_size)]
            oneline[i] = 1
            list_.append(oneline)
        newjson["poem"] = list_
        print >> fo, json.dumps(newjson,ensure_ascii = False).encode('utf-8')

fo.close()


