#!/usr/bin/env python
#encoding=utf-8

import sys
import re
import json

one_poem = {}
title = ""
author = ""
poem = ""
pattern=re.compile(u"第[一二三四五六七八九十]卷")
pattern_t = re.compile(u'◎卷.\d+【(.*)】(.*)')
pattern_f = re.compile(u'(.*)（.*')
for lines in sys.stdin:
	lines = lines.strip().decode('utf-8').strip()
	if pattern.match(lines):
		continue
	match = pattern_t.match(lines)
	if pattern_t.match(lines):
		#print match.groups()
		one_poem["poem"] = poem
		print json.dumps(one_poem,ensure_ascii = False).encode('utf-8')
		poem = ""
		title = match.group(1)
		author = match.group(2)
		one_poem["title"] = title
		one_poem["author"] = author
		continue
	match = pattern_f.match(lines)
	if title and author:
		if match:
			lines = match.group(1)
		poem += lines
	#if not lines.startswith(' '):
#		print lines
#		continue
#	print lines
