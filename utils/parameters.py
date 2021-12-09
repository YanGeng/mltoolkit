#!/usr/bin/env python
# -*- coding: utf-8 -*-

OOV = '_OOV_'
START = '</s>'
END = '</s>'
GOLD_TAG = 'GoldNER'
PRED_TAG = 'NER'
task = 'ner'
MAX_LEN = 175  # Max sequence length

LogP_ZERO = float('-inf')
LogP_INF = float('inf')
LogP_ONE = 0.0
FloatX = 'float32'
