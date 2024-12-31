import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import csv
import random
import re
import os
import unicodedata
import itertools
import json



PAD_token = 0
SOS_token = 1
EOS_token = 2
corpus_name = "Corpus"

class Vocablary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        self.word2count = {}

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        keep_words = [k for k, v in self.word2count.items() if v >= min_count]
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.addWord(word)

def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Vocablary(corpus_name)
    return voc, pairs

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def filterPair(p, max_length):
    return len(p[0].split()) < max_length and len(p[1].split()) < max_length

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s







def maskNLLLoss(decoder_out, target, mask):
    nTotal = mask.sum()
    target = target.view(-1, 1)
    gathered_tensor = torch.gather(decoder_out, 1, target)
    crossEntropy = -torch.log(gathered_tensor)
    loss = crossEntropy.masked_select(mask)
    loss = loss.mean()
    return loss, nTotal.item()
