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
from utils import Vocablary,filterPairs,readVocs


PAD_token = 0
SOS_token = 1
EOS_token = 2
corpus_name = "Corpus"
voc = Vocablary(corpus_name)

def loadLinesAndConversations(fileName):
    lines = {}
    conversations = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            lineJson = json.loads(line)
            lineObj = {
                "lineID": lineJson["id"],
                "characterID": lineJson["speaker"],
                "text": lineJson["text"]
            }
            lines[lineObj['lineID']] = lineObj

            if lineJson["conversation_id"] not in conversations:
                convObj = {
                    "conversationID": lineJson["conversation_id"],
                    "movieID": lineJson["meta"]["movie_id"],
                    "lines": [lineObj]
                }
            else:
                convObj = conversations[lineJson["conversation_id"]]
                convObj["lines"].insert(0, lineObj)
            conversations[convObj["conversationID"]] = convObj

    return lines, conversations

def extractSentencePairs(conversations):
  qa_pairs = []
  for conversation in conversations.values():
      # Iterate over all the lines of the conversation
      for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
          inputLine = conversation["lines"][i]["text"].strip()
          targetLine = conversation["lines"][i+1]["text"].strip()
          # Filter wrong samples (if one of the lists is empty)
          if inputLine and targetLine:
              qa_pairs.append([inputLine, targetLine])
  return qa_pairs


def loadPrepareData( corpus_name, datafile):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs,max_length=10)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')



def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


voc = Vocablary(corpus_name)
datafile = "./formatted_movie_lines.txt"
print("Reading and processing file")
lines =  open(datafile, encoding = 'utf-8').read().strip().split('\n')
pairs = [[normalizeString(s) for s in pair.split('\t')] for pair in lines]
pairs = [pair for pair in pairs if len(pair)>1]
print("There are {} pairs/conversations in the dataset".format(len(pairs)))



Min_Count = 3
def trimRareWords(voc,pairs,Min_Count):
  voc.trim(Min_Count)
  keep_pairs = []
  for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True
    
    for word in input_sentence.split(' '):
      if word not in voc.word2index:
        keep_input = False
        break
        
    for word in output_sentence.split(' '):
      if word not in voc.word2index:
        keep_output = False
        break
    if keep_input and keep_output: 
      keep_pairs.append(pair)
      

  print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs),len(keep_pairs),len(keep_pairs)/len(pairs)))
  return keep_pairs

voc, pairs = loadPrepareData(corpus_name, datafile)
print("After filtering, there are {} pairs/conversations left".format(len(pairs)))
pairs = trimRareWords(voc,pairs,Min_Count)

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=0):
    return [[0 if token == PAD_token else 1 for token in seq] for seq in l]



def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = zip(*pair_batch)
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len