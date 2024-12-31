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

"""CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

# Utility Functions

def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations.values():
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs



# Vocabulary Class

# Preprocessing Functions"""
tensor = torch.rand(3,4)
print(tensor)












# Main Function
def main():
    print("Complete the steps here...")
