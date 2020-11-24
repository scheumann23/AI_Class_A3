#!/usr/local/bin/python3
# CSCI B551 Fall 2020
#
# Authors: PLEASE PUT YOUR NAMES AND USERIDS HERE
#
# based on skeleton code by D. Crandall, 11/2020
#
# ./break_code.py : attack encryption
#


import random
import math
import copy 
import sys
import encode
from dictionaries import initial_dict, trans_dict

# put your code here!

def break_code(string, corpus):
    return "Not implemented yet!"

def prob(string, initial_dict, trans_dict):
    doc_prob = 0
    word_list = string.split()
    for word in word_list:
        word_prob = 0
        for i in range(len(word)):
            if i == 0:
                word_prob += math.log(initial_dict[word[i]])
            else:
                word_prob += math.log(trans_dict[word[i-1]][word[i]])
        doc_prob += word_prob
    return doc_prob


if __name__== "__main__":
    if(len(sys.argv) != 4):
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = encode.read_clean_file(sys.argv[1])
    corpus = encode.read_clean_file(sys.argv[2])
    decoded = break_code(encoded, corpus)

    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)

