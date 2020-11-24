#!/usr/local/bin/python3
# CSCI B551 Fall 2020
#
# Authors:  Neelan Scheumann (nscheuma)
#           Cody Harris (harrcody)       
#           Vishal Bhalla (vibhalla)
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
from collections import defaultdict

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def train(corpus):
    data = corpus.split()

    # learn the initial probabilities
    initial_dict = defaultdict(lambda: 0)
    for word in data:
        beg_pos = word[0]
        initial_dict[beg_pos] += 1
        for l in letters:
            if l in initial_dict.keys():
                pass
            else:
                initial_dict[l] = 1

    total = sum(initial_dict.values())
    for key in initial_dict.keys():
        initial_dict[key] = initial_dict[key] / total

    # learn the transition probailities
    trans_dict = {}
    for word in data:
        for i in range(len(word)-1):
            if word[i] in trans_dict.keys():
                if word[i+1] in trans_dict[word[i]].keys():
                    trans_dict[word[i]][word[i+1]] += 1
                else:
                    trans_dict[word[i]][word[i+1]] = 1
            else:
                trans_dict[word[i]] = {word[i+1]: 1}

    for value in trans_dict.values():
        for letter in letters:
            if letter in value.keys():
                pass
            else:
                value[letter] = 1

    for value in trans_dict.values():
        total = sum(value.values())
        for key in value.keys():
            value[key] = value[key] / total

    return initial_dict, trans_dict

def decode(str, replace, rearrange):
    str2 = str.translate({ ord(i):ord(replace[i]) for i in replace })
    str2 +=  ' ' * (len(rearrange)-(len(str2) %  len(rearrange)))
    return "".join(["".join([str2[rearrange[j] + i] for j in range(0, len(rearrange))]) for i in range(0, len(str), len(rearrange))])

def swap_dict(dictionary):
    i, j = random.choices(list(dictionary.keys()), k=2)
    dictionary[i], dictionary[j] = dictionary[j], dictionary[i]

def break_code(string, corpus):

    #learn the probabilities based on the corpus
    initial_dict, trans_dict = train(corpus)

    # initialize replacement table
    letters_copy = letters[:]
    random.shuffle(letters_copy)
    replace_table = dict(zip(letters, letters_copy))

    # initialize rearrange table
    rearrange_table = [0,1,2,3]
    random.shuffle(rearrange_table)

    for i in range(75000):
        # create new replacement table by swapping two values
        new_replace_table = copy.deepcopy(replace_table)
        swap_dict(new_replace_table)

        # create new rearrange table but only every 100 iterations
        if i % 100 == 0:
            new_rearrange_table = [0,1,2,3]
            random.shuffle(new_rearrange_table)
        else:
            new_rearrange_table = copy.deepcopy(rearrange_table)
        
        # create potentially decoded messages using the two sets of tables
        decode1 = decode(string, replace_table, rearrange_table)
        decode2 = decode(string, new_replace_table, new_rearrange_table)

        # calculate probs for each potentially decoded message
        prob1 = prob(decode1, initial_dict, trans_dict)
        prob2 = prob(decode2, initial_dict, trans_dict)

        if prob2 > prob1:
            rearrange_table = copy.deepcopy(new_rearrange_table)
            replace_table = copy.deepcopy(new_replace_table)
        elif random.random() < math.exp(prob2 - prob1):
            rearrange_table = copy.deepcopy(new_rearrange_table)
            replace_table = copy.deepcopy(new_replace_table)

    return decode(string, replace_table, rearrange_table)

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

