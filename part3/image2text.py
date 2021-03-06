#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors:  Neelan Scheumann (nscheuma)
#           Cody Harris (harrcody)
#           Vishal Bhalla (vibhalla)
#
# (based on skeleton code by D. Crandall, Oct 2020)

from PIL import Image, ImageDraw, ImageFont
import sys
import random
import math
import copy
import re
import numpy as np
from collections import defaultdict

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print(im.size)
    #print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# Strip out unneccessary labels from the bc.train file so we are left with only grammatically correct english sentences
def strip_labels(string):
    output = re.sub(' ADJ | ADV | ADP | CONJ | DET | NOUN | NUM | PRON | PRT | VERB | X |\n', ' ', string)
    output = re.sub(' \'\' . ', '\" ', output)
    output = re.sub(r' \`\` . ', ' \"', output)
    output = re.sub(r'\`\` . ', '\"', output)
    output = re.sub(' , . ', ', ', output)
    output = re.sub(r' \? . ', '? ', output)
    output = re.sub(r' \! . ', '! ', output)
    output = re.sub(' . . ', '. ', output)
    output = re.sub('    ', '', output)
    return output

# read in the data
def read_data(fname):
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        exemplars += [ strip_labels(line) ]
    return exemplars

# Do the training!

initial_dict = defaultdict(lambda: 0)
trans_dict = {}

def train(data):
    #print(data)
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

    # learn the initial counts
    for row in data:
        beg_pos = row[0]
        initial_dict[beg_pos] += 1
    for l in TRAIN_LETTERS:
        if l in initial_dict.keys():
            pass
        else:
            initial_dict[l] = 1
    
    # convert to probabilities
    total = sum(initial_dict.values())
    for key in initial_dict.keys():
        initial_dict[key] = initial_dict[key] / total
    
    # learn the transition counts
    for row in data:
        for i in range(len(row)-1):
            if row[i] in trans_dict.keys():
                if row[i+1] in trans_dict[row[i]].keys():
                        trans_dict[row[i]][row[i+1]] += 1
                else:
                    trans_dict[row[i]][row[i+1]] = 1
            else:
                trans_dict[row[i]] = {row[i+1]: 1}

    # make sure each transition is accounted for at least once
    for value in trans_dict.values():
        for letter in TRAIN_LETTERS:
            if letter in value.keys():
                pass
            else:
                value[letter] = 1

    # convert to probabilities
    for value in trans_dict.values():
        total = sum(value.values())
        for key in value.keys():
            value[key] = value[key] / total

    return initial_dict, trans_dict

# the emission probabilities are conputer on the fly instead of learned from the training data
# this takes two 400-pixel images, in this case one ground truth iamge and one fuzzy image, and counts how many
# pixels are the same. Then, assuming that m% of the pixels are fuzzy it computes a probability score that the
# images are the same
def emis_prob(pic1, pic2, m):
    counter = sum([1 if pic1[r][c] == pic2[r][c] else 0 for c in range(len(pic1[0])) for r in range(len(pic1))])
    log_prob = 350 * math.log(m/100) + counter * math.log((100-m)/m)
    return log_prob

# this model compares each character in the fuzzy image to each of the ground truth training examples and returns
# the character that has the highest probability
def simple_model(fuzzy_img, train_letters, m):
    seq = []
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    for i in range(len(fuzzy_img)):
        max_let = ''
        max_prob = -100000000
        for let in TRAIN_LETTERS:
            temp_prob = emis_prob(fuzzy_img[i], train_letters[let], m)
            if temp_prob > max_prob:
                max_prob = temp_prob
                max_let = let
        seq.append(max_let)
    return "".join(seq)

# implements the Viterbi algorithm 
# based on the same code we used in Part 1
def hmm_viterbi(fuzzy_img, train_letters):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    # initialize matrices
    score = np.zeros([len(TRAIN_LETTERS), len(fuzzy_img)])
    trace = np.zeros([len(TRAIN_LETTERS), len(fuzzy_img)], dtype=int)
    for i, obs in enumerate(fuzzy_img):
        for j, st in enumerate(TRAIN_LETTERS):
            if i == 0:
                score[j][i] = emis_prob(obs, train_letters[st], 25) + math.log(initial_dict[st])
            else:
                prev_list = [score[a][i - 1] for a in range(len(TRAIN_LETTERS))]
                max_this = []
                for p in range(len(prev_list)):
                    max_this.append(prev_list[p] + math.log(trans_dict[TRAIN_LETTERS[p]][st]))
                
                trace[j][i] = np.argmax(max_this)
                score[j][i] = max(max_this) + emis_prob(obs, train_letters[st], 25)

    z = np.argmax(score[:, -1])
    hidden = [TRAIN_LETTERS[z]]
    for h in range(len(fuzzy_img) - 1, 0, -1):
        prev_z = z
        z = trace[prev_z, h]
        hidden += [TRAIN_LETTERS[z]]

    return hidden[::-1]

#####
# main program
train_img_fname, train_txt_fname, test_img_fname = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

# Read training data and train the model
train_data = read_data(train_txt_fname)
train(train_data)

# The final two lines of your output should look something like this:
print("Simple: " + simple_model(test_letters, train_letters, 25))
print("   HMM: " + ''.join(hmm_viterbi(test_letters, train_letters)) )