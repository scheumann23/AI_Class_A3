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

def emis_prob(pic1, pic2, m):
    counter = sum([1 if pic1[r][c] == pic2[r][c] else 0 for c in range(len(pic1[0])) for r in range(len(pic1))])
    log_prob = 350 * math.log(m/100) + counter * math.log((100-m)/m)
    return log_prob

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


# Read the training data
def read_data(fname):
    file = open(fname, 'r');
    text = []
    for line in file:
        data = tuple([w for w in line.split()])
        row = [r for r in data[0::2]]
        text.append(row)
    return text

# Do the training!

initial_dict = defaultdict(lambda: 0)
trans_dict = {}

def train(data):
    #print(data)
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

    # learn the initial probabilities

    for row in data:
        for word in row:
            beg_pos = word[0]
            initial_dict[beg_pos] += 1
        for l in TRAIN_LETTERS:
            if l in initial_dict.keys():
                pass
            else:
                initial_dict[l] = 1
        
    total = sum(initial_dict.values())
    for key in initial_dict.keys():
        initial_dict[key] = initial_dict[key] / total
    
    # learn the transition probailities
    
    for row in data:
        for word in row:
            for i in range(len(word)-1):
                if word[i] in trans_dict.keys():
                    if word[i+1] in trans_dict[word[i]].keys():
                         trans_dict[word[i]][word[i+1]] += 1
                    else:
                        trans_dict[word[i]][word[i+1]] = 1
                else:
                   trans_dict[word[i]] = {word[i+1]: 1}

        for value in trans_dict.values():
            for letter in TRAIN_LETTERS:
                if letter in value.keys():
                    pass
                else:
                    value[letter] = 1

    for value in trans_dict.values():
        total += sum(value.values())
    

    for value in trans_dict.values():
        for key in value.keys():
            value[key] = value[key] / total



#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

# Read training data and train the model
train_data = read_data(train_txt_fname)
train(train_data)

# The final two lines of your output should look something like this:
print("Simple: " + simple_model(test_letters, train_letters, 5))
print("   HMM: " + "Sample simple result") 


