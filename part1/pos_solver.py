###################################
# CS B551 Fall 2020, Assignment #3
#
# Your names and user ids:
# Neelan Scheumann (nscheuma)
# Vishal Bhalla (vibhalla)
# Cody Harris (harrcody)
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
from collections import defaultdict


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
vocab_dict = {}
emis_dict = {}
initial_dict = defaultdict(lambda: 0)
trans_dict = {}
posNum = {}
pos = ['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']

class Solver:
    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        #Get total number of words in the vocab
        totalWords = sum(sum(c.values()) for c in vocab_dict.values())
        if model == "Simple":
            condProb = 0
            for i in range(0,len(sentence)):
                prob = 0
                try:
                    if emis_dict[label[i]]:
                        prob += (emis_dict[label[i]][sentence[i]])
                except KeyError:
                    prob += (1/(totalWords + 1))
                
                condProb =  (condProb + math.log(prob))
            return condProb

        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!

    def train(self, data):
        for row in data:
            # learn the vocab dictionary {word1: {pos1: #, pos2: #}, word2: {pos1: #, pos2: #}, ...}
            for i in range(0,len(row[0])):
                if row[0][i] in vocab_dict:
                    if row[1][i] in vocab_dict[row[0][i]]:
                        vocab_dict[row[0][i]][row[1][i]] += 1
                    else:
                        vocab_dict[row[0][i]][row[1][i]] = 1
                else:
                    vocab_dict[row[0][i]] = {row[1][i]:1}

            # learn the emission probability dictionary {pos1: {word1: #, word2: #}, pos2: {word1: #, word2: #}, ...}
            for i in range(0,len(row[0])):
                if row[1][i] in emis_dict:
                    if row[0][i] in emis_dict[row[1][i]]:
                        emis_dict[row[1][i]][row[0][i]] += 1
                    else:
                        emis_dict[row[1][i]][row[0][i]] = 1
                else:
                    emis_dict[row[1][i]] = {row[0][i]:1}
            
            # learn the initial probability dictionary {pos1: # pos2: #, pos3: #, ...}
            beg_pos = row[1][0]
            initial_dict[beg_pos] += 1
            for p in pos:
                if p in initial_dict.keys():
                    pass
                else:
                    initial_dict[p] = 1

            # learn the transition probability dictionary {pos1: {pos1: #, pos2: #}, pos2: {pos1: #, pos2: #}, ...}
            for i in range(len(row[1])-1):
                if row[1][i] in trans_dict.keys():
                    if row[1][i+1] in trans_dict[row[1][i]].keys():
                        trans_dict[row[1][i]][row[1][i+1]] += 1
                    else:
                        trans_dict[row[1][i]][row[1][i+1]] = 1
                else:
                    trans_dict[row[1][i]] = {row[1][i+1]: 1}

        # convert initial_dict values to probabilities 
        total = sum(initial_dict.values())
        for key in initial_dict.keys():
            initial_dict[key] = initial_dict[key] / total

        # convert vocab_dict values to probabilities 
        for value in vocab_dict.values():
            total = sum(value.values())
            for key in value.keys():
                value[key] = value[key] / total

        # convert emis_dict values to probabilities 
        for value in emis_dict.values():
            total = sum(value.values())
            for key in value.keys():
                value[key] = value[key] / total

        # convert trans_dict values to probabilities 
        for value in trans_dict.values():
            total = sum(value.values())
            for key in value.keys():
                value[key] = value[key] / total

        for p in pos:
                    try:
                        pNum = (sum(x[p] for x in vocab_dict.values() if p in x.keys()))
                    except KeyError:
                        pass
                    posNum[p] = pNum if pNum != 0 else 1
    

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        parts = []
        for word in sentence:
            try:
                if vocab_dict[word]:
                    # This code to get the key with max value is taken from https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
                    parts.append(max(vocab_dict[word], key=lambda k: vocab_dict[word][k]))
            except KeyError:
                parts.append(max(posNum, key=posNum.get))
        return parts

    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)

    def confidence(self, sentence, answer):
        return [ 0.999 ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

