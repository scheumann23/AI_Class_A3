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
vocab = {}
pos = ['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
posProb = {}
totalWords = 0
class Solver:
    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):

        if model == "Simple":
            condProb = 0
            for i in range(0,len(sentence)):
                try:
                    if vocab[sentence[i]]:
                        maxLabel = (max(vocab[sentence[i]], key=lambda k: vocab[sentence[i]][k]))
                        maxCnt = (vocab[sentence[i]][maxLabel])
                        allCnt = (sum(vocab[sentence[i]].values()))
                        prob = (maxCnt/allCnt)*posProb[maxLabel]
                except KeyError:
                    prob = (1/(totalWords + 1))
                condProb += (round(math.log(prob),2))
            return condProb
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #

    def learn_init_probs(self, data):
        initial_dict = defaultdict(lambda: 0)
        for sent in data:
            beg_word = sent[1][0]
            initial_dict[beg_word] += 1
        for p in pos:
            if p in initial_dict.keys():
                pass
            else:
                initial_dict[p] = 1
        total_words = sum(initial_dict.values())
        for key in initial_dict.keys():
            initial_dict[key] = initial_dict[key] / total_words
        return initial_dict

    def train(self, data):
        for row in data:
            for i in range(0,len(row[0])):
                if row[0][i] in vocab:
                    if row[1][i] in vocab[row[0][i]]:
                        vocab[row[0][i]][row[1][i]] += 1
                    else:
                        vocab[row[0][i]][row[1][i]] = 1
                else:
                    vocab[row[0][i]] = {row[1][i]:1}
        
        totalWords = sum(sum(c.values()) for c in vocab.values())
        for p in pos:
            try:
                pProb = (sum(x[p] for x in vocab.values() if p in x.keys()))
            except KeyError:
                pass
            posProb[p] = round(pProb/totalWords if pProb != 0 else 1/(totalWords+1),2)


    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        parts = []
        for word in sentence:
            try:
                if vocab[word]:
                    # This code to get the key with max value is taken from https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
                    parts.append(max(vocab[word], key=lambda k: vocab[word][k]))
            except KeyError:
                parts.append('unknown')
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

