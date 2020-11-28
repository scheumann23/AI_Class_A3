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
from numpy import zeros, argmax


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
vocab_dict = {}
emis_dict = {}
initial_dict = defaultdict(lambda: 0)
trans_dict = {}
posNum = {}
missing_emis = 1/10000000
missing_tran = 1/10000000
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
            condProb = 0
            for i in range(len(sentence)):
                emis_bool = label[i] in emis_dict.keys() and sentence[i] in emis_dict[label[i]].keys()
                if i == 0:
                    if emis_bool:
                        prob = emis_dict[label[i]][sentence[i]] * initial_dict[label[i]]
                    else:
                        prob = missing_emis * initial_dict[label[i]]
                else:
                    trans_bool = label[i-1] in trans_dict.keys() and label[i] in trans_dict[label[i-1]].keys()
                    if trans_bool and emis_bool:
                        prob = trans_dict[label[i-1]][label[i]] * emis_dict[label[i]][sentence[i]]
                    elif not trans_bool and emis_bool:
                        prob = missing_tran * emis_dict[label[i]][sentence[i]]
                    elif trans_bool and not emis_bool:
                        prob = trans_dict[label[i-1]][label[i]] * missing_emis
                    elif not trans_bool and not emis_bool:
                        prob = missing_tran * missing_emis
                condProb = (condProb + math.log(prob))
            return condProb
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
        totalWords = sum(sum(c.values()) for c in vocab_dict.values())
        # initialize matrices
        score = zeros([len(pos), len(sentence)])
        trace = zeros([len(pos), len(sentence)], dtype=int)
        for i, obs in enumerate(sentence):
            for j, st in enumerate(pos):
                emis_bool = st in emis_dict.keys() and obs in emis_dict[st].keys()
                if i == 0:
                    if emis_bool:
                        score[j][i] = emis_dict[st][obs] * initial_dict[st]
                    else:
                        score[j][i] = (missing_emis) * initial_dict[st]
                else:
                    prev_list = [score[a][i - 1] for a in range(len(pos))]
                    max_this = []
                    for p in range(len(prev_list)):
                        if pos[p] in trans_dict.keys() and st in trans_dict[pos[p]].keys():
                            max_this.append(prev_list[p] * trans_dict[pos[p]][st])
                        else:
                            max_this.append(prev_list[p] * (missing_tran))
                    trace[j][i] = argmax(max_this)
                    if emis_bool:
                        score[j][i] = max(max_this) * emis_dict[st][obs]
                    else:
                        score[j][i] = max(max_this) * (missing_emis)
        z = argmax(score[:, -1])
        hidden = [pos[z]]
        for h in range(len(sentence) - 1, 0, -1):
            prev_z = z
            z = trace[prev_z, h]
            hidden += [pos[z]]

        return hidden[::-1]

    def confidence(self, sentence, answer):
        #Code based on example of variable elimination from variable_elimination.py
        #Provided by David Crandall in Fall 2020 CSCI
        conf_list = []
        tau = {}
        N = len(sentence)
        for M in range(len(sentence)):
            for i in range(0, M):
                tau[i+1] = { s:0 for s in pos }
                for s in pos:
                    for s2 in pos:
                        emis_bool = s2 in emis_dict.keys() and sentence[i] in emis_dict[s2].keys()
                        trans_bool = s2 in trans_dict.keys() and s in trans_dict[s2].keys()
                        if emis_bool and trans_bool:
                            tau[i+1][s] += (tau[i][s2] if i > 0 else initial_dict[s2]) * emis_dict[s2][sentence[i]] * trans_dict[s2][s]
                        elif emis_bool and not trans_bool:
                            tau[i+1][s] += (tau[i][s2] if i > 0 else initial_dict[s2]) * emis_dict[s2][sentence[i]] * missing_tran
                        elif not emis_bool and trans_bool:
                            tau[i+1][s] += (tau[i][s2] if i > 0 else initial_dict[s2]) * missing_emis * trans_dict[s2][s]
                        elif not emis_bool and not trans_bool:
                            tau[i+1][s] += (tau[i][s2] if i > 0 else initial_dict[s2]) * missing_emis * missing_tran
                        
            for i in range(N-1, M, -1):
                tau[i] = { s:0 for s in pos }
                for s in pos:
                    for s2 in pos:
                        emis_bool = s2 in emis_dict.keys() and sentence[i] in emis_dict[s2].keys()
                        trans_bool = s in trans_dict.keys() and s2 in trans_dict[s].keys()
                        if emis_bool and trans_bool:
                            tau[i][s] += (tau[i+1][s2] if i+1 < N else 1) * emis_dict[s2][sentence[i]] * trans_dict[s][s2]
                        elif emis_bool and not trans_bool:
                            tau[i][s] += (tau[i+1][s2] if i+1 < N else 1) * emis_dict[s2][sentence[i]] * missing_tran
                        elif not emis_bool and trans_bool:
                            tau[i][s] += (tau[i+1][s2] if i+1 < N else 1) * missing_emis * trans_dict[s][s2]
                        elif not emis_bool and not trans_bool:
                            tau[i][s] += (tau[i+1][s2] if i+1 < N else 1) * missing_emis * missing_tran
            
            joint = {}
            for s in pos:
                emis_bool = s in emis_dict.keys() and sentence[M] in emis_dict[s].keys()
                if emis_bool:
                    joint[s] = (1 if M == 0 else tau[M][s]) * (1 if M == len(sentence)-1 else tau[M+1][s]) * emis_dict[s][sentence[M]]
                else:
                    joint[s] = (1 if M == 0 else tau[M][s]) * (1 if M == len(sentence)-1 else tau[M+1][s]) * missing_emis
            j_sum = sum(joint.values())
            total_joints = { j:joint[j] / j_sum for j in joint }
            conf_list +=  [0.999 if round(total_joints[answer[M]], 3) == 1.0 else round(total_joints[answer[M]], 3)]
        return conf_list


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

