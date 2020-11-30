# Part 1: Part-of-speech tagging

## Problem Formulation

The problem consists of three subparts. We are asked to:
- Perform simple POS tagging based on the simple Bayes Net 
- Perform POS tagging using the sophisticatedBayes Net using HMM/Viterbi Algorithm 
- Calculate the confidence level(marginal probabilities) on our tagging using Viterbi 

## How the Program Works

First we train the program on the test data provided. As part of the training we create multiple dictionaries, one for words and POS they are tagged against, 2nd we have a dictionary with emission probabilities and one with transition probabilities.

There are specific functions that perform the Viterbi, HMM based probability calculation and simple probability calculations.

To determine HMM posterior probability, we simply take the model and the predicted parts of speech and multiply the initial part of speech probability by the emmission probability of that part of speech for that word. Then we go through the rest of the words in the sentence multiplying by the transition to the next part of speech and the emission probability for that word. As we go along we are taking the log of each probability and summing it.

The Viterbi function is a classic version of the Viterbi algorithm where all possible parts of speech are evaluated for each word and then the most probable chain of parts of speech are selected as the output.

Lastly the confidence uses a variable elimination algorithm pased upon the code provided in variable_elimination.py.

## Problems, Assumptions, Simplifications, and Design Choices

The first problem that we faced was to decide on a data structure to use. We considered list of lists, dictionary of tuples etc and decided on dictionary of dictionaries given its ease of lookup. We had earlier been storing the numbers/counts in the dictionary and with having to convert the same to probabilities at multiple places we decided to save the probabilities directly. 

The major assumption in this program is the probability to use when there is a missing transition, or emission probability. We handled all possible initial probabilities when we created the dictionary. If the transition or emission probability used when the actual probability is missing in the training set is too large, then it creates bad results for the path taken in the Viterbi algorithm. After experimentation, we settled on 1e-7 as the probability when a transmission or emission probability is missing and this gave us good results.

# Part 2: Code Breaking

## Problem Formulation

The problem provides us with some text files that have been encoded using a replacement table and rearrangement tables which shuffle the original text two ways, first by changing the alphabetical mapping and then rearranging the alphabets in sets of 4. The character mapping table is unknown and so is the rearrangement table. The program should take the encrypted file as input and a file containing a set of english words for training. The output of the program is expected the file unencrypted.

## How the Program Works

First we train the program on the test data provided. As part of the training we create a dictionary with transition probabilities word by word and one with initial probabilities. We randomly prepare a alphabet mapping table and a rearrangement table and unscramble the input based on these tables. Thereafter we calculate the probabilities of the unscrambled file and compare with initial probabilities. If the new probabilities are greater we set the initial probabilities equal to the new probabilities and the new rearrangement and replacement tables replace the earlier ones. If the new probabilities are smaller we take the ratio of newer to initial probabilities and based on a threshold we shuffle the rearrangement and replacement tables.


## Problems, Assumptions, Simplifications, and Design Choices
As the number of combinations for replacement table is many magnitudes higher than the rearrangement table we rearrange the table once every 25 iterations. The program takes up to 10000 iterations for us to successfully unscramble the files. 

# Part 3: Reading Text

## Problem Formulation

A set if noisy images have been provided as input and our program should be able to read and predict the correct text in the images.

## How the Program Works

From the train data we strip out the POS tags and the doubled punctuations, and then use the text to calculate the initial and transition probabilities. Thereafter there are 2 processes for calculating the probabilities, the simple method which takes the train letters and compares them to the input image pixel by pixel and returns the string with highest probabilities. The second process uses hmm/viterbi to predict the sentences from the test images.


## Problems, Assumptions, Simplifications, and Design Choices

The data from the Part 1 above has been chosen as the training data as it has properly formed sentences instead of words as in case of Part 2. We also chose to use a "noise" parameter in the emmission probability calc of 25. This produced the best results. If it was too low, it failed to distinguish similar letters and if it was too high, the code assumed everything was one letter, e.g. 'AAAAAAAAAAAAAAAAAAAAAA'
