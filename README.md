# Part 1: Part-of-speech tagging

## Problem Formulation

The problem consists of three subparts. We are asked to:
	### Perform simple POS tagging based on the simple Bayes Net
	### Perform POS tagging using the sophisticatedBayes Net using HMM/Viterbi Algorithm
	### Calculate the confidence level(marginal probabilities) on our tagging using Viterbi

## How the Program Works

First we train the program on the test data provided. As part of the trainign we create multiple dictionaries, one for words and POS they are tagged against, 2nd we have a dictionary with emission probabilities and one with transition probabilities.

There are specific functions that perform the VIterbi, HMM based probability calculation and simple probability calculations.

## Problems, Assumptions, Simplifications, and Design Choices

The first problem that we faced was to decide on a data structure to use. We considered list of lists, dictionary of tuples etc and decided on dictionary of dictionaries given its ease of lookup. We had earlier been storing the numbers/counts in the dictionary and with having to convert the same to probabilities at multiple places we decided to save the probabilities directly. 


