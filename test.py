# -*- coding: UTF-8 -*-

import math
from filecmp import cmp
import pandas as pd
import numpy as np
import random
import time
import os
import collections


def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x


def findCategory(filePath, stopWordsPath, categoriesPath):
    # get the text file content
    fileContent = ""
    with open(filePath) as file:
        for line in file:
            fileContent += line

    # count times of appearance of each word
    frequency = dict(collections.Counter(fileContent.split()))

    # get the list of each single word in the file
    keys = list(frequency)

    # get the times of appearance of each word
    values = list(frequency.values())

    # count the possibility of each word by using softmax
    softmaxValues = softmax(np.array(values))

    # translate to softmax probability
    for i, k in enumerate(frequency):
        frequency[k] = softmaxValues[i]

    # get meaningless vocabulary
    stopword = []
    with open(stopWordsPath) as file:
        for word in file:
            stopword.append(word[:-2])

    # 1st version for deleting meaningless words, which only works for part of the text for some unclear reasons
    # frequencyWithoutStopWords = frequency.copy()
    # for i, k in enumerate(frequency):
    #     if k in stopword:
    #         del frequencyWithoutStopWords[k]

    # 2nd version for deleting meaningless words, which works fine
    frequencyWithoutStopWords = []
    for i, k in enumerate(frequency):
        if k not in stopword:
            frequencyWithoutStopWords.append((k, frequency[k]))
    frequencyWithoutStopWords = dict(frequencyWithoutStopWords)

    # sort the words by its corresponding possibility
    frequencyWithoutStopWords = sorted(frequencyWithoutStopWords.items(), key=lambda x: x[1], reverse=True)

    # get the whole default categories
    categories = []
    with open(categoriesPath) as file:
        for category in file:
            categories.append(category[:-1])

    # determine category
    for i, (word, percentage) in enumerate(frequencyWithoutStopWords):
        if word in categories:
            # we got the category by finding the most possible word in the list of category
            return word
    return None


if __name__ == "__main__":
    print("category: ", findCategory("textFiles/texte.txt", "stopWordsFR.txt", "categories.txt"))
