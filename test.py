# encoding=utf8

import math
import pandas as pd
import numpy as np
import random
import time
import os

import collections


def stopwords(filePath):
    pass


def rm_stopwords(self, file_path, word_dict):
    # read stop word dict and save in stop_dict
    stop_dict = {}
    with open(word_dict) as d:
        for word in d:
            stop_dict[word.strip("\n")] = 1

    # # remove tmp file if exists
    # if os.path.exists(file_path + ".tmp"):
    #     os.remove(file_path + ".tmp")
    #
    # print
    # "now remove stop words in %s." % file_path
    # # read source file and rm stop word for each line.
    # with nested(open(file_path), open(file_path + ".tmp", "a+"))  as (f1, f2):
    #     for line in f1:
    #         tmp_list = []  # save words not in stop dict
    #         words = line.split()
    #         for word in words[1:]:
    #             if word not in stop_dict:
    #                 tmp_list.append(word)
    #         words_without_stop = " ".join(tmp_list)
    #         f2.write(words[0] + " " + words_without_stop + "\n")
    #
    # # overwrite origin file with file been removed stop words
    # shutil.move(file_path + ".tmp", file_path)
    # print
    # "stop words in %s has been removed." % file_path
    # return


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


def findCategory(filePath):
    fileContent = ""
    with open(filePath) as file:
        for line in file:
            fileContent += line
    # print("fileContent:\n", fileContent)
    frequency = dict(collections.Counter(fileContent.split()))
    keys = list(frequency)
    values = list(frequency.values())
    softmaxValues = softmax(np.array(values))
    for i, k in enumerate(frequency):
        frequency[k] = softmaxValues[i]
    return sorted(frequency.items(), key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    # # x = np.array([[2, 8, 7, 6], [64, 78, 13, 4]])
    # # print(x)
    # # print(softmax(x))
    #
    # text = "I'm a hand some boy! I'm a hand some boy!I'm a ! \n I'm a hand some boy!"
    # frequency = dict(collections.Counter(text.split()))
    # keys = list(frequency)
    # values = list(frequency.values())
    # softmaxValues = softmax(np.array(values))
    #
    # print("values ", values, "\nsoftmaxValues ", softmaxValues)
    #
    # # print("keys ",keys)
    # # print("frequency ",frequency)
    # # print("frequency[keys[0]] ",frequency[keys[0]],"\n\n\n")
    #
    # for i, k in enumerate(frequency):
    #     frequency[k] = softmaxValues[i]
    #
    # print(sorted(frequency.items(), key=lambda item: item[1], reverse=True))
    # print(sorted(frequency.items(), key=lambda item: item[1]))
    #
    # # frequency = collections.defaultdict(int)
    # # text = "I'm a hand some boy!"
    # # for word in text.split():
    # #     frequency[word] += 1
    # # print(frequency)
    print(findCategory("texte.txt"))
