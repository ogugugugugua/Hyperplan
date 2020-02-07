import os
import argparse
import json
import numpy as np
import csv
import codecs
import time
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, default = None)
parser.add_argument('--clusters',type = int, default = 20)
args = parser.parse_args()

labels = [] #table that stores all labels for corresponding converted inputs
contents = [] #table that store all converted inputs
encodingError = []
path = args.path
#endings: all possible file suffix
endings = [".doc",".DOC",".pdf",".rtf",".odt",".docx",".txt",".PDF",".DOCX"]

#all directories that contains converted input && ground labels
dossiers = os.listdir(path)
for i in range(len(dossiers)):
    dossiers[i] = path + dossiers[i]


for i in range(len(dossiers)):
    #In each directory which contains converted input && ground labels, we get them both out:

    #--------------------------------converted part begins-----------------------------------#
    if dossiers[i][-1] =='/':
        converted = dossiers[i] + 'converted/'
    else:
        converted = dossiers[i] + '/converted/'
    # print("converted: ",converted)

    #get all possible files in the 'converted' directory in this current father directory
    files = os.listdir(converted)
    if files == []:
        raise Exception

    #complet the full name
    for j in range(len(files)):
        files[j] = converted + files[j]

    mark = False

    #get all converted file contents out
    for file in files:
        if os.path.splitext(os.path.splitext(file)[0])[-1] in endings:
            mark = True
            with open(file, 'r') as File:
                try:
                    contents.append(File.readlines())
                except:
                    encodingError.append(i)
                    raise Exception
    if mark==False:
        print("404: ",i,"th directory: ",dossiers[i])
    # ---------------------------------converted part ends----------------------------------------#

    # ---------------------------------groundtruth part begins------------------------------------#
    if i not in encodingError:
        if dossiers[i][-1] == '/':
            label = dossiers[i] + 'groundtruth/'
        else:
            label = dossiers[i] + '/groundtruth/'
        # print("label: ",label)

        # get all possible files in the 'label' directory in this current father directory
        fileLabels = os.listdir(label)

        # complet the full name
        for i in range(len(fileLabels)):
            fileLabels[i] = label + fileLabels[i]

        # get all label file contents out
        for file in fileLabels:
            if os.path.splitext(file)[-1] == ".json":
                with open(file, 'r') as jsonFile:
                    jsonFileContent = json.load(jsonFile)
                    labels.append(jsonFileContent['guessed']['what'])

    # ---------------------------------groundtruth part ends----------------------------------------#


# Cast the contents type from list into table
for j, _ in enumerate(contents):
    temp = []
    for i, li in enumerate(contents[j]):
        if i == 0:
            temp = contents[j][i]
        temp = temp + contents[j][i]
    contents[j] = temp


#Store data in a csv file
with open("test.csv",'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(contents)

#Read the stored data in csv file
with open("test.csv",'r') as csvfile:
    reader = csv.reader(csvfile)
    # for line in reader:
        # print(line)




vectorizer =  CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(contents))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()

# print("Features length: ",str(len(word)))
resultName = "Tfidf_Result.txt"
result = codecs.open(resultName, 'w', 'utf-8')

#write feature vector text value
for j in range(len(word)):
    result.write(word[j] + ' ')
result.write('\r\n\r\n')

#write weights for all tf-idf text, the first for goes through all the text file, the second for goes through all words in that file
for i in range(len(weight)):
    for j in range(len(word)):
        result.write(str(weight[i][j]) + ' ')
    result.write('\r\n\r\n')

#close file stream
result.close()

print("Start Kmeans:")
clusters = args.clusters
clf = KMeans(n_clusters = clusters)
s = clf.fit(weight)

#print cluster centers
print("centers for the ",clusters," clusters: \n",clf.cluster_centers_)

#print clusters for each text file
i = 1
while i<=len(clf.labels_):
    print("\nThe ",i,"th file --> ",clf.labels_[i-1]," cluster")
    print(labels[i-1])
    i = i+1

print("inertia: ",clf.inertia_)


