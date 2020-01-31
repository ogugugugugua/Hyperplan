import os
import argparse
import json
import numpy as np
import csv
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, default = None)
parser.add_argument('--clusters',type = int, default = 20)
args = parser.parse_args()

truths = []
path = args.path

#Print all ground truth:
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if os.path.splitext(name)[-1]==".json":
            pathComplet = os.path.join(root,name)
            #print(pathComplet)
            with open(pathComplet,'r') as jsonFile:
                content = json.load(jsonFile)
                print(content['guessed']['what'])
                truths.append(content['guessed']['what'])

#truths: all the unique labels
truths = np.unique(truths)
print(np.size(truths))

#endings: all possible file suffix
endings = [".doc",".DOC",".pdf",".rtf",".odt",".docx"] 

#textPaths: paths of all the file to be trained
textPaths = []

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if os.path.splitext(os.path.splitext(name)[0])[-1] in endings:
            pathComplet = os.path.join(root,name)
            textPaths.append(pathComplet)
print(textPaths)

#in the following part we are gonna get all the content of those files
#remember to use try/except because there are some character that are not in UTF-8 encoding
contents = []
for File in textPaths:
    with open(File, 'r') as f:
        try:
            contents.append(f.readlines())
            #print(File)
        except:
            continue

#Cast the contents list into contents table
for j,_ in enumerate(contents):
    temp = []
    for i,li in enumerate(contents[j]):
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
    for line in reader:
        print(line)


vectorizer =  CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(contents))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()

print("Features length: ",str(len(word)))
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
print("centers for the ",clusters," clusters: ",clf.cluster_centers_)

#print clusters for each text file
i = 1
while i<=len(clf.labels_):
    print("The ",i,"th file belongs to ",clf.labels_[i-1]," cluster")
    i = i+1

print("inertia: ",clf.inertia_)


