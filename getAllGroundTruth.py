import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, default = None)
args = parser.parse_args()

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


