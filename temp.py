f = os.listdir(path_input)
converted_i = f[i]+'/converted'
files = os.listdir(coverted_i)

for i in range(len(files)):
    files[i] = converted_i + '/' + files[i]


for file in files:
    if os.path.splitext(os.path.splitext(file)[0])[-1] in endings:
        print(file)
        with open(file,'r') as File:
            try:
                contents.append(File.readlines())
            except:
                continue

label_i = f[i]+'/groundtruth'
files = os.listdir(label_i)
for i in range(len(files)):
    files[i] = label_i + '/' + files[i]

for file in files:
    if os.path.splitext(file)[-1] == ".json":
        with open(file,'r') as jsonFile:
            label = json.load(jsonFile)
            labels.append(label['guessed']['what'])

labels = np.unique(labels)

