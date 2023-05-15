import pickle
import torch
"""
if 1 in set(x):
    one_counts += 1
    new_target[key].append(item[i])
if 2 in set(x):
    two_counts += 1
    new_target[key].append(item[i])
if 3 in set(x):
    three_counts += 1
    new_target[key].append(item[i])
if 4 in set(x):
    four_counts += 1
    new_target[key].append(item[i])
if 5 in set(x):
    five_counts += 1
    new_target[key].append(item[i])
if 6 in set(x):
    six_counts += 1
    new_target[key].append(item[i])
if 7 in set(x):
    seven_counts += 1
    new_target[key].append(item[i])
#else:
    #print(item[i])
"""
zero_counts = 0

path = "C:/Users/Meier/PycharmProjects/gan-for-event-detection/"
with open(path + "source_final_lower.pickle", "rb") as f: # "source_final_lower.pickle" # source2
    source = pickle.load(f)
with open(path + "target_final_lower.pickle", "rb") as f:
    target = pickle.load(f)
with open(path + "vocabulary_final_lower.pickle", "rb") as f:
    vocabulary = pickle.load(f)
print(len(target))
total = []


new_target = {}
index = {}
new_source = {}

for key, item in target.items():
    new_target[key] = []
    index[key] = []
    for i in range(len(item)):
    #for sub in item:
        #total.append(sub)
        x = set(item[i].tolist())
        if len(x) == 1 and 0 in x:
            zero_counts += 1
            if zero_counts%10 == 0:
                new_target[key].append(item[i])
                index[key].append(i)
        if len(x) > 1:
            new_target[key].append(item[i])
            index[key].append(i)

new_data = []
for key, item in source.items():
    for i in range(len(item)):
        if i in index[key]:
            if key in new_source:
                new_source[key].append(item[i])
                new_data.append(item[i].tolist())
            else:
                new_source[key] = []
                new_source[key].append(item[i])
                new_data.append(item[i].tolist())
word_to_id = {}
word_to_id["Argument"] = 1
word_to_id["Related_to"] = 2
word_to_id["Set"] = 3
word_to_id["Member"] = 4
word_to_id["Whole"] = 5
word_to_id["Part"] = 6
counter = 7
print(len(vocabulary))
vocabulary = {v: k for k, v in vocabulary.items()}
for sent in new_data:
    for word in sent:
        w = vocabulary[word]
        if w not in word_to_id:
            word_to_id[vocabulary[word]] = counter
            counter += 1
print(counter)
print("len word 2 id: ", len(word_to_id))
#print("new source: ", new_source)
#vocabulary = {v: k for k, v in vocabulary.items()}
#word_to_id = {v: k for k, v in word_to_id.items()}
# reindexing
new_source_translated = {}
for key, value in new_source.items():
    new_source_translated[key] = []
    for val in value:
        y = []
        for w in val:
            x = int(w)
            y.append(vocabulary[x])
        new_source_translated[key].append(y)

print(new_source_translated)

new_source_f = {}
for key, value in new_source_translated.items():
    new_source_f[key] = []
    for val in value:
        y = []
        for w in val:
            y.append(word_to_id[w])
        y = torch.LongTensor(y)
        new_source_f[key].append(y)

"""
with open("new_target.pickle", "wb") as f:
    pickle.dump(new_target, f)

with open("new_source.pickle","wb") as f:
    pickle.dump(new_source, f)
"""

def create_counts(data):
    total = 0
    zero_counts = 0
    one_counts = 0
    two_counts = 0
    three_counts = 0
    four_counts = 0
    five_counts = 0
    six_counts = 0
    seven_counts = 0
    eight_counts = 0
    nine_counts = 0
    for key, item in data.items():
        new_target[key] = []
        index[key] = []
        for i in range(len(item)):
            # for sub in item:
            # total.append(sub)
            x = set(item[i].tolist())
            if 0 in x and len(x) == 1:
                zero_counts += 1
            if 1 in set(x):
                one_counts += 1
                new_target[key].append(item[i])
            if 2 in set(x):
                two_counts += 1
                new_target[key].append(item[i])
            if 3 in set(x):
                three_counts += 1
                new_target[key].append(item[i])
            if 4 in set(x):
                four_counts += 1
                new_target[key].append(item[i])
            if 5 in set(x):
                five_counts += 1
                new_target[key].append(item[i])
            if 6 in set(x):
                six_counts += 1
                new_target[key].append(item[i])
            if 7 in set(x):
                seven_counts += 1
                new_target[key].append(item[i])
            if 8 in set(x):
                eight_counts += 1
                new_target[key].append(item[i])
            total += 1
    print("zero: ", zero_counts)
    print("one: ", one_counts)
    print("two: ", two_counts)
    print("three: ", three_counts)
    print("four: ", four_counts)
    print("five: ", five_counts)
    print("six: ", six_counts)
    print("seven: ", seven_counts)
    print("eight: ", eight_counts)
    print("total: ", total)


    # else:
    # print(item[i])



#print(target)


print(len(total))

#print(create_counts(new_target))
#"""
with open("new_target_10_shrink.pickle", "wb") as f:
    pickle.dump(new_target, f)

with open("new_source_10_shrink.pickle","wb") as f:
    pickle.dump(new_source_f, f)

with open("vocabulary_10_shrink.pickle","wb") as f:
    pickle.dump(word_to_id, f)
print(len(new_source_f), len(new_target))
#"""
#print(new_source)
word_to_id = {v: k for k, v in word_to_id.items()}
print(word_to_id)
for i,j in zip(new_source_f.values(), new_target.values()):
    for k,h in zip(i,j):
        print(len(k), len(h))
        if len(k) != len(h):
            print("poiadjlsa: ", len(k), len(h))
    #print(len(i),len(j))
