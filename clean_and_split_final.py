from preprocess_final import find_instances, extract_from_source, clean_data
import os
import glob
import pickle
import spacy
from collections import defaultdict
import torch

"""
Main preprocessing module. Uses 'preprocess_final' methods.
The preprocessing consists of the main module 'clean_and_split_final.py' and the submodule 'preprocess_final.py'. Calling 'clean_and_split_final.py' will create the directories 'cleaned_red', 'cleaned_red_sentences', 'red_target', 'final_red', 'red_target_NN'.

cleaned_red: Contains the cleaned red corpus
cleaned_red_sentences: Contains the cleaned red corpus, where each line contains a sentences
red_target: Contains the human readable target version of the red corpus
red_target_NN: Contains the targets for neural network training
"""


# Configuration
nlp = spacy.load("en")
path = "D:/Korpora/red/data"
annotations = path + "/annotation"
source = path + "/source"
splits = "D:/Korpora/red/docs/splits.txt"

cleaned = "D:/Korpora/cleaned_red/"
sents = "D:/Korpora/cleaned_red_sentences/"

def collect_words(filename):
    """

    :param filename: String
    :return:
    """
    with open(filename, "r", encoding = "utf-8") as f:
        data = f.read()
    data = data.lower()
    words = data.split()
    doc = nlp(data)
    sentences = list(doc.sents)
    #print(sentences)
    fil = filename.split("cleaned_red/")[1]
    filename = fil.split("mpdf.txt")[0]
    with open("D:/Korpora/cleaned_red_sentences/" + filename+"_sentence_splits.txt", "w+", encoding="utf-8") as f:
        for sent in sentences:
            #print(sent)
            f.write(sent.text + "\n")
    return words

def build_dictionary(words):
    word_to_id = {}
    word_to_id["Argument"] = 1
    word_to_id["Related_to"] = 2
    word_to_id["Set"] = 3
    word_to_id["Member"] = 4
    word_to_id["Whole"] = 5
    word_to_id["Part"] = 6
    counter = 7
    for sent in words:
        for word in sent:
            if word not in word_to_id:
                word_to_id[word] = counter
                counter += 1
    return word_to_id

def create_tgs(filename,dictionary):
    """
    Creates the .tgs files in the BIO format.
    :param filename:
    :return:
    """
    # muss man nach lines machen!
    tags  = ["member", "argument", "related_to", "set", "whole", "part"]
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line != "\n":
                data.append(line)
    # jezt umstellen
    final = []
    final2 = []
    to_remove = []
    tag_dict = {"0":7, "argument":1, "related_to": 2, "set":3, "member":4, "whole":5, "part":6}
    for line in data:
        x = line.split() # creates normal target
        for i in range(len(x)):
            #print(len(x), i)
            if x[i] not in tags and x[i] in dictionary:
                #print(x[i])
                x[i] = "0"
            if x[i] not in dictionary and x[i] != "0":
                to_remove.append(x[i])
        for item in to_remove:
            if item in x:
                x.remove(item)

        y = line.split() # creates target in BIO schema
        for i in range(len(y)):
            #print(len(x), i)
            if y[i] not in tags and y[i] in dictionary:
                #print(x[i])
                y[i] = "0"
            if y[i] not in dictionary and y[i] != "0":
                to_remove.append(y[i])
            if y[i] in tags and y[i] in tag_dict:
                y[i] = str(tag_dict[y[i]])
        for item in to_remove:
            if item in y:
                y.remove(item)

        line = " ".join(x)
        final.append(line)

        line2 = " ".join(y)
        final2.append(line2)
    #print(final)
    fil = filename.split("cleaned_red_sentences/")[1]
    filename = fil.split("mpdf.txt")[0]
    #print(filename)
    with open("D:/Korpora/red_target/" + filename + "tgs", "w+", encoding="utf-8") as f:
        for line in final:
            f.write(line+"\n")

    with open("D:/Korpora/red_target_NN/" + filename + "tgs", "w+", encoding="utf-8") as f:
        for line in final2:
            f.write(line+"\n")
    return final, final2


def convert_data(filename, dictionary):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line != "\n":
                data.append(line)
    final = []
    to_remove = []
    for line in data:
        x = line.split()
        for i in range(len(x)):
            #print(x[i])
            if x[i] in dictionary:
                x[i] = str(dictionary[x[i]])
            else:
                to_remove.append(x[i])
        for item in to_remove:
            if item in x:
                x.remove(item)
        line = " ".join(x)
        final.append(line)
    fil = filename.split("cleaned_red_sentences/")[1]
    filename = fil.split("._sentence_splits")[0]
    with open("D:/Korpora/final_red/" + filename + ".txt", "w+", encoding="utf-8") as f:
        for line in final:
            f.write(line+"\n")
    return final




if __name__ == "__main__":
    #"""
    subdirectories = ['deft', 'pilot', 'proxy']

    xml = []
    source = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if name.endswith(".xml"):
                xml.append(os.path.join(path, name).replace("\\", "/"))
            else:
                source.append(os.path.join(path, name).replace("\\", "/"))
    print(xml)
    print(source)

    for i in range(len(xml)):
        rel, ent = find_instances(xml[i])
        data = extract_from_source(rel, ent, source[i])
        data = clean_data(data, source[i])
        print(data)

    #"""
    # read in dev and test set
    with open(splits, "r", encoding="utf-8") as f:
        data = f.read()
    dev = data.split("test set:")[0].split("\n")
    test = data.split("test set:")[1].split("\n")
    dev.remove("development set:")
    dev.remove("")
    test.remove("")


    cleaned_files = [f for f in os.listdir(cleaned) if os.path.isfile(os.path.join(cleaned, f))]
    #print(cleaned_files)
    all_words = []
    for file in cleaned_files:
        all_words.append(collect_words(cleaned+file))
    dictionary = build_dictionary(all_words)

    with open("vocabulary_final_lower.pickle","wb") as f:
        pickle.dump(dictionary, f)
    print(len(dictionary))


    source = defaultdict(list)
    target = defaultdict(list)
    complete = defaultdict(list)
    sent_split_files = [f for f in os.listdir(sents) if os.path.isfile(os.path.join(sents, f))]
    for file in sent_split_files:
        tgs, tgs2 = create_tgs(sents+file, dictionary)
    #for file in sent_split_files:
        conv = convert_data(sents+file, dictionary)
        for i,j in zip(conv, tgs2):
            a = [int(z) for z in i.split()]
            b = [int(z) for z in j.split()]
            #b = [int(z) for z in j.split()]
            complete[file].append(torch.as_tensor([a,b]))
            source[file].append(torch.as_tensor(a))
            target[file].append(torch.as_tensor(b))
    print(len(source), len(target))
    with open("source_final_lower.pickle","wb") as f:
        pickle.dump(source, f)
    with open("target_final_lower.pickle","wb") as f:
        pickle.dump(target, f)
    with open("complete_final:lower.pickle","wb") as f:
        pickle.dump(complete, f)
    print(len(dictionary))

