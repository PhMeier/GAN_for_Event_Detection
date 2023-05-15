import pickle
import torch
import numpy as np
import torch.nn as nn
from model2 import Generator, Discriminator
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn import preprocessing

import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score 
import torch.nn.functional as F

"""
Evaluation procedure for the trained models.
"""


class EventData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y


    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


BATCH_SIZE = 16
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 256 #182 # 512 #128 # 512
NUM_LAYERS = 2  # 2
EPOCHS = 100
LAMBDA = 0.0001  # 0.0001
CLASSES = 7

criterion_gen_coop = torch.nn.CrossEntropyLoss()



def pick_subsets(data, wanted_files):
    res = {}
    keys = []
    for key, value in data.items():
        filename = key.split(".txt_sentence_splits.txt")[0]
        filename = filename.split("._sentence_splits.txt")[0]
        if filename in wanted_files:
            res[filename] = value
            keys.append(key)
    for k in keys:
        del data[k]
    return res, data


def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0 ,1), y, lengths



def translate(instance, vocabulary):
    inp = instance.tolist()
    res = ""
    for item in inp:
        res += vocabulary[item] + " "
    return res


def eval(gen_1, disc_1, test_data, vocabulary):
    all_predictions = []
    all_true = []
    gen_1.eval()
    disc_1.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_prec = 0.0
    sum_rec = 0.0
    sum_f1 = 0.0
    #sum_rmse = 0.0
    counter = 0
    vocabulary = {v: k for k, v in vocabulary.items()}
    for (batch, (inp, targ)) in enumerate(test_data):
        #targ = torch.as_tensor(targ)
        counter += 1
        feature_rep, hidden  = gen_1(inp) #predicition
        #print(feature_rep)
        prediction= disc_1(feature_rep)
        #print(prediction)
        loss = F.cross_entropy(prediction, targ)
        pred = torch.max(prediction, 1)[1]
        print(pred)#, targ, inp)#, translate(inp, vocabulary), inp)
        correct += (pred == targ).float().sum()
        total += targ.shape[0]
        sum_loss += loss.item()*targ.shape[0]
        prec = precision_score(pred.tolist(), targ.tolist(), average = "macro")
        rec = recall_score(pred.tolist(), targ.tolist(), average = "macro")
        sum_prec += prec
        sum_rec += rec
        all_predictions.append(pred.tolist())
        all_true.append(targ.tolist())
    f1 = 2 * (sum_prec / counter * sum_rec / counter) / (sum_prec / counter + sum_rec / counter)
    return sum_loss / total, "accuracy: ", correct / total, "precision: ", sum_prec / counter, "Recall: ", sum_rec / counter, "F1-Score: ", f1



if __name__ == "__main__":

    dev_set = ["17a2dc40635ec239e9e16d10b6dd45e8", "96bf72399b104346f3e79022e0c08e5a",
               "NYT_ENG_20130424.0047", "NYT_ENG_20130613.0153", "NYT_ENG_20131029.0091",
               "PROXY_AFP_ENG_20020128_0449", "PROXY_AFP_ENG_20020406_0538",
               "57026b7bcb8f855de3e26d572db35285", "soc.culture.china_20050203.0639"]

    test_set = ["NYT_ENG_20130426.0143", "dd0b65f632f64369c530f9bbb4b024b4", "c06e8bbdf69f73a69cd3d5dbb4d06a21",
                "NYT_ENG_20130709.0087", "NYT_ENG_20130712.0047",  "NYT_ENG_20131003.0269",
                "PROXY_AFP_ENG_20020210_0074", "PROXY_AFP_ENG_20020408_0348", "d21dc2cb6e6435da7f9d9b0e5759e214",
                "soc.culture.iraq_20050211.0445"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = "C:/Users/Meier/PycharmProjects/gan-for-event-detection/"
    with open(path + "source_final_lower.pickle", "rb") as f: #source2 actuaL:source_final_lower.pickle
        source = pickle.load(f)
    with open(path + "target_final_lower.pickle", "rb") as f: #target_NN2, acutal: target_final_lower.pickle
        target = pickle.load(f)
    with open(path + "vocabulary_final_lower.pickle", "rb") as f: # vocabulary_final_lower
        vocabulary = pickle.load(f)



    #print(source.keys())
    development_set_X, source = pick_subsets(source, dev_set)
    development_set_Y, target = pick_subsets(target, dev_set)
    test_X, source = pick_subsets(source, test_set)
    test_Y, target = pick_subsets(target, test_set)
    #print(len(development_set_X))
    #print(len(source))
    X_ = list(source.values())
    Y_ = list(target.values())

    X_TEST = list(test_X.values())
    Y_DEV = list(test_Y.values())
    print(len(X_TEST))

    X_test= [item for sublist in X_TEST for item in sublist if len(item) > 1]
    y_test = [item for sublist in Y_DEV for item in sublist if len(item) > 1]

    X = [item for sublist in X_ for item in sublist if len(item) > 1]
    Y = [item for sublist in Y_ for item in sublist if len(item) > 1]

    # randomize
    train_pl = list(zip(X,Y))
    X, Y = zip(*train_pl)
    test_pl = list(zip(X_TEST, Y_DEV))
    X_TEST, Y_DEV = zip(*test_pl)
    VOC_SIZE = len(vocabulary)
    test_data = EventData(X_test, y_test)

    # torch load my model

    gen_1 = Generator(35, HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_SIZE, VOC_SIZE) #24
    disc_1 = Discriminator(35, EMBEDDING_SIZE, 1, CLASSES, HIDDEN_SIZE) #24

    gen_1.load_state_dict(torch.load(path + "saved_models/gen_1_complete_stock_new_weight", map_location=torch.device('cpu'))) #best: gen_1_complete_stock_new_weight  #gen_1_complete_stock gen_1_complete_stock_no_weight
    disc_1.load_state_dict(torch.load(path + "saved_models/disc_1_complete_stock_new_weight", map_location=torch.device('cpu')))# best: disc_1_complete_stock_new_weight #disc_1_complete_stock disc_1_complete_stock_no_weight

    #gen_1.load_state_dict(torch.load(path + "saved_models/gen_1_complete_stock", map_location=torch.device('cpu'))) #best: gen_1_complete_stock_new_weight  #gen_1_complete_stock gen_1_complete_stock_no_weight
    #disc_1.load_state_dict(torch.load(path + "saved_models/disc_1_complete_stock", map_location=torch.device('cpu')))
    #print(vocabulary)
    print(eval(gen_1, disc_1, test_data, vocabulary))
    #vocabulary = {v: k for k, v in vocabulary.items()}
    #for (batch, (inp, targ)) in enumerate(test_data):
        #print(targ, translate(inp, vocabulary), inp)






