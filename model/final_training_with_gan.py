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
from sklearn.metrics import precision_score, recall_score, multilabel_confusion_matrix, precision_recall_fscore_support
import torch.nn.functional as F


class EventData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        #self.length = [np.sum(1 - np.equal(x, 0)) for x in X]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        #x_len = self.length[index]
        return x, y

    def __len__(self):
        return len(self.data)


BATCH_SIZE = 16
EMBEDDING_SIZE = 300
HIDDEN_SIZE =  256  #128 # 512
LEARNING_RATE = 1.0  # 0.001
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


def loss_self_reg_learning(output1, output2):
    A = (torch.matmul(output1.squeeze(), output2.squeeze().T )**2)
    res = torch.norm(A, p="fro")
    return res**0.5
    # res = A**0.5
    # return res
    # x = torch.abs(output1 - output2.T)**2

def validate(gen_1, disc_1, val_data):
    #all_predictions = []
    #all_true = []
    gen_1.eval()
    disc_1.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_prec = 0.0
    sum_rec = 0.0
    #sum_rmse = 0.0
    counter = 0
    for (batch, (inp, targ)) in enumerate(val_data):
        counter += 1
        inp = inp.to(device)
        targ = targ.to(device)
        feature_rep, hidden  = gen_1(inp) #predicition
        #print(feature_rep)
        prediction= disc_1(feature_rep)
        #print(prediction)
        loss = F.cross_entropy(prediction, targ)
        pred = torch.max(prediction, 1)[1]
        #print(pred, targ)
        pred = pred.to(device)
        correct += (pred == targ).float().sum()
        total += targ.shape[0]
        sum_loss += loss.item()*targ.shape[0]
        prec = precision_score(pred.tolist(), targ.tolist(), average = "macro")
        rec = recall_score(pred.tolist(), targ.tolist(), average = "macro")
        sum_prec += prec
        sum_rec += rec
    return sum_loss / total, "accuracy: ", correct / total, "precision: ", sum_prec / counter, "Recall: ", sum_rec / counter



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
    path = "/home/students/meier/EventProc/gan-for-event-detection/"
    with open(path + "source_final_lower.pickle", "rb") as f: #source2 actuaL: new_source # _23
        source = pickle.load(f)
    with open(path + "target_final_lower.pickle", "rb") as f: #target_NN2, acutal:new_target # _23
        target = pickle.load(f)
    with open(path + "vocabulary_final_lower.pickle", "rb") as f: # _23
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

    X_dev = list(development_set_X.values())
    Y_dev = list(development_set_Y.values())


    development_set_X = [item for sublist in X_dev for item in sublist if len(item) > 1]
    development_set_Y = [item for sublist in Y_dev for item in sublist if len(item) > 1]

    X = [item for sublist in X_ for item in sublist if len(item) > 1]
    Y = [item for sublist in Y_ for item in sublist if len(item) > 1]

    #print(X)
    #print(Y)
    X = X  
    Y = Y  

    # print(X)

    # X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X, X_test, Y, y_test = train_test_split(X,Y, train_size = 1.0, random_state=42)

    X_test = development_set_X
    y_test = development_set_Y


    # randomize
    train_pl = list(zip(X,Y))
    X, Y = zip(*train_pl)
    val_pl = list(zip(X_dev, Y_dev))
    X_dev, Y_dev = zip(*val_pl)


    dataset = EventData(X, Y)
    val_data = EventData(X_test, y_test)
    print("X Len: ", len(X))

    # print(dataset)
    # for (batch, (inp, targ, inp_len)) in enumerate(dataset):
    #    print(batch, (inp, targ, inp_len))

    le = preprocessing.LabelEncoder()

    VOC_SIZE = len(vocabulary)
    x_train = list(source.values())[0][0]
    # x_train = x_train[:930]
    x_train = torch.LongTensor(x_train)
    y_target = list(target.values())[0][0]
    y_target = y_target[:930]
    #print(y_target)

    params = {"batch_size": 1,
              "shuffle": False,
              "num_workers": 1}


    # cooperative network
    gen_1 = Generator(len(x_train), HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_SIZE, VOC_SIZE)
    disc_1 = Discriminator(len(x_train), EMBEDDING_SIZE, 1, CLASSES, HIDDEN_SIZE)

    gen_1.to(device)
    disc_1.to(device)


    # GAN
    gen_2 = Generator(len(x_train), HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_SIZE, VOC_SIZE) # the troublemaker
    disc_2 = Discriminator(len(x_train), EMBEDDING_SIZE, 1, CLASSES, HIDDEN_SIZE) # unlucky prof


    gen_1_opt = optim.Adam(gen_1.parameters(), 0.001) #optim.Adadelta(gen_1.parameters(), lr=LEARNING_RATE) # 0.001
    disc_1_opt = optim.Adam(disc_1.parameters(), 0.001) #optim.Adadelta(disc_1.parameters(), lr=LEARNING_RATE)

    # Optimizer for GAN:
    gen_2_opt = optim.Adam(gen_2.parameters(), 0.001)#optim.Adadelta(gen_2.parameters(), lr=LEARNING_RATE)
    disc_2_opt = optim.Adam(disc_2.parameters(), 0.001)#optim.Adadelta(disc_2.parameters(), lr=LEARNING_RATE)

    gen_2.to(device)
    disc_2.to(device)

    total_inst = 1 #5612 #5314
    weight = torch.tensor([total_inst/3689, total_inst/303, total_inst/299, total_inst/638, total_inst/834, total_inst/369, total_inst/385])
    weight = weight.to(device)

    class_counts = [3757, 299, 296, 637, 826, 366, 381]
    channel_1_crit = torch.nn.CrossEntropyLoss(weight)
    channel_2_crit = torch.nn.CrossEntropyLoss(weight)

    # dataset stuff
    torch.autograd.set_detect_anomaly(True)



    for epoch in range(EPOCHS):

        total_loss_g = 0
        total_loss_d = 0
        total_loss_g2 = 0
        total_loss_d2 = 0
        gen_1.train()
        gen_2.train()
        disc_1.train()
        disc_2.train()
        total_loss = 0

        total_loss_g = 0
        for (batch, (inp, targ)) in enumerate(dataset):

            loss = 0

            # zero grad for optimizers
            gen_1_opt.zero_grad()
            disc_1_opt.zero_grad()
            gen_2_opt.zero_grad()
            disc_2_opt.zero_grad()


            inp = torch.LongTensor(inp)
            inp = inp.to(device)
            targ = targ.to(device)

            out, hidden = gen_1(inp.to(device))
            fake, hidden2 = gen_2(inp.to(device))  # changes real to fake


            l_diff = loss_self_reg_learning(out, fake)  # memory supressor, both should be as dissimilar as possible

            # discriminator channel 2 on true data
            res_true = disc_2(out)  # inp?
            disc_real_error = channel_2_crit(res_true, targ)
            disc_real_error.backward(retain_graph=True)

            # discriminator channel 2 on fake data
            res_fake = disc_2(fake.detach())
            disc_fake_error = channel_2_crit(res_fake, targ)
            disc_fake_error.backward(retain_graph=True)
            disc_2_opt.step()

            total_loss_d2 += disc_real_error + disc_fake_error

            final_res = disc_1(out)  


            # update G
            res_fake_D = disc_2(fake)


            loss += channel_1_crit(final_res, targ) + (channel_1_crit(final_res, targ) + LAMBDA * l_diff) + channel_2_crit(res_fake_D, targ)
            loss.backward()


            # optimizer steps --> update the model parameters
            gen_1_opt.step()
            disc_1_opt.step()

            #gen_2_opt.step()

            total_loss_g += channel_1_crit(final_res, targ)
            total_loss_d += channel_1_crit(final_res, targ) + LAMBDA * l_diff
            #total_loss_d2 += disc_real_error
            total_loss_g2 += channel_2_crit(res_fake_D, targ)
            total_loss += loss


        print("LOSS: ", loss/len(X), epoch)
        print("total_loss_g: ", total_loss_g / len(X))
        print("total_loss_d: ", total_loss_d / len(X))
        print("total_loss_g2: ", total_loss_g2 / len(X))
        print("total_loss_d2: ", total_loss_d2 / len(X))
        if epoch % 10 == 0 or epoch == 1:
            print(validate(gen_1, disc_1, val_data))

    torch.save(gen_1.state_dict(), path+"gen_1_complete_stock_512")
    torch.save(disc_1.state_dict(), path+"disc_1_complete_stock_512")
    torch.save(gen_2.state_dict(), path+"gen_2_complete_stock_512")
    torch.save(disc_2.state_dict(), path+"disc2_complete_stock_512")










