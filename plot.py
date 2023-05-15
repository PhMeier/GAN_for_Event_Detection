"""
This module creates the validation plots.
"""


import matplotlib.pyplot as plt
import numpy as np


def get_validation(data, searched_string):
    res = []
    for sub in data:
        if searched_string in sub:
            res.append(float(sub.split(searched_string + ": ")[1]))
            # print(sub)
    return res


if __name__ == "__main__":


    epochs = [i for i in range(0, 100) if i % 10 == 0]

    epochs = np.array(epochs)
    #plt.plot(epochs, x, "r", label="Baseline")
    acc = [0.9397, 0.9397, 0.9397, 0.9397, 0.9397, 0.9397, 0.9397, 0.9397, 0.9397, 0.9397] #best [0.9426, 0.9525, 0.9527, 0.9567, 0.9567, 0.9567, 0.9567, 0.9567, 0.9567, 0.9567]#[0.9408, 0.9408, 0.9513, 0.9513, 0.9513, 0.9513, 0.9513, 0.9513, 0.9513, 0.9513] #[0.9395, 0.9395, 0.9395 ,0.9395, 0.9395, 0.9395, 0.9395, 0.9395, 0.9395, 0.9395]#[0.9837, 1.0, 1.0, 0.9998, 0.9991, 0.9978, 0.9939, 0.9909, 0.9869, 0.9857] #[0.9646, 0.9826, 0.9994, 0.9998, 1.0, 1.0, 1.0, 0.9998]
    prec = [0.808, 0.808, 0.808, 0.808, 0.808, 0.808, 0.808, 0.808, 0.808, 0.808]#best [0.82, 0.85, 0.851, 0.866, 0.866, 0.866, 0.866, 0.866, 0.866, 0.866]# [0.8115, 0.8115, 0.8447, 0.8447, 0.8447, 0.8447, 0.8447, 0.8447, 0.8447, 0.8447] #[0.805, 0.805, 0.805, 0.805, 0.805, 0.805, 0.805, 0.805, 0.805, 0.805] #[0.942, 1.0, 1.0, 0.998, 0.994, 0.988, 0.972, 0.964, 0.959, 0.943]#[0.892, 0.934, 0.998, 0.999, 1.0, 1.0, 1.0, 0.998]
    rec = [0.788, 0.788, 0.788, 0.788, 0.788, 0.788, 0.788, 0.788, 0.788, 0.788] #best [0.80, 0.836, 0.837, 0.854, 0.854, 0.854, 0.854, 0.854, 0.854, 0.854]#[0.793, 0.793, 0.828, 0.828, 0.828, 0.828, 0.828, 0.828, 0.828, 0.828] #[0.785, 0.785, 0.785, 0.785, 0.785, 0.785, 0.785, 0.785, 0.785, 0.785]#[0.938, 1.0, 1.0, 0.998, 0.995, 0.990, 0.97, 0.967, 0.963, 0.948] #[0.881, 0.935, 0.998, 0.998, 1.0, 1.0, 1.0, 0.988]

    plt.plot(epochs, acc, marker='o', color='skyblue', linewidth=1, label = "accuracy")
    plt.plot(epochs, prec, marker='o', color='red', linewidth=1, label = "precision")
    plt.plot(epochs, rec, marker='o', color='green', linewidth=1, label = "recall")
    plt.legend()

    plt.xlabel("Epochs")
    plt.ylabel("Validation Scores")
    plt.legend(loc="middle right")
    plt.show()
    print(epochs)
    # """
