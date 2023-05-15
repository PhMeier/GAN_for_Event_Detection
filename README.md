# GAN for Event Detection

Project Implementation of Self-regulation: Employing a Generative Adversarial Network to Improve Event Detection
in Pytorch for the seminar 'Event Processing'.

All trained models can be found under 'saved models'. The test Results and training logs can be found in the directory 'logs'.
Best performing model is 'gen_1_complete_stock_new_weight' and 'disc_1_complete_stock_new_weight'
The stagnating model is 'gen_1_complete_stock_no_weight_512' and disc_1_complete_stock_no_weight_512
Model using smaller vocabulary is 'gen_1_complete_stock' and 'disc_1_complete_stock'

In order to run the code, Python 3.5.3 or higher is required as well a installation of Pytorch and SciKit-learn.

# Data #

RED corpus, contains splits in test, validation and training.

* 3982 training instances
* 466 test instances
* 244 validation instances
* Vocabulary of 7033 words

# Preprocessing #

The preprocessing consists of the main module 'clean_and_split_final.py' and the submodule 'preprocess_final.py'. Calling 'clean_and_split_final.py' will create the directories 'cleaned_red', 'cleaned_red_sentences', 'red_target', 'final_red', 'red_target_NN'.

* cleaned_red: Contains the cleaned red corpus
* cleaned_red_sentences: Contains the cleaned red corpus, where each line contains a sentences
* red_target: Contains the human readable target version of the red corpus
* red_target_NN: Contains the targets for neural network training

# Training time # 

The training time was roughly 210 minutes(3,5h) to 240 minutes (4h).

# Results #

| Model  | Accuracy | Prevision | Recall | F1
| -------------------|------- |------- |------- |------- |
| 256 units, 2 layers, weighted Cross Entropy Loss | 0.95 | 0.84 | 0.82 | 0.83 |
| 256 units, 2 layers, no weighted Cross Entropy Loss | 0.93 | 0.77 | 0.75 | 0.76 |

