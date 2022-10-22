"""
Código para dividir propiamente los conjuntos de training y test utilizando los CSVs
generados con genPairsCSV.py

Mezcla los datos del 80% de usuarios, de pares con mismos usuarios y diferentes para
entrenamiento y el 20% de test, de usuarios que no aparecen en entrenamiento.
"""

import pandas as pd
import numpy as np
import sklearn.model_selection as skms

leftSameSub80 = pd.read_csv("./dataset/left-same-sub80.csv", sep=",", header=None)
leftSameTop80 = pd.read_csv("./dataset/left-same-top80.csv", sep=",", header=None)
leftDiffSub80 = pd.read_csv("./dataset/left-diff-sub80.csv", sep=",", header=None)
leftDiffTop80 = pd.read_csv("./dataset/left-diff-top80.csv", sep=",", header=None)

rightSameSub80 = pd.read_csv("./dataset/right-same-sub80.csv", sep=",", header=None)
rightSameTop80 = pd.read_csv("./dataset/right-same-top80.csv", sep=",", header=None)
rightDiffSub80 = pd.read_csv("./dataset/right-diff-sub80.csv", sep=",", header=None)
rightDiffTop80 = pd.read_csv("./dataset/right-diff-top80.csv", sep=",", header=None)


leftSameSub80 = leftSameSub80.to_numpy()
leftSameTop80 = leftSameTop80.to_numpy() 
leftDiffSub80 = leftDiffSub80.to_numpy() 
leftDiffTop80 = leftDiffTop80.to_numpy() 

np.random.shuffle(leftDiffSub80)
np.random.shuffle(leftDiffTop80)

rightSameSub80 = rightSameSub80.to_numpy() 
rightSameTop80 = rightSameTop80.to_numpy()
rightDiffSub80 = rightDiffSub80.to_numpy() 
rightDiffTop80 = rightDiffTop80.to_numpy() 

np.random.shuffle(rightDiffSub80)
np.random.shuffle(rightDiffTop80)

#training = leftSameSub80 + rightSameSub80[:780] + leftDiffSub80[:780] + rightDiffSub80[:780]
#testing = leftSameTop80 + rightSameTop80 + leftDiffTop80[:200] + rightDiffTop80[:200]

training = np.vstack((leftSameSub80, rightSameSub80[:780]))
training = np.vstack((training, leftDiffSub80[:780]))
training = np.vstack((training, rightDiffSub80[:780]))

testing = np.vstack((leftSameTop80, rightSameTop80[:200]))
testing = np.vstack((testing, leftDiffTop80[:200]))
testing = np.vstack((testing, rightDiffTop80[:200]))

trainFile = open('./dataset/train.csv', 'w')

for i in training:
    trainFile.write("{},{},{}\n".format(i[0], i[1], i[2]))
    
trainFile.close()

testFile = open('./dataset/test.csv', 'w')
for i in testing:
    testFile.write("{},{},{}\n".format(i[0], i[1], i[2]))
    
testFile.close()