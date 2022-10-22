import pandas as pd
import numpy as np
import sklearn.model_selection as skms

leftSame = pd.read_csv("./dataset/left-same.csv", sep=",", header=None)
leftDiff = pd.read_csv("./dataset/left-diff.csv", sep=",", header=None)
rightSame = pd.read_csv("./dataset/right-same.csv", sep=",", header=None)
rightDiff = pd.read_csv("./dataset/right-diff.csv", sep=",", header=None)

minSize = len(leftSame)

leftSame = leftSame.to_numpy()
leftDiff = leftDiff.to_numpy()
rightSame = rightSame.to_numpy()
rightDiff = rightDiff.to_numpy()

rightSame = rightSame[:minSize]
np.random.shuffle(leftDiff)
leftDiff = leftDiff[:minSize]
np.random.shuffle(rightDiff)
rightDiff = rightDiff[:minSize]

fullData = np.vstack((leftSame, rightSame))
fullData = np.vstack((fullData, leftDiff))
fullData = np.vstack((fullData, rightDiff))

fullLabels = fullData[:,2]
fullData = fullData[:,:2]

x_train, x_test, y_train, y_test = skms.train_test_split(fullData, 
                                                       fullLabels, 
                                                       stratify=fullLabels,
                                                       test_size=0.2,
                                                       random_state=16)


trainFile = open('./dataset/train.csv', 'w')

for i,j in zip(x_train, y_train):
    trainFile.write("{},{},{}\n".format(i[0], i[1], j))
    
trainFile.close()

testFile = open('./dataset/test.csv', 'w')
for i,j in zip(x_test, y_test):
    testFile.write("{},{},{}\n".format(i[0], i[1], j))
    
testFile.close()