"""
Código para generar los pares de imágenes.

Se dividen los datos en imágenes de ojos izquierdos y derechos, del mismo
usuario y de diferentes usuarios evitando siempre repetir.

Se dividen en 20/80 para que la red no observe un 20% de usuarios a la hora
de probar, se entrena con un 80% de usuarios.
"""
import pandas as pd
import numpy as np

def makeSamePermut(df, person):
    df = df[df["person"] == person]
    df.set_index("pic", inplace = True)

    perList = []

    for i in range(1, len(df)+1):
        for j in range(i+1, len(df)+1):
            temp1 = df.loc[[i]]["filepath"].to_list()[0]
            temp2 = df.loc[[j]]["filepath"].to_list()[0]
            perList.append([temp1, temp2, 1])
    
    return perList


def makeDiffPermit(df, person1, person2):
    per1 = df[df["person"] == person1]
    per2 = df[df["person"] == person2]

    per1.set_index("pic", inplace = True)
    per2.set_index("pic", inplace = True)


    perList = []

    for i in range(1, len(per1)+1):
        for j in range(1, len(per2)+1):
            temp1 = per1.loc[[i]]["filepath"].to_list()[0]
            temp2 = per2.loc[[j]]["filepath"].to_list()[0]
            perList.append([temp1, temp2, 0])

    return perList


path = "./dataset/MMU-Iris-2.csv"
dataset = pd.read_csv(path, sep=",", header=0)

leftSide = dataset[dataset["lat"] == 1]
rightSide = dataset[dataset["lat"] == 2]

leftPerson = np.unique(leftSide["person"])
rightPerson = np.unique(rightSide["person"])

leftPersonSameX = []
rightPersonSameX = []

leftPersonSameY = []
rightPersonSameY = []

leftPersonDiffX = []
leftPersonDiffY = []

rightPersonDiffX = []
rightPersonDiffY = []

for i in leftPerson:
    if(i < 80):
        leftPersonSameX += makeSamePermut(leftSide, i)
    else:
        leftPersonSameY += makeSamePermut(leftSide, i)

for i in rightPerson:
    if (i < 80):
        rightPersonSameX += makeSamePermut(rightSide, i)
    else:
        rightPersonSameY += makeSamePermut(rightSide, i)

lsX = open('./dataset/left-same-sub80.csv', 'w')
lsY = open('./dataset/left-same-top80.csv', 'w')
rsX = open('./dataset/right-same-sub80.csv', 'w')
rsY = open('./dataset/right-same-top80.csv', 'w')

for i in leftPersonSameX:
    lsX.write("{},{},{}\n".format(i[0], i[1], i[2]))
lsX.close()

for i in leftPersonSameY:
    lsY.write("{},{},{}\n".format(i[0], i[1], i[2]))
lsY.close()

for i in rightPersonSameX:
    rsX.write("{},{},{}\n".format(i[0], i[1], i[2]))
rsX.close()

for i in rightPersonSameY:
    rsY.write("{},{},{}\n".format(i[0], i[1], i[2]))
rsY.close()

for i in leftPerson:
    print(i)
    for j in leftPerson:
        if(i != j):
            if(i < 80):
                leftPersonDiffX += makeDiffPermit(leftSide, i, j)
            elif(i >= 80 and j >= 80):
                leftPersonDiffY += makeDiffPermit(leftSide, i, j)


for i in rightPerson:
    print(i)
    for j in rightPerson:
        if(i != j):
            if(i < 80 and j < 80):
                rightPersonDiffX += makeDiffPermit(rightSide, i, j)
            elif(i >= 80 and j >= 80):
                rightPersonDiffY += makeDiffPermit(rightSide, i, j)

ldX = open('./dataset/left-diff-sub80.csv', 'w')
ldY = open('./dataset/left-diff-top80.csv', 'w')
rdX = open('./dataset/right-diff-sub80.csv', 'w')
rdY = open('./dataset/right-diff-top80.csv', 'w')

for i in leftPersonDiffX:
    ldX.write("{},{},{}\n".format(i[0], i[1], i[2]))
ldX.close()

for i in leftPersonDiffY:
    ldY.write("{},{},{}\n".format(i[0], i[1], i[2]))
ldY.close()

for i in rightPersonDiffX:
    rdX.write("{},{},{}\n".format(i[0], i[1], i[2]))
rdX.close()

for i in rightPersonDiffY:
    rdY.write("{},{},{}\n".format(i[0], i[1], i[2]))
rdY.close()