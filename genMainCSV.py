"""
Código para obtener el CSV principal del dataset
"""
import glob 

RAW_DATA_PATH = "./dataset/MMU-Iris-2/"

files = glob.glob("{}*.bmp".format(RAW_DATA_PATH))
files.sort()

irisData = []

for f in files:
    tempFile = (f.split("/")[3]).split(".")[0]
    person = tempFile[0:2]
    lat = tempFile[2:4]
    pic = tempFile[4:6]
    print(person, lat, pic, f)
    irisData.append([person, lat, pic, f])

mainFile = open('./dataset/MMU-Iris-2.csv', 'w')

mainFile.write("{},{},{},{}\n".format("person", "lat", "pic", "filepath"))
for i in irisData:
    mainFile.write("{},{},{},{}\n".format(i[0], i[1], i[2], i[3]))

mainFile.close()