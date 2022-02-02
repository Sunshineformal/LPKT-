import os
import csv
import numpy as np

fromK2 = []
fromPEBG = []
#read relationFromK2.csv
with open("relationFromK2.csv", encoding = "utf-8") as file:
    rows = csv.reader(file)
    for row in rows:
        fromK2.append(row)

#read relationFromPEBG.csv
with open("relationFromPEBG.csv", encoding = "utf-8") as file:
    rows = csv.reader(file)
    for row in rows:
        fromPEBG.append(row)

#convert to numpy
fromK2 = np.array(fromK2, dtype = "float")
fromK2 = fromK2[1:,1:]
fromPEBG = np.array(fromPEBG, dtype = "float")

#caculate * matrix
finalMatrix = fromK2 * fromPEBG
for i in range(finalMatrix.shape[0]):
    finalMatrix[i,i] = 1


#convert matrix shape
# finalMatrix = np.append(np.zeros((1,189)), finalMatrix, axis = 0)
# finalMatrix = np.append(np.zeros((189,1)), finalMatrix, axis = 1)

#write file
finalMatrix = finalMatrix.tolist()
file = open("finalRelation.csv","w",newline = "", encoding = "utf-8")
writer = csv.writer(file)
for each in finalMatrix:
    writer.writerow(each)
file.close()

print("Complated")