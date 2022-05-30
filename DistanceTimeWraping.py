import numpy as np
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
def d_spilt(d) :
    f = []
    for i in range(len(d)):
        if i > 0:
            b = d[i].split()
            c = [float(i) for i in b]
            f.append(c)
            return f
path =r'/home/malini/Downloads/train_2.mfcc'
filenames = glob.glob(path + "/*.mfcc")
train=[]
for i in filenames:
    #print(filenames)
    file=open(i+"",'r')
    d=file.readlines()
    n_f=d_spilt(d)
    train.append(n_f)
path =r'/home/malini/Downloads/dev_2.mfcc'
filenames = glob.glob(path + "/*.mfcc")
test=[]
for i in filenames:
    file=open(i+"",'r')
    d=file.readlines()
    n_f=d_spilt(d)
    test.append(n_f)
i=0
j=0
whole_mat=[]
while(j<len(test)):
    while(i< len(train)):
        mat=np.zeros((len(train[i]),len(n_f)))
        for d in range(len(train[i])):
            arr1=np.array(train[i][d])
        for d1 in range(len(n_f)):
            arr2=np.array(n_f[d1])
            distance=np.sqrt(np.sum(np.square(arr1-arr2)))
            mat[d][d1]=distance
        print(mat)
    whole_mat.append(mat)
    #print(len(mat))
    #print(len(mat[0]))
    row = (len(mat)-1)
    col = (len(mat[0]) - 1)
    last_ele = mat[row][col]
print( "last element of mat " + str(last_ele))
c = 1
list_path=[]
while(row >= 0 and col >= 0 ):
    if mat[row-1][col]<mat[row][col-1]and mat[row-1][col]<mat[row-1][col-1]:
        row = row -1
        col = col
    elif mat[row][col-1]<mat[row-1][col-1]:
        row = row
        col = col-1
    else:
        row = row-1
        col = col-1
    c = c+1
    path_rate = last_ele / c
    list_path.append(path_rate)
    i = i+1
    j=j+1
len_list=len(list_path)
print(len_list)
avg=np.sum(list_path)/(len_list)
print("avg_value:"+str(avg))
print("path_rate :" + str(list_path))
plt.plot(list_path,c='r')
plt.title('Distance matrix with optimal warping path')
plt.xlabel('Test')
plt.ylabel('Train')
plt.show()