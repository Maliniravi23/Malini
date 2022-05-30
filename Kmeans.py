import numpy as np
import os

f = []


def d_spilt(d):
    for i in range(len(d)):
        if i > 0:
            b = d[i].split()
            c = [float(i) for i in b]
            f.append(c)
    return f


file = os.listdir("/home/Malini/IITM course/Assignment/Mfcc/d/")
for i in file:
    a = open("/home/Malini/IITM course/Assignment/Mfcc/d/" + i, 'r')
    d = a.readlines()
    nf_4 = d_spilt(d)
dictionary = {i: nf_4[i] for i in range(len(nf_4))}
mylist = []
for i in range(len(nf_4)):
    mylist.append(i)
k4 = np.random.choice(mylist)
k7 = np.random.choice(mylist)
k8 = np.random.choice(mylist)

k = 0
l1 = np.array(dictionary[k4])
l2 = np.array(dictionary[k7])
l3 = np.array(dictionary[k8])

# print(l1)
x = []
x1 = []
x2 = []
for i in range(0, 38):
    x.append(0)
# x=np.array(x)
for i in range(0, 38):
    x1.append(0)
# x1=np.array(x1)
for i in range(0, 38):
    x2.append(0)
# x2=np.array(x2)
c1 = 0
c2 = 0
c3 = 0
# print(x)

for i in range(len(nf_4)):
    arr1 = np.array(nf_4[i])
    # print(arr1)
    dist1 = np.sqrt(np.sum(np.square(arr1 - l1)))
    dist2 = np.sqrt(np.sum(np.square(arr1 - l2)))
    dist3 = np.sqrt(np.sum(np.square(arr1 - l3)))

    mini = min(dist1, dist2, dist3)
    if (dist1 == mini):
        x += arr1
        c1 += 1

    elif (dist2 == mini):
        x1 += arr1
        c2 += 1

    else:
        x2 += arr1
        c3 += 1
li = []
for i in range(len(x)):
    avg = x[i] / c1
    li.append(avg)
l1 = np.array(li)
li_1 = []
for i in range(len(x1)):
    avg = x1[i] / c1
    li_1.append(avg)
l2 = np.array(li_1)
li_2 = []
for i in range(len(x2)):
    avg = x2[i] / c1
    li_2.append(avg)
l3 = np.array(li_2)
code_1 = []
code_2 = []
code_3 = []
code_1.append(l1)
code_2.append(l2)
code_3.append(l3)

print(x)
print(li)
print(x1)
print(li_1)
print(x2)
print(li_2)
print(c1)
print(c2)
print(c3)


