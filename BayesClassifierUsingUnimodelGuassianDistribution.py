import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a1 = pd.read_csv("5.1_ls.frm.csv")
a2 = pd.read_csv("5.2_ls.frm.csv")
a3 = pd.read_csv("5.3_ls.frm.csv")
a4 = pd.read_csv("5_ls.frm.csv")
a1 = np.array(a1)
a2 = np.array(a2)
a3 = np.array(a3)
a4 = np.array(a4)
a_x = a1[:,0]
a_y = a1[:,1]
i_x = a2[:,0]
i_y = a2[:,1]
u_x = a3[:,0]
u_y = a3[:,1]
all_x = a4[:,0]
all_y = a4[:,1]
n = len(a_x)
n1 = len(i_x)
n2 = len(u_x)
mean_1 = np.mean(a_x)
mean_1
mean_2 = np.mean(a_y)
mean_2
mean_3 = np.mean(i_x)
mean_3
mean_4 = np.mean(i_y)
mean_4
mean_5 = np.mean(u_x)
mean_5
mean_6 = np.mean(u_y)
mean_6
a_ax = 0
a_ay = 0
a_axy = 0
a_ayx = 0
for i,j in zip(a_x,a_y):
    cov_ax = (i-mean_1)**2
    cov_xy = (i-mean_1)*(j-mean_2)
    cov_yx = (j-mean_2)*(i-mean_1)
    cov_ay = (j-mean_2)**2
    a_ax+=cov_ax/(n-1)
    a_axy+=cov_xy/(n-1)
    a_ayx+=cov_yx/(n-1)
    a_ay+=cov_ay/(n-1)
    lis_a = [a_ax,a_axy,a_ayx,a_ay]
n_a = len(lis_a)
i_ix = 0
i_iy = 0
i_ixy = 0
i_iyx = 0
for i,j in zip(i_x,u_y):
    cov_ix = (i-mean_3)**2
    cov_ixy = (i-mean_3)*(j-mean_4)
    cov_iyx = (j-mean_4)*(i-mean_3)
    cov_iy = (j-mean_4)**2
    i_ix+=cov_ix/(n1-1)
    i_ixy+=cov_ixy/(n1-1)
    i_iyx+=cov_iyx/(n1-1)
    i_iy+=cov_iy/(n1-1)
    lis_i = [i_ix,i_ixy,i_iyx,i_iy]
n_i = len(lis_i)
u_ux = 0
u_uy = 0
u_uxy = 0
u_uyx = 0
for i,j in zip(u_x,u_y):
    cov_ux = (i-mean_5)**2
    cov_uxy = (i-mean_5)*(j-mean_6)
    cov_uyx = (j-mean_6)*(i-mean_5)
    cov_uy = (j-mean_6)**2
    u_ux+=cov_ux/(n2-1)
    u_uxy+=cov_uxy/(n2-1)
    u_uyx+=cov_uyx/(n2-1)
    u_uy+=cov_uy/(n2-1)
    lis_u = [u_ux,u_uxy,u_uyx,u_uy]
n_u = len(lis_u)
cov_a = np.cov(a_x,a_y)
print(cov_a)
cov_i = np.cov(i_x,i_y)
print(cov_i)
cov_u = np.cov(u_x,u_y)
print(cov_u)
det_a = np.linalg.det(cov_a)
print(det_a)
det_i = np.linalg.det(cov_i)
print(det_i)
det_u = np.linalg.det(cov_u)
print(det_u)
inv_a = np.linalg.inv(cov_a)
print(inv_a)
inv_i = np.linalg.inv(cov_i)
print(inv_i)
inv_u = np.linalg.inv(cov_u)
print(inv_u)
all_min_f1 = min(all_x)
print(all_min_f1)
all_max_f1 = max(all_x)
print(all_max_f1)
all_min_f2 = min(all_y)
print(all_min_f2)
all_max_f2 = max(all_y)
print(all_max_f2)
sample_f1=[]
sample_f2=[]
while all_min_f1 < all_max_f1 or all_min_f2 < all_max_f2:
    sample_f1.append(all_min_f1)
    sample_f2.append(all_min_f2)
    all_min_f1+=0.1
    all_min_f2+=0.1
print(len(sample_f1))
print(len(sample_f2))
sample_x=[]
sample_y=[]
for i in sample_f1:
    for j in sample_f2:
        sample_x.append(i)
        sample_y.append(j)
#non covariance matrix
maximum_ax = []
maximum_ay = []
maximum_ix = []
maximum_iy = []
maximum_ux = []
maximum_uy = []
for i,j in zip(sample_x,sample_y):
    mean_axy = np.matrix([i-mean_1,j-mean_2])
    half_axy = np.log(mean_axy*inv_a*mean_axy.T)
    half_a = np.exp(-half_axy)
    pdf_axy = 1/(2*np.pi*(det_a)**1/2)*half_a
    #print(pdf_axy)
    mean_ixy = np.matrix([i-mean_3,j-mean_4])
    half_ixy = np.log(mean_ixy*inv_i*mean_ixy.T)
    half_i = np.exp(-half_ixy)
    pdf_ixy = 1/(2*np.pi*(det_i)**1/2)*half_i
    #print(pdf_ixy)
    mean_uxy = np.matrix([i-mean_5,j-mean_6])
    half_uxy = np.log(mean_uxy*inv_u*mean_uxy.T)
    half_u = np.exp(-half_uxy)
    pdf_uxy = 1/(2*np.pi*(det_u)**1/2)*half_u
    #print(pdf_uxy)
    if pdf_axy > pdf_ixy and pdf_axy > pdf_uxy:
        maximum_ax.append(i)
        maximum_ay.append(j)
    #print(maximum_axy)
    elif pdf_ixy > pdf_axy and pdf_ixy > pdf_uxy:
        maximum_ix.append(i)
        maximum_iy.append(j)
    #print(maximum_ixy)
    else:
        maximum_ux.append(i)
        maximum_uy.append(j)
#print(maximum_uxy)
plt.scatter(maximum_ax,maximum_ay,c='g')
plt.scatter(maximum_ix,maximum_iy,c='b')
plt.scatter(maximum_ux,maximum_uy,c='r')
plt.scatter(a_x,a_y,c='black')
plt.scatter(i_x,i_y,c='yellow')
plt.scatter(u_x,u_y,c='pink')
x,y=np.meshgrid(a_x,a_y)
x1,y1=np.meshgrid(i_x,i_y)
x2,y2=np.meshgrid(u_x,u_y)
z= np.ndarray(shape=(len(x),len(y)))
z1=np.ndarray(shape=(len(x1),len(y1)))
z2=np.ndarray(shape=(len(x2),len(y2)))
for i in range(len(x)):
    for j in range(len(y)):
        mean_axy = np.matrix([x[i][j]-mean_1,y[i][j]-mean_2])
        #print(mean_axy)
        half_axy = np.log(mean_axy*inv_a*mean_axy.T)
        half_a = np.exp(-half_axy)
        z[i][j] = 1/(2*np.pi*det_a**1/2)*half_a
for i in range(len(x1)):
    for j in range(len(y1)):
        mean_axy = np.matrix([x1[i][j]-mean_1,y1[i][j]-mean_2])
        #print(mean_axy)
        half_axy = np.log(mean_axy*inv_a*mean_axy.T)
        half_a = np.exp(-half_axy)
        z1[i][j] = 1/(2*np.pi*det_a**1/2)*half_a
for i in range(len(x2)):
    for j in range(len(y2)):
        mean_axy = np.matrix([x2[i][j]-mean_1,y2[i][j]-mean_2])
        #print(mean_axy)
        half_axy = np.log(mean_axy*inv_a*mean_axy.T)
        half_a = np.exp(-half_axy)
        z2[i][j] = 1/(2*np.pi*det_a**1/2)*half_a
plt.contourf(x,y,z)
plt.contourf(x1,y1,z1)
plt.contourf(x2,y2,z2)
#diagonal covariance matrix
iden = np.identity(2)
eigen_a = cov_a * iden
eigen_i = cov_i * iden
eigen_u = cov_u * iden
print(eigen_a)
print(eigen_i)
print(eigen_u)
dt_a = np.linalg.det(eigen_a)
print(dt_a)
dt_i = np.linalg.det(eigen_i)
print(dt_i)
dt_u = np.linalg.det(eigen_u)
print(dt_u)
iv_a = np.linalg.inv(eigen_a)
print(iv_a)
iv_i = np.linalg.inv(eigen_i)
print(iv_i)
iv_u = np.linalg.inv(eigen_u)
print(iv_u)
maximum_ax = []
maximum_ay = []
maximum_ix = []
maximum_iy = []
maximum_ux = []
maximum_uy = []
for i,j in zip(sample_x,sample_y):
    mean_axy = np.matrix([i-mean_1,j-mean_2])
    half_axy = np.log(mean_axy*iv_a*mean_axy.T)
    half_a = np.exp(-half_axy)
    pdf_axy = 1/(2*np.pi*dt_a**1/2)*half_a
    #print(pdf_axy)
    mean_ixy = np.matrix([i-mean_3,j-mean_4])
    half_ixy = np.log(mean_ixy*iv_i*mean_ixy.T)
    half_i = np.exp(-half_ixy)
    pdf_ixy = 1/(2*np.pi*dt_i**1/2)*half_i
    #print(pdf_ixy)
    mean_uxy = np.matrix([i-mean_5,j-mean_6])
    half_uxy = np.log(mean_uxy*iv_u*mean_uxy.T)
    half_u = np.exp(-half_uxy)
    pdf_uxy = 1/(2*np.pi*dt_u**1/2)*half_u
    #print(pdf_uxy)
if pdf_axy > pdf_ixy and pdf_axy > pdf_uxy:
    maximum_ax.append(i)
    maximum_ay.append(j)
    #print(maximum_axy)
    elif pdf_ixy > pdf_axy and pdf_ixy > pdf_uxy:
        maximum_ix.append(i)
        maximum_iy.append(j)
        #print(maximum_ixy)
    else:
        maximum_ux.append(i)
        maximum_uy.append(j)
        #print(maximum_uxy)
        #print(len(maximum_ix))
plt.scatter(maximum_ax,maximum_ay,c='g')
plt.scatter(maximum_ix,maximum_iy,c='b')
plt.scatter(maximum_ux,maximum_uy,c='r')
plt.scatter(a_x,a_y,c='black')
plt.scatter(i_x,i_y,c='yellow')
plt.scatter(u_x,u_y,c='pink')
x,y=np.meshgrid(a_x,a_y)
x1,y1=np.meshgrid(i_x,i_y)
x2,y2=np.meshgrid(u_x,u_y)
z= np.ndarray(shape=(len(x),len(y)))
z1=np.ndarray(shape=(len(x1),len(y1)))
z2=np.ndarray(shape=(len(x2),len(y2)))
for i in range(len(x)):
    for j in range(len(y)):
        mean_axy = np.matrix([x[i][j]-mean_1,y[i][j]-mean_2])
        #print(mean_axy)
        half_axy = np.log(mean_axy*inv_a*mean_axy.T)
        half_a = np.exp(-half_axy)
        z[i][j] = 1/(2*np.pi*det_a**1/2)*half_a
for i in range(len(x1)):
    for j in range(len(y1)):
        mean_axy = np.matrix([x1[i][j]-mean_1,y1[i][j]-mean_2])
        #print(mean_axy)
        half_axy = np.log(mean_axy*inv_a*mean_axy.T)
        half_a = np.exp(-half_axy)
        z1[i][j] = 1/(2*np.pi*det_a**1/2)*half_a
for i in range(len(x2)):
    for j in range(len(y2)):
        mean_axy = np.matrix([x2[i][j]-mean_1,y2[i][j]-mean_2])
        #print(mean_axy)
        half_axy = np.log(mean_axy*inv_a*mean_axy.T)
        half_a = np.exp(-half_axy)
        z2[i][j] = 1/(2*np.pi*det_a**1/2)*half_a
plt.contourf(x,y,z)
plt.contourf(x1,y1,z1)
plt.contourf(x2,y2,z2)
#using same covariance for 2 different classes
maximum_ax = []
maximum_ay = []
maximum_ix = []
maximum_iy = []
maximum_ux = []
maximum_uy = []
for i,j in zip(sample_x,sample_y):
    mean_axy = np.matrix([i-mean_1,j-mean_2])
    half_axy = np.log(mean_axy*inv_a*mean_axy.T)
    half_a = np.exp(-half_axy)
    pdf_axy = 1/(2*np.pi*det_a**1/2)*half_a
    #print(pdf_axy)
    mean_ixy = np.matrix([i-mean_3,j-mean_4])
    half_ixy = np.log(mean_ixy*inv_a*mean_ixy.T)
    half_i = np.exp(-half_ixy)
    pdf_ixy = 1/(2*np.pi*det_a**1/2)*half_i
    #print(pdf_ixy)
    mean_uxy = np.matrix([i-mean_5,j-mean_6])
    half_uxy = np.log(mean_uxy*inv_u*mean_uxy.T)
    half_u = np.exp(-half_uxy)
    pdf_uxy = 1/(2*np.pi*det_u**1/2)*half_u
    #print(pdf_uxy)
if pdf_axy > pdf_ixy and pdf_axy > pdf_uxy:
    maximum_ax.append(i)
    maximum_ay.append(j)
    #print(maximum_axy)
elif pdf_ixy > pdf_axy and pdf_ixy > pdf_uxy:
    maximum_ix.append(i)
    maximum_iy.append(j)
    #print(maximum_ixy)
else:
    maximum_ux.append(i)
    maximum_uy.append(j)
    #print(maximum_uxy)
    #print(len(maximum_ix))
plt.scatter(maximum_ax,maximum_ay,c='g')
plt.scatter(maximum_ix,maximum_iy,c='b')
plt.scatter(maximum_ux,maximum_uy,c='r')
plt.scatter(a_x,a_y,c='black')
plt.scatter(i_x,i_y,c='yellow')
plt.scatter(u_x,u_y,c='pink')
x,y=np.meshgrid(a_x,a_y)
x1,y1=np.meshgrid(i_x,i_y)
x2,y2=np.meshgrid(u_x,u_y)
z= np.ndarray((len(x),len(y)))
z1=np.ndarray((len(x1),len(y1)))
z2=np.ndarray((len(x2),len(y2)))
for i in range(len(x)):
    for j in range(len(y)):
        mean_axy = np.matrix([x[i][j]-mean_1,y[i][j]-mean_2])
        #print(mean_axy)
        half_axy = np.log(mean_axy*inv_a*mean_axy.T)
        half_a = np.exp(-half_axy)
        z[i][j] = 1/(2*np.pi*det_a**1/2)*half_a
for i in range(len(x1)):
    for j in range(len(y1)):
        mean_axy = np.matrix([x1[i][j]-mean_1,y1[i][j]-mean_2])
        #print(mean_axy)
        half_axy = np.log(mean_axy*inv_a*mean_axy.T)
        half_a = np.exp(-half_axy)
        z1[i][j] = 1/(2*np.pi*det_a**1/2)*half_a
for i in range(len(x2)):
    for j in range(len(y2)):
        mean_axy = np.matrix([x2[i][j]-mean_1,y2[i][j]-mean_2])
        #print(mean_axy)
        half_axy = np.log(mean_axy*inv_a*mean_axy.T)
        half_a = np.exp(-half_axy)
        z2[i][j] = 1/(2*np.pi*det_a**1/2)*half_a
plt.contourf(x,y,z)
plt.contourf(x1,y1,z1)
plt.contourf(x2,y2,z2)