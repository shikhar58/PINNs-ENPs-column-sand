# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 18:34:57 2022

@author: shikhar
"""


from pyDOE import lhs
import sys
sys.path.append(r"C:\Users\shikhar\PycharmProjects\mesh\autoencoder-for-denoising-coarser-mesh-based-numerical-solution")
#importing dependency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras as K
import tensorflow.python.keras.backend as K
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pyDOE import lhs
#tf.enable_eager_execution()
#tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


"""
N_f=5000

minval=np.array([0,0,0])
maxval=np.array([10,4,6000])

XX_f = np.tile(maindata[:,0:1], (1,len(t_dc))) # N x T
YY_f = np.tile(maindata[:,1:2], (1,len(t_dc))) # N x 
TT_f = np.tile(t_dc, (1,len(maindata[:,0:1]))).T # N x T

x = XX_f.flatten()[:,None] # NT x 1
y = YY_f.flatten()[:,None] # NT x 1
t = TT_f.flatten()[:,None] # NT x 1



idx = np.random.choice(len(t_dc)*len(maindata[:,0:1]), N_f, replace=False)
x_fp = x[idx,:]
y_fp = y[idx,:]
t_fp = t[idx,:]

"""




layers_c=[2, 50, 50, 50, 50, 50, 1]

layers_s=[2, 50, 50, 50, 50,50, 1]

x=np.linspace(0, 1, num=101)

x=x[:,None]


t=np.linspace(0, 6000, num=3001)

t=t[:,None]
"""
xx, tt = np.meshgrid(x, t)

X = np.hstack((xx.flatten()[:,None], tt.flatten()[:,None]))
"""
ic = np.concatenate((x, 0*x), 1)
x_ic=ic[:,0:1]
t_ic=ic[:,1:2]

boundary=(0,1)
cond_lb = np.concatenate((0*t + boundary[0], t), 1)
x_lb=cond_lb[:,0:1]
t_lb=cond_lb[:,1:2]
cond_rb = np.concatenate((0*t + boundary[1], t), 1)
x_rb=cond_rb[:,0:1]
t_rb=cond_rb[:,1:2]


"""
c_dcl=1/(1 + np.exp(-0.1*(t-500)))

c_dcr=1/(1 + np.exp(-0.1*(-t+1500)))

c_dc=c_dcl*c_dcr
"""
#c_dc=1/(1 + np.exp(-0.1*(t-500)))


c_dcl=1/(1 + np.exp(-0.02*(t-500)))

c_dcr=1/(1 + np.exp(-0.02*(-t+4100)))

c_dc=c_dcl*c_dcr

def sampling(minval,maxval,var,N):
    X = minval + (maxval-minval)*lhs(var, N)
    return X

"""
data_f=sampling(np.array([0,0]),np.array([1,3600]),2,20000)
x_fp=data_f[:,0:1]
t_fp=data_f[:,1:2]

X_fp = np.concatenate((x_fp,t_fp), 1)
"""

x=np.linspace(0, 1, num=51)

x=x[:,None]

t_f=np.linspace(0, 6000, num=301)

t_f=t_f[:,None]

xx,tt=np.meshgrid(x,t_f)
xx_f = xx.flatten()[:,None] # NT x 1
tt_f = tt.flatten()[:,None] # NT x 1


fp=np.concatenate((xx_f,tt_f),1)

def initialize_NN(layers):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases

def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.compat.v1.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

#chamka..sigmoid function canged everything

def neural_net( X, weights, biases):
    num_layers = len(weights) + 1
    H = (X-minval)/(maxval-minval)
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        #H = tf.tanh(tf.add(tf.matmul(H, W), b))
        H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


"""
def net_NS_IC( x, y, t):
    c_ic = neural_net(tf.concat([x,y,t], 1), weights, biases)
    return c_ic
  """  
def net_NS( x, t,c_out,por,alpha):
    cx = tf.gradients(c_out, x)[0]
    ct = tf.gradients(c_out, t)[0]

    cxx = tf.gradients(cx, x)[0]
    ux=0.0003
    uy=0
    adv=ux*cx
    Dd=alpha*ux

    De=1e-9
   # ka=0.0008
   # kd=0.0001

    Jd=-(Dd+De)*(cxx)
    print("count",1)
    #breakpoint()
    fc=por*ct+adv+Jd
    return fc, Jd
#rho_b*s is state variable as s

def callback(loss):
    print('Loss: %.3e' % (loss))
#X = np.concatenate([x_train, y_train, t_train], 1)


#DEFINE THIS ONLY FOR INPUT DATA, IE FEATURES AND TAGER WHICH Has to be minimised
x_i=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

t_i=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

x_dcb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

t_dcb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

x_neb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

t_neb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

x_f=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
t_f=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])



weights_c, biases_c = initialize_NN(layers_c)  


w_ic=tf.Variable(5.0)

w_dc=tf.Variable(3.0)
w_fc=tf.Variable(10000.4)


w_j=tf.Variable(1.0)



alpha=0.01
por=0.3


#conka1=tf.Variable(0)
#conkd1=tf.Variable(0)


sess=tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

minval=np.array([0,0])
maxval=np.array([1,6000])

#tf_dict = { x_i: x_ic, y_i: y_ic, x_f: x_fp, y_f:y_fp, t_f:t_fp}


tf_dict = {x_dcb: x_lb, t_dcb: t_lb, x_neb: x_rb, t_neb: t_rb, x_i: x_ic, t_i: t_ic, x_f: xx_f, t_f: tt_f}


tf_dict0 = {x_dcb: x_lb, t_dcb: t_lb, x_neb: x_rb, t_neb: t_rb, x_i: x_ic, t_i: t_ic, x_f: xx_f, t_f: tt_f}
#dict0 is define so that the samping happens on the original mesh, not the updated mesh with refine points
#tf_dict = { x_i: x_ic, y_i: y_ic, x_f: x_fp, y_f:y_fp, t_f:t_fp}


#c_dcb=neural_net(tf.concat([x_dcb, y_dcb], 1), weights_ib, biases_ib,np.array([0,1.5]),np.array([0,2.5]))
#c_dcb=neural_net(tf.concat([x_dcb, y_dcb], 1), weights_dcb, biases_dcb,np.array([0,1.5]),np.array([0,2.5]))
c_dcb=neural_net(tf.concat([x_dcb, t_dcb], 1), weights_c, biases_c)
c_neb=neural_net(tf.concat([x_neb, t_neb], 1), weights_c, biases_c)

c_ic=neural_net(tf.concat([x_i, t_i], 1), weights_c, biases_c)
c_f=neural_net(tf.concat([x_f, t_f], 1), weights_c,biases_c)

f,_=net_NS(x_f,t_f,c_f,por,alpha)

_,j=net_NS(x_neb,t_neb,c_neb,por,alpha)
"""
#adding 36 really worked out
loss_ic=36*tf.reduce_sum(abs(c_ic))
loss_dc=tf.reduce_sum(abs(c_dcb-c_dc))
#adding 36 really worked out
loss_f=tf.reduce_sum(abs(f))
loss_j=tf.reduce_sum(abs(j))
loss_all=loss_ic+loss_dc+loss_f+loss_j
ind_dc=loss_all/loss_dc
ind_ic=loss_all/loss_ic
ind_f=loss_all/loss_f
ind_j=loss_all/loss_j
"""
"""
conka=ka.eval(session=sess)
if conka<0 and conka>1:
    print("great")
    conka1=100
else:
    conka1=0

conkd=kd.eval(session=sess)
if conkd<0 and conkd>1:
    print("great")
    conkd1=100

else:
    conkd1=0
    
    """
#loss = 36*tf.reduce_sum(abs(c_ic))+tf.reduce_sum(abs(c_dcb-c_dc))+tf.reduce_sum(abs(f))+tf.reduce_sum(abs(j))
#tf.reduce_sum(abs(f)).eval(feed_dict=tf_dict,session=sess)
#add square 
#loss = w_ic*tf.reduce_sum(tf.square(c_ic))+w_is*tf.reduce_sum(tf.square(s_ic))+w_dc*tf.reduce_sum(tf.square(c_dcb-c_dc))+w_fc*tf.reduce_sum(tf.square(fc))+w_fs*tf.reduce_sum(tf.square(fs))+w_j*tf.reduce_sum(tf.square(j))
loss = w_ic*tf.reduce_sum(tf.square(c_ic))+w_dc*tf.reduce_sum(tf.square(c_dcb-c_dc))+w_fc*tf.reduce_sum(tf.square(f))+w_j*tf.reduce_sum(tf.square(j))

#loss = tf.reduce_sum(tf.square(c_ic))+10*tf.reduce_sum(tf.square(s_ic))+tf.reduce_sum(tf.square(c_dcb-c_dc))+100*tf.reduce_sum(tf.square(fc))+10000*tf.reduce_sum(tf.square(fs))+tf.reduce_sum(tf.square(j))



optimizer_Adam = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.005, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam')



train_op_Adam = optimizer_Adam.minimize(loss,var_list=[weights_c,biases_c])  


import random
import math

def errorpoints(r_value):
    r_sum= np.sum(r_value**k)

    prob_value=(r_value**k/r_sum)+c

    new=np.zeros(target)
    for i in range(target):
        new[i]=int(random.choices(range(len(prob_value)),prob_value)[0])  #index of fp selected according to the fp
    
    prob_selected=[prob_value[int(i)] for i in new]      #probability of the points which got resampled

    newpt=np.array([fp[int(i)] for i in new])     
        
    return(newpt)

refine=2
def adap(x,t):
    gen=[]
    fpp=np.array([x/0.02,t/20])
    #print(fpp)
    count=0
    while True:    # number of refinement
        if count>refine-1:
            break
        px=random.uniform(-1, 1)
        pt=random.uniform(-1, 1)
        print(px)
        xnew=fpp[0]+px
        tnew=fpp[1]+pt
        fppnew=np.array([xnew,tnew])
        #random points generated with range of 1,1 to -1-1 for the point 0 0
        #now we need to eliminate teh points who are far away
        print(np.shape(fppnew),np.shape(fpp))
        dist=math.dist(fpp, fppnew)
        if dist<1:
            a=np.array([fppnew[0]*0.02,fppnew[1]*20])
            if a[0]>0 and a[0]<1 and a[1]>0 and a[1]<6000 and a[1]>0:
                gen.append(a)
                count=count+1
    return np.array(gen)

#write code for definite number of refinement

def resam(fp):
    gen_fin=[]
    for i in range(len(fp)):
        #print(i)
        gen=np.array(adap(fp[i,0],fp[i,1]))  #receiving n individual points in the dimension (n,2), where n is the number of point seeked for an individual original nesh point
        gen_fin.append(gen)  #repeating the activity for each original mesh point. Th mesh point now represnet, all refined ppoint ie for m points, m*n points of xt
    #to unbox gen from list to numpy
    ada=np.zeros([1,2])
    for n,i in enumerate(gen_fin):
        #print(n)
        if len(i)>0:
            ada=np.concatenate((ada,i),0)
    return ada

sess=tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)   
target=500
k=2
c=0
losstot=[]
nIter=1800
for it in range(1,nIter):
    sess.run(train_op_Adam, tf_dict)
    print(it)
    #fv=f.eval(feed_dict=tf_dict,session=sess)
    #print(it,loss_value,tf.reduce_sum(tf.square(c_dcb-c_dc)).eval(feed_dict=tf_dict,session=sess),tf.reduce_sum(tf.square(f)).eval(feed_dict=tf_dict,session=sess))
    loss_value=loss.eval(feed_dict=tf_dict0,session=sess)
    if it%300==0:
        losstot.append(loss_value)
        r_val=f.eval(feed_dict=tf_dict0,session=sess)
       #actual points which got resampled
        resamp=errorpoints(r_val)
        
        refinep=resam(resamp)   #here gen_fin is list of refined point for each oriningal point, and aa is the list of all those refined points

        #resampled=resam(newp)
        #print(resampled)
        #print("%%%%%%%", len(resampled))
        xx_new=np.concatenate((xx_f,resamp[:,0:1]),0)
        tt_new=np.concatenate((tt_f,resamp[:,1:2]),0)
        
        tf_dict = {x_dcb: x_lb, t_dcb: t_lb, x_neb: x_rb, t_neb: t_rb, x_i: x_ic, t_i: t_ic, x_f: xx_new, t_f: tt_new}

    print(it,loss_value)
    if abs(loss_value)<0.01:
        break



"""
to CONFIRM THE DISTANCE IS ONE out thedistance
fp=np.concatenate((xx_f/0.02,tt_f/20),1)
a=np.zeros(500)
b=np.zeros(500)
for j in range(1):
    print(j)
    dist=[math.dist(fp[j], fp[i]) for i in range(len(fp)) if i!=j]
    #a[j]=np.min([math.dist(fp[j], fp[i]) for i in range(len(fp)) if i!=j])
    b[j]=np.argmin(dist)
    a[j]=dist[int(b[j])]

plt.scatter(xx_f[:500]/0.01,tt_f[:500]/20)
"""


aa=np.array([x for x in range(3001)])
aa=aa[:,None]
#c=neural_net(tf.concat([x1, t1], 1), weights, biases)
cneb=c_neb.eval(feed_dict=tf_dict,session=sess)
cdcb=c_dcb.eval(feed_dict=tf_dict,session=sess)
plt.plot(aa[:,:], cneb[:,:], marker='.', label="actual")
plt.plot(aa[:,:], cdcb[:,:], 'r', label="actual")
plt.plot(aa[:,:], c_dc[:,:], 'g', label="actual")


plt.scatter(xx_f*5000,tt_f, marker='.')
#plt.scatter(xx_new,tt_new, marker='.')
plt.scatter(resamp[:,0]*5000,resamp[:,1], marker='*')
plt.scatter(refinep[:,0]*5000,refinep[:,1], marker='.')
plt.ylim(-1, 1000.0)
plt.xlim(-0.01, 1000)
plt.show()


