# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 00:56:41 2022

@author: shikhar
"""

"""
Created on Mon Aug 29 20:07:07 2022

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
def net_NS( x, t,c_out,s_out):
    cx = tf.gradients(c_out, x)[0]
    ct = tf.gradients(c_out, t)[0]
    st=  tf.gradients(s_out, t)[0]
    cxx = tf.gradients(cx, x)[0]
    ux=0.0003
    uy=0
    alpha=0.01
    adv=ux*cx
    Dd=alpha*ux
    por=0.3
    De=1e-9
    ka=0.0008
    kd=0.0001
    rho_b=2650*(1-por)
    Jd=-(Dd+De)*(cxx)
    print("count",1)
    #breakpoint()
    fc=por*ct+adv+Jd+ka*c_out*por-kd*s_out
    fs=ka*c_out*por-kd*s_out-st
    return fc, fs,Jd

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
weights_s, biases_s = initialize_NN(layers_s)  

#w_fc is 10000 but considering nomralization of loss it will be 10000*(15535/3000)=50000multiplied. then it gives best result

w_ic=tf.Variable(1.0)
w_is=tf.Variable(10.0)
w_dc=tf.Variable(1.0)
w_fc=tf.Variable(10000.0)
w_fs=tf.Variable(10000.0)
w_j=tf.Variable(1.0)


sess=tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

minval=np.array([0,0])
maxval=np.array([1,6000])
tf_dict = {x_dcb: x_lb, t_dcb: t_lb, x_neb: x_rb, t_neb: t_rb, x_i: x_ic, t_i: t_ic, x_f: xx_f, t_f: tt_f }

#tf_dict = { x_i: x_ic, y_i: y_ic, x_f: x_fp, y_f:y_fp, t_f:t_fp}


#c_dcb=neural_net(tf.concat([x_dcb, y_dcb], 1), weights_ib, biases_ib,np.array([0,1.5]),np.array([0,2.5]))
#c_dcb=neural_net(tf.concat([x_dcb, y_dcb], 1), weights_dcb, biases_dcb,np.array([0,1.5]),np.array([0,2.5]))
c_dcb=neural_net(tf.concat([x_dcb, t_dcb], 1), weights_c, biases_c)
c_neb=neural_net(tf.concat([x_neb, t_neb], 1), weights_c, biases_c)
s_neb=neural_net(tf.concat([x_neb, t_neb], 1), weights_s,biases_s)
c_ic=neural_net(tf.concat([x_i, t_i], 1), weights_c, biases_c)
c_f=neural_net(tf.concat([x_f, t_f], 1), weights_c,biases_c)
s_f=neural_net(tf.concat([x_f, t_f], 1),weights_s,biases_s)
s_ic=neural_net(tf.concat([x_i, t_i], 1), weights_s,biases_s)
fc, _,_=net_NS(x_f,t_f,c_f,s_f)
_,fs,_=net_NS(x_f,t_f,c_f,s_f)
_,_,j=net_NS(x_neb,t_neb,c_neb,s_neb)

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


#loss = 36*tf.reduce_sum(abs(c_ic))+tf.reduce_sum(abs(c_dcb-c_dc))+tf.reduce_sum(abs(f))+tf.reduce_sum(abs(j))
#tf.reduce_sum(abs(f)).eval(feed_dict=tf_dict,session=sess)
#add square 
loss = w_ic*tf.reduce_sum(tf.square(c_ic))+w_is*tf.reduce_sum(tf.square(s_ic))+w_dc*tf.reduce_sum(tf.square(c_dcb-c_dc))+w_fc*tf.reduce_sum(tf.square(fc))+w_fs*tf.reduce_sum(tf.square(fs))+w_j*tf.reduce_sum(tf.square(j))

#loss = tf.reduce_sum(tf.square(c_ic))+10*tf.reduce_sum(tf.square(s_ic))+tf.reduce_sum(tf.square(c_dcb-c_dc))+100*tf.reduce_sum(tf.square(fc))+10000*tf.reduce_sum(tf.square(fs))+tf.reduce_sum(tf.square(j))


loss_max=10000-loss



optimizer_Adam = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.005, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam')
optimizer_Adam_max = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam')



train_op_Adam = optimizer_Adam.minimize(loss,var_list=[weights_c,biases_c,weights_s,biases_s])  


train_op_Adam_max = optimizer_Adam_max.minimize(loss_max,var_list=[w_ic,w_dc,w_is,w_fc,w_fs,w_j])


sess=tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)   

w_icb=[]
w_isb=[]
w_dcb=[]
w_fcb=[]
w_fsb=[]
w_jb=[]


losstot=[]
nIter=20000
for it in range(nIter):
    sess.run(train_op_Adam, tf_dict)
    print(it)
    #fv=f.eval(feed_dict=tf_dict,session=sess)
    #print(it,loss_value,tf.reduce_sum(tf.square(c_dcb-c_dc)).eval(feed_dict=tf_dict,session=sess),tf.reduce_sum(tf.square(f)).eval(feed_dict=tf_dict,session=sess))
    loss_value=loss.eval(feed_dict=tf_dict,session=sess)
    if it%10==0:
        losstot.append(loss_value)
        w_icb.append(w_ic.eval(session=sess))
        w_isb.append(w_is.eval(session=sess))
        w_dcb.append(w_dc.eval(session=sess))
        w_fcb.append(w_fc.eval(session=sess))
        w_fsb.append(w_fs.eval(session=sess))
        w_jb.append(w_j.eval(session=sess))
    if loss_value>1:
        if it%20==0:
            sess.run(train_op_Adam_max, tf_dict)
    print(it,loss_value)
    if abs(loss_value)<0.05:
        break


x1=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
t1=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])



tf1 = {x1: x_ic, t1: t1np}


aa=np.array([x for x in range(3001)])
aa=aa[:,None]
#c=neural_net(tf.concat([x1, t1], 1), weights, biases)
cneb=c_neb.eval(feed_dict=tf_dict,session=sess)
cdcb=c_dcb.eval(feed_dict=tf_dict,session=sess)
plt.plot(aa[:,:], cneb[:,:], marker='.', label="actual")
plt.plot(aa[:,:], cdcb[:,:], 'r', label="actual")
plt.plot(aa[:,:], c_dc[:,:], 'g', label="actual")

aa=np.array([x for x in range(934)])
plt.plot(aa, [i/3000 for i in losstot], 'black', label="actual",linewidth=4)
plt.rcParams['font.size'] = 18

aa=np.array([x for x in range(934)])
plt.plot(aa, (w_icb-w_icb[0]), label="w_ic", linewidth=4)
plt.plot(aa, w_isb-w_isb[0], label="w_is",linewidth=4)
plt.plot(aa, (w_dcb-w_dcb[0]), label="w_dc",linewidth=4)
plt.plot(aa, 5*(w_fcb-w_fcb[0]), label="w_fc",linewidth=4)
plt.plot(aa, 5*(w_fsb-w_fsb[0]), label="w_fs",linewidth=4)
plt.plot(aa, w_jb-w_jb[0], label="w_j",linewidth=4)
plt.rcParams['font.size'] = 18




plt.legend(loc="upper left", mode = "expand", ncol = 3)
plt.ylim(0, 50.0)
plt.show()

t1np=np.array([x for x in range(6001)])
t1np=t1np[:,None]
x1np = np.empty((6001,1))
x1np.fill(1)
tf1 = {x1: x1np, t1: t1np}
cat=neural_net(tf.concat([x1, t1], 1), weights_c, biases_c)
catp=cat.eval(feed_dict=tf1,session=sess)
plt.plot(t1np, catp, 'g', label="actual")


#how to save

cat6000 = pd.DataFrame(catp6000)
cat6000.to_csv(r'C:\Users\shikhar\PycharmProjects\mesh\autoencoder-for-denoising-coarser-mesh-based-numerical-solution\def\forward result\cat6000.csv')


"""
saver = tf.compat.v1.train.Saver()
saver.save(sess, 'modelbkl',global_step=1000)

sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.import_meta_graph('modelbkl-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

"""
"""
import matplotlib.pyplot as plt
plot = plt.pcolormesh(output[,], y, Z, cmap='RdBu', shading='flat')

from scipy.interpolate import NearestNDInterpolator
interp = LinearNDInterpolator(list(zip(output[:,0], output[:,1],output[:,2])), output[:,3])


x1, y1 = np.meshgrid(output[:,0],output[:,1])
t_plot = np.empty(len(x1))
t_plot.fill(3600)
c_plot=interp(x1,y1,t_plot)
import matplotlib.pyplot as plt
plt.contourf(output[:,0],output[:,1],c_plot)
"""