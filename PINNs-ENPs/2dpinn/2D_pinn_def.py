# -*- coding: utf-8 -*-
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


data = pd.read_csv("2Ddata.csv")

maindata=data.iloc[:,:].values

"""
XX_dc = np.tile(boundary_dc[:,0:1], (1,len(t_dc))) # N x T
YY_dc = np.tile(boundary_dc[:,1:2], (1,len(t_dc))) # N x 
TT_dc = np.tile(t_dc, (1,len(boundary_dc))).T # N x T

x_dc = XX_dc.flatten()[:,None] # NT x 1
y_dc = YY_dc.flatten()[:,None] # NT x 1
t_dc = TT_dc.flatten()[:,None] # NT x 1
"""


layers=[3, 100,100, 100, 1]


time=np.arange(start=0, stop=36001, step=600)

N_f=5000

minval=np.array([0,0,0])
maxval=np.array([10,4,36000])

XX_f = np.tile(maindata[:,0:1], (1,len(time))) # N x T
YY_f = np.tile(maindata[:,1:2], (1,len(time))) # N x 
TT_f = np.tile(time, (1,len(maindata[:,0:1]))).T # N x T

x_fp = XX_f.flatten()[:,None] # NT x 1
y_fp = YY_f.flatten()[:,None] # NT x 1
t_fp = TT_f.flatten()[:,None] # NT x 1
c_fp=maindata[:,2:].flatten()[:,None]


"""
c=maindata[:,2:].flatten()[:,None]
cc=maindata[:,2:]
cct=np.tranpose(cc)
cct=np.transpose(cc)
"""
def sampling(minval,maxval,var,N):
    X = minval + (maxval-minval)*lhs(var, N)
    return X

data_up=x_up=sampling(np.array([0,0]),np.array([10,6000]),2,1000)
x_up=data_up[:,0:1]
t_up=data_up[:,1:2]
y_up = np.empty((len(x_up),1))
y_up.fill(4)


data_down=sampling(np.array([0,0]),np.array([10,6000]),2,1000)
x_down=data_down[:,0:1]
t_down=data_down[:,1:2]
y_down=np.zeros((len(x_down),1))

data_right=sampling(np.array([0,0]),np.array([4,6000]),2,1000)
y_right=data_right[:,0:1]
t_right=data_right[:,1:2]
x_right = np.empty((len(y_right),1))
x_right.fill(10)

x_neu=np.concatenate((x_up,x_down,x_right),0)
y_neu=np.concatenate((y_up,y_down,y_right),0)
t_neu=np.concatenate((t_up,t_down,t_right),0)

data_dc=sampling(np.array([1.5,0]),np.array([2.5,6000]),2,3000)
y_dc=data_dc[:,0:1]
t_dc=data_dc[:,1:2]
x_dc=np.zeros((len(y_dc),1))

data_neudc1=sampling(np.array([0,0]),np.array([1.5,6000]),2,500)
y_neudc1=data_neudc1[:,0:1]
t_neudc1=data_neudc1[:,1:2]
x_neudc1=np.zeros((len(y_neudc1),1))

data_neudc2=sampling(np.array([2.5,0]),np.array([4,6000]),2,500)
y_neudc2=data_neudc2[:,0:1]
t_neudc2=data_neudc2[:,1:2]
x_neudc2=np.zeros((len(y_neudc2),1))

x_neudc=np.concatenate((x_neudc1,x_neudc2),0)
y_neudc=np.concatenate((y_neudc1,y_neudc2),0)
t_neudc=np.concatenate((t_neudc1,t_neudc2),0)

data_ic=sampling(np.array([0,0]),np.array([10,4]),2,3000)
x_ic=data_ic[:,0:1]
y_ic=data_ic[:,1:2]
t_ic=np.zeros((len(y_ic),1))


c_dc=1/(1 + np.exp(-0.1*(t_dc-500)))
"""
"""


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

def neural_net( X, weights, biases):
    num_layers = len(weights) + 1
    H = (X-minval)/(maxval-minval)
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
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
def net_NS( x, y, t,c_out, por, alpha):
    cx = tf.gradients(c_out, x)[0]
    cy = tf.gradients(c_out, y)[0]
    ct = tf.gradients(c_out, t)[0]
    cxx = tf.gradients(cx, x)[0]
    cyy = tf.gradients(cy, y)[0]
    cxy = tf.gradients(cx, y)[0]
    cyx = tf.gradients(cy, x)[0]
    ux=0.0001
    uy=0

    adv=ux*cx+uy*cy
    Dd=alpha*ux

    De=1e-9
    Jd=-(Dd+De)*(cxx+cyy+cxy+cyx)
    print("count",1)
    #breakpoint()
    f=por*ct+adv+Jd
    return f
def callback(loss):
    print('Loss: %.3e' % (loss))
#X = np.concatenate([x_train, y_train, t_train], 1)
   

#DEFINE THIS ONLY FOR INPUT DATA, IE FEATURES AND TAGER WHICH Has to be minimised
x_i=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
y_i=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])
t_i=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])
c_dct=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])

x_dcb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
y_dcb=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])
t_dcb=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])

x_neb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
y_neb=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])
t_neb=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])

x_nebdc=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
y_nebdc=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])
t_nebdc=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])


x_f=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
y_f=tf.compat.v1.placeholder(tf.float32, shape=[None, y_ic.shape[1]])
t_f=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])


weights, biases = initialize_NN(layers)  

sess=tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

alpha=tf.Variable(0.005)
por=tf.Variable(0.5)

minval=np.array([0,0,0])
maxval=np.array([10,4,6000])
tf_pred = {x_i: x_ic, y_i: y_ic, t_i: t_ic,x_dcb: x_dc, y_dcb: y_dc, t_dcb: t_dc, x_neb: x_neu, y_neb: y_neu, t_neb: t_neu, x_nebdc: x_neudc, y_nebdc: y_neudc, t_nebdc: t_neudc, x_i: x_ic, y_i: y_ic, t_i: t_ic, x_f: x_fp, y_f:y_fp, t_f:t_fp}

#tf_dict = { x_i: x_ic, y_i: y_ic, x_f: x_fp, y_f:y_fp, t_f:t_fp}


#c_dcb=neural_net(tf.concat([x_dcb, y_dcb], 1), weights_ib, biases_ib,np.array([0,1.5]),np.array([0,2.5]))
#c_dcb=neural_net(tf.concat([x_dcb, y_dcb], 1), weights_dcb, biases_dcb,np.array([0,1.5]),np.array([0,2.5]))
c_dcb=neural_net(tf.concat([x_dcb, y_dcb, t_dcb], 1), weights, biases)
c_neb=neural_net(tf.concat([x_neb, y_neb, t_neb], 1), weights, biases)
c_nebdc=neural_net(tf.concat([x_nebdc, y_nebdc, t_nebdc], 1), weights, biases)
c_ic=neural_net(tf.concat([x_i, y_i, t_i], 1), weights, biases)
c_f=neural_net(tf.concat([x_f, y_f, t_f], 1), weights,biases)
f=net_NS(x_f,y_f,t_f,c_f, por, alpha)

loss = 10*tf.reduce_sum(tf.square(c_ic))+100*tf.reduce_sum(tf.square(c_f-c_fp))+tf.reduce_sum(tf.square(c_neb))+tf.reduce_sum(tf.square(c_nebdc))+tf.reduce_sum(tf.square(c_dcb-c_dct))


#loss = tf.reduce_sum(abs(c_dcb-1))
#loss = tf.reduce_sum(c_dcb-1)

optimizer_Adam = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.005, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam')

train_op_Adam = optimizer_Adam.minimize(loss,var_list=[weights,biases,por,alpha])  

sess=tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)   
bs=20740
tot=20740
nIter=20000

for it in range(nIter):
    print("............")
    for i in range(0,tot,bs):
        tf_dict = {x_i: x_ic, y_i: y_ic, t_i: t_ic, x_dcb: x_dc, y_dcb: y_dc, t_dcb: t_dc,c_dct:c_dc, x_neb: x_neu, y_neb: y_neu, t_neb: t_neu, x_nebdc: x_neudc, y_nebdc: y_neudc, t_nebdc: t_neudc, x_f: x_fp[i:i+bs,:], y_f:y_fp[i:i+bs,:], t_f:t_fp[i:i+bs,:]}
        sess.run(train_op_Adam, tf_dict)
        loss_value=loss.eval(feed_dict=tf_dict,session=sess)
        por_value=por.eval(session=sess)
        print(it,i,loss_value,por_value)
    if it%20==0:
        c_out=(c_f).eval(feed_dict=tf_pred,session=sess)
        output=np.concatenate((x_fp,y_fp,t_fp,c_out),1)
        df = pd.DataFrame(output)
        df.to_csv("output_bm_in.csv")
    if abs(loss_value)<0.1:
        break    

"""
    if it%50==0:
        c_out=(c_f).eval(feed_dict=tf_pred,session=sess)
        output=np.concatenate((x_fp,y_fp,t_fp,c_out),1)
        df = pd.DataFrame(output)
        df.to_csv("output_bm.csv")
    if abs(loss_value)<0.1:
        break
  """  
import tensorflow_probability as tfp

tfp.optimizer.lbfgs_minimize(sess,
                                 tf_dict,
                                 [loss])


c_out=(c_f).eval(feed_dict=tf_dict,session=sess)
output=np.concatenate((x_fp,y_fp,t_fp,c_out),1)
import pandas as pd
df = pd.DataFrame(output)
df.to_csv("output.csv")
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
