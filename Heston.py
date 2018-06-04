import tensorflow as tf
import math
import numpy as np
import sys
from datetime import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

nT=365
T=1.0
dt=T/nT
sqrtDt=math.sqrt(dt)
K=1
sigma=0.2
rho=-0.5
rhobar=math.sqrt(1-rho*rho)
V0=0.04
theta=0.04
kappa=1
alpha=0.5


class HestonNp:
    def price(self, n):
        W1 = np.random.randn(nT, n).astype(np.float32)
        W2 = rhobar*np.random.randn(nT, n).astype(np.float32) + rho*W1

        V = np.full((nT, n), V0)
        for i in range(1, nT):
            V[i,:] = np.fabs(V[i-1,:] + kappa * (theta - V[i-1,:])*dt + alpha * np.sqrt(V[i-1,:]) * sqrtDt * W2[i-1,:])
     
        R = np.sqrt(V)*sqrtDt*W1 - 0.5*V*dt
        C = np.cumsum(R, axis=0)
        S = np.exp(C)
        A = np.mean(S, axis=0)
        P = np.mean(np.maximum(A - K, 0))
        return P


class HestonTf:

    def updateV(prevV, currW):
        return tf.abs(prevV + kappa * (theta - prevV) * dt + alpha * tf.sqrt(prevV) * sqrtDt * currW)

    def __init__(self, tf_device_name):
        with tf.device(tf_device_name):
            self.n = tf.placeholder(tf.int32)
            W1t = tf.random_normal((nT, self.n), dtype=tf.float32)
            W2t = rhobar*tf.random_normal((nT, self.n), dtype=tf.float32) + rho*W1t

            V0t = tf.fill([1, self.n], V0)
            Vt = tf.scan(HestonTf.updateV, tf.concat([V0t, W2t[:-1,:]], 0))

            Rt = tf.multiply(tf.sqrt(Vt), sqrtDt*W1t) - 0.5*Vt*dt
            Ct = tf.cumsum(Rt, axis=0)
            St = tf.exp(Ct)
            At = tf.reduce_mean(St, axis=0)
            self.Pt = tf.reduce_mean(tf.maximum(At - K, 0))

    def price(self, n):
        p = sess.run(self.Pt, feed_dict={self.n: n})
        return p


if __name__ == "__main__":

    device_name = sys.argv[1]
    print("Heston", device_name)

    if device_name == "np":
        engine = HestonNp()
    elif device_name == "cpu":
        engine = HestonTf("/cpu:0")
    else:
        engine = HestonTf("/gpu:0")

    with tf.Session() as sess:
        for n in [10000, 25000, 50000, 100000, 250000]:

            t0 = datetime.now()
            p = engine.price(n)
            t1 = datetime.now()
            print("num sims: {}, price: {}, time: {}".format(n, p, t1-t0))
