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
K2=1.05
sigma=0.2
rho=-0.5
rhobar=math.sqrt(1-rho*rho)
V0=0.04
theta=0.04
kappa=1.0
alpha=0.5


class HestonNp:
    def priceSingle(self, n, V0, alpha):
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

    def price(self, n, d):
        P = self.priceSingle(n, V0, alpha)
        if d == 0:
            PupV0 = self.priceSingle(n, V0+0.001, alpha)
            return (PupV0-P)/0.001
        else:
            PupV0 = self.priceSingle(n, V0+0.001, alpha)
            PupAlpha = self.priceSingle(n, V0, alpha+0.001)
            return (PupV0-P)/0.001, (PupAlpha-P)/0.001


class HestonTf:

    def __init__(self, tf_device_name):

        with tf.device(tf_device_name):
            self.n = tf.placeholder(tf.int32)
            W1t = tf.random_normal((nT, self.n), dtype=tf.float32)
            W2t = rhobar*tf.random_normal((nT, self.n), dtype=tf.float32) + rho*W1t

            def updateV(prevV, currW):
                return tf.abs(prevV + kappa * (theta - prevV) * dt + alphav * tf.sqrt(prevV) * sqrtDt * currW)

            alphav = tf.Variable(alpha)
            V0v = tf.Variable(V0)
            V0t = tf.fill([1, self.n], V0v)
            Vt = tf.scan(updateV, tf.concat([V0t, W2t[:-1,:]], 0))

            Rt = tf.multiply(tf.sqrt(Vt), sqrtDt*W1t) - 0.5*Vt*dt
            Ct = tf.cumsum(Rt, axis=0)
            St = tf.exp(Ct)
            At = tf.reduce_mean(St, axis=0)
            Pt = tf.reduce_mean(tf.maximum(At - K, 0))
            P2t = tf.reduce_mean(tf.maximum(At - K2, 0))
            self.Deriv0 = tf.gradients(Pt, [V0v])
            self.Deriv1 = tf.gradients(Pt, [V0v, alphav])
            self.Deriv2 = tf.gradients([Pt, P2t], [V0v, alphav])
            self.init = tf.global_variables_initializer()

    def price(self, n, d):
        self.init.run()
        return sess.run({0: self.Deriv0, 1: self.Deriv1, 2: self.Deriv2}[d], feed_dict={self.n: n})


if __name__ == "__main__":

    device_name = sys.argv[1]
    print("Heston auto diff", device_name)

    if device_name == "np":
        engine = HestonNp()
    elif device_name == "cpu":
        engine = HestonTf("/cpu:0")
    else:
        engine = HestonTf("/gpu:0")

    deriv = int(sys.argv[2])
    with tf.Session() as sess:
        for n in [10000, 25000, 50000, 100000]:

            t0 = datetime.now()
            d = engine.price(n, deriv)
            t1 = datetime.now()
            print("num sims: {}, derivs: {}, time: {}".format(n, d, t1-t0))
