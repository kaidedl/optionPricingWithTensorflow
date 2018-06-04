import tensorflow as tf
import math
import numpy as np
from numpy.linalg import inv
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
sigmabar=0.2


class HestonLVNp:
    def price(self, n):
        W1 = np.random.randn(nT, n).astype(np.float32)
        W2 = rhobar*np.random.randn(nT, n).astype(np.float32) + rho*W1

        V = np.full((nT, n), V0)
        for i in range(1, nT):
            V[i,:] = np.fabs(V[i-1,:] + kappa * (theta - V[i-1,:])*dt + alpha * np.sqrt(V[i-1,:]) * sqrtDt * W2[i-1,:])

        reg_coeff = np.array([V0, 0, 0]).T
        X = np.ones((3, n))
        logS = np.zeros([1, n])
        S = np.zeros((nT, n))
        for i in range(nT):
            var = V[i,:] * sigmabar*sigmabar / np.maximum(0.0001, np.matmul(X.T, reg_coeff))
            logS = logS + np.sqrt(var) * sqrtDt * W1[i,:] - 0.5*var*dt
            S[i,:] = np.exp(logS)
            X = np.vstack([np.ones(n), S[i,:], np.square(S[i,:])])
            reg_coeff = np.matmul(inv(np.matmul(X, X.T)), np.matmul(X, V[i,:]))

        A = np.mean(S, axis=0)
        P = np.mean(np.maximum(A - K, 0))

        return P

class HestonLVTf:

    def updateV(prevV, currW):
        return tf.abs(prevV + kappa * (theta - prevV)*dt + alpha * tf.sqrt(prevV) * sqrtDt * currW)

    def __init__(self, tf_device_name, n):
        with tf.device(tf_device_name):
            W1t = tf.random_normal((nT, n), dtype=tf.float32)
            W2t = rhobar*tf.random_normal((nT, n), dtype=tf.float32) + rho*W1t

            V0t = tf.fill([1, n], V0)
            Vt = tf.scan(HestonLVTf.updateV, tf.concat([V0t, W2t[:-1,:]], 0))

            def cond(i, _1, _2, _3, _4):
                return i < nT
            
            def body(i, X, reg_coeff, lastLogS, ta):
                condExp = tf.reshape(tf.maximum(0.0001, tf.matmul(X, reg_coeff)), [n])
                var = tf.divide(sigmabar*sigmabar * Vt[i,:], condExp)
                logS = lastLogS + tf.multiply(tf.sqrt(var), sqrtDt * W1t[i,:]) - 0.5*var*dt
                ta = ta.write(i, logS)
                S = tf.exp(logS)
                X = tf.stack([tf.ones(S.shape), S, tf.square(S)], axis=1)
                reg_coeff = tf.matmul(tf.matrix_inverse(tf.matmul(X, X, transpose_a=True)),
                    tf.matmul(X, tf.reshape(Vt[i,:],[n,1]), transpose_a=True))
                return i+1, X, reg_coeff, logS, ta

            ta = tf.TensorArray(dtype=tf.float32, size=nT)
            _0, _1, _2, _3, ta = tf.while_loop(cond, body, [0, tf.ones((n, 3)), tf.constant([[V0], [0], [0]]), tf.zeros(n), ta])

            Ct = ta.stack()
            St = tf.exp(Ct)
            At = tf.reduce_mean(St, axis=0)
            self.Pt = tf.reduce_mean(tf.maximum(At - K, 0))

    def price(self, n):
        p = sess.run(self.Pt)
        return p


if __name__ == "__main__":

    device_name=sys.argv[1]
    print("Heston with local vol", device_name)

    with tf.Session() as sess:
        for n in [10000, 25000, 50000, 100000, 250000]:

            if device_name == "np":
                engine = HestonLVNp()
            elif device_name == "cpu":
                engine = HestonLVTf("/cpu:0", n)
            else:
                engine = HestonLVTf("/gpu:0", n)

            t0 = datetime.now()
            p = engine.price(n)
            t1 = datetime.now()
            print("num sims: {}, price: {}, time: {}".format(n, p, t1-t0))
