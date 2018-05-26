import tensorflow as tf
import math
import numpy as np
import sys
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

nT = 365
T = 1.0
dt = T/nT
sqrtDt = math.sqrt(dt)
K = 1
sigma = 0.2


class BlackScholesNp:
    def price(self, n):
        R = sigma*sqrtDt*np.random.randn(nT, n) - 0.5*sigma*sigma*dt
        C = np.cumsum(R, axis=0)
        S = np.exp(C)
        A = np.mean(S, axis=0)
        P = np.mean(np.maximum(A - K, 0))
        #S2 = S*S
        #A2 = np.mean(S2, axis=0)
        #P2 = np.mean(np.maximum(A2 - K, 0))
        return P


class BlackScholesTf:

    def __init__(self, tf_device_name):
        with tf.device(tf_device_name):
            self.n = tf.placeholder(tf.int32)
            Rt = sigma*sqrtDt*tf.random_normal((nT, self.n)) - 0.5*sigma*sigma*dt
            Ct = tf.cumsum(Rt, axis=0)
            St = tf.exp(Ct)
            At = tf.reduce_mean(St, axis=0)
            self.Pt = tf.reduce_mean(tf.maximum(At - K, 0))
            #S2t = St*St
            #A2t = tf.reduce_mean(S2t, axis=0)
            #self.P2t = tf.reduce_mean(tf.maximum(A2t - K, 0))

    def price(self, n):
        p = sess.run(self.Pt, feed_dict={self.n: n})
        #p = sess.run([self.Pt,self.P2t], feed_dict={self.n: n})
        return p


if __name__ == "__main__":

    device_name = sys.argv[1]
    print("Black Scholes " + device_name)

    with tf.Session() as sess:
        for n in [10000, 25000, 50000, 100000, 250000]:
            if device_name == "np":
                engine = BlackScholesNp()
            elif device_name=="cpu":
                engine = BlackScholesTf("/cpu:0")
            else:
                engine = BlackScholesTf("/gpu:0")

            t0 = datetime.now()
            p = engine.price(n)
            t1 = datetime.now()
            print("num sims: {}, price: {}, time: {}".format(n, p, t1-t0))
