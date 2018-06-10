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


class HestonTf:

    def updateV(prevV, currW):
        return tf.abs(prevV + kappa * (theta - prevV) * dt + alpha * tf.sqrt(prevV) * sqrtDt * currW)

    def __init__(self, tf_device_name):
        with tf.device(tf_device_name):
            with tf.name_scope("spotProcess"):
                self.n = tf.placeholder(tf.int32, name="numSims")
                W1rt = tf.random_normal((nT, self.n), dtype=tf.float32, name="W1")
                W2rt = tf.random_normal((nT, self.n), dtype=tf.float32, name="W2")
                W1t = tf.identity(W1rt, name="wienerSpot")
                W2t = tf.add(rhobar*W2rt, rho*W1rt, name="wienerVar")

                with tf.name_scope("varProcess"):
                    V0t = tf.fill([1, self.n], V0)
                    Vt = tf.scan(HestonTf.updateV, tf.concat([V0t, W2t[:-1,:]], 0))

                Rt = tf.add(tf.multiply(tf.sqrt(Vt), sqrtDt*W1t), -0.5*Vt*dt, name="returns")
                Ct = tf.cumsum(Rt, axis=0, name="cumReturns")
                St = tf.exp(Ct, name="spot")

            with tf.name_scope("payoff"):
                At = tf.reduce_mean(St, axis=0, name="asianing")
                payofft = tf.maximum(At - K, 0)

            self.Pt = tf.reduce_mean(payofft, name="price")

    def price(self, n):
        writer = tf.summary.FileWriter("graphs", sess.graph)
        p = sess.run(self.Pt, feed_dict={self.n: n})
        writer.close()
        return p


if __name__ == "__main__":

    device_name = sys.argv[1]
    print("Heston", device_name)

    if device_name == "cpu":
        engine = HestonTf("/cpu:0")
    else:
        engine = HestonTf("/gpu:0")

    with tf.Session() as sess:
        for n in [10000, 25000, 50000, 100000, 250000]:

            t0 = datetime.now()
            p = engine.price(n)
            t1 = datetime.now()
            print("num sims: {}, price: {}, time: {}".format(n, p, t1-t0))
