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
dt_lv = 7.0 / 365.0
dx_lv = 0.5

t_lv = [i * dt_lv for i in range(54)]
x_lv = [-10 + i * dx_lv for i in range(42)]
sigma_lv = [[0.2 for i in range(42)] for j in range(54)]

class LocalVolNp:
    def __init__(self):
        self.x_lv = np.array(x_lv)
        self.sigma_lv = np.array(sigma_lv).astype(np.float32)

    def interpolate_lv(self, t, R):
        index_t = int(t // dt_lv)
        indices_x = (R // dx_lv).astype(np.int32) + 21

        sigma_td = (self.x_lv[indices_x+1]-R) / dx_lv * self.sigma_lv[index_t, indices_x] + \
                   (R-self.x_lv[indices_x]) / dx_lv * self.sigma_lv[index_t, indices_x+1]
        sigma_tu = (self.x_lv[indices_x+1]-R) / dx_lv * self.sigma_lv[index_t+1, indices_x] + \
                   (R-self.x_lv[indices_x]) / dx_lv * self.sigma_lv[index_t+1, indices_x+1]

        sigma_t = (t_lv[index_t+1]-t) / (t_lv[index_t+1]-t_lv[index_t]) * sigma_td + \
                  (t-t_lv[index_t]) / (t_lv[index_t+1]-t_lv[index_t]) * sigma_tu
        return sigma_t

    def price(self, n):
        W = np.random.randn(nT, n).astype(np.float32)

        R = np.zeros((nT, n))
        for i in range(1, nT):
            sigma_t = self.interpolate_lv(i / 365.0, R[i-1,:])
            R[i, :] = sqrtDt*sigma_t*W[i-1,:] - 0.5*np.square(sigma_t)*dt

        C = np.cumsum(R, axis=0)
        S = np.exp(C)
        A = np.mean(S, axis=0)
        P = np.mean(np.maximum(A - K, 0))
        return P


class LocalVolTf:

    def __init__(self, tf_device_name):
        with tf.device(tf_device_name):
            self.x_lv = tf.constant(x_lv, dtype=tf.float32)
            self.sigma_lv = tf.constant(sigma_lv, dtype=tf.float32)

            self.n = tf.placeholder(tf.int32)
            Wt = tf.random_normal((nT, self.n), dtype=tf.float32)

            def updateLVAndR(prevR, currW):
                self.t += 1.0/365.0
                index_t = int(self.t // dt_lv)
                indices_x = tf.cast(prevR / dx_lv, dtype=tf.int32) + 21

                sigma_td = (tf.gather(self.x_lv, indices_x+1)-prevR) / dx_lv * tf.gather(self.sigma_lv[index_t, :], indices_x) + \
                           (prevR-tf.gather(self.x_lv, indices_x)) / dx_lv * tf.gather(self.sigma_lv[index_t, :], indices_x+1)
                sigma_tu = (tf.gather(self.x_lv, indices_x+1)-prevR) / dx_lv * tf.gather(self.sigma_lv[index_t+1, :], indices_x) + \
                           (prevR-tf.gather(self.x_lv, indices_x)) / dx_lv * tf.gather(self.sigma_lv[index_t+1, :], indices_x+1)

                sigma_t = (t_lv[index_t+1]-self.t) / (t_lv[index_t+1]-t_lv[index_t]) * sigma_td + \
                          (self.t-t_lv[index_t]) / (t_lv[index_t+1]-t_lv[index_t]) * sigma_tu

                return sqrtDt*sigma_t*currW - 0.5*np.square(sigma_t)*dt
            self.t = 0
            R0t = tf.zeros([1, self.n])
            Rt = tf.scan(updateLVAndR, tf.concat([R0t, Wt], 0))

            Ct = tf.cumsum(Rt, axis=0)
            St = tf.exp(Ct)
            At = tf.reduce_mean(St, axis=0)
            self.Pt = tf.reduce_mean(tf.maximum(At - K, 0))

    def price(self, n):
        p = sess.run(self.Pt, feed_dict={self.n: n})
        return p


if __name__ == "__main__":

    device_name = sys.argv[1]
    print("Local Vol", device_name)

    if device_name == "np":
        engine = LocalVolNp()
    elif device_name == "cpu":
        engine = LocalVolTf("/cpu:0")
    else:
        engine = LocalVolTf("/gpu:0")

    with tf.Session() as sess:
        for n in [10000, 25000, 50000, 100000, 250000]:
            t0 = datetime.now()
            p = engine.price(n)
            t1 = datetime.now()
            print("num sims: {}, price: {}, time: {}".format(n, p, t1-t0))
