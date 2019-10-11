from os import listdir
from PIL import Image
from matplotlib import pyplot as plt
import random as rand
import numpy as np
import dataloader as dl
import pca

dim = 5             #dimensions
k = 1               #categories
w = np.full(k, np.zeros(dim))   #vector of weight vectors
train_set = []
holdout_set = []
test_set = [] 

def batch_gd_sm(T, a):
    """
    Args:
        T: trials/epochs
        a: learning rate
    """
    global w
    global k
    global train_set
    global holdout_set
    for _ in range(T):
        for i in range(k):
            w[i] = np.add(w[i], np.multiply(a, (sigma(train_set, lambda p: gradient_sm(p[0], p[1], w, i)))))
    return w

def softmax_activation(x, k, w):
    """
    Args:
        x: input vector
        k: class
        w: vector of weight vectors
    Return:
        y_k
    """
    exp_k = np.power(np.e, np.dot(w[k],x))
    sigma_exp_k = sigma(w, lambda n: np.power(np.e, np.dot(n, x)))
    return exp_k / sigma_exp_k


def sigma(s, f):
    """
    Args:
        s: set to be summated over
        f: function that takes in set's tuple elements
    """
    global dim
    total = np.zeros(dim)
    for i in range(len(s)):
        total = np.add(total, f(s[i]))
    return total

def gradient_sm(x, ts, w, k):
    """
    Args:
        x: input vector
        ts: teaching signal
        w: vector of weight vectors
        k: class
    Returns:
        Delta(E(w)_k)
    """
    return np.multiply((ts - softmax_activation(x, k, w)), x)

def main():
    global dim
    global train_set
    global holdout_set
    global test_set

    images, labels = dl.load_data()
    data = zip(images, labels)
    ht_m_data = [(i, 1) for i, l in data if 'ht' in l.strip('pgm')]
    ht_m_data += [(i, 0) for i, l in data if 'm' in l.strip('pgm')]
    for _ in range(int(len(ht_m_data) * .6)):
        random_index = rand.randint(0, len(ht_m_data)-1)
        train_set.append(ht_m_data[random_index])
        del ht_m_data[random_index]
    for _ in range(int(len(ht_m_data) * .5)):
        random_index = rand.randint(0, len(ht_m_data)-1)
        holdout_set.append(ht_m_data[random_index])
        del ht_m_data[random_index]
    test_set = ht_m_data

    pcaConv = pca.PCA(dim)
    pcaConv.fit(np.array([i for i, l in train_set][:]))
    train_set = [(pcaConv.transform(i)[0], ts) for i, ts in train_set]
    holdout_set = [(pcaConv.transform(i)[0], ts) for i, ts in holdout_set]
    test_set = [(pcaConv.transform(i)[0], ts) for i, ts in test_set]
    print(batch_gd_sm(10, .01)) 





if __name__ == '__main__':
	main()
