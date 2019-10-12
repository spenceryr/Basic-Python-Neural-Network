from os import listdir
from PIL import Image
from matplotlib import pyplot as plt
import random as rand
import numpy as np
import dataloader as dl
import pca

test_single = input("Single test?(y/n): ").lower()
test_single = test_single == 'y' or test_single == 'yes'
test_learning_rates = None
learning_rates = []
k_pcas = None
if not test_single:
    test_learning_rates = input("Testing learning rates?(y/n): ").lower()
    test_learning_rates = test_learning_rates == 'y' or test_learning_rates == 'yes'
    learning_rates = [.001, .08, 4] if test_learning_rates else [.01, .02, .03, .1]
    k_pcas = 8 if test_learning_rates else [1,4,8,16]
else:
    learning_rates = .5
    k_pcas = 8

c = 6               #categories
train_set = []
holdout_set = []
test_set = [] 
epochs = 20
holdout_errors = []
train_errors = []

def unit_vector(i, d):
    """
    Args:
        i: index of vector to be set to 1
        d: dimension of vector
    Returns:
        vector with all zeros, except for the ith element, which is 1
    """
    v = np.zeros(d)
    v[i] = 1
    return v
one_hot_dict = {
    "a": unit_vector(0, c),
    "d": unit_vector(1, c),
    "f": unit_vector(2, c),
    "h": unit_vector(3, c),
    "m": unit_vector(4, c),
    "s": unit_vector(5, c),
}

def batch_gd_sm(T, a, dim):
    """
    Args:
        T: trials/epochs
        a: learning rate
    """
    global c
    global train_set
    global holdout_set
    global holdout_errors
    global train_errors

    w = [np.zeros(dim) for _ in range(c)]   #list of weight vectors
    minError = ce_error(w, holdout_set)
    holdout_errors[0].append(minError)
    train_errors[0].append(ce_error(w, train_set))
    for e in range(T):
        v = [np.zeros(dim) for _ in range(c)]
        for i in range(c):
            v[i] = np.add(w[i], np.multiply(a, (sigma(train_set, lambda p: gradient_sm(p[0], one_hot(p[1])[i], w, i)))))
        curError = ce_error(v, holdout_set)
        if curError < minError:
            w = v
            minError = curError
        holdout_errors[e+1].append(curError)
        train_errors[e+1].append(ce_error(w, train_set))
    return w

def ce_error(w, s):
    """
    Args:
        w: list of weight vectors
        s: set to test error on
    Returns:
        Average error
    """
    global c
    E = -1 * sigma(s, lambda p: sigma([k for k in range(c)], lambda k: one_hot(p[1])[k]*np.log(softmax_activation(p[0], k, w))))
    return (E / (len(s)*c))


def softmax_activation(x, k, w):
    """
    Args:
        x: input vector
        k: class
        w: list of weight vectors
    Return:
        y_k
    """
    exp_k = np.power(np.e, np.dot(w[k],x[0]))
    sigma_exp_k = sigma(w, lambda n: np.power(np.e, np.dot(n,x[0])))
    return exp_k / sigma_exp_k


def sigma(s, f):
    """
    Args:
        s: set to be summated over
        f: function that takes in set's tuple elements
    """
    total = f(s[0])
    for i in range(len(s) - 1):
        total += f(s[i + 1])
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

def one_hot(label):
    """
    Args:
        label: label to find the one hot encoding of
    Returns:
        the unit vector associated with this label
    """
    for l in one_hot_dict:
        if l in label.strip('pgm'):
            return one_hot_dict[l]

def main():

    images, labels = dl.load_data()
    data = zip(images, labels)
    slice_data = [(i, l) for i, l in data if 'ht' not in l.strip('pgm') and 'n' not in l.strip('pgm')]
    if test_single:
        single_test(slice_data)
    else:
        if test_learning_rates:
            learning_rate_test(slice_data)
        else:
            pca_test(slice_data)


def single_test(data):
    global train_errors
    global holdout_errors
    global epochs
    fig, ax_list = plt.subplots(nrows=1, ncols=1, figsize=(15,6), sharex=True)
    fig.suptitle("Average Holdout and Training Errors", y=1)
    fig.tight_layout()
    fig.subplots_adjust(top=.85, wspace=.3)

    train_errors = [[] for _ in range(epochs + 1)]
    holdout_errors = [[] for _ in range(epochs + 1)]
    percent_correct = []

    percent_correct = train_model(data, k_pcas, learning_rates, random=False)

    train_mean, train_std = (list(map(np.mean, train_errors)), list(map(np.std, train_errors)))
    holdout_mean, holdout_std = (list(map(np.mean, holdout_errors)), list(map(np.std, holdout_errors)))
    print(train_mean)
    print(holdout_mean)

    graph = ax_list
    graph.set_title("Learning Rate: " + str(learning_rates))
    graph.set_ylabel("Cross Entropy Loss")
    graph.set_xlabel("# of Epochs")
    graph.errorbar(list(range(epochs + 1)), holdout_mean, holdout_std, linestyle='-', color='red', marker='o', label="holdout")
    graph.errorbar(list(range(epochs + 1)), train_mean, train_std, linestyle='-', color='blue', marker='o', label="train")
    graph.fill_between(list(range(epochs + 1)), np.asarray(holdout_mean) + np.asarray(holdout_std), np.asarray(holdout_mean) - np.asarray(holdout_std), facecolor='r',alpha=0.5)
    graph.fill_between(list(range(epochs + 1)), np.asarray(train_mean) + np.asarray(train_std), np.asarray(train_mean) - np.asarray(train_std), facecolor='b',alpha=0.5)
    mean = np.mean(percent_correct)
    std_dev = np.std(percent_correct)
    print("PCA={}, average % correct= {} ({})".format(k_pcas, mean, std_dev))
    plt.savefig("single_test.png", bbox_inches='tight')

def learning_rate_test(data):
    global train_errors
    global holdout_errors
    global epochs
    fig, ax_list = plt.subplots(nrows=1, ncols=3, figsize=(15,6), sharex=True)
    fig.suptitle("Average Holdout and Training Errors", y=1)
    fig.tight_layout()
    fig.subplots_adjust(top=.85, wspace=.3)
    percent_correct = []
    for lr in learning_rates:
        train_errors = [[] for _ in range(epochs + 1)]
        holdout_errors = [[] for _ in range(epochs + 1)]

        pc = train_model(data, k_pcas, lr, random=False)

        percent_correct.append(pc)
        train_mean, train_std = (list(map(np.mean, train_errors)), list(map(np.std, train_errors)))
        holdout_mean, holdout_std = (list(map(np.mean, holdout_errors)), list(map(np.std, holdout_errors)))

        graph = ax_list[learning_rates.index(lr)]
        graph.set_title("Learning Rate: " + str(lr))
        graph.set_ylabel("Cross Entropy Loss")
        graph.set_xlabel("# of Epochs")
        graph.errorbar(list(range(epochs + 1)), holdout_mean, holdout_std, linestyle='-', color='red', marker='o', label="holdout")
        graph.errorbar(list(range(epochs + 1)), train_mean, train_std, linestyle='-', color='blue', marker='o', label="train")
        graph.fill_between(list(range(epochs + 1)), np.asarray(holdout_mean) + np.asarray(holdout_std), np.asarray(holdout_mean) - np.asarray(holdout_std), facecolor='r',alpha=0.5)
        graph.fill_between(list(range(epochs + 1)), np.asarray(train_mean) + np.asarray(train_std), np.asarray(train_mean) - np.asarray(train_std), facecolor='b',alpha=0.5)

    plt.savefig("lr_test.png", bbox_inches='tight')
    for correctness_list, i in zip(percent_correct, range(3)):
        mean = np.mean(correctness_list)
        std_dev = np.std(correctness_list)
        print("Learning Rate={}, average % correct= {} ({})".format(learning_rates[i], mean, std_dev))


def pca_test(data):
    global train_errors
    global holdout_errors
    fig, ax_list = plt.subplots(nrows=1, ncols=4, figsize=(15,6), sharex=True)
    fig.suptitle("Average Holdout and Training Errors", y=1)
    fig.tight_layout()
    fig.subplots_adjust(top=.85, wspace=.3)
    percent_correct = []
    for k in k_pcas:
        train_errors = [[] for _ in range(epochs + 1)]
        holdout_errors = [[] for _ in range(epochs + 1)]

        pc = train_model(data, k, learning_rates[k_pcas.index(k)])

        percent_correct.append(pc)
        holdout_mean, holdout_std = (list(map(np.mean, holdout_errors)), list(map(np.std, holdout_errors)))
        train_mean, train_std = (list(map(np.mean, train_errors)), list(map(np.std, train_errors)))

        graph = ax_list[k_pcas.index(k)]
        graph.set_title(str(k) + " Principle Components")
        graph.set_ylabel("Cross Entropy Loss")
        graph.set_xlabel("# of Epochs")
        graph.errorbar(list(range(epochs + 1)), holdout_mean, holdout_std, linestyle='-', color='red', marker='o', label="holdout")
        graph.errorbar(list(range(epochs + 1)), train_mean, train_std, linestyle='-', color='blue', marker='o', label="train")
        graph.fill_between(list(range(epochs + 1)), np.asarray(holdout_mean) + np.asarray(holdout_std), np.asarray(holdout_mean) - np.asarray(holdout_std), facecolor='r',alpha=0.5)
        graph.fill_between(list(range(epochs + 1)), np.asarray(train_mean) + np.asarray(train_std), np.asarray(train_mean) - np.asarray(train_std), facecolor='b',alpha=0.5)
        graph.legend()

    plt.savefig("pca_test.png", bbox_inches='tight')
    for correctness_list, i in zip(percent_correct, range(4)):
        mean = np.mean(correctness_list)
        std_dev = np.std(correctness_list)
        print("PCA={}, average % correct= {} ({})".format(k_pcas[i], mean, std_dev))

def train_model(slice_data, k, lr, random=True):
    global test_set
    global holdout_set
    global train_set
    global epochs
    percent_correct = []
    slice_data = bucket(slice_data, 6)
    for i in range(5):
        test_start_i = 2*i
        test_end_i = 2*i + 2
        test_set = unbucket(slice_data[test_start_i:test_end_i], 6)
        remaining = slice_data[:test_start_i] + slice_data[test_end_i:]
        rand.shuffle(remaining)
        train_set = unbucket(remaining[:6], 6)
        holdout_set = unbucket(remaining[6:], 6)
        

        p_c_a = pca.PCA(k)
        p_c_a.fit(np.array([image for image,_ in train_set]))
        train_set = append_one([(p_c_a.transform(np.array(image)), label) for image, label in train_set])
        holdout_set = append_one([(p_c_a.transform(np.array(image)), label) for image, label in holdout_set])
        test_set = append_one([(p_c_a.transform(np.array(image)), label) for image, label in test_set])

        w = batch_gd_sm(epochs, lr, p_c_a.k + 1)
        percent_correct.append(sum([correct_category_sm(x, w) for x in test_set])/len(test_set))
    return percent_correct

def correct_category_sm(x, w):
    global c
    isCorrect = 0
    currentMax = 0
    for k in range(c):
        y = softmax_activation(x[0], k, w)
        if y >= currentMax:
            currentMax = y
            if one_hot(x[1])[k] == 1:
                isCorrect = 1
            else:
                isCorrect = 0
    return isCorrect

def append_one(image_set):
    for (image,label), i in zip(image_set, range(len(image_set))):
        image_set[i] = (np.array([np.insert(image, 0, 1)]), label)
    return image_set

def bucket(l, n):
    """
    Args:
        l: list to convert into bucketed list
        n: size of bucket
    """ 
    bl = [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n )]
    return bl

def unbucket(bl, n):
    """
    Args:
        l: bucket list to convert into regular list
        n: size of bucket
    """ 
    l = []
    for b in bl:
        for i in range(n):
            l.append(b[i])
    return l

    """
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

    pcaConv = pca.PCA(k_pcas)
    pcaConv.fit(np.array([i for i, l in train_set][:]))
    train_set = [(pcaConv.transform(i)[0], ts) for i, ts in train_set]
    holdout_set = [(pcaConv.transform(i)[0], ts) for i, ts in holdout_set]
    test_set = [(pcaConv.transform(i)[0], ts) for i, ts in test_set]
    print(batch_gd_sm(10, .01)) 
    """





if __name__ == '__main__':
	main()
