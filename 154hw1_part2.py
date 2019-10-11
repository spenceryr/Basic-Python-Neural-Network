import pca
import random as rand
from dataloader import load_data
import numpy as np
import matplotlib.pyplot as plt

test_learning_rates = input("Testing learning rates?(y/n): ").lower()
test_learning_rates = test_learning_rates == 'y' or test_learning_rates == 'yes'

train_set = []
holdout_set = []
test_set = []
learning_rates = [.001, .1, 4] if test_learning_rates else [.01, .05, .8, .5]
k_pcas = 8 if test_learning_rates else [1,2,4,8]
epochs = 10
c1 = 'ht'
c2 = 'm'
holdout_errors = []
train_errors = []

def logistic_reg(w, x):
    return 1/(1+pow(np.e, -1 * (w.T.dot(x[0]))))

def correct_category(w, x, t):
    output = logistic_reg(w, x)
    return 1 if (output > .5 and t == 1) or (output <= .5 and t==0) else 0

def gradient(t, x, w):
    """
    param: t: teacher scalar
    param: w: numpy array of all weights for a node
    param: x: numpy array of all inputs for a node
    """
    return (t - logistic_reg(w, x)) * x[0]

def calc_error(w, which_set):
    return (-1/len(which_set)) * sum([(((1 if c1 in label else 0) *
                                    np.log(logistic_reg(w, x))) +
                                    ((1-(1 if c1 in label else 0)) * np.log(1-logistic_reg(w, x)))) for x, label in which_set])

def batch_gradient_descent(epochs, learning_rate, dim_input):
    """
    param: epochs: number of trials to run
    param: learning_rate: learning rate of update function
    param: dim_input: dimension of input
    """
    w = np.zeros(dim_input)
    current_error = calc_error(w, holdout_set)
    for e in range(epochs):
        r = (learning_rate * sum([gradient((1 if c1 in label else 0),image,w) for image, label in train_set]))
        v = w + r
        new_error = calc_error(v, holdout_set)
        if new_error < current_error:
            w = v
            current_error = new_error
        if (e+1)%2 == 0:
            holdout_errors[e//2].append(current_error)
            train_errors[e//2].append(calc_error(w, train_set))

    return w

def learning_rate_test(data):
    global train_errors
    global holdout_errors
    fig, ax_list = plt.subplots(nrows=1, ncols=3, figsize=(15,6), sharex=True)
    fig.suptitle("Average Training Errors", y=1)
    fig.tight_layout()
    fig.subplots_adjust(top=.85, wspace=.3)
    for lr in learning_rates:
        train_errors = [[],[],[],[],[]]
        holdout_errors = [[],[],[],[],[]]

        train_model(data, k_pcas, lr, random=False)

        train_mean, train_std = (list(map(np.mean, train_errors)), list(map(np.std, train_errors)))
        graph = ax_list[learning_rates.index(lr)]
        graph.set_title("Learning Rate: " + str(lr))
        graph.set_ylabel("Cross Entropy Loss")
        graph.set_xlabel("# of Epochs")
        graph.errorbar([2,4,6,8,10], train_mean, train_std, linestyle='-', color='red', marker='o',)

    plt.savefig("lr_test.png", bbox_inches='tight')

def pca_test(data):
    global train_errors
    global holdout_errors
    fig, ax_list = plt.subplots(nrows=1, ncols=4, figsize=(15,6), sharex=True)
    fig.suptitle("Average Holdout and Training Errors", y=1)
    fig.tight_layout()
    fig.subplots_adjust(top=.85, wspace=.3)
    percent_correct = []
    for k in k_pcas:
        holdout_errors = [[],[],[],[],[]]
        train_errors = [[],[],[],[],[]]

        pc = train_model(data, k, learning_rates[k_pcas.index(k)])

        percent_correct.append(pc)
        holdout_mean, holdout_std = (list(map(np.mean, holdout_errors)), list(map(np.std, holdout_errors)))
        train_mean, train_std = (list(map(np.mean, train_errors)), list(map(np.std, train_errors)))

        graph = ax_list[k_pcas.index(k)]
        graph.set_title(str(k) + " Principle Components")
        graph.set_ylabel("Cross Entropy Loss")
        graph.set_xlabel("# of Epochs")
        graph.errorbar([2,4,6,8,10], holdout_mean, holdout_std, linestyle='-', color='red', marker='o', label="holdout")
        graph.errorbar([2,4,6,8,10], train_mean, train_std, linestyle='-', color='blue', marker='o', label="train")
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
    percent_correct = []
    for i in range(5):
        holdout_set = []
        test_starting_index = (i * 4)
        test_ending_index = test_starting_index + 4
        test_set = slice_data[test_starting_index:test_ending_index]
        remaining = slice_data[:test_starting_index] + slice_data[test_ending_index:]
        for _ in range(2):
            rand_index = rand.randint(0, len(remaining)-1) if random else i*2
            holdout_set += [remaining[::2][rand_index//2]] + [remaining[1::2][rand_index//2]]
            del remaining[(rand_index if rand_index % 2 == 0 else rand_index-1):(rand_index+2 if rand_index % 2 == 0 else rand_index+1)]
        train_set = remaining

        p_c_a = pca.PCA(k)
        p_c_a.fit(np.array([image for image,_ in train_set]))
        train_set = [(p_c_a.transform(np.array(image)), label) for image, label in train_set]
        holdout_set = [(p_c_a.transform(np.array(image)), label) for image, label in holdout_set]
        test_set = [(p_c_a.transform(np.array(image)), label) for image, label in test_set]

        w = batch_gradient_descent(epochs, lr, p_c_a.k)
        percent_correct.append(sum([correct_category(w, image, 1 if c1 in label else 0) for image,label in test_set])/len(test_set))
    return percent_correct

def main():
    images, labels = load_data(data_dir="./CAFE/")
    data = zip(images,labels)
    slice_data = [(image, label) for image, label in data if c2 in label.strip('pgm') or c1 in label.strip('pgm')]
    if test_learning_rates:
        learning_rate_test(slice_data)
    else:
        pca_test(slice_data)





if __name__ == '__main__':
    main()
