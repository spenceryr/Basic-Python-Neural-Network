import pca
import random as rand
from dataloader import load_data
import numpy as np

train_set = []
holdout_set = []
test_set = []
learning_rate = .01
k_pca = 8
epochs = 10
c1 = 'ht'
c2 = 'm'

def logistic_reg(w, x):
    return 1/(1+np.power(np.e, -1 * (w.T.dot(x[0]))))

def gradient(t, x, w):
    """
    param: t: teacher scalar
    param: w: numpy array of all weights for a node
    param: x: numpy array of all inputs for a node
    """
    return (t - logistic_reg(w, x)) * x[0]

def calc_error(w):
    return (-1/len(holdout_set)) * sum([(((1 if c1 in label else 0) *
                                    np.log(logistic_reg(w, x))) +
                                    ((1-(1 if c1 in label else 0)) * np.log(1-logistic_reg(w, x)))) for x, label in holdout_set])

def batch_gradient_descent(epochs, learning_rate, dim_input):
    """
    param: epochs: number of trials to run
    param: learning_rate: learning rate of update function
    param: dim_input: dimension of input
    """
    w = np.zeros(dim_input)
    current_error = calc_error(w)
    for _ in range(epochs):
        r = (learning_rate * sum([gradient(1 if c1 in label else 0,image,w) for image, label in train_set]))
        print(r)
        v = w + r
        new_error = calc_error(v)
        w = v if new_error < current_error else w
    return w

def main():
    global train_set
    global holdout_set
    global test_set
    images, labels = load_data(data_dir="./CAFE/")
    data = zip(images,labels)
    slice_data = [(image, label) for image, label in data if c2 in label.strip('pgm') or c1 in label.strip('pgm')]
    for i in range(5):
        test_starting_index = (i * 4)
        test_ending_index = test_starting_index + 4
        test_set = slice_data[test_starting_index:test_ending_index]
        remaining = slice_data[:test_starting_index] + slice_data[test_ending_index:]
        rand_index = rand.randint(0, len(remaining)-1)
        holdout_set = [remaining[::2][rand_index//2]] + [remaining[1::2][rand_index//2]]
        del remaining[(rand_index if rand_index % 2 == 0 else rand_index-1):(rand_index+2 if rand_index % 2 == 0 else rand_index+1)]
        rand_index = rand.randint(0, len(remaining)-1)
        holdout_set += [remaining[::2][rand_index//2]] + [remaining[1::2][rand_index//2]]
        del remaining[(rand_index if rand_index % 2 == 0 else rand_index-1):(rand_index+2 if rand_index % 2 == 0 else rand_index+1)]
        train_set = remaining

        p_c_a = pca.PCA(k_pca)
        p_c_a.fit(np.array([image for image,_ in train_set]))
        train_set = [(p_c_a.transform(np.array(image)), label) for image, label in train_set]
        holdout_set = [(p_c_a.transform(np.array(image)), label) for image, label in holdout_set]
        test_set = [(p_c_a.transform(np.array(image)), label) for image, label in test_set]

        w = batch_gradient_descent(epochs, learning_rate, p_c_a.k)
        print(calc_error(w))


if __name__ == '__main__':
    main()
