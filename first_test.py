import pca
import random as rand
from dataloader import load_data
import numpy as np

train_set = []
holdout_set = []
test_set = []
pca_k = 50
learning_rate = 0.01
epochs = 10

def logistic_reg(w, x):
    return 1/(1+np.power(np.e, -1 * (w.T.dot(x[0]))))

def gradient(t, x, w):
    """
    param: t: teacher scalar
    param: w: numpy array of all weights for a node
    param: x: numpy array of all inputs for a node
    """
    return (t - logistic_reg(w, x)) * x[0]

def batch_gradient_descent(epochs, learning_rate, dim_input):
    """
    param: epochs: number of trials to run
    param: learning_rate: learning rate of update function
    param: dim_input: dimension of input
    """
    w = np.zeros(dim_input)
    for _ in range(epochs):
        v = w + (learning_rate * (1/len(train_set)) * sum([gradient(1 if 'ht' in label else 0,image,w) for image, label in train_set]))
        testv = 0
        testw = 0
        for image, label in holdout_set:
              log_reg_v = logistic_reg(v, image)
              testv += 1 if (log_reg_v > .5 and 'ht' in label) or (log_reg_v <= .5 and 'm' in label) else 0
              log_reg_w = logistic_reg(w, image)
              testw += 1 if (log_reg_w > .5 and 'ht' in label) or (log_reg_w <= .5 and 'm' in label) else 0
        w = v if testv > testw else w
    return w

def main():
    global train_set
    global holdout_set
    global test_set
    images, labels = load_data(data_dir="./CAFE/")
    data = zip(images,labels)
    ht_m_data = [(image, label) for image, label in data if 'm' in label.strip('pgm') or 'ht' in label.strip('pgm')]
    for _ in range(int(len(ht_m_data) * .6)):
        random_index = rand.randint(0, len(ht_m_data)-1)
        train_set.append(ht_m_data[random_index])
        del ht_m_data[random_index]
    for _ in range(int(len(ht_m_data) * .5)):
        random_index = rand.randint(0, len(ht_m_data)-1)
        holdout_set.append(ht_m_data[random_index])
        del ht_m_data[random_index]
    test_set = ht_m_data

    p_c_a = pca.PCA(len(train_set)-1)
    p_c_a.fit(np.array([image for image,_ in train_set]))
    train_set = [(p_c_a.transform(np.array(image)), label) for image, label in train_set]
    holdout_set = [(p_c_a.transform(np.array(image)), label) for image, label in holdout_set]
    test_set = [(p_c_a.transform(np.array(image)), label) for image, label in test_set]

    w = batch_gradient_descent(epochs, learning_rate, p_c_a.k)
    total = 0
    for image,label in (test_set + holdout_set):
        output = logistic_reg(w, image)
        print(output)
        print(label)
        total += 1 if (output > .5 and 'ht' in label) or (output <= .5 and 'm' in label) else 0
    print(total/len(test_set + holdout_set))
        

if __name__ == '__main__':
    main()
