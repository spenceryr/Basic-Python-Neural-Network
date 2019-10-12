import pca
import random as rand
from dataloader import load_data
import numpy as np
import matplotlib.pyplot as plt

# Used if you only want to run a single test with a single learning rate and # of pcas
test_single = input("Single test?(y/n): ").lower()
test_single = test_single == 'y' or test_single == 'yes'
test_learning_rates = None
learning_rates = []
k_pcas = None
if not test_single:
    #Used if you want to test various learning rates with fixed PCA or various PCA with set learning rates
    test_learning_rates = input("Testing learning rates?(y/n): ").lower()
    test_learning_rates = test_learning_rates == 'y' or test_learning_rates == 'yes'
    learning_rates = [.001, .1, 4] if test_learning_rates else [.01, .02, .03, .1]
    k_pcas = 8 if test_learning_rates else [1,2,4,8]
else:
    learning_rates = .1
    k_pcas = 8

train_set = []
holdout_set = []
test_set = []
epochs = 10
c1 = 'f' #First emotion category based on semantics of filename
c2 = 's' #Second emotion category based on semantics of filename
holdout_errors = []
train_errors = []

def append_one(image_set):
    """
    Takes in a set of transformed image vectors and prepends a 1 to the beginning of each
    param: image_set: set of images that have already been transformed by pca
    return: the new set of images with 1's prepends to each image vector
    """
    for (image,label), i in zip(image_set, range(len(image_set))):
        image_set[i] = (np.array([np.insert(image, 0, 1)]), label)
    return image_set

def logistic_reg(w, x):
    """
    Calculates the logistic regression given a weight and image vector
    param: w: weight vector
    param: x: image vector
    return: result of logistic regression
    """
    return 1/(1+np.exp(-1 * (w.T.dot(x[0]))))

def correct_category(w, x, t):
    """
    Determines if an image is predicted to be in the correct category using logistic regression
    param: w: weight vector
    param: x: image vector
    param: t: binary integer that should be 1 if image is in first category and 0 if in second
    return: 1 if predicted in correct category and 0 if not
    """
    output = logistic_reg(w, x)
    return 1 if (output > .5 and t == 1) or (output <= .5 and t==0) else 0

def gradient(t, x, w):
    """
    Calculates the gradient based on a weight vector and an image vector
    param: t: teacher scalar (0 or 1)
    param: w: weight vector
    param: x: image vector
    return: vector result of gradient calculation
    """
    return (t - logistic_reg(w, x)) * x[0] # Note that (t - logistic_reg(w, x)) results in a scalar
                                           # while x[0] is a vector

def calc_error(w, which_set):
    """
    Calculates the cross-entropy error given a weight vector and a set of images
    param: w: weight vector
    param: which_set: set of images to calculate the error based on
    return: Average cross-entropy error calculated
    """
    return (-1/len(which_set)) * sum([(((1 if c1 in label else 0) *
                                    np.log(logistic_reg(w, x))) +
                                    ((1-(1 if c1 in label else 0)) * np.log(1-logistic_reg(w, x)))) for x, label in which_set])

def batch_gradient_descent(epochs, learning_rate, dim_input):
    """
    Runs batch gradient descent algorithm through a given number of empochs
    param: epochs: number of trials to run
    param: learning_rate: learning rate of update function
    param: dim_input: dimension of input
    return: the best weight vector based on the error of the holdout set
    """
    w = np.zeros(dim_input)
    best_w = w
    best_error = calc_error(w, holdout_set)
    # Calculates starting error for holdout and training sets
    holdout_errors[0].append(best_error)
    train_errors[0].append(calc_error(best_w, train_set))
    for e in range(epochs):
        r = (learning_rate * sum([gradient((1 if c1 in label else 0),image,w) for image, label in train_set]))
        v = w + r
        new_error = calc_error(v, holdout_set)
        # Keeps track of the best error and best weights
        if new_error < best_error:
            best_w = v
            best_error = new_error
        w = v
        # Adds error for each epoch to array for later use
        holdout_errors[e+1].append(new_error)
        train_errors[e+1].append(calc_error(w, train_set))

    return best_w

def single_test(data):
    """
    Runs a single test and creates a single graph with the holdout and training errors
    Also prints out the accuracy
    param: data: the image set to use
    """
    global train_errors
    global holdout_errors
    fig, ax_list = plt.subplots(nrows=1, ncols=1, figsize=(7,6), sharex=True)
    fig.suptitle("Average Holdout and Training Errors", y=1)
    fig.tight_layout()
    fig.subplots_adjust(top=.85, wspace=.3)

    train_errors = [[],[],[],[],[],[],[],[],[],[],[]]
    holdout_errors = [[],[],[],[],[],[],[],[],[],[],[]]
    percent_correct = []

    percent_correct = train_model(data, k_pcas, learning_rates, random=False)

    train_mean, train_std = (list(map(np.mean, train_errors)), list(map(np.std, train_errors)))
    holdout_mean, holdout_std = (list(map(np.mean, holdout_errors)), list(map(np.std, holdout_errors)))

    graph = ax_list
    graph.set_title("Learning Rate: " + str(learning_rates))
    graph.set_ylabel("Cross Entropy Loss")
    graph.set_xlabel("# of Epochs")
    graph.errorbar(list(range(11)), holdout_mean, holdout_std, linestyle='-', color='red', marker='o', label="holdout")
    graph.errorbar(list(range(11)), train_mean, train_std, linestyle='-', color='blue', marker='o', label="train")
    graph.fill_between(list(range(11)), np.asarray(holdout_mean) + np.asarray(holdout_std), np.asarray(holdout_mean) - np.asarray(holdout_std), facecolor='r',alpha=0.5)
    graph.fill_between(list(range(11)), np.asarray(train_mean) + np.asarray(train_std), np.asarray(train_mean) - np.asarray(train_std), facecolor='b',alpha=0.5)
    mean = np.mean(percent_correct) * 100
    std_dev = np.std(percent_correct) * 100
    print("PCA={}, average % correct= {} ({})".format(k_pcas, mean, std_dev))
    plt.savefig("single_test.png", bbox_inches='tight')

def learning_rate_test(data):
    """
    Runs a test through multiple learning rates and creates a graph of holdout/training error per learning rate
    Also prints out the accuracy for each learning rate
    param: data: the image set to use
    """
    global train_errors
    global holdout_errors
    fig, ax_list = plt.subplots(nrows=1, ncols=3, figsize=(15,6), sharex=True)
    fig.suptitle("Average Holdout and Training Errors", y=1)
    fig.tight_layout()
    fig.subplots_adjust(top=.85, wspace=.3)
    percent_correct = []
    for lr in learning_rates:
        train_errors = [[],[],[],[],[],[],[],[],[],[],[]]
        holdout_errors = [[],[],[],[],[],[],[],[],[],[],[]]

        pc = train_model(data, k_pcas, lr, random=False)

        percent_correct.append(pc)
        train_mean, train_std = (list(map(np.mean, train_errors)), list(map(np.std, train_errors)))
        holdout_mean, holdout_std = (list(map(np.mean, holdout_errors)), list(map(np.std, holdout_errors)))

        graph = ax_list[learning_rates.index(lr)]
        graph.set_title("Learning Rate: " + str(lr))
        graph.set_ylabel("Cross Entropy Loss")
        graph.set_xlabel("# of Epochs")
        graph.errorbar(list(range(11)), holdout_mean, holdout_std, linestyle='-', color='red', marker='o', label="holdout")
        graph.errorbar(list(range(11)), train_mean, train_std, linestyle='-', color='blue', marker='o', label="train")
        graph.fill_between(list(range(11)), np.asarray(holdout_mean) + np.asarray(holdout_std), np.asarray(holdout_mean) - np.asarray(holdout_std), facecolor='r',alpha=0.5)
        graph.fill_between(list(range(11)), np.asarray(train_mean) + np.asarray(train_std), np.asarray(train_mean) - np.asarray(train_std), facecolor='b',alpha=0.5)

    plt.savefig("lr_test.png", bbox_inches='tight')
    for correctness_list, i in zip(percent_correct, range(3)):
        mean = np.mean(correctness_list) * 100
        std_dev = np.std(correctness_list) * 100
        print("Learning Rate={}, average % correct= {} ({})".format(learning_rates[i], mean, std_dev))


def pca_test(data):
    """
    Runs a test through multiple # of PCA's (using best learning rate for each) and creates a graph of holdout/training error per PCA
    Also prints out the accuracy for each # of PCA's
    param: data: the image set to use
    """
    global train_errors
    global holdout_errors
    fig, ax_list = plt.subplots(nrows=1, ncols=4, figsize=(15,6), sharex=True)
    fig.suptitle("Average Holdout and Training Errors", y=1)
    fig.tight_layout()
    fig.subplots_adjust(top=.85, wspace=.3)
    percent_correct = []
    for k in k_pcas:
        train_errors = [[],[],[],[],[],[],[],[],[],[],[]]
        holdout_errors = [[],[],[],[],[],[],[],[],[],[],[]]

        pc = train_model(data, k, learning_rates[k_pcas.index(k)])

        percent_correct.append(pc)
        holdout_mean, holdout_std = (list(map(np.mean, holdout_errors)), list(map(np.std, holdout_errors)))
        train_mean, train_std = (list(map(np.mean, train_errors)), list(map(np.std, train_errors)))

        graph = ax_list[k_pcas.index(k)]
        graph.set_title(str(k) + " Principle Components")
        graph.set_ylabel("Cross Entropy Loss")
        graph.set_xlabel("# of Epochs")
        graph.errorbar(list(range(11)), holdout_mean, holdout_std, linestyle='-', color='red', marker='o', label="holdout")
        graph.errorbar(list(range(11)), train_mean, train_std, linestyle='-', color='blue', marker='o', label="train")
        graph.fill_between(list(range(11)), np.asarray(holdout_mean) + np.asarray(holdout_std), np.asarray(holdout_mean) - np.asarray(holdout_std), facecolor='r',alpha=0.5)
        graph.fill_between(list(range(11)), np.asarray(train_mean) + np.asarray(train_std), np.asarray(train_mean) - np.asarray(train_std), facecolor='b',alpha=0.5)
        graph.legend()

    plt.savefig("pca_test.png", bbox_inches='tight')
    for correctness_list, i in zip(percent_correct, range(4)):
        mean = np.mean(correctness_list) * 100
        std_dev = np.std(correctness_list) * 100
        print("PCA={}, average % correct= {} ({})".format(k_pcas[i], mean, std_dev))

def train_model(slice_data, k, lr, random=True):
    """
    Performs 5 runs to create 5 different models, using a different train/holdout/test split each time
    param: slice_data: data to slice up
    param: k: number of principle components
    param: lr: learning rate
    param: random: boolean that determines if holdout set should be chosen randomly, default is True
    return: average of all 5 runs percent correct based on test set
    """
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
            # Either chooses random index or a set index
            # This is used when you want to not use the data itself as a factor when testing just the effects of changing the learning rate
            rand_index = rand.randint(0, len(remaining)-1) if random else i*2
            holdout_set += [remaining[::2][rand_index//2]] + [remaining[1::2][rand_index//2]]
            del remaining[(rand_index if rand_index % 2 == 0 else rand_index-1):(rand_index+2 if rand_index % 2 == 0 else rand_index+1)]
        train_set = remaining

        p_c_a = pca.PCA(k)
        p_c_a.fit(np.array([image for image,_ in train_set]))
        train_set = append_one([(p_c_a.transform(np.array(image)), label) for image, label in train_set])
        holdout_set = append_one([(p_c_a.transform(np.array(image)), label) for image, label in holdout_set])
        test_set = append_one([(p_c_a.transform(np.array(image)), label) for image, label in test_set])
        # Only shuffles if order of images can be a factor
        # Turned off random when only testing learning rate effect
        if random:
            rand.shuffle(train_set)

        w = batch_gradient_descent(epochs, lr, p_c_a.k + 1)
        percent_correct.append(sum([correct_category(w, image, 1 if c1 in label else 0) for image,label in test_set])/len(test_set))
    return percent_correct

def main():
    """
    Grabs data from data loader and runs appropriate function
    """
    images, labels = load_data(data_dir="./CAFE/")
    data = zip(images,labels)
    slice_data = [(image, label) for image, label in data if c2 in label.strip('pgm') or c1 in label.strip('pgm')]
    if test_single:
        single_test(slice_data)
    else:
        if test_learning_rates:
            learning_rate_test(slice_data)
        else:
            pca_test(slice_data)





if __name__ == '__main__':
    main()
