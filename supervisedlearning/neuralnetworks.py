from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time
import common
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np

LABEL = "Neural Networks"
LBFGS = "Quasi Newton"
SGD = "Gradient Descent"
ADAM = "Stochastic Gradient"

scenarios = [LBFGS, SGD, ADAM]
# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/


def grid_search(dataset_load):
    param_grid = {
        'activation': ['tanh', 'relu'],
        'hidden_layer_sizes': [(20,), (30,), (50,), (75,), (100,)],
        'max_iter': [200, 300, 500, 800, 1000, 1500],
        'alpha': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'solver': ['sgd', 'adam'],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter' : range(1, 100, 5)
    }

    covid_example, covid_classes = dataset_load(max_rows=2500)
    X_train, X_test, y_train, y_test = train_test_split(covid_example, covid_classes, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf  = HalvingGridSearchCV(MLPClassifier(batch_size='auto', random_state=1, activation='tanh', solver='sgd', learning_rate='adaptive', alpha = 1e-5), param_grid, cv=5, factor = 2)
    clf.fit(X_train, y_train)
    print(clf.best_params_)


def run_analysis(dataset_load, num_sample, hidden_layer_sizes = (75,), max_iter = 200, solver='sgd', learning_rate= 'adaptive'):
    covid_example, covid_classes = dataset_load(max_rows=num_sample)
    X_train, X_test, y_train, y_test = train_test_split(covid_example, covid_classes, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # (training_examples, training_classes), (test_examples, test_classes) = common.generate_cross_validation_leave_p(covid_example, covid_classes)
    clf = MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', random_state=1, solver=solver, learning_rate= learning_rate, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    start_time = time.time()
    clf = clf.fit(X_train, y_train)
    training_time = common.time_to_ms(time.time() - start_time)

    start_time = time.time()
    training_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    testing_time = common.time_to_ms(time.time() - start_time)
    return training_score, test_score, training_time, testing_time

def sample_size_analysis(dataset_load, prefix, solver='sgd', max_iter = 200, learning_rate= 'adaptive', hidden_layer_sizes= (75,)):
    num_samples = range(100, 2500, 100)
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for num_sample in num_samples:
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_sample, solver=solver, max_iter=max_iter, hidden_layer_sizes= hidden_layer_sizes, learning_rate= learning_rate)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(num_samples, test_scores, label="Test Set")
    ax.plot(num_samples, training_scores, label="Training Set")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - NN - Accuracy vs num of samples".format(prefix))
    ax.legend()
    fig.savefig("{} - NN - Accuracy vs num of samples.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(num_samples, testing_times, label="Testing")
    ax.plot(num_samples, training_times, label="Training")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - Time taken vs num of samples.".format(prefix))
    ax.legend()
    fig.savefig("{} - Time taken vs num of samples.png".format(prefix))


def number_nodes_layer_analysis(dataset_load, prefix, solver='sgd', max_iter = 200, learning_rate= 'adaptive'):
    number_nodes = range(5, 251, 5)
    num_sample = 2500

    test_scores = []
    training_scores = []
    testing_times = []
    training_times = []

    for number_node in number_nodes:
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_sample, hidden_layer_sizes=(number_node,), solver=solver, max_iter=max_iter, learning_rate= learning_rate)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(number_nodes, test_scores, label="Test Set")
    ax.plot(number_nodes, training_scores, label="Training Set")
    ax.set_xlabel("Number of Nodes in hidden layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - NN - Accuracy vs num of nodes in hidden layer".format(prefix))
    ax.legend()
    fig.savefig("{} - NN - Accuracy vs num of nodes in hidden layer.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(number_nodes, testing_times, label="Testing")
    ax.plot(number_nodes, training_times, label="Training")
    ax.set_xlabel("Number of Nodes in hidden layer")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - Time taken vs num of nodes in hidden layer.".format(prefix))
    ax.legend()
    fig.savefig("{} - Time taken vs num of nodes in hidden layer.png".format(prefix))

def hidden_layer_analysis(dataset_load, prefix, solver='sgd', max_iter = 200, learning_rate= 'adaptive', number_node = 20):
    hidden_layers = range(1, 20)
    num_sample = 2500

    test_scores = []
    training_scores = []
    testing_times = []
    training_times = []

    for hidden_layer in hidden_layers:
        layer = []
        for i in range(hidden_layer):
            layer.append(number_node)
        # print("testing for {}".format(layer))
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_sample, hidden_layer_sizes=layer, solver=solver, max_iter=max_iter, learning_rate= learning_rate)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(hidden_layers, test_scores, label="Test Set")
    ax.plot(hidden_layers, training_scores, label="Training Set")
    ax.set_xlabel("Number of hidden layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - NN - Accuracy vs num of hidden layers".format(prefix))
    ax.legend()
    fig.savefig("{} - NN - Accuracy vs num of hidden layers.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(hidden_layers, testing_times, label="Testing")
    ax.plot(hidden_layers, training_times, label="Training")
    ax.set_xlabel("Number of hidden layer")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - Time taken vs num of hidden.".format(prefix))
    ax.legend()
    fig.savefig("{} - Time taken vs num of hidden.png".format(prefix))


def iter_analysis(dataset_load, prefix, solver='sgd', learning_rate= 'adaptive', hidden_layer_sizes= (75,)):
    iterations = list(range(10, 300, 10)) + list(range(400, 800, 100))
    num_sample = 2500

    test_scores = []
    training_scores = []

    for max_iteration in iterations:
        training_score, test_score, _, _ = run_analysis(dataset_load, num_sample, max_iter=max_iteration, hidden_layer_sizes= hidden_layer_sizes, solver=solver, learning_rate= learning_rate)
        training_scores.append(training_score)
        test_scores.append(test_score)

    fig, ax = plt.subplots()
    ax.plot(iterations, test_scores, label="Test Set")
    ax.plot(iterations, training_scores, label="Training Set")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - NN - Accuracy vs num of iterations ".format(prefix))
    ax.legend()
    fig.savefig("{} - NN - Accuracy vs num of iterations .png".format(prefix))


if __name__ == "__main__":
    # grid_search(common.load_pollution_csv)
    sample_size_analysis(common.load_pollution_csv, "Pollution Dataset")
    number_nodes_layer_analysis(common.load_pollution_csv, "Pollution Dataset")
    hidden_layer_analysis(common.load_pollution_csv, "Pollution Dataset")
    iter_analysis(common.load_pollution_csv, "Pollution Dataset")

    # grid_search(common.load_UCI_CC)
    sample_size_analysis(common.load_UCI_CC, "Credit Card Dataset", solver='adam', max_iter=86, learning_rate= 'constant', hidden_layer_sizes=(50,))
    number_nodes_layer_analysis(common.load_UCI_CC, "Credit Card Dataset", solver='adam', max_iter=86, learning_rate= 'constant')
    iter_analysis(common.load_UCI_CC, "Credit Card Dataset", solver='adam', hidden_layer_sizes=(50,), learning_rate= 'constant')
    hidden_layer_analysis(common.load_UCI_CC,  "Credit Card Dataset", solver='adam', max_iter=86, learning_rate= 'constant', number_node=50)

