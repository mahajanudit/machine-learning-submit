import time
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import common
from sklearn.model_selection import train_test_split
import numpy as np


def run_analysis(dataset_load, num_sample, num_neighbors = 5, weights = 'uniform'):
    covid_example, covid_classes = dataset_load(max_rows=num_sample)
    X_train, X_test, y_train, y_test = train_test_split(covid_example, covid_classes, random_state=0)
    # (training_examples, training_classes), (test_examples, test_classes) = common.generate_cross_validation_leave_p(covid_example, covid_classes)
    clf = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weights)
    start_time = time.time()
    clf = clf.fit(X_train, y_train)
    training_time = common.time_to_ms(time.time() - start_time)

    start_time = time.time()
    training_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    testing_time = common.time_to_ms(time.time() - start_time)
    return training_score, test_score, training_time, testing_time


def sample_size_analysis(dataset_load, prefix):
    num_samples = range(100, 5000, 50)
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for num_sample in num_samples:
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_sample)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(num_samples, test_scores, label="Test Set")
    ax.plot(num_samples, training_scores, label="Training Set")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - KNN -  Accuracy vs num of samples".format(prefix))
    ax.legend()
    fig.savefig("{} - KNN - Accuracy vs num of samples.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(num_samples, testing_times, label="Testing")
    ax.plot(num_samples, training_times, label="Training")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - Time taken vs num of samples for training testing.".format(prefix))
    ax.legend()
    fig.savefig("{} - Time taken vs num of samples for training testing.png".format(prefix))


def n_analysis(dataset_load, prefix, weights = 'uniform'):
    num_samples = 2500
    num_neighbors = range(1, 200)
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for num_neighbor in num_neighbors:
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_sample=num_samples, num_neighbors=num_neighbor, weights=weights)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(num_neighbors, test_scores, label="Test Set")
    ax.plot(num_neighbors, training_scores, label="Training Set")
    ax.set_xlabel("Number of Neighbors")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - KNN -  Accuracy vs num of neighbors ({}) weights".format(prefix, weights))
    ax.legend()
    fig.savefig("{} - KNN - Accuracy vs num of neighbors({}) weights.png".format(prefix, weights))

    fig, ax = plt.subplots()
    ax.plot(num_neighbors, testing_times, label="Testing")
    ax.plot(num_neighbors, training_times, label="Training")
    ax.set_xlabel("Number of Neighbors")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - Time taken vs num of neighbors. ({}) weights".format(prefix, weights))
    ax.legend()
    fig.savefig("{} - Time taken vs num of neighbors ({}) weights.png".format(prefix, weights))

if __name__== "__main__":
    sample_size_analysis(common.load_pollution_csv, "Pollution Dataset")
    n_analysis(common.load_pollution_csv, "Pollution Dataset")
    n_analysis(common.load_pollution_csv, "Pollution Dataset", weights = 'distance')


    sample_size_analysis(common.load_UCI_CC, "Credit Card Dataset")
    n_analysis(common.load_UCI_CC, "Credit Card Dataset")
    n_analysis(common.load_UCI_CC, "Credit Card Dataset", weights = 'distance')
