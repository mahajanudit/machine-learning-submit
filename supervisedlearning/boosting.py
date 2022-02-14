import time
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import common
from sklearn import tree
from sklearn.model_selection import train_test_split
import decisiontree
import numpy as np


LABEL = "Boosting"

ADA = "AdaBoost"
scenarios = [ADA]


def dt_max_depth_analysis(dataset_load, prefix):
    max_depths = range(1, 30)
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for max_depth in max_depths:
        training_score, test_score, training_time, testing_time = decisiontree.run_analysis(dataset_load, num_sample=2500, max_depth = max_depth, alpha=0.0)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(max_depths, test_scores, label="Test Set")
    ax.plot(max_depths, training_scores, label="Training Set")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - Accuracy vs max depth".format(prefix))
    ax.legend()
    fig.savefig("{} - Accuracy vs max depth.png".format(prefix))


def run_analysis(dataset_load, num_sample, num_estimators = 50, learning_rate=1.0):
    covid_example, covid_classes = dataset_load(max_rows=num_sample)
    weak_learner = tree.DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=num_estimators,learning_rate=learning_rate)
    X_train, X_test, y_train, y_test = train_test_split(covid_example, covid_classes, random_state=0)
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
        print("testing for {}".format(num_sample))
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
    ax.set_title("{} - Accuracy vs num of samples".format(prefix))
    ax.legend()
    fig.savefig("{} - Accuracy vs num of samples.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(num_samples, testing_times, label="Testing")
    ax.plot(num_samples, training_times, label="Training")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - Time taken vs num of samples.".format(prefix))
    ax.legend()
    fig.savefig("{} - Time taken vs num of samples.png".format(prefix))


def num_estimators_analysis(dataset_load, prefix, max_estimators = 80):
    num_estimators = range(1, max_estimators)
    num_samples = 10000
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for num_estimator in num_estimators:
        print("testing for {}".format(num_estimator))
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_sample=num_samples, num_estimators=num_estimator)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(num_estimators, test_scores, label="Test Set")
    ax.plot(num_estimators, training_scores, label="Training Set")
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - AdaBoost - Accuracy vs num of estimators".format(prefix))
    ax.legend()
    fig.savefig("{} - AdaBoost - Accuracy vs num of estimators.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(num_estimators, testing_times, label="Testing")
    ax.plot(num_estimators, training_times, label="Training")
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - AdaBoost - Time taken vs num of estimators.".format(prefix))
    ax.legend()
    fig.savefig("{} - AdaBoost - Time taken vs num of estimators.png".format(prefix))


def learningrate_analysis(dataset_load, prefix):
    learning_rates = np.arange(0.1, 2.1, 0.01)
    num_samples = 10000
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for learning_rate in learning_rates:
        print("testing for {}".format(learning_rate))
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_sample=num_samples, num_estimators=20, learning_rate=learning_rate)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(learning_rates, test_scores, label="Test Set")
    ax.plot(learning_rates, training_scores, label="Training Set")
    ax.set_xlabel("Learning rates")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - AdaBoost - Accuracy vs Learning rates".format(prefix))
    ax.legend()
    fig.savefig("{} - AdaBoost - Accuracy vs Learning rates.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(learning_rates, testing_times, label="Testing")
    ax.plot(learning_rates, training_times, label="Training")
    ax.set_xlabel("Learning rates")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - AdaBoost - Time taken vs Learning rates.".format(prefix))
    ax.legend()
    fig.savefig("{} - AdaBoost - Time taken vs Learning rates.png".format(prefix))


if __name__ == "__main__":
    dt_max_depth_analysis(common.load_pollution_csv, "Pollution Dataset")
    sample_size_analysis(common.load_pollution_csv, "Pollution Dataset")
    num_estimators_analysis(common.load_pollution_csv, "Pollution Dataset")
    learningrate_analysis(common.load_pollution_csv, "Pollution Dataset")


    dt_max_depth_analysis(common.load_UCI_CC, "Credit Card Dataset")
    sample_size_analysis(common.load_UCI_CC, "Credit Card Dataset")
    num_estimators_analysis(common.load_UCI_CC, "Credit Card Dataset", max_estimators=200)
    learningrate_analysis(common.load_UCI_CC, "Credit Card Dataset")