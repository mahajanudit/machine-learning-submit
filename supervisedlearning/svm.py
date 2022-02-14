from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time
import common
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np

def grid_search(dataset_load):
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], # taken from scikit _classes.py
        'C': np.arange(0.25, 2.25, 0.25),
        'degree': [1, 2, 3, 4, 5], # only used for poly
        'gamma': ['scale', 'auto']
    }

    covid_example, covid_classes = dataset_load(max_rows=2500)
    X_train, X_test, y_train, y_test = train_test_split(covid_example, covid_classes, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf  = HalvingGridSearchCV(svm.SVC(), param_grid, cv=5, factor = 2)
    clf.fit(X_train, y_train)
    print(clf.best_params_)


def run_analysis(dataset_load, num_sample, kernel, C, degree, gamma):
    covid_example, covid_classes = dataset_load(max_rows=num_sample)
    X_train, X_test, y_train, y_test = train_test_split(covid_example, covid_classes, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # (training_examples, training_classes), (test_examples, test_classes) = common.generate_cross_validation_leave_p(covid_example, covid_classes)
    clf = svm.SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
    start_time = time.time()
    clf = clf.fit(X_train, y_train)
    training_time = common.time_to_ms(time.time() - start_time)

    start_time = time.time()
    training_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    testing_time = common.time_to_ms(time.time() - start_time)
    return training_score, test_score, training_time, testing_time


def sample_size_analysis(dataset_load, prefix, kernel='rbf', C=1.75, degree=4, gamma='auto'):
    num_samples = range(100, 2500, 100)
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for num_sample in num_samples:
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_sample, kernel, C, degree, gamma)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(num_samples, test_scores, label="Test Set")
    ax.plot(num_samples, training_scores, label="Training Set")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - SVM - Accuracy vs num of samples".format(prefix))
    ax.legend()
    fig.savefig("{} - SVM - Accuracy vs num of samples.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(num_samples, testing_times, label="Testing")
    ax.plot(num_samples, training_times, label="Training")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - SVM Time taken vs num of samples.".format(prefix))
    ax.legend()
    fig.savefig("{} - SVM Time taken vs num of samples.png".format(prefix))


def degree_poly_kernel_analysis(dataset_load, prefix, C=1.75, gamma='auto'):
    degrees = range(1, 19)
    kernel='poly'
    num_samples = 2500
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for degree in degrees:
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_samples, kernel, C, degree, gamma)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(degrees, test_scores, label="Test Set")
    ax.plot(degrees, training_scores, label="Training Set")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - SVM - Accuracy vs degree for poly kernel".format(prefix))
    ax.legend()
    fig.savefig("{} - SVM - Accuracy vs degree for poly kernel.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(degrees, testing_times, label="Testing")
    ax.plot(degrees, training_times, label="Training")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - SVM Time taken vs degree for poly kernel.".format(prefix))
    ax.legend()
    fig.savefig("{} - SVM Time taken vs degree for poly kernel.png".format(prefix))


def gamma_kernel_analysis(dataset_load, prefix, C=1.75, degree=4.0, kernel='poly'):
    gammas = np.arange(0.001, 0.8, 0.005)
    num_samples = 2500
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for gamma in gammas:
        print(gamma)
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_samples, kernel, C, degree, gamma)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(gammas, test_scores, label="Test Set")
    ax.plot(gammas, training_scores, label="Training Set")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Gamma")
    ax.set_title("{} - SVM - Accuracy vs gammas for {} kernel".format(prefix, kernel))
    ax.legend()
    fig.savefig("{} - SVM - Accuracy vs gammas for {} kernel.png".format(prefix, kernel))

    fig, ax = plt.subplots()
    ax.plot(gammas, testing_times, label="Testing")
    ax.plot(gammas, training_times, label="Training")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - SVM Time taken vs gammas for {} kernel.".format(prefix, kernel))
    ax.legend()
    fig.savefig("{} - SVM Time taken vs gammas for {} kernel.png".format(prefix, kernel))



def c_kernel_analysis(dataset_load, prefix, degree=4.0, kernel='poly'):
    Cs = np.arange(0.1, 2.25, 0.1)
    num_samples = 2500
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for c in Cs:
        print(c)
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, num_samples, kernel, C=c, degree=degree, gamma = 'auto')
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(Cs, test_scores, label="Test Set")
    ax.plot(Cs, training_scores, label="Training Set")
    ax.set_xlabel("C")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - SVM - Accuracy vs C for {} kernel".format(prefix, kernel))
    ax.legend()
    fig.savefig("{} - SVM - Accuracy vs C for {} kernel.png".format(prefix, kernel))

    fig, ax = plt.subplots()
    ax.plot(Cs, testing_times, label="Testing")
    ax.plot(Cs, training_times, label="Training")
    ax.set_xlabel("C")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - SVM Time taken vs C for {} kernel.".format(prefix, kernel))
    ax.legend()
    fig.savefig("{} - SVM Time taken vs C for {} kernel.png".format(prefix, kernel))


if __name__ == "__main__":
    # grid_search(common.load_pollution_csv)
    sample_size_analysis(common.load_pollution_csv, "Pollution Dataset")
    gamma_kernel_analysis(common.load_pollution_csv, "Pollution Dataset", kernel = 'rbf')
    c_kernel_analysis(common.load_pollution_csv, "Pollution Dataset", kernel = 'rbf')
    degree_poly_kernel_analysis(common.load_pollution_csv, "Pollution Dataset")
    gamma_kernel_analysis(common.load_pollution_csv, "Pollution Dataset")

    # grid_search(common.load_UCI_CC)
    sample_size_analysis(common.load_UCI_CC, "Credit Card Dataset")
    gamma_kernel_analysis(common.load_UCI_CC, "Credit Card Dataset", kernel = 'rbf')
    c_kernel_analysis(common.load_UCI_CC, "Credit Card Dataset", kernel = 'rbf')
    degree_poly_kernel_analysis(common.load_UCI_CC, "Credit Card Dataset")
    gamma_kernel_analysis(common.load_UCI_CC, "Credit Card Dataset")