import matplotlib.pyplot as plt
import time
import common
from sklearn import tree
from sklearn.model_selection import train_test_split


def run_analysis(dataset_load, alpha, num_sample, max_depth = None):
    covid_example, covid_classes = dataset_load(max_rows=num_sample)
    X_train, X_test, y_train, y_test = train_test_split(covid_example, covid_classes, random_state=0)
    # (training_examples, training_classes), (test_examples, test_classes) = common.generate_cross_validation_leave_p(covid_example, covid_classes)
    clf = tree.DecisionTreeClassifier(ccp_alpha=alpha, max_depth=max_depth)
    start_time = time.time()
    clf = clf.fit(X_train, y_train)
    training_time = common.time_to_ms(time.time() - start_time)

    start_time = time.time()
    training_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    testing_time = common.time_to_ms(time.time() - start_time)
    return training_score, test_score, training_time, testing_time

def sample_size_analysis(dataset_load, alpha, prefix):
    num_samples = range(100, 5000, 50)
    training_times = []
    test_scores = []
    training_scores = []
    testing_times = []
    for num_sample in num_samples:
        training_score, test_score, training_time, testing_time = run_analysis(dataset_load, alpha, num_sample)
        training_scores.append(training_score)
        test_scores.append(test_score)
        training_times.append(training_time)
        testing_times.append(testing_time)

    fig, ax = plt.subplots()
    ax.plot(num_samples, test_scores, label="Test Set")
    ax.plot(num_samples, training_scores, label="Training Set")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("{} - Accuracy vs num of samples for training and testing sets".format(prefix))
    ax.legend()
    fig.savefig("{} - Accuracy vs num of samples for training and testing sets.png".format(prefix))

    fig, ax = plt.subplots()
    ax.plot(num_samples, testing_times, label="Testing")
    ax.plot(num_samples, training_times, label="Training")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Time taken")
    ax.set_title("{} - Time taken vs num of samples for training testing.".format(prefix))
    ax.legend()
    fig.savefig("{} - Time taken vs num of samples for training testing.png".format(prefix))

def pruning_analysis(dataset_load, prefix):
    covid_example, covid_classes = dataset_load(max_rows=5000)
    X_train, X_test, y_train, y_test = train_test_split(covid_example, covid_classes, random_state=0)
    # (training_examples, training_classes), (test_examples, test_classes) = common.generate_cross_validation_leave_p(covid_example, covid_classes)
    clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(training_examples, training_classes)
    # calculate error in training set
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    # predicted_classes = clf.predict(training_examples)
    # training_error.append(common.get_error(predicted_classes, training_classes))
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("Effective alpha")
    ax.set_ylabel("Total Impurity of leaves")
    ax.set_title("{} - Total Impurity vs effective alpha for training set".format(prefix))
    fig.savefig("{} - Total Impurity vs effective alpha for training set.png".format(prefix))

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("{} - Depth vs alpha".format(prefix))
    fig.tight_layout()
    fig.savefig("{} - Number of nodes and Dept vs alpha.png".format(prefix))

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("{} - Accuracy vs alpha for training and testing sets".format(prefix))
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    fig.savefig("{} - Accuracy vs alpha for training and testing sets.png".format(prefix))
    plt.show()


if __name__ == "__main__":

    sample_size_analysis(common.load_pollution_csv, 0.002, "Pollution Dataset")
    sample_size_analysis(common.load_UCI_CC, 0.002, "Credit Card Default Dataset")


    # pruning_analysis(common.load_pollution_csv, "Pollution Dataset")
    # pruning_analysis(common.load_UCI_CC, "Credit Card Default Dataset")


    print(run_analysis(common.load_UCI_CC, 0.01, 2200))
