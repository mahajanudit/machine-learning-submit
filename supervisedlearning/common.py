import random

import numpy as np


def load_pollution_csv(max_rows = None):
    # handle = open(data_file_path, 'r')
    # contents = handle.read()
    # handle.close()
    # rows = contents.split('\n')

    cols = range(1, 9)
    out = np.loadtxt('../data/LSTM-Multivariate_pollution.csv', comments='#', delimiter=',', skiprows=1, usecols=cols, converters= { 1: converter_pollution, 5: converter_wind}, max_rows=max_rows)

    classes = out[:, 0]
    examples = out[:, 1:]

    return examples, classes


def converter_pollution(pollution_val):
    pollution_val = float(pollution_val)
    if pollution_val < 51:
        return  0  # acceptable
    if pollution_val < 101:
        return  1  # moderate
    if pollution_val < 151:
        return 0  # moderately unhealthy
    if pollution_val < 201:
        return 3  # unhealthy
    if pollution_val < 301:
        return 4  # very unhealthy
    return  5  # hazardous

def converter_wind(wind_strval):
    wind_strval = wind_strval.decode('UTF-8')
    if wind_strval == 'cv':
        return  0
    if wind_strval == 'N':
        return  1
    if wind_strval == 'NE':
        return  2
    if wind_strval == 'E':
        return  3
    if wind_strval == 'SE':
        return  4
    if wind_strval == 'S':
        return  5
    if wind_strval == 'SW':
        return  6
    if wind_strval == 'W':
        return  7
    if wind_strval == 'NW':
        return  8

def load_breast_csv(max_rows = None):
    # handle = open(data_file_path, 'r')
    # contents = handle.read()
    # handle.close()
    # rows = contents.split('\n')
    out = np.loadtxt('../data/BreastCancerDataSet.csv', comments='#', delimiter=',', skiprows=1, max_rows=max_rows, converters= { 1: converter_diagnosis})

    classes = out[:, 0]
    examples = out[:, 1:]

    return examples, classes


def converter_diagnosis(diagnosis_strval):
    diagnosis_strval = diagnosis_strval.decode('UTF-8')
    if diagnosis_strval == 'M':
        return  0
    return  1


def load_wine_csv(max_rows = None):
    # handle = open(data_file_path, 'r')
    # contents = handle.read()
    # handle.close()
    # rows = contents.split('\n')
    out = np.loadtxt('../data/wine_dataset.csv', comments='#', delimiter=',', skiprows=1, max_rows=max_rows)

    classes = out[:, 0]
    examples = out[:, 1:]

    return examples, classes


def load_UCI_CC(max_rows = None):
    # handle = open(data_file_path, 'r')
    # contents = handle.read()
    # handle.close()
    # rows = contents.split('\n')
    out = np.loadtxt('../data/UCI_Credit_Card.csv', comments='#', delimiter=',', skiprows=1, max_rows=max_rows, converters= { 1: converter_diagnosis})

    classes = out[:, -1]
    examples = out[:, :-1]

    return examples, classes

def process_class_to_binary(classes):
    unique_values = np.unique(classes)
    if len(unique_values) > 2:
        raise ValueError("More than 2 values in class, not a binary classification.")

    new_classes = np.where(classes == unique_values[0], 0, classes)
    new_classes = np.where(new_classes == unique_values[1], 1, new_classes)
    return new_classes


def generate_cross_validation_leave_p(examples, classes, p = 20):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        p (int): percent of test sample.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    n =examples.shape[0]  # num samples
    num_test_examples_to_pick = int( (n * p) / 100)

    training_indices = random.sample(range(n), n - num_test_examples_to_pick)
    training_classes = classes[training_indices]
    training_examples = examples[training_indices, :]

    test_indices = random.sample(range(n), num_test_examples_to_pick)
    test_classes = classes[test_indices]
    test_examples = examples[test_indices, :]

    return (training_examples, training_classes), (test_examples, test_classes)


def generate_multiple_test_sets(dataset, n):
    test_sets = []
    classes = np.copy(dataset[1])
    examples = np.copy(dataset[0])
    training_indices = random.sample(range(len(dataset[0])), n)
    training_classes = classes[training_indices]
    training_examples = examples[training_indices, :]

    for p in range(1, 20):
        num_test_examples_to_pick = int((n * p) / 100)
        test_indices = random.sample(range(len(dataset[0])), num_test_examples_to_pick)
        test_classes = classes[test_indices]
        test_examples = examples[test_indices, :]

        test_sets.append((test_examples, test_classes))
    return (training_examples, training_classes), test_sets


def get_error(output_set, class_set):
    total = len(output_set)
    error = 0.
    for i in range(total):
        if output_set[i] != class_set[i]:
            error += 1.
    return round(100 * (error / total), 2)


def time_to_ms(t):
    return round(t * 1000)
