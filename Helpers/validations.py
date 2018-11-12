from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def k_cross_validate(algorithm, images, labels):
    f_scores = 0
    for train, test in KFold(n_splits=3).split(X=images, y=labels):
        train_images, train_labels = images[train], labels[train]
        algorithm.fit(train_images, train_labels)
        test_images, test_labels = images[test], labels[test]
        predictions = algorithm.predict(test_images)
        f_scores += fbeta_score(test_labels, predictions, beta=0.5, average='macro')
    return f_scores / 3


def strat_k_cross_validate(algorithm, images, labels):
    f_scores = 0
    for train, test in StratifiedKFold(n_splits=3).split(X=images, y=labels):
        train_images, train_labels = images[train], labels[train]
        algorithm.fit(train_images, train_labels)
        test_images, test_labels = images[test], labels[test]
        predictions = algorithm.predict(test_images)
        f_scores += fbeta_score(test_labels, predictions, beta=0.5, average='macro')
    return f_scores / 3


def train_test_split(algorithm, images, labels, test_size=0.3):
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size)
    algorithm.fit(train_images, train_labels)
    predictions = algorithm.predict(test_images)
    return fbeta_score(test_labels, predictions, beta=0.5, average='macro')
