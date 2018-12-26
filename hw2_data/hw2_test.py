import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sys import argv



""" Load data and transfer them into numpy """
def load_data(data_train_name, data_test_name) :
    # data_train.shape = (n, 57) , data_test.shape(m, 56)
    data_train = pd.read_csv(data_train_name, header=1, index_col=0)
    data_test = pd.read_csv(data_test_name, header=1, index_col=0)
    x_train = np.array(data_train.iloc[:, :-1])
    y_train = np.array(data_train.iloc[:, -1])
    x_test = np.array(data_test)
    return x_train, y_train, x_test



""" Logistic Regression """
def lrClassification(x_train, y_train, x_test, y_test=[]):
    ss = StandardScaler()
    x_train_std= ss.fit_transform(x_train)
    x_test_std = ss.fit_transform(x_test)
    pipe_lr = Pipeline([
        ('lr', LogisticRegression(penalty='l2', multi_class='ovr' ))
    ])
    param_range = [10 ** i for i in range(-6,5)]
    train_scores, test_scores = validation_curve(
        estimator=pipe_lr,
        X=x_train_std,
        y=y_train,
        param_name='lr__C',
        param_range=param_range
    )
    param_best = param_range[np.argmax(test_scores.sum(axis=1))]
    print (param_best)
    ### Test case

    lr = LogisticRegression(penalty='l2', multi_class='ovr', C=param_best)
    lr.fit(x_train_std, y_train)
    score = lr.score(x_test_std, y_test)
    lr_ori = LogisticRegression(penalty='l2', multi_class='ovr')
    lr_ori.fit(x_train, y_train)
    score_ori = lr_ori.score(x_test, y_test)
    print ('score for mod:', score)
    print ('score for non-mod:', score_ori)

    y_pred = lr.predict(x_test)
    y_pred_pd = pd.DataFrame(y_pred.flatten())
    y_pred_pd.to_csv('predict.csv', index=False)    #remember to set index=False



""" Decision Tree """
def treeClassifier(x_train, y_train, x_test):
    ### Basic Params
    train_size = x_train.shape[0]
    k = 10

    ### Pipe 1 of decision tree
    pipe_tree = Pipeline([
        ('tree', DecisionTreeClassifier(criterion='gini', splitter='best',
                                        min_samples_split=10,
                                        max_features=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0))
    ])

    ### Param range for max_depth
    param_range = range(2, int(train_size ** 0.5), 1)
    train_scores, test_scores = validation_curve(
        estimator=pipe_tree,
        X=x_train,
        y=y_train,
        param_name='tree__max_depth',
        param_range=param_range,
        cv=k
    )
    test_scores = test_scores.sum(axis=1) / k
    param_max_depth = param_range[np.argmax(test_scores)]

    ### Pipe 2 of decision tree
    pipe_tree_2 = Pipeline([
        ('tree', DecisionTreeClassifier(
            criterion='gini', splitter='best',
            max_depth=param_max_depth,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0))
    ])
    param_min_sample_range = range(2, train_size // k // 2, (train_size // k // 2 // 15) + 1)

    ### Param range for min_samples_split
    train_scores, test_scores = validation_curve(
        estimator=pipe_tree_2,
        X=x_train,
        y=y_train,
        param_name='tree__min_samples_split',
        param_range=param_min_sample_range,
        cv=k
    )
    test_scores = test_scores.sum(axis=1) / k
    param_min_samples = param_min_sample_range[np.argmax(test_scores)]

    ### Final tree with the params above
    tree_fin = DecisionTreeClassifier(
        criterion='gini', splitter='best',
        min_samples_split=param_min_samples,
        max_depth=param_max_depth,
        max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0
    )
    tree_fin.fit(x_train, y_train)

    ### Output the y_pred
    y_pred = tree_fin.predict(x_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv('predict.csv', index=False)



### SVM Classification
def svmClassification(x_train, y_train, x_test):
    ### Standardization
    ss = StandardScaler()
    x_train_std = ss.fit_transform(x_train)
    x_test_std = ss.fit_transform(x_test)

    ### Pipe of svm
    pipe_svm = Pipeline([
        ('svm', SVC(kernel='rbf', C=1.0))
    ])

    ### Param range for `gamma`
    param_range = [10 ** i for i in range(-4, 0)]
    train_scores, test_scores = validation_curve(
        estimator=pipe_svm,
        X=x_train_std,
        y=y_train,
        param_name='svm__gamma',
        param_range=param_range,
        cv=5
    )
    param_gamma = param_range[np.argmax(test_scores.sum(axis=1))]

    ### Param range 2 based on the param obtained below
    param_range_2 = np.linspace(param_gamma * 1.1 / 2, param_gamma * 11 / 2, num=10)
    train_scores, test_scores = validation_curve(
        estimator=pipe_svm,
        X=x_train_std,
        y=y_train,
        param_name='svm__gamma',
        param_range=param_range_2,
        cv=5
    )
    param_gamma_fin = param_range_2[np.argmax(test_scores.sum(axis=1))]

    ### Final SVM
    svm_fin = SVC(kernel='rbf', C=10.0, gamma=param_gamma_fin)
    svm_fin.fit(x_train_std, y_train)

    ### Output the y_pred
    y_pred = svm_fin.predict(x_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv('predict.csv', index=False)



""" NeuralNetwork Classification """
def neuralNetworkClaasification(x_train, y_train, x_test):
    ### Standardization
    ss = StandardScaler()
    x_train_std = ss.fit_transform(x_train)
    x_test_std = ss.fit_transform(x_test)

    ### Pipe of MLPClassification
    pipe_mlps = Pipeline([
        ('mlps', MLPC(hidden_layer_sizes=(50, 5), activation='logistic',
                      solver='lbfgs', batch_size='auto',
                      learning_rate='constant', learning_rate_init=0.001,
                      max_iter=200, shuffle=True))
    ])

    ### Param range for `alpha`
    param_range = [10 ** i for i in range(-5, 0)]
    train_scores, test_scores = validation_curve(
        estimator=pipe_mlps,
        X=x_train_std,
        y=y_train,
        param_name='mlps__alpha',
        param_range=param_range,
        cv=5
    )
    param_alpha = param_range[np.argmax(test_scores.sum(axis=1))]

    ### MLP Classification
    mlpc_fin = MLPC(hidden_layer_sizes=(50, 5), activation='logistic',
                 solver='lbfgs', alpha=param_alpha, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.001,
                 max_iter=200, shuffle=True)
    mlpc_fin.fit(x_train_std, y_train)

    ### Output the y_pred
    y_pred = mlpc_fin.predict(x_test_std)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv('predict.csv', index=False)


data = pd.read_csv('spambase.csv')
x = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16)
lrClassification(x_train, y_train, x_test, y_test)

"""
if __name__ == '__main__' :
    if len(argv) > 1 :
        mode = argv[1]
        data_train_name, data_test_name = argv[2], argv[3]
        x_train, y_train, x_test = load_data(data_train_name, data_test_name)

        if mode.lower() == 'r' :
            lrClassification(x_train, y_train, x_test)
        elif mode.lower() == 'd' :
            treeClassifier(x_train, y_train, x_test)
        elif mode.lower() == 's' :
            svmClassification(x_train, y_train, x_test)
        elif mode.lower() == 'n' :
            neuralNetworkClaasification(x_train, y_train, x_test)
"""