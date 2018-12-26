import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC


### Load data and transfer them into numpy
data = pd.read_csv('spambase.csv', header=1, index_col=0)
x = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16)




### Logistic Regression
def lrClassification() :
    pipe_lr = Pipeline([
        ('lr',LogisticRegression(penalty='l2', multi_class='ovr', verbose=0, n_jobs=1))
    ])

    ### validation for C
    param_range = [10.0 ** i for i in range(-3, 2)]
    train_scores, test_scores = validation_curve(
        estimator=pipe_lr,
        X=x_train,
        y=y_train,
        param_name='lr__C',
        param_range=param_range,
        cv=10
    )
    test_scores = test_scores.sum(axis=1) / 10
    param_C = param_range[np.argmax(test_scores)]
    print (param_C)
    param_range_2 = np.linspace(param_C/10.0, param_C*10.0, num=30)
    print (param_range_2)
    train_scores, test_scores = validation_curve(
        estimator=pipe_lr,
        X=x_train,
        y=y_train,
        param_name='lr__C',
        param_range=param_range_2,
        cv=10
    )
    test_scores = test_scores.sum(axis=1) / 10
    param_C_fin = param_range_2[np.argmax(test_scores)]
    print (param_C_fin)
    lr = LogisticRegression(penalty='l2', multi_class='ovr', verbose=0, n_jobs=1, C=param_C_fin)
    lr.fit(x_train, y_train)
    score = lr.score(x_test, y_test)
    print (score)
    """
    lr = LogisticRegression(penalty='l2', multi_class='ovr', verbose=0, n_jobs=1)
    skf = StratifiedKFold(n_splits=10, random_state=0)
    scores, coefs = [], []
    for train_index, test_index in skf.split(x_train, y_train) :
        lr.fit(x_train[train_index], y_train[train_index])
        score = lr.score(x_train[test_index], y_train[test_index])
        scores.append(score)
        coefs.append(lr.coef_)
    pick = 3
    coefs_selected = sorted(zip(scores, coefs), key=lambda x: x[0])[:-(pick+1):-1]
    coef_fin = coefs_selected.pop()[1]
    for coef in coefs_selected :
        coef_fin += coef[1]
    lr.coef_ = coef_fin/float(pick)
    score = lr.score(x_test, y_test)
    print (score)
    """

"""
lrClassification()
l2 = LogisticRegression(penalty='l2', multi_class='ovr', verbose=0, n_jobs=1, C=1.0)
l2.fit(x_train, y_train)
print (l2.score(x_test, y_test))

"""


### Decision Tree
def treeClassifier() :
    test_size = x_test.shape[0]
    k = 10
    pipe_tree = Pipeline([
        ('tree', DecisionTreeClassifier(criterion='gini', splitter='best',
                                        min_samples_split=10,
                                        max_features=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0))
    ])

    ### validation for max_depth
    param_range = range(2, int(test_size**0.5), 1)
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

    ### Pipe for decision tree
    pipe_tree_2 = Pipeline([
        ('tree', DecisionTreeClassifier(criterion='gini', splitter='best',
                                        max_depth=param_max_depth,
                                        max_features=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0))
    ])
    param_min_sample_range = range(2,test_size//k//2,(test_size//k//2//15)+1)
    train_scores, test_scores = validation_curve(
        estimator=pipe_tree_2,
        X=x_train,
        y=y_train,
        param_name='tree__min_samples_split',
        param_range=param_min_sample_range,
        cv=k)
    test_scores = test_scores.sum(axis=1) / k
    param_min_samples = param_min_sample_range[np.argmax(test_scores)]
    print (param_max_depth, param_min_samples)

    ### final tree with the params below
    tree_fin = DecisionTreeClassifier(criterion='gini', splitter='best',
                                        min_samples_split=param_min_samples,
                                        max_depth=param_max_depth,
                                        max_features=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0)
    tree_fin.fit(x_train, y_train)
    score = tree_fin.score(x_test, y_test)
    print (score)
"""
treeClassifier()
tree = DecisionTreeClassifier(criterion='gini', splitter='best',
                                        max_features=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0)
tree.fit(x_train, y_train)
score = tree.score(x_test, y_test)
print (score)
"""
"""
for i in range(10) :
    print ('___epoch %s___' % (i+1))
    x = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16)
    treeClassifier()
    tree = DecisionTreeClassifier(criterion='gini', splitter='best',
                                            min_samples_split=2, max_depth=None,
                                            max_features=None, max_leaf_nodes=None,
                                            min_impurity_decrease=0.0)
    tree.fit(x_train, y_train)  
    print (tree.score(x_test, y_test))
"""



### SVM Classification
def svmClassification() :
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
    print ('test_scores', test_scores.sum(axis=1))
    print ('arg', np.argmax(test_scores.sum(axis=1)))
    param_gamma = param_range[np.argmax(test_scores.sum(axis=1))]

    ### Param range 2 based on the param obtained below
    param_range_2 = np.linspace(param_gamma*1.1/2, param_gamma*11/2, num=10)
    train_scores, test_scores = validation_curve(
        estimator=pipe_svm,
        X=x_train_std,
        y=y_train,
        param_name='svm__gamma',
        param_range=param_range_2,
        cv=5
    )
    param_gamma_fin = param_range_2[np.argmax(test_scores.sum(axis=1))]

    print (param_gamma, param_gamma_fin)
    svm = SVC(kernel='rbf', C=1.0, gamma=param_gamma_fin)
    svm.fit(x_train_std, y_train)
    score = svm.score(x_test_std, y_test)
    print ('score for svm modified : %s' % score)


svmClassification()

ss = StandardScaler()
x_train_std = ss.fit_transform(x_train)
x_test_std = ss.fit_transform(x_test)
svm3 = SVC(kernel='rbf', C=1.0)
svm3.fit(x_train_std, y_train)
print (svm3.score(x_test_std, y_test))



### NeuralNetwork Classification
def neuralNetworkClaasification() :
    ### Standardization
    ss = StandardScaler()
    x_train_std = ss.fit_transform(x_train)
    x_test_std = ss.fit_transform(x_test)

    ### Pipe of MLPClassification
    pipe_mlps = Pipeline([
        ('mlps', MLPC(hidden_layer_sizes=(50,5), activation='logistic',
                solver='lbfgs', batch_size='auto', alpha=0.0001,
                learning_rate='constant', learning_rate_init=0.001,
                shuffle=True))
    ])

    ### Param range for `alpha`
    param_range = [10 * i for i in range(1, 20)]
    train_scores, test_scores = validation_curve(
        estimator=pipe_mlps,
        X=x_train_std,
        y=y_train,
        param_name='mlps__max_iter',
        param_range=param_range,
        cv=5
    )
    print ('test scores ', test_scores.sum(axis=1))
    param_alpha = param_range[np.argmax(test_scores.sum(axis=1))]
    mlpc = MLPC(hidden_layer_sizes=(50,5), activation='logistic',
                solver='lbfgs', alpha=0.0001, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.001,
                max_iter=param_alpha, shuffle=True)
    mlpc.fit(x_train_std, y_train)
    score = mlpc.score(x_test_std, y_test)
    print ('score for pipe with alpha %s: %s' % (param_alpha, score))

    ### MLP Classification
    mlpc2 = MLPC(hidden_layer_sizes=(50,5), activation='logistic',
                solver='lbfgs', alpha=0.0001, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.001,
                max_iter=100, shuffle=True)
    mlpc2.fit(x_train_std, y_train)
    score = mlpc2.score(x_test_std, y_test)
    print ('score: %s' % score)

neuralNetworkClaasification()