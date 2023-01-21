import streamlit as st
from streamlit_extras.switch_page_button import switch_page

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

from collections import Counter


# some functions
def lower_n_first(string, n):
    string = list(string)
    for i in range(n): string[i] = string[i].lower()
    return "".join(string)

# pages output

def kmeans_page():
    st.markdown("# KMeans Clustering")
    st.write("Applying KMeans (from sklearn) on our dataset")

    try:
        data = st.session_state["data"]
    except Exception as e:
        switch_page("Home")
    inertia_lst = [KMeans(n_clusters=i, random_state=0, n_init="auto").fit(data).inertia_ for i in range(1, 11)]
    fig= plt.figure(figsize=(10, 4))
    sns.scatterplot(x=range(1, 11), y=inertia_lst)
    st.pyplot(fig)

    nb_clusters = 3
    try:
        nb_clusters  = int(st.text_input('Number of clusters', 
                            placeholder="Specify the number of clusters"))
    except Exception as e :
        print(e)
    st.write('## The current number of clusters is', nb_clusters)

    # Classification based on the number of cluster specified
    model_kmeans = KMeans(n_clusters=nb_clusters, random_state=0, n_init="auto").fit(data)
    st.session_state["model_kmeans"] = model_kmeans
    c_labels = model_kmeans.labels_
    counter = Counter(c_labels)
    fig= plt.figure(figsize=(10, 4))
    sns.barplot(x=list(counter.keys()), y=list(counter.values())).set(title='Distrubition of flowers classes')
    del counter
    st.write("### Distibution of classes based on the Kmeans results")
    st.pyplot(fig)

    #Visualisation
    st.write("### Visualisation of data with PCA dimension reduction")
    option = st.selectbox(
        'Select the number of dimensions:',
        ('2D', '3D'),)
    st.write('#### %s plot :' %option)
    if option=='2D':
        data_pca_2d=st.session_state['data_pca_2d']
        x, y = data_pca_2d.T
        fig, axes= plt.subplots(1, 2, figsize=(9,9), sharex=True)
        axes[0].set_title('Original labels')
        axes[1].set_title('New labels')
        sns.set(style='whitegrid')
        sns.scatterplot(x=x, y=y, c=st.session_state["iris"].target, ax=axes[0])
        sns.scatterplot(x=x, y=y, c=c_labels, ax=axes[1])
        st.pyplot(fig)
    else:
        data_pca_3d=st.session_state['data_pca_3d']
        fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(*data_pca_3d.T, c=st.session_state["iris"].target, marker='o', depthshade=False, cmap='Paired')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title("Original labels")
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(*data_pca_3d.T, c=c_labels, marker='o', depthshade=False, cmap='Paired')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title("New labels")
        st.pyplot(fig)

    if nb_clusters==3:
        accuracy_kmeans = 1 - np.sum(np.abs(np.array(list(Counter(c_labels).values())) - 50))/len(c_labels)
        st.write("### Approximated accurary:", accuracy_kmeans)

def decision_tree_page():
    st.markdown("# Decision tree classification")
    st.write("Applying Decision tree (from sklearn) on our dataset")

    try:
        data_target = st.session_state["data_target"]
        iris = st.session_state["iris"]
    except Exception as e:
        switch_page("Home")

    try:
        tree_depth  = int(st.text_input('Depth of the decision tree', 
                        placeholder="Specify the depth of the tree (0 for automatic depth)"))
    except Exception as e:
        tree_depth = 0

    data_shuffle = data_target.sample(frac=1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data_shuffle.iloc[:, :-1].to_numpy(), data_shuffle.iloc[:, -1], test_size=0.25)

    from sklearn import tree
    if not tree_depth: tree_depth = None
    clf = tree.DecisionTreeClassifier(max_depth=tree_depth)
    clf = clf.fit(X_train, y_train)
    st.session_state['model_decision_tree'] = clf

    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf, 
                    feature_names=iris.feature_names,  
                    class_names=iris.target_names,
                    label=None,
                    proportion=True,
                    filled=True)
    st.pyplot(fig)

    accuracy_decision_tree = 1 - np.sum(np.abs(clf.predict(X_test) - y_test)) / len(y_test)
    st.write("### Accurarcy of decision tree: ", accuracy_decision_tree)

def random_forest_page():
    st.markdown("# Random forest classification")
    st.write("Applying random forest (from sklearn) on our dataset")

    try:
        data = st.session_state["data"]
        iris = st.session_state["iris"]
    except Exception as e:
        switch_page("Home")

    try:
        tree_depth  = st.text_input('Depth of the decision tree', 
                        placeholder="Specify the depth of the tree (0 for automatic depth)", key="tree_depth")

        n_t  = st.text_input('Number of trees in the random forest',
                        placeholder="Specify the numbers of trees (estimators) in the random forest(100 by default)", key='n_t')

        option = st.selectbox(
            'Select the criterion type:',
            ("Gini", "Entropy", "Log_loss"),)
        option = list(option)
        option[0] = option[0].lower()
        option = ''.join(option)
    except Exception as e:
        print(e)
        n_t = 100
        option = "gini" #{“gini”, “entropy”, “log_loss”}, default=”gini”
        tree_depth = None

    tree_depth = None if tree_depth in ["", 0] else int(tree_depth)
    n_t = 100 if n_t=="" else int(n_t)

    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn import tree




    clf = RFC(max_depth=tree_depth, random_state=0, criterion=option, n_estimators=n_t)
    clf.fit(data.to_numpy(), iris.target)
    st.session_state["model_random_forest"]=clf


    try:
        i  = int(st.text_input('Tree number in the random forest to show', 
            placeholder="Specify the tree number (1 to %i)"%n_t))
        i-=1

        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf.estimators_[i], 
                        feature_names=iris.feature_names,  
                        class_names=iris.target_names,
                        label=None,
                        proportion=True,
                        filled=True)
        st.pyplot(fig)
    except Exception as e:
        pass

    st.write("### Accurarcy of the random forest classifier: ", clf.score(data.to_numpy(), iris.target))

def tree_page():
    tab1, tab2 = st.tabs(["Decision tree", "Random forest"])
    with tab1:
        decision_tree_page()

    with tab2:
        random_forest_page()

def logistic_regression_page():
        st.markdown("# Logistic regression")
        st.write("Applying logistic regression on our dataset")

        from sklearn import datasets
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing

        iris = datasets.load_iris(as_frame=True)
        min_max_scaler = preprocessing.MinMaxScaler()
        data_target=pd.concat([pd.DataFrame(min_max_scaler.fit_transform(iris.data), columns=iris.data.columns), iris.target], axis=1)
        data_shuffle = data_target.sample(frac=1)
        X_train, X_test, y_train, y_test = train_test_split(data_shuffle.iloc[:, :-1].to_numpy(), data_shuffle.iloc[:, -1], test_size=0.25)
        
        try:
            penalty  = st.selectbox(
                'Select the criterion type:',
                ("L1", "L2", "elasticnet"),)

            if penalty == "elasticnet":
                l1_ratio = st.slider("L1 ratio (0 to 1)", 0.0, 1.0, step=0.01)

            max_iter  = st.text_input('Maximum number of iterations',
                            placeholder="Specify the maximum number of iterations taken for the solvers to converge (default 100)")
        except Exception as e:
            penalty = "l2" #{‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’
            l1_ratio = None # 0 <= l1_ratio <= 1 if penalty='elasticnet'
            max_iter = 100

        penalty = lower_n_first(penalty, 1)
        l1_ratio = float(l1_ratio) if penalty == "elasticnet" else None
        max_iter = int(max_iter) if max_iter!="" else 100

        clf = LR(random_state=0, solver="saga", penalty=penalty, l1_ratio=l1_ratio,max_iter=max_iter).fit(X_train, y_train)
        st.session_state["model_logistic_regression"] = clf

        features_target_cofficients = pd.DataFrame(clf.coef_, columns=iris.data.columns, index=iris.target_names)
        st.dataframe(features_target_cofficients)
        RL_precision = np.sum(clf.predict(X_test) == y_test) / len(X_test)
        st.write('Logistic regression precision', RL_precision)

def neural_network():
    st.markdown("# Neural Network")
    st.write("Applying Neural network (MLPClassifier) on our dataset")

    from sklearn import datasets
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    iris = datasets.load_iris(as_frame=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_target=pd.concat([pd.DataFrame(min_max_scaler.fit_transform(iris.data), columns=iris.data.columns), iris.target], axis=1)
    data_shuffle = data_target.sample(frac=1)
    X_train, X_test, y_train, y_test = train_test_split(data_shuffle.iloc[:, :-1].to_numpy(), data_shuffle.iloc[:, -1], test_size=0.25)

    try:
        activation  = st.selectbox(
            'Select the activation type:',
            ('Identity', 'Logistic', 'Tanh', 'Relu'),)

        learning_rate  = st.selectbox(
            'Select the learning rate type:',
            ('Constant', 'Invscaling', 'Adaptive'),)

        alpha = st.slider("L2 ratio (0 to 1)", 0.0, 1.0, step=0.01)

        max_iter  = st.text_input('Maximum number of iterations',
                        placeholder="Specify the maximum number of iterations taken for the solvers to converge (default 200)")
    except Exception as e:
        activation = "relu"
        learning_rate = "constant"
        alpha = 0.0001
        max_iter = 200

    activation = lower_n_first(activation, 1)
    learning_rate = lower_n_first(learning_rate, 1)
    alpha = float(alpha) if alpha != "" else 0.0001
    max_iter = int(max_iter) if max_iter!="" else 200


    clf = MLPClassifier(solver='lbfgs', alpha=alpha, activation=activation, max_iter=max_iter, learning_rate=learning_rate,
                    hidden_layer_sizes=(5, 10), random_state=1)
    clf.fit(X_train, y_train)
    st.session_state["model_neural_network"]=clf

    NN_precision = clf.score(X_test, y_test)
    st.write('Logistic regression precision', NN_precision)

page_names_to_funcs = {
    "Kmeans": kmeans_page,
    # "Fuzzy Kmeans": decision_tree_page,
    "Decision Tree & Random Forest Classifier": tree_page,
    "Logistic Regression": logistic_regression_page,
    "Neural network": neural_network,
}

selected_page = st.sidebar.selectbox("Algorithm for classification", page_names_to_funcs.keys(), key="1")
page_names_to_funcs[selected_page]()