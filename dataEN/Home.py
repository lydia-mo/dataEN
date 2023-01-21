import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import datasets

from collections import Counter

st.write("""
# Simple Iris Flower Prediction App with different models and parameters

This app allows to predict the **Iris flower** type by **user** input values\n
It uses different machine learning algorithms to achieve that prediction :
""")
st.markdown("- Kmeans")
st.markdown("- Fuzzy Kmeans")
st.markdown("- Decision Tree")
st.markdown("- Random Forest Classifier")
st.markdown("- Logistic Regression")
st.markdown("- Neural network")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)

st.write("## Iris dataset:")

st.write("### 1. Data overview")
#Load data
iris = datasets.load_iris(as_frame=True)
data_target=pd.concat([iris.data, iris.target], axis=1)
data = iris.data
st.dataframe(data_target)

st.session_state['iris'] = datasets.load_iris(as_frame=True)
st.session_state['data_target'] = pd.concat([iris.data, iris.target], axis=1)
st.session_state['data'] = iris.data

# Classes disribution
st.write("### 2. Classes disribution")
counter = Counter(iris.target)
fig = plt.figure(figsize=(10, 4))
sns.barplot(x=list(counter.keys()), y=list(counter.values())).set(title='Distrubition of flowers classes')
st.pyplot(fig)
del counter
st.write("We see that classes are evenly distributed")

# Box plot for each feature by class
st.write("### 3. Box plot for each feature by class")
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Flower features by Iris class')
sns.boxplot(ax=axes[0, 0], data=data_target, x='target', y='sepal length (cm)')
sns.boxplot(ax=axes[0, 1], data=data_target, x='target', y='sepal width (cm)')
sns.boxplot(ax=axes[1, 0], data=data_target, x='target', y='petal length (cm)')
sns.boxplot(ax=axes[1, 1], data=data_target, x='target', y='sepal width (cm)')
st.pyplot(fig)
st.write("We notice that some features like 'Petal length' change drasticly between some spiecies of iris flowers")



# Correlation plot
st.write("### 4. Correlation plot")
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
st.pyplot(f)
st.write("The different features are not heavily correlated")


st.write("### 5. Visualisation with dimension reduction (Using PCA)")

# 2d plot after dim red with PCA
st.write("#### 2D Plot")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data)
data_pca_2d = pca.fit_transform(data)
st.session_state['data_pca_2d']=data_pca_2d
x, y = data_pca_2d.T

fig = plt.figure(figsize=(10, 4))
sns.set(style='whitegrid')
sns.scatterplot(x=x, y=y, c=iris.target)
st.pyplot(fig)

pca_explained_var= pd.DataFrame(
        pca.explained_variance_ratio_, 
        columns=["Expl Var"], 
        index=["PC %i"%i for i in range(len(pca.explained_variance_ratio_))]
)
st.write("PCA explained variance for 2 components: ", pca_explained_var)
st.write("PCA components: ")
pca_components = pd.DataFrame(pca.components_, 
            columns=data.columns, 
            index=["PC %i"%i for i in range(len(pca.explained_variance_ratio_))])
st.write(pca_components)

# 3d plot after dim red with PCA
st.write("#### 3D plot")
pca = PCA(n_components=3)
pca.fit(data)
data_pca_3d = pca.fit_transform(data)
st.session_state['data_pca_3d']=data_pca_3d

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(projection='3d')
ax.scatter(*data_pca_3d.T, c=iris.target, marker='o', depthshade=False, cmap='Paired')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
st.pyplot(fig)

pca_explained_var= pd.DataFrame(
        pca.explained_variance_ratio_, 
        columns=["Expl Var"], 
        index=["PC %i"%i for i in range(len(pca.explained_variance_ratio_))]
)
st.write("PCA explained variance for 3 components: ", pca_explained_var)
pca_components = pd.DataFrame(pca.components_, 
            columns=data.columns, 
            index=["PC %i"%i for i in range(len(pca.explained_variance_ratio_))])
st.write("PCA components: ")
pca_components = pd.DataFrame(pca.components_, 
            columns=data.columns, 
            index=["PC %i"%i for i in range(len(pca.explained_variance_ratio_))])
st.write(pca_components)

st.write("\n\n")
st.write("**We notice that the 1st component in PCA has more impact in explaining the variance**")


