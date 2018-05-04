import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.model_selection import learning_curve

#Visualization helps to observe the relationship between different features

def visualizeRelations(data):
    print()
    print('Relationship between age, gender and empathy: ')
    sns.barplot(x="Age", y="Empathy", hue="Gender", data=data)
    plt.show()
    
    print()
    print('Relationship between Happiness in life, Education and empathy: ')
    sns.barplot(x="Happiness in life", y="Empathy", hue="Education", data=data)
    plt.show()
    
    print()
    print('Analyzing the relationship between village/town residents and empathy')
    
    fig = plt.figure(figsize=(15,4))
    i = 1
    for lr in data['Village - town'].unique():
        fig.add_subplot(1, 3, i)
        plt.title('Village - town : {}'.format(lr))
        data.Empathy[data['Village - town'] == lr].value_counts().plot(kind='pie')
        i += 1
    plt.show()
    
    print('Analyzing the relationship between gender and empathy')
    pie = plt.figure(figsize=(10,4))
    pie.add_subplot(121)
    plt.title('Gender : Male')
    data.Empathy[data['Gender'] == 'male'].value_counts().plot(kind='pie')
    pie.add_subplot(122)
    plt.title('Gender : Female')
    data.Empathy[data['Gender'] == 'female'].value_counts().plot(kind='pie')
    print('Left pie chart shows emapthy in men and the right pit char shows empathy in women')
    plt.show()
    
    print()
    print('Visualize correlation between some features: ')
    print()
    correlation_matrix = data[data.columns.to_series().sample(5)].corr()
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
    plt.title('Correlation matrix between the features', fontsize=20)
    plt.show()
    print()



# Plotting learning curve - included in notebook



def plotLearningCurve(X, Y):
    train_sizes, train_scores, test_scores = learning_curve(SVC(kernel = 'rbf'), 
                                                        X, Y,
                                                        cv=10,
                                                        scoring='accuracy',
                                                        n_jobs=-1, 
                                                        train_sizes=[50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350, 380, 410, 440, 470, 500, 530, 560, 590, 620, 650, 680, 710, 740, 770, 800, 830, 870])

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    


# Visualize top 15 Features - Included in notebook



def visualizeTopFeatures(X_train, y_train):
    logreg = LogisticRegression()
    rfe = RFE(logreg, 15)
    rfe = rfe.fit(X_train, y_train)
    inclusion = rfe.support_
    cols = X_train.columns.values
    X_train_selected = pd.DataFrame()
    X_train = X_train.reset_index(drop=True)
    for i in range(0, len(inclusion)):
        if(inclusion[i] == True):
            X_train_selected[cols[i]] = X_train[cols[i]].copy()
    print()
    print('Visualize correlation between top 15 features in Training data: ')
    print()
    correlation_matrix = X_train_selected.corr()
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
    plt.title('Correlation matrix between the features', fontsize=20)
    plt.show()


# Plot confusion matrix to observe data that are incorrectly classified



def plotConfusionMatrix(cnf_matrix, classifier_name):
    print('Plotting the confusion matrix to observe which data points are incorrectly classified')
    print()
    ax= plt.subplot()
    sns.heatmap(cnf_matrix, annot=True, ax = ax);
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix for {}'.format(classifier_name));
    ax.xaxis.set_ticklabels(['1', '2', '3', '4', '5']); ax.yaxis.set_ticklabels(['1', '2', '3', '4', '5']);
    plt.show()
