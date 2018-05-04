import pandas as pd
import numpy as np
from sys import argv
import Visualization
from sklearn.metrics import f1_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn import svm, grid_search
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve


#Read the content of the files in a Pandas DataFrame


def readData(data_file_path, columns_file_path):
    df = pd.read_csv(data_file_path)
    df_features = pd.read_csv(columns_file_path)
    return df



#Impute Numeric Features and remove categorical features/Emapathy rows which have null values


def preProcess(data):
    print('Imputing numeric data - imputing missing values with most frequent values except for Empathy column....')
    print()
    numerical_columns = data._get_numeric_data().columns.values
    index, = np.where(numerical_columns == 'Empathy')
    numerical_columns = np.delete(numerical_columns, index)
    numeric_imputer = Imputer(strategy='most_frequent')
    data[numerical_columns] = numeric_imputer.fit_transform(data[numerical_columns])
    columns = data.columns.values
    series_nan = data.isnull().any()
    print('Columns having null data after imputing numeric values are: ')
    for column in columns:
        if(series_nan[column] == True):
            print(column)
    print()
    print('Dropping rows which have null values... ')
    data = data.dropna(how='any')
    print()
    print('Final Dimension of data is : ', data.shape)
    return data


#Encoded categorical features as they cannot be directly passed to the classifiers - Used One Hot Encoding and Label Encoding

##Based on above observation we can use label encoding on Only child, Village - town, House - Block of flats, 
#Left - right handed as these feature can be encoded using 0's and 1's and to prevent blow up of feature space


def EncodeCategoricalData(data):
    categorical_data = data.select_dtypes(include=['object'])
    for column in categorical_data.columns:
        data[column] = pd.Categorical(data[column])
    category_data = data.select_dtypes(include=['category'])
    print()
    print('Displaying categorial data with various labels and the number of occurences: ')
    for columns in category_data.columns:
        print()
        print(category_data[columns].value_counts(dropna=False))
        print("--------------------------------------------------------")
    
    print()
    print('Encoding Categorical Features....')
    features_to_encode = ['Only child','Gender', 'Village - town', 'House - block of flats', 'Left - right handed']
    df_encoded = data
    for feature in features_to_encode:
        df_encoded[feature] = LabelEncoder().fit_transform(data[feature])
    
    df_encoded = pd.get_dummies(df_encoded, columns = list(set(category_data.columns) - set(features_to_encode)))
    print()
    print('Categorial features have been encoded')
    print()
    print('printing a summary of the encoded data :')
    print()
    print(df_encoded.head())
    return df_encoded



#Split the data into training data and testing data



def getTrainTest(data):
    print()
    print('Splitting the data further into train/test...')
    X = data[data.loc[:, data.columns != 'Empathy'].columns]
    Y = data['Empathy']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state = 25)
    return X, Y, X_train, X_test, y_train, y_test




# Trying PCA - Works the best for feature selection

def doFeatureSelection(X_train,X_test):
    print()
    print('Selecting features by using Principal Component Analysis...')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(0.88)
    pca.fit(X_train)
    X_train_transformed = pca.transform(X_train_scaled)
    X_test_transformed = pca.transform(X_test_scaled)
    return X_train_transformed, X_test_transformed




# Using differnt classifiers to train and predict the Empathy labels


def classifier_train_predict(X_train, y_train, X_test, y_test) :
    print()
    print('Running various classifiers to predict the labels of the test data...')
    classifiers = {
        "Baseline Classifier": DummyClassifier(strategy='most_frequent', random_state=0),
        "Logistic Regression": LogisticRegression(),
        "Linear SVM": SVC(),
        "Non Linear SVM": SVC(kernel='rbf'),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Neural Network": MLPClassifier(alpha = 1),
        "Naive Bayes": GaussianNB(),
        "Decision tree" : DecisionTreeClassifier(criterion='entropy'),
        "Kernelized SVM" : SVC(decision_function_shape='ovo'),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost" : XGBClassifier()
    }
   
    for i in classifiers :
        clf = classifiers[i]
        clf = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        print()
        print (i," Accuracy Score: ",accuracy_score(y_test, predicted))
        cnf_matrix = confusion_matrix(y_test, predicted)
        print()
        plotConfusionMatrix(cnf_matrix, i)
        


# Also trying the voting classifier - hard margin


def votingClassifier(X_train, y_train, X_test, y_test):
    clf1 = RandomForestClassifier(n_estimators=50)
    clf2 = SVC(kernel = 'rbf')
    clf3 = SVC()
    
    print()
    print('Voting Classifier with hard voting:')
    vote_model = VotingClassifier(estimators=[('Random', clf1), ('kernelized', clf2), ('SVC', clf3)], voting='hard')
    vote_model.fit(X_train,y_train)
    y_pred = vote_model.predict(X_test)
    print()
    print("Accuracy Score of Voting Classifier: ",accuracy_score(y_test, y_pred))
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print()
    plotConfusionMatrix(cnf_matrix, 'voting classifier')


# Tuning the hyperparameters of SVC using GridSearchCV


def svc_param_selection(X, y, nfolds):
    print('Using GridSearchCV for tuning the hyperparameters...')
    print()
    print('This might take some time (approx 5-10 mins)...')
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = {'C': C_range, 'gamma' : gamma_range}
    model = svm.SVC(kernel = 'rbf')
    grid_search = GridSearchCV(model, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print("Best Params")
    print(grid_search.best_params_)
    return grid_search.best_params_


# Tuning the hyperparameters and displaying the final results


def tuneSVC(X_train_selected, y_train, X_test_selected, y_test):
    print()
    print('SVC gives better accuracy as compared to other classifiers, tuning hyperparameters of SVC...')
    print()
    param = svc_param_selection(X_train_selected, y_train, 3)
    model = svm.SVC(C=param['C'], kernel= 'rbf', gamma=param['gamma'])
    model = model.fit(X_train_selected, y_train)
    predicted = model.predict(X_test_selected)
    print()
    print("Accuracy Score of SVC with hyperparameters tuned: ",accuracy_score(y_test, predicted))
    cnf_matrix = confusion_matrix(y_test, predicted)
    print()
    plotConfusionMatrix(cnf_matrix, 'Tuned SVC')


if __name__ == "__main__" :
    data_file_path = argv[1]
    columns_file_path = argv[2]
    df = readData(data_file_path, columns_file_path)
    df = preProcess(df)
    visualizeRelations(df)
    df_encoded = EncodeCategoricalData(df)
    X, Y, X_train, X_test, y_train, y_test = getTrainTest(df_encoded)
    X_train_selected, X_test_selected = doFeatureSelection(X_train,X_test)
    classifier_train_predict(X_train_selected, y_train, X_test_selected, y_test)
    votingClassifier(X_train_selected, y_train, X_test_selected, y_test)
    tuneSVC(X_train_selected, y_train, X_test_selected, y_test)

