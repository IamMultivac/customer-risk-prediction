import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from scipy import stats

from statsmodels.stats.proportion import proportions_ztest



def Benchmark(models,X,y,scoring='roc_auc',cv=3,return_resultados=True):
    """
    Evaluate a list of machine learning models using cross-validation and return their performance metrics.

    Parameters:
    - models (list of estimators): List of machine learning models to evaluate.
    - X (array-like or pd.DataFrame): Input features.
    - y (array-like or pd.Series): Target variable.
    - scoring (str, optional): Scoring metric for evaluation. Default is 'roc_auc'.
    - cv (int, optional): Number of cross-validation folds. Default is 3.
    - return_resultados (bool, optional): Whether to return the results as a DataFrame. Default is True.

    Returns:
    - resultados (pd.DataFrame or None): DataFrame containing cross-validation results, or None if return_resultados is False.
    """
    names,roc,cv_min,cv_max,cv_score=[],[],[],[],[]
  
  
    for model in models:
        names.append(str(model)[:str(model).find('(')])
  
    for model in models:
        model.fit(X,y)
        cv_score.append(np.mean(cross_val_score(X=X,y=y,estimator=model,scoring=scoring,cv=cv)))
        cv_min.append(np.min(cross_val_score(X=X,y=y,estimator=model,scoring=scoring,cv=cv)))
        cv_max.append(np.max(cross_val_score(X=X,y=y,estimator=model,scoring=scoring,cv=cv)))
    resultados=pd.DataFrame(data={'CV Score':cv_score,'CV Min':cv_min,'CV max':cv_max},index=names)
    resultados.plot.barh(figsize=(12,8))
    if return_resultados:
        return resultados



def ROCs(models,X,y,test_size=.3):
    """
    Plot ROC curves for a list of machine learning models.

    Parameters:
    - models (list of estimators): List of machine learning models to evaluate.
    - X (array-like or pd.DataFrame): Input features.
    - y (array-like or pd.Series): Target variable.
    - test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.3.

    Returns:
    - None
    """
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=12)
    ls=[]
    names=[]

    for model in models:
        names.append(str(model)[:str(model).find('(')])

    plt.figure(figsize=(10,5))
    for name,model in zip(names,models):
        model.fit(X_train,y_train)
        escore=roc_auc_score(y_test,model.predict_proba(X_test)[:,1])
        fpr,tpr,thresshold=roc_curve(y_test,model.predict_proba(X_test)[:,1])
        ls.append(escore)
        plt.plot(fpr,tpr,label=name+ ': '+ str(round(escore,4)))
    plt.plot([0,1],[0,1],'--',color='red',label='Threshold')
      #plt.xlim([0,1])
      #plt.ylim([0,1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')


def FancyImputer(method,df,colum):
    """
    Impute missing values in a DataFrame using a specified imputation method.

    Parameters:
    - method (sklearn estimator): The imputation method or estimator to be used.
    - df (pd.DataFrame): The input DataFrame containing missing values.
    - colum (str): The name of the column to impute.

    Returns:
    - imputed_values: An array-like object containing the imputed values for the specified column.
    """
    aux=df.dropna(axis=0,how='any')
    X=aux.drop(colum,axis=1)
    y=aux[colum]
    method.fit(X,y)
    Ximp=df[df[colum].isnull()].drop(colum,axis=1).fillna(df.mean())
    return method.predict(Ximp)


def Class_balanced(df,target,mayority_class=0):
    """
    Balance a DataFrame by resampling the majority and minority classes based on the target variable.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target (str): The name of the column representing the target variable.
    - majority_class (int, optional): The majority class label. Default is 0.

    Returns:
    - balanced_df (pd.DataFrame): A balanced DataFrame with an equal number of samples for each class.
    """
    mayority=df[df[target]==mayority_class].sample(df[df[target] !=mayority_class].shape[0])
    minority=df[df[target] != mayority_class]
    balanced=pd.concat([minority,mayority])
    balanced.reset_index(inplace=True)
    return balanced


def Calibrated_classifier(models,method='sigmoid',return_names=True):
    """
    Create calibrated versions of machine learning models and optionally return their names.

    Parameters:
    - models (list of estimators): List of machine learning models to be calibrated.
    - method (str, optional): The calibration method to use ('sigmoid' or 'isotonic'). Default is 'sigmoid'.
    - return_names (bool, optional): Whether to return the names of calibrated models along with the models themselves. Default is True.

    Returns:
    - calibrated_models (list or list of tuples): List of calibrated models or a list of tuples containing model names and calibrated models.
    """
    calibrated,names=[],[]
    for model in models:
        names.append(str(model)[:str(model).find('(')])

    for model in models:
        clf=CalibratedClassifierCV(base_estimator=model,method=method)
        calibrated.append(clf)
    if return_names:
        return list(zip(names,calibrated))
    else: 
        return calibrated


def PlotTime(col1,col2):
    """
    Plot the time-series data of two columns from a DataFrame.

    Parameters:
    - col1 (str): The name of the first column to plot.
    - col2 (str): The name of the second column to plot.

    Returns:
    - None
    """
    plt.figure(figsize=(16,7))

    plt.subplot(1,2,1)
    our_customers[(our_customers.account_opening_year>=2007) & (our_customers.account_opening_year<=2018) & (our_customers.clase==1)].resample('M')[col1].sum().plot(color='green')
    our_customers[(our_customers.account_opening_year>=2007) & (our_customers.account_opening_year<=2018)  & (our_customers.clase==0)].resample('M')[col1].sum().plot(color='red')
    plt.title('Cedit limit of our good and bad customers through time')
    plt.ylabel('Credit limit')

    plt.subplot(1,2,2)
    our_customers[(our_customers.account_opening_year>=2007) & (our_customers.account_opening_year<=2018) & (our_customers.clase==1)].resample('M')[col2].count().plot(color='green')
    our_customers[(our_customers.account_opening_year>=2007) & (our_customers.account_opening_year<=2018)  & (our_customers.clase==0)].resample('M')[col2].count().plot(color='red')
    plt.title('Number of credits owned good and bad customers \n through time')
    plt.ylabel('Number of credits')

    plt.tight_layout()


def BusinessUplot(df,x,y,clase):
    """
    Visualize and compare monthly income and outcome between good and bad customers.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing relevant data.
    - x (str): The name of the column representing monthly income.
    - y (str): The name of the column representing monthly outcome.
    - clase (str): The name of the column representing customer classification.

    Returns:
    - None
    """
    plt.figure(figsize=(16,7))

    p_value_inc=stats.ttest_ind(np.log(df[x][df[clase]==0]),np.log(df[x][df[clase]==1]))[1]
    p_value_out=stats.ttest_ind(np.log(df[y][df[clase]==0]),np.log(df[y][df[clase]==1]))[1]


    plt.subplot(2,2,1)
    sns.kdeplot(data=np.log(df[x][df[clase]==1]),shade=True,color='green')
    sns.rugplot(a=np.log(df[x][df[clase]==1]),color='green',alpha=.3)
    sns.kdeplot(data=np.log(df[x][df[clase]==0]),shade=True,color='red')
    sns.rugplot(a=np.log(df[x][df[clase]==0]),color='red',alpha=.3)
    plt.title('Monthly income of good vs bad customers  \n p-value t : %.4f' % p_value_inc)
    plt.xlabel('Log of monthly income')

    plt.subplot(2,2,2)
    sns.kdeplot(data=np.log(df[y][df[clase]==1]),shade=True,color='green')
    sns.rugplot(a=np.log(df[y][df[clase]==1]),color='green',alpha=.3)
    sns.kdeplot(data=np.log(df[y][df[clase]==0]),shade=True,color='red')
    sns.rugplot(a=np.log(df[y][df[clase]==0]),color='red',alpha=.3)
    plt.title('Monthly outcome of good vs bad customers \n p-value t : %.4f' % p_value_out)
    plt.xlabel('Log of monthly outcome')

    plt.subplot(2,2,3)
    plt.scatter(x=np.log(df[x][df[clase]==1]),y=np.log(df[y][df[clase]==1]),facecolors='none',edgecolors='green')
    plt.title('Monthly Income vs Outcome good customers \n log scled')
    plt.xlabel('Log of monthly income')
    plt.ylabel('Log of monthly outcome')

    plt.subplot(2,2,4)
    plt.scatter(x=np.log(df[x][df[clase]==0]),y=np.log(df[y][df[clase]==0]),facecolors='none',edgecolors='red')
    plt.title('Monthly Income vs Outcome bad customers \n log scaled')
    plt.xlabel('Log of monthly income')
    plt.ylabel('Log of monthly outcome')

    plt.tight_layout()


def Pieplot(a,b,x,y):
    """
    Create a side-by-side pie chart to visualize the distribution of customers based on credit status.

    Parameters:
    - a (int): Number of good customers with at least one credit.
    - b (int): Number of bad customers with at least one credit.
    - x (int): Number of good customers with no credit.
    - y (int): Number of bad customers with no credit.

    Returns:
    - None
    """
    plt.figure(figsize=(12,7))
    count = [a,x]
    nobs = [a+b,x+y]
    stat, pval = proportions_ztest(count, nobs) 

    plt.subplot(1,2,1)
    labels = ['Good customers','Bad customers']
    sizes = [a,b]
    plt.pie(sizes,labels=labels, autopct='%1.1f%%',colors=['palegreen','lightcoral'],explode=(0,.1),)
    plt.title('Distribution of customers that have \n at least one credit with Konfio \n z-test for proportion of good customers : %.4f' % pval)

    plt.subplot(1,2,2)
    labels = ['Good customers','Bad customers']
    sizes = [x,y]
    plt.pie(sizes,labels=labels, autopct='%1.1f%%',colors=['palegreen','lightcoral'],explode=(0,.1),)
    plt.title('Distribution of customers that have \n NO credit with Konfio')


def Logtransformation(df,columna):
    """
    Perform a log transformation on a specified column and return the transformed DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the column to be transformed.
    - columna (str): The name of the column to be log-transformed.

    Returns:
    - transformed_df (pd.DataFrame): A DataFrame with the specified column log-transformed.
    """
    X=df.drop(columna,axis=1)
    return (X+1).apply(np.log)


def FeatureSelection(X,y,models,cv=3,scorer='roc_auc',steps=1):
    """
    Perform feature selection using Recursive Feature Elimination (RFE) with cross-validation
    and visualize the results for multiple machine learning models.

    Parameters:
    - X (pd.DataFrame): The input feature matrix.
    - y (pd.Series or array-like): The target variable.
    - models (list of estimators): List of machine learning models to evaluate.
    - cv (int, optional): Number of cross-validation folds. Default is 3.
    - scorer (str, optional): The scoring metric for feature selection. Default is 'roc_auc'.
    - steps (int, optional): The number of features to remove at each step. Default is 1.

    Returns:
    - None
    """
    names=[]
    features_opt=[]
    for model in models:
        names.append(str(model)[:str(model).find('(')])

    for name,model in zip(names,models):
        rfecv=RFECV(estimator=model,step=steps,scoring=scorer,cv=StratifiedKFold(cv)).fit(X,y)
        features_opt.append(rfecv.n_features_)
        plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"],marker='o',label=name)
    plt.xlabel('Number of features')
    plt.ylabel(scorer)
    plt.legend(loc='best')
    plt.title('Optimal number of features based on '+str(scorer)+ ' is :%d \n' % int(np.mean(features_opt)))


def FeatureSelector(X,y,model,cv=3,scorer='roc_auc',steps=1):
    """
    Perform feature selection using Recursive Feature Elimination (RFE) with cross-validation
    and return a list of selected features.

    Parameters:
    - X (pd.DataFrame): The input feature matrix.
    - y (pd.Series or array-like): The target variable.
    - model: The machine learning model used for feature selection.
    - cv (int, optional): Number of cross-validation folds. Default is 3.
    - scorer (str, optional): The scoring metric for feature selection. Default is 'roc_auc'.
    - steps (int, optional): The number of features to remove at each step. Default is 1.

    Returns:
    - selected_features (list): List of selected feature names.
    """
    ls=[]
    rfecv=RFECV(estimator=model,step=steps,scoring=scorer,cv=StratifiedKFold(cv)).fit(X,y)
    for i,col in enumerate(X.columns):
        if rfecv.support_[i]:
            ls.append(col)
    return ls 



def Plot2D(X,y):
    """
    Visualize a 2D scatter plot of data points using PCA.

    Parameters:
    - X (pd.DataFrame): The input feature matrix.
    - y (pd.Series or array-like): The target variable.

    Returns:
    - None
    """
    X2D=PCA(n_components=2,).fit_transform(X)
    plt.scatter(X2D[:,0],X2D[:,1],c=y,cmap='RdYlGn',alpha=.4)
    plt.xlabel('first component')
    plt.ylabel('second component')
    plt.title('Values of X previously scaled by clase')
    plt.tight_layout()
    plt.colorbar()


def PlotAnomalies(X,return_values=False):
    """
    Visualize anomalies in data using Local Outlier Factor (LOF) and PCA.

    Parameters:
    - X (pd.DataFrame): The input feature matrix.
    - return_values (bool, optional): If True, return the outlier scores. Default is False.

    Returns:
    - outliyers (array-like): Array of outlier scores if return_values is True, else None.
    """
    if not return_values:
        outliyers=LocalOutlierFactor().fit_predict(X)
        X2D=PCA(n_components=2,).fit_transform(X)
        plt.scatter(X2D[:,0],X2D[:,1],c=outliyers,cmap='RdGy',alpha=.4)
        plt.xlabel('first component')
        plt.ylabel('second component')
        plt.title('Values of X in 2D to visualize outliers')
        plt.tight_layout()
        plt.colorbar()
    else:
        outliyers=LocalOutlierFactor().fit_predict(X)
        return outliyers

def PlotK(X,n_clusters_test=8):
    """
    Visualize the inertia of K-Means clustering for different values of K.

    Parameters:
    - X (pd.DataFrame): The input feature matrix.
    - n_clusters_test (int, optional): The maximum number of clusters to test. Default is 8.

    Returns:
    - None
    """
    ls=[]
    for k in range(1,n_clusters_test+1):
        kmeans=KMeans(n_clusters=k).fit(X)
        ls.append(kmeans.inertia_)
    plt.plot(range(1,n_clusters_test+1),ls,marker='o')
    plt.title('K- means inertia for different k')
    plt.xlabel('k-value')
    plt.ylabel('inertia')


def PlotClusterScore(X,n_clusters_test=8):
    """
    Visualize silhouette scores for different values of K in K-Means clustering.

    Parameters:
    - X (pd.DataFrame): The input feature matrix.
    - n_clusters_test (int, optional): The maximum number of clusters to test. Default is 8.

    Returns:
    - None
    """
    ls=[]
    for k in range(2,n_clusters_test+1):
        kmeans=KMeans(n_clusters=k).fit(X)
        ls.append(silhouette_score(X,kmeans.labels_))
    plt.plot(range(2,n_clusters_test+1),ls,marker='s',color='red')
    plt.title('Siluhete score for different values of k')
    plt.xlabel('k-value')
    plt.ylabel('score')


def PlotConfusionMatrix(model,X,y):
    """
    Visualize a confusion matrix for a machine learning model's predictions.

    Parameters:
    - model: The machine learning model.
    - X (pd.DataFrame): The input feature matrix for prediction.
    - y (pd.Series or array-like): The true target variable.

    Returns:
    - None
    """
    y_pred=model.predict(X)
    cm=pd.crosstab(index=y,columns=y_pred,margins=False)
    sns.heatmap(cm,annot=True,cmap='Greens',fmt='g')
    plt.title('Confusion matrix: \n rows: True lable columns: predicted label')



def RiskPlot(df,column,clase):
    """
    Visualize the risk associated with good and bad customers based on a probability column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the risk column.
    - column (str): The name of the risk/probability column.
    - clase (str): The name of the target variable indicating good and bad customers.

    Returns:
    - None
    """
    plt.figure(figsize=(8,6))
    sns.kdeplot(data=df[column][df[clase]==1],shade=True,color='green')
    sns.rugplot(a=df[column][df[clase]==1],color='green',alpha=.3)
    sns.kdeplot(data=df[column][df[clase]==0],shade=True,color='red')
    sns.rugplot(a=df[column][df[clase]==0],color='red',alpha=.3)
    plt.title('Risk associated to good and bad customers \n Probability ot the customer to payback the loan')
    plt.ylabel('Number of customers')
    plt.xlabel('Probability');

