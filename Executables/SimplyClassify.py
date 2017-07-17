

import pandas as pd
import numpy as np
import StringIO
io = StringIO.StringIO()
import operator
import math
from sklearn.metrics import roc_curve, auc
import sys, ast, getopt, types
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer as IM
from sklearn.ensemble import AdaBoostClassifier as Classifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.ensemble import BaggingClassifier as Classifier
from sklearn.ensemble import RandomForestClassifier as Classifier
from sklearn.ensemble import GradientBoostingClassifier as Classifier
from sklearn.naive_bayes import GaussianNB as Classifier
from sklearn.naive_bayes import BernoulliNB as Classifier
from sklearn.linear_model import LogisticRegression as Classifier
import xgboost as xgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import interp
import xlsxwriter


#fname='D:\Gurpreet\MS-LAMP\data\Newdata\New_Data.xlsx'
HOME_DIR="D:/Gurpreet/SimplyClassify"
#SAVE_DIR="D:/Gurpreet/SimplyClassify"
total = len(sys.argv)
cmdargs = str(sys.argv)
fname=sys.argv[1]
Option_Array=eval(sys.argv[2])
Run_Array=eval(sys.argv[3])
KFoldValue=eval(sys.argv[4])
TestTrainSplit=eval(sys.argv[5])
SAVE_DIR=sys.argv[6]
# In[2]:
print Run_Array
def OPTION_1(fname):
    train_ws=pd.read_excel(fname, sheetname='Training_Data')
    test_ws=pd.read_excel(fname, sheetname='Test_Data')

    noc=list(train_ws)

    y_train=train_ws.loc[:,noc[0]]
    X_train=train_ws.loc[:,:]
    X_train=X_train.drop([noc[0]],axis=1)
    X_train=pd.DataFrame(X_train)
    y_train=pd.DataFrame(y_train)

    y_test=test_ws.loc[:,noc[0]]
    X_test=test_ws.loc[:,:]
    X_test=X_test.drop([noc[0]],axis=1)
    X_test=pd.DataFrame(X_test)
    y_test=pd.DataFrame(y_test)

    Train_df=pd.concat([y_train,X_train],axis=1)
    Test_df=pd.concat([y_test,X_test],axis=1)
    
    return(X_train, y_train, X_test, y_test, Train_df, Test_df)
# In[3]:
def OPTION_2(fname,splitperc):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import Imputer as IM
    All_ws=pd.read_excel(fname, sheetname='All_Data')
    
    noc=list(All_ws)
    
    y=All_ws.loc[:,noc[0]]
    X=All_ws.loc[:,:]
    X=X.drop([noc[0]],axis=1)
    
    imp = IM(missing_values='NaN', strategy='median', axis=1)
    imp.fit(X)
    X=imp.fit_transform(X)
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    
    X.columns = noc[1:len(noc)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitperc,random_state=42)
    
    Train_df=pd.concat([y_train,X_train],axis=1)
    Test_df=pd.concat([y_test,X_test],axis=1)
    
    return(X_train, y_train, X_test, y_test, Train_df, Test_df)
# In[4]:
def AdaBoo(X_train,y_train,X_T,y_T,KFoldSplit,figname):
    T_figname=figname+"_T.jpg"
    figname=figname+".jpg"
    from sklearn.ensemble import AdaBoostClassifier as Classifier
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    from scipy import interp
    
    FN= list(X_train)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr_T = dict()
    tpr_T = dict()
    roc_auc_T = dict()
    ADA = Classifier()
    lw=2
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(1)
    kf = KFold(n_splits=KFoldSplit,shuffle=True,random_state=1254)
    X=X_train
    y=y_train
    i=1
    Kfold_Sets={}
    Kfold_All={}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        Kfold_Sets['X_train']=X_train
        Kfold_Sets['X_test']=X_test
        Kfold_Sets['Y_test']=y_test
        Kfold_Sets['Y_train']=y_train
        Kfold_All[i]=Kfold_Sets
        ADA_MODEL=ADA.fit(X_train, np.ravel(y_train))
        y_pred=ADA.predict_proba(X_test)
        
        # Compute ROC curve and area the curve
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred[:,1])
        
        mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
        mean_tpr[0] = 0.0
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color='gray', linestyle='-.',lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc[i]))
        i=i+1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    mean_tpr /= kf.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    fig.savefig(figname, dpi=100)
    #plt.show()
    
    FI = ADA.feature_importances_
    feature_importance=pd.concat([pd.DataFrame(FN),pd.DataFrame(FI)],axis=1)
    feature_importance.columns = ['Feature', 'Gain']
    FN_df=list(feature_importance)
  
    feature_importance=feature_importance.sort_values(FN_df[1],axis=0,ascending=False)
    
    y_pred_T=ADA.predict_proba(X_T)
    fpr_T, tpr_T, thresholds_T = roc_curve(y_T, y_pred_T[:,1])
    roc_auc_T = auc(fpr_T, tpr_T)
    plt.figure(2)
    plt.plot(fpr_T, tpr_T, color='gray', linestyle='-.',lw=lw,label='Test ROC (area = %0.2f)' %(roc_auc_T))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig2 = plt.gcf()
    fig2.set_size_inches(6,4)
    
    fig2.savefig(T_figname, dpi=100)
    #plt.show()
    Train_set={}
    Train_set['TPR']=tpr
    Train_set['FPR']=fpr
    Train_set['ROC']=roc_auc
    Train_set['mean_TPR']=mean_tpr
    Train_set['mean_FPR']=mean_fpr
    Train_set['mean_AUC']=mean_auc
    Train_set['Feature_importance']=feature_importance
    
    Test_set={}
    Test_set['TPR']=tpr_T
    Test_set['FPR']=fpr_T
    Test_set['ROC']=roc_auc_T
    
    Figures={}
    Figures['Train']=figname
    Figures['Test']=T_figname
    plt.close("all")
    #print feature_importance
    return(Train_set,Test_set,Figures)
# In[5]:
def RandomFores(X_train,y_train,X_T,y_T,KFoldSplit,figname):
    T_figname=figname+"_T.jpg"
    figname=figname+".jpg"
    from sklearn.ensemble import RandomForestClassifier as Classifier
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    from scipy import interp
    FN= list(X_train)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr_T = dict()
    tpr_T = dict()
    roc_auc_T = dict()
    ADA = Classifier()
    lw=2
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(1)
    kf = KFold(n_splits=KFoldSplit,shuffle=True,random_state=1254)
    X=X_train
    y=y_train
    i=1
    Kfold_Sets={}
    Kfold_All={}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        Kfold_Sets['X_train']=X_train
        Kfold_Sets['X_test']=X_test
        Kfold_Sets['Y_test']=y_test
        Kfold_Sets['Y_train']=y_train
        Kfold_All[i]=Kfold_Sets
        ADA_MODEL=ADA.fit(X_train, np.ravel(y_train))
        y_pred=ADA.predict_proba(X_test)
        
        # Compute ROC curve and area the curve
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred[:,1])
        
        mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
        mean_tpr[0] = 0.0
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color='gray', linestyle='-.',lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc[i]))
        i=i+1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    mean_tpr /= kf.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    fig.savefig(figname, dpi=100)
    #plt.show()
    
    FI = ADA.feature_importances_
    feature_importance=pd.concat([pd.DataFrame(FN),pd.DataFrame(FI)],axis=1)
    feature_importance.columns = ['Feature', 'Gain']
    FN_df=list(feature_importance)
  
    feature_importance=feature_importance.sort_values(FN_df[1],axis=0,ascending=False)
    
    y_pred_T=ADA.predict_proba(X_T)
    fpr_T, tpr_T, thresholds_T = roc_curve(y_T, y_pred_T[:,1])
    roc_auc_T = auc(fpr_T, tpr_T)
    plt.figure(2)
    plt.plot(fpr_T, tpr_T, color='gray', linestyle='-.',lw=lw,label='Test ROC (area = %0.2f)' %(roc_auc_T))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig2 = plt.gcf()
    fig2.set_size_inches(6,4)
    fig2.savefig(T_figname, dpi=100)
    #plt.show()
    Train_set={}
    Train_set['TPR']=tpr
    Train_set['FPR']=fpr
    Train_set['ROC']=roc_auc
    Train_set['mean_TPR']=mean_tpr
    Train_set['mean_FPR']=mean_fpr
    Train_set['mean_AUC']=mean_auc
    Train_set['Feature_importance']=feature_importance
    
    Test_set={}
    Test_set['TPR']=tpr_T
    Test_set['FPR']=fpr_T
    Test_set['ROC']=roc_auc_T
    
    Figures={}
    Figures['Train']=figname
    Figures['Test']=T_figname
    plt.close("all")
    return(Train_set,Test_set,Figures)
# In[6]:
def BaggingClassifi(X_train,y_train,X_T,y_T,KFoldSplit,figname):
    T_figname=figname+"_T.jpg"
    figname=figname+".jpg"
    from sklearn.ensemble import BaggingClassifier as Classifier
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    from scipy import interp
        
    FN= list(X_train)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr_T = dict()
    tpr_T = dict()
    roc_auc_T = dict()
    ADA = Classifier()
    lw=2
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(1)
    kf = KFold(n_splits=KFoldSplit,shuffle=True,random_state=1254)
    X=X_train
    y=y_train
    i=1
    Kfold_Sets={}
    Kfold_All={}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        Kfold_Sets['X_train']=X_train
        Kfold_Sets['X_test']=X_test
        Kfold_Sets['Y_test']=y_test
        Kfold_Sets['Y_train']=y_train
        Kfold_All[i]=Kfold_Sets
        ADA_MODEL=ADA.fit(X_train, np.ravel(y_train))
        y_pred=ADA.predict_proba(X_test)
        
        # Compute ROC curve and area the curve
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred[:,1])
        
        mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
        mean_tpr[0] = 0.0
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color='gray', linestyle='-.',lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc[i]))
        i=i+1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    mean_tpr /= kf.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    fig.savefig(figname, dpi=100)
    #plt.show()
    
    #FI = ADA.feature_importances_
    #feature_importance=pd.concat([pd.DataFrame(FN),pd.DataFrame(FI)],axis=1)
    #feature_importance.columns = ['Feature', 'Gain']
    #FN_df=list(feature_importance)
  
    #feature_importance=feature_importance.sort_values(FN_df[1],axis=0,ascending=False)
    
    y_pred_T=ADA.predict_proba(X_T)
    fpr_T, tpr_T, thresholds_T = roc_curve(y_T, y_pred_T[:,1])
    roc_auc_T = auc(fpr_T, tpr_T)
    plt.figure(2)
    plt.plot(fpr_T, tpr_T, color='gray', linestyle='-.',lw=lw,label='Test ROC (area = %0.2f)' %(roc_auc_T))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig2 = plt.gcf()
    fig2.set_size_inches(6,4)
    fig2.savefig(T_figname, dpi=100)
    #plt.show()
    Train_set={}
    Train_set['TPR']=tpr
    Train_set['FPR']=fpr
    Train_set['ROC']=roc_auc
    Train_set['mean_TPR']=mean_tpr
    Train_set['mean_FPR']=mean_fpr
    Train_set['mean_AUC']=mean_auc
    #Train_set['Feature_importance']=feature_importance
    
    Test_set={}
    Test_set['TPR']=tpr_T
    Test_set['FPR']=fpr_T
    Test_set['ROC']=roc_auc_T
    
    Figures={}
    Figures['Train']=figname
    Figures['Test']=T_figname
    plt.close("all")
    return(Train_set,Test_set,Figures)
# In[7]:
def GradientBoo(X_train,y_train,X_T,y_T,KFoldSplit,figname):
    T_figname=figname+"_T.jpg"
    figname=figname+".jpg"
    from sklearn.ensemble import GradientBoostingClassifier as Classifier
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    from scipy import interp
        
    FN= list(X_train)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr_T = dict()
    tpr_T = dict()
    roc_auc_T = dict()
    ADA = Classifier()
    lw=2
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(1)
    kf = KFold(n_splits=KFoldSplit,shuffle=True,random_state=1254)
    X=X_train
    y=y_train
    i=1
    Kfold_Sets={}
    Kfold_All={}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        Kfold_Sets['X_train']=X_train
        Kfold_Sets['X_test']=X_test
        Kfold_Sets['Y_test']=y_test
        Kfold_Sets['Y_train']=y_train
        Kfold_All[i]=Kfold_Sets
        ADA_MODEL=ADA.fit(X_train, np.ravel(y_train))
        y_pred=ADA.predict_proba(X_test)
        
        # Compute ROC curve and area the curve
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred[:,1])
        
        mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
        mean_tpr[0] = 0.0
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color='gray', linestyle='-.',lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc[i]))
        i=i+1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    mean_tpr /= kf.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    fig.savefig(figname, dpi=100)
    #plt.show()
    
    FI = ADA.feature_importances_
    feature_importance=pd.concat([pd.DataFrame(FN),pd.DataFrame(FI)],axis=1)
    feature_importance.columns = ['Feature', 'Gain']
    FN_df=list(feature_importance)
  
    feature_importance=feature_importance.sort_values(FN_df[1],axis=0,ascending=False)
    
    y_pred_T=ADA.predict_proba(X_T)
    fpr_T, tpr_T, thresholds_T = roc_curve(y_T, y_pred_T[:,1])
    roc_auc_T = auc(fpr_T, tpr_T)
    plt.figure(2)
    plt.plot(fpr_T, tpr_T, color='gray', linestyle='-.',lw=lw,label='Test ROC (area = %0.2f)' %(roc_auc_T))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig2 = plt.gcf()
    fig2.set_size_inches(6,4)
    fig2.savefig(T_figname, dpi=100)
    #plt.show()
    Train_set={}
    Train_set['TPR']=tpr
    Train_set['FPR']=fpr
    Train_set['ROC']=roc_auc
    Train_set['mean_TPR']=mean_tpr
    Train_set['mean_FPR']=mean_fpr
    Train_set['mean_AUC']=mean_auc
    Train_set['Feature_importance']=feature_importance
    
    Test_set={}
    Test_set['TPR']=tpr_T
    Test_set['FPR']=fpr_T
    Test_set['ROC']=roc_auc_T
    
    Figures={}
    Figures['Train']=figname
    Figures['Test']=T_figname
    plt.close("all")
    return(Train_set,Test_set,Figures)
# In[8]:
def Graussiannb(X_train,y_train,X_T,y_T,KFoldSplit,figname):
    T_figname=figname+"_T.jpg"
    figname=figname+".jpg"
    from sklearn.naive_bayes import GaussianNB as Classifier
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    from scipy import interp
        
    FN= list(X_train)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr_T = dict()
    tpr_T = dict()
    roc_auc_T = dict()
    ADA = Classifier()
    lw=2
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(1)
    kf = KFold(n_splits=KFoldSplit,shuffle=True,random_state=1254)
    X=X_train
    y=y_train
    i=1
    Kfold_Sets={}
    Kfold_All={}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        Kfold_Sets['X_train']=X_train
        Kfold_Sets['X_test']=X_test
        Kfold_Sets['Y_test']=y_test
        Kfold_Sets['Y_train']=y_train
        Kfold_All[i]=Kfold_Sets
        ADA_MODEL=ADA.fit(X_train, np.ravel(y_train))
        y_pred=ADA.predict_proba(X_test)
        
        # Compute ROC curve and area the curve
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred[:,1])
        
        mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
        mean_tpr[0] = 0.0
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color='gray', linestyle='-.',lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc[i]))
        i=i+1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    mean_tpr /= kf.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    fig.savefig(figname, dpi=100)
    #plt.show()
    
    #FI = ADA.feature_importances_
    #feature_importance=pd.concat([pd.DataFrame(FN),pd.DataFrame(FI)],axis=1)
    #feature_importance.columns = ['Feature', 'Gain']
    #FN_df=list(feature_importance)
  
    #feature_importance=feature_importance.sort_values(FN_df[1],axis=0,ascending=False)
    
    y_pred_T=ADA.predict_proba(X_T)
    fpr_T, tpr_T, thresholds_T = roc_curve(y_T, y_pred_T[:,1])
    roc_auc_T = auc(fpr_T, tpr_T)
    plt.figure(2)
    plt.plot(fpr_T, tpr_T, color='gray', linestyle='-.',lw=lw,label='Test ROC (area = %0.2f)' %(roc_auc_T))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig2 = plt.gcf()
    fig2.set_size_inches(6,4)
    fig2.savefig(T_figname, dpi=100)
    #plt.show()
    Train_set={}
    Train_set['TPR']=tpr
    Train_set['FPR']=fpr
    Train_set['ROC']=roc_auc
    Train_set['mean_TPR']=mean_tpr
    Train_set['mean_FPR']=mean_fpr
    Train_set['mean_AUC']=mean_auc
    #Train_set['Feature_importance']=feature_importance
    
    Test_set={}
    Test_set['TPR']=tpr_T
    Test_set['FPR']=fpr_T
    Test_set['ROC']=roc_auc_T
    
    Figures={}
    Figures['Train']=figname
    Figures['Test']=T_figname
    plt.close("all")
    return(Train_set,Test_set,Figures)
# In[9]:
def Bernoullinb(X_train,y_train,X_T,y_T,KFoldSplit,figname):
    T_figname=figname+"_T.jpg"
    figname=figname+".jpg"
    from sklearn.naive_bayes import BernoulliNB as Classifier
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    from scipy import interp
        
    FN= list(X_train)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr_T = dict()
    tpr_T = dict()
    roc_auc_T = dict()
    ADA = Classifier()
    lw=2
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(1)
    kf = KFold(n_splits=KFoldSplit,shuffle=True,random_state=1254)
    X=X_train
    y=y_train
    i=1
    Kfold_Sets={}
    Kfold_All={}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        Kfold_Sets['X_train']=X_train
        Kfold_Sets['X_test']=X_test
        Kfold_Sets['Y_test']=y_test
        Kfold_Sets['Y_train']=y_train
        Kfold_All[i]=Kfold_Sets
        ADA_MODEL=ADA.fit(X_train, np.ravel(y_train))
        y_pred=ADA.predict_proba(X_test)
        
        # Compute ROC curve and area the curve
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred[:,1])
        
        mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
        mean_tpr[0] = 0.0
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color='gray', linestyle='-.',lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc[i]))
        i=i+1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    mean_tpr /= kf.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    fig.savefig(figname, dpi=100)
    #plt.show()
    
    #FI = ADA.feature_importances_
    #feature_importance=pd.concat([pd.DataFrame(FN),pd.DataFrame(FI)],axis=1)
    #feature_importance.columns = ['Feature', 'Gain']
    #FN_df=list(feature_importance)
  
    #feature_importance=feature_importance.sort_values(FN_df[1],axis=0,ascending=False)
    
    y_pred_T=ADA.predict_proba(X_T)
    fpr_T, tpr_T, thresholds_T = roc_curve(y_T, y_pred_T[:,1])
    roc_auc_T = auc(fpr_T, tpr_T)
    plt.figure(2)
    plt.plot(fpr_T, tpr_T, color='gray', linestyle='-.',lw=lw,label='Test ROC (area = %0.2f)' %(roc_auc_T))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig2 = plt.gcf()
    fig2.set_size_inches(6,4)
    fig2.savefig(T_figname, dpi=100)
    #plt.show()
    Train_set={}
    Train_set['TPR']=tpr
    Train_set['FPR']=fpr
    Train_set['ROC']=roc_auc
    Train_set['mean_TPR']=mean_tpr
    Train_set['mean_FPR']=mean_fpr
    Train_set['mean_AUC']=mean_auc
    #Train_set['Feature_importance']=feature_importance
    
    Test_set={}
    Test_set['TPR']=tpr_T
    Test_set['FPR']=fpr_T
    Test_set['ROC']=roc_auc_T
    
    Figures={}
    Figures['Train']=figname
    Figures['Test']=T_figname
    plt.close("all")
    return(Train_set,Test_set,Figures)
# In[10]:
def LogisticRegressi(X_train,y_train,X_T,y_T,KFoldSplit,figname):
    T_figname=figname+"_T.jpg"
    figname=figname+".jpg"
    from sklearn.linear_model import LogisticRegression as Classifier
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    from scipy import interp
        
    FN= list(X_train)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr_T = dict()
    tpr_T = dict()
    roc_auc_T = dict()
    ADA = Classifier()
    lw=2
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(1)
    kf = KFold(n_splits=KFoldSplit,shuffle=True,random_state=1254)
    X=X_train
    y=y_train
    i=1
    Kfold_Sets={}
    Kfold_All={}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        Kfold_Sets['X_train']=X_train
        Kfold_Sets['X_test']=X_test
        Kfold_Sets['Y_test']=y_test
        Kfold_Sets['Y_train']=y_train
        Kfold_All[i]=Kfold_Sets
        ADA_MODEL=ADA.fit(X_train, np.ravel(y_train))
        y_pred=ADA.predict_proba(X_test)
        
        # Compute ROC curve and area the curve
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred[:,1])
        
        mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
        mean_tpr[0] = 0.0
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color='gray', linestyle='-.',lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc[i]))
        i=i+1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    mean_tpr /= kf.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    fig.savefig(figname, dpi=100)
    #plt.show()
    
   # FI = ADA.feature_importances_
    #feature_importance=pd.concat([pd.DataFrame(FN),pd.DataFrame(FI)],axis=1)
    #feature_importance.columns = ['Feature', 'Gain']
   # FN_df=list(feature_importance)
  
   # feature_importance=feature_importance.sort_values(FN_df[1],axis=0,ascending=False)
    
    y_pred_T=ADA.predict_proba(X_T)
    fpr_T, tpr_T, thresholds_T = roc_curve(y_T, y_pred_T[:,1])
    roc_auc_T = auc(fpr_T, tpr_T)
    plt.figure(2)
    plt.plot(fpr_T, tpr_T, color='gray', linestyle='-.',lw=lw,label='Test ROC (area = %0.2f)' %(roc_auc_T))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig2 = plt.gcf()
    fig2.set_size_inches(6, 4)
    fig2.savefig(T_figname, dpi=100)
    #plt.show()
    Train_set={}
    Train_set['TPR']=tpr
    Train_set['FPR']=fpr
    Train_set['ROC']=roc_auc
    Train_set['mean_TPR']=mean_tpr
    Train_set['mean_FPR']=mean_fpr
    Train_set['mean_AUC']=mean_auc
    #Train_set['Feature_importance']=feature_importance
    
    Test_set={}
    Test_set['TPR']=tpr_T
    Test_set['FPR']=fpr_T
    Test_set['ROC']=roc_auc_T
    
    Figures={}
    Figures['Train']=figname
    Figures['Test']=T_figname
    plt.close("all")
    return(Train_set,Test_set,Figures)
# In[11]:
def XGBOOST(X_train,y_train,X_T,y_T,KFoldSplit,figname):
        T_figname=figname+"_T.jpg"
        figname=figname+".jpg"
        import xgboost as xgb
        import matplotlib.pyplot as plt
        param={}
        param['objective']='binary:logitraw'
        param['eta']=0.1
        param['max_depth']=5
        param['eval_metric']='auc'
        param['gamma']=0
        param['min_child_weight']=10
        param['subsample']=0.6
        param['colsample_bytree']=0.6
        param['colsample_bylevel']=1
        param['seed']=42
        num_boost_round=200
        plst=list(param.items())
        lw=2
        xgmat_train=xgb.DMatrix(X_train, label=y_train)
        xgmat_test=xgb.DMatrix(X_test, label=y_test)

        xgb.cv(plst,xgmat_train,num_boost_round,nfold=KFoldSplit,verbose_eval=False, metrics={'auc'});

        clf_train = xgb.train(plst,xgmat_train,num_boost_round=200)                    
        predit_train=clf_train.predict(xgmat_train)
        
        fpr, tpr, thresholds = roc_curve(y_train, predit_train)
        roc_auc = auc(fpr, tpr)
        plt.figure(1)
        plt.plot(fpr, tpr, color='gray', linestyle='-.',lw=lw,label='Train ROC (area = %0.2f)' %(roc_auc))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    
    
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches(6,4)
        fig.savefig(figname, dpi=100)
        #plt.show()
        
        predit_test=clf_train.predict(xgmat_test)
        
        fpr_T, tpr_T, thresholds_T = roc_curve(y_T,predit_test)
        roc_auc_T = auc(fpr_T, tpr_T)
        plt.figure(2)
        plt.plot(fpr_T, tpr_T, color='gray', linestyle='-.',lw=lw,label='Test ROC (area = %0.2f)' %(roc_auc_T))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')
    
    
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        fig2 = plt.gcf()
        fig2.set_size_inches(6, 4)
        fig2.savefig(T_figname, dpi=100)
        #plt.show()
        
               
        feature_importance = clf_train.get_score(importance_type='gain')
        feature_importance = sorted(feature_importance.items(), key=operator.itemgetter(1))
        feature_importance = pd.DataFrame(feature_importance, columns=['Feature', 'Gain'])
        feature_importance['Gain'] = feature_importance['Gain']/feature_importance['Gain'].sum()
               
            
        Train_set={}
        Train_set['TPR']=tpr
        Train_set['FPR']=fpr
        Train_set['ROC']=roc_auc
        #Train_set['mean_TPR']=mean_tpr
        #Train_set['mean_FPR']=mean_fpr
        #Train_set['mean_AUC']=mean_auc
        Train_set['Feature_importance']=feature_importance
    
        Test_set={}
        Test_set['TPR']=tpr_T
        Test_set['FPR']=fpr_T
        Test_set['ROC']=roc_auc_T
    
        Figures={}
        Figures['Train']=figname
        Figures['Test']=T_figname   
        plt.close("all")                   
        return(Train_set,Test_set,Figures)
# In[12]:
def wxlsx(Train_df,Test_df,Train_set,Test_set,Figures,nameoffile,featureset):
        import xlsxwriter
        writer = pd.ExcelWriter(nameoffile, engine='xlsxwriter')
        workbook  = writer.book
        bold = workbook.add_format({'bold': True})

        Train_img=Figures['Train']
        Test_img=Figures['Test'] 
        #---------------------Write Train and Test Data-------------------------------
        Train_data = pd.DataFrame(Train_df)        
        Train_data.to_excel(writer,sheet_name='Train_Data')
        Train_data_Handle = writer.sheets['Train_Data']

        Test_data = pd.DataFrame(Test_df)        
        Test_data.to_excel(writer,sheet_name='Test_Data')
        Test_data_Handle = writer.sheets['Test_Data']

        #---------------------Test Results-------------------------------
        tpr_test=Test_set['TPR']
        fpr_test=Test_set['FPR']
        roc_auc_test=Test_set['ROC']
        fpr_test_df=pd.DataFrame(fpr_test)
        tpr_test_df=pd.DataFrame(tpr_test)
        ROC_test_df=pd.concat([fpr_test_df,tpr_test_df],axis=1)
        MLX=[0,1]
        ROC_AUC_Test_0=[0,1]
        ROC_AUC_Test_1=[roc_auc_test,roc_auc_test]
        ROC_Test_data = pd.DataFrame(ROC_test_df)
        MLX_data=pd.DataFrame(MLX)
        ROC_AUC_Test_data_0=pd.DataFrame(ROC_AUC_Test_0)
        ROC_AUC_Test_data_1=pd.DataFrame(ROC_AUC_Test_1)
        ROC_AUC_Test_df=pd.concat([ROC_AUC_Test_data_0,ROC_AUC_Test_data_1],axis=1)
        ROC_Test_data.to_excel(writer,sheet_name='ROC_Test')
        ROC_AUC_Test_df.to_excel(writer,sheet_name='ROC_AUC_Test')
        ROC_Test_Handle = writer.sheets['ROC_Test']
        
        ROC_AUC_Test_Handle=writer.sheets['ROC_AUC_Test']
        ROC_Test_Handle.write('B1','FPR',bold)
        ROC_Test_Handle.write('C1','TPR',bold)
        fpr_test_range='=ROC_Test!$B2:$B'+str(fpr_test.size+1)
        tpr_test_range='=ROC_Test!$C2:$C'+str(tpr_test.size+1)
        ROC_AUC_Test_y_range='=ROC_AUC_Test!$C2:$C3'
        ROC_AUC_Test_Handle.hide()
        
        ROC_AUC_Test_x_range='=ROC_AUC_Test!$B2:$B3'
        ROC_Test_Handle.insert_image('F2', Test_img)
        
        if featureset == 1:
        #---------------------Feature importance Results-------------------------------
            feature_importance=Train_set['Feature_importance']
            Feature_Importance_data = pd.DataFrame(feature_importance)
            FN_df=list(Feature_Importance_data)
            Feature_Importance_data=Feature_Importance_data.sort_values(FN_df[1],axis=0,ascending=False)
            Feature_Importance_data.to_excel(writer,sheet_name='Feature_Importance')
            Feature_Importance_Handle=writer.sheets['Feature_Importance']
            Feature_Importance_Handle.write('B1','Feature',bold)
            Feature_Importance_Handle.write('C1','Gain',bold)
            Col_FG=Feature_Importance_data['Gain']
            count_rows=1;
            total_rows=(Col_FG.size+1)
            Val=0;
            for j in range(total_rows):
                if Val < 0.7:
                    Val=Val+Col_FG[j]
                    #print Val
                    count_rows+=1
            #print count_rows
            feature_feature_importance_range='=Feature_Importance!$B2:$B'+str(count_rows)
            gain_feature_importance_range='=Feature_Importance!$C2:$C'+str(count_rows)    
            #feature_feature_importance_range='=Feature_Importance!$B2:$B'+str((feature_importance.size/2)+1)
            #gain_feature_importance_range='=Feature_Importance!$C2:$C'+str((feature_importance.size/2)+1)
            Feature_Importance_chart1 = workbook.add_chart({'type': 'radar','subtype':'filled'})
            # Configure the first series.
            Feature_Importance_chart1.set_style(11)
            Feature_Importance_chart1.set_size({'x_scale':3, 'y_scale': 3})
            Feature_Importance_chart1.add_series({'name':'Feature Importance','data_labels': {'category': False},'categories': feature_feature_importance_range,'values':gain_feature_importance_range,})
            Feature_Importance_Handle.insert_chart('D50', Feature_Importance_chart1)


            Feature_Importance_chart = workbook.add_chart({'type': 'bar'})
            Feature_Importance_chart.set_style(11)
            Feature_Importance_chart.set_size({'x_scale': 3, 'y_scale': 3})
            Feature_Importance_chart.add_series({'name':'Feature Importance','data_labels': {'category': False},'categories': feature_feature_importance_range,'values':gain_feature_importance_range,})
            Feature_Importance_Handle.insert_chart('D2', Feature_Importance_chart)

        workbook.close()
 


# In[39]:
if Option_Array==1:
    [X_train, y_train, X_test, y_test, Train_df, Test_df]=OPTION_1(fname)
    print("inside")
else:
    [X_train, y_train, X_test, y_test, Train_df, Test_df]=OPTION_2(fname,TestTrainSplit)
        
if Run_Array[0]==1:
    Fig_save=SAVE_DIR+"Results"+"_RandomForestClassifier"
    Xls_save=SAVE_DIR+"Results"+"_RandomForestClassifier.xlsx"
    [Train_dict,Test_dict,Figures]=RandomFores(X_train,y_train,X_test,y_test,KFoldValue,Fig_save)
    wxlsx(Train_df,Test_df,Train_dict,Test_dict,Figures,Xls_save,1)
if Run_Array[1]==1:
    Fig_save=SAVE_DIR+"Results"+"_AdaBoostClassifier"
    Xls_save=SAVE_DIR+"Results"+"_AdaBoostClassifier.xlsx"	
    [Train_dict,Test_dict,Figures]=AdaBoo(X_train,y_train,X_test,y_test,KFoldValue,Fig_save)
    wxlsx(Train_df,Test_df,Train_dict,Test_dict,Figures,Xls_save,1)
if Run_Array[2]==1:
    Fig_save=SAVE_DIR+"Results"+"_BaggingClassifier"
    Xls_save=SAVE_DIR+"Results"+"_BaggingClassifier.xlsx"
    [Train_dict,Test_dict,Figures]=BaggingClassifi(X_train,y_train,X_test,y_test,KFoldValue,Fig_save)
    wxlsx(Train_df,Test_df,Train_dict,Test_dict,Figures,Xls_save,0)
if Run_Array[3]==1:
    Fig_save=SAVE_DIR+"Results"+"_GradientBoostClassifier"
    Xls_save=SAVE_DIR+"Results"+"_GradientBoostClassifier.xlsx"
    [Train_dict,Test_dict,Figures]=GradientBoo(X_train,y_train,X_test,y_test,KFoldValue,Fig_save)
    wxlsx(Train_df,Test_df,Train_dict,Test_dict,Figures,Xls_save,1)
if Run_Array[4]==1:
    Fig_save=SAVE_DIR+"Results"+"_GuassianNBClassifier"
    Xls_save=SAVE_DIR+"Results"+"_GuassianNBClassifier.xlsx"
    [Train_dict,Test_dict,Figures]=Graussiannb(X_train,y_train,X_test,y_test,KFoldValue,Fig_save)
    wxlsx(Train_df,Test_df,Train_dict,Test_dict,Figures,Xls_save,0)
if Run_Array[5]==1:
    Fig_save=SAVE_DIR+"Results"+"_BernoulliNBClassifier"
    Xls_save=SAVE_DIR+"Results"+"_BernoulliNBClassifier.xlsx"
    [Train_dict,Test_dict,Figures]=Bernoullinb(X_train,y_train,X_test,y_test,KFoldValue,Fig_save)
    wxlsx(Train_df,Test_df,Train_dict,Test_dict,Figures,Xls_save,0)
if Run_Array[6]==1:
    Fig_save=SAVE_DIR+"Results"+"_LogisticRegressionClassifier"
    Xls_save=SAVE_DIR+"Results"+"_LogisticRegressionClassifier.xlsx"
    wxlsx(Train_df,Test_df,Train_dict,Test_dict,Figures,Xls_save,0)
    [Train_dict,Test_dict,Figures]=LogisticRegressi(X_train,y_train,X_test,y_test,KFoldValue,Fig_save)
if Run_Array[7]==1:
    Fig_save=SAVE_DIR+"Results"+"_XGBoostClassifier"
    Xls_save=SAVE_DIR+"Results"+"_XGBoostClassifier.xlsx"
    [Train_dict,Test_dict,Figures]=XGBOOST(X_train,y_train,X_test,y_test,KFoldValue,Fig_save)
    wxlsx(Train_df,Test_df,Train_dict,Test_dict,Figures,Xls_save,1)


# In[ ]:



