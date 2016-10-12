import numpy as np
import pandas as pd
from pandas import DataFrame

def create_title_classes(df):
    df['Mil'] = np.where((df['Name'].str.contains('Capt') | df['Name'].str.contains('Major\.') | df['Name'].str.contains('Col\.') | df['Name'].str.contains('Brig\.') | df['Name'].str.contains('Gen\.')), 1, 0)
    df['Rev'] = np.where(df['Name'].str.contains('Rev'), 1, 0)
    df['Dr'] = np.where(df['Name'].str.contains('Dr\.'), 1, 0) 
    df['Master'] = np.where(df['Name'].str.contains('Master'), 1, 0) 
    df['Mrs'] = np.where(df['Name'].str.contains('Mrs\.') | df['Name'].str.contains('Mme\.'), 1, 0) 
    df['Mr'] = np.where(df['Name'].str.contains('Mr\.') | ((df['Sex'] == 'male') & (df['Dr'] == 0) & (df['Mil'] == 0) & (df['Rev'] == 0) & (df['Master'] == 0)),1,0)
    df['Miss'] = np.where(df['Name'].str.contains('Miss') | ((df['Sex'] == 'female') & (df['Dr'] == 0) & (df['Mrs'] == 0)), 1, 0) 

    print "Number of Mr.: %d " % len(df[df['Mr'] == 1])
    print "Number of Mrs.: %d " % len(df[df['Mrs'] == 1])
    print "Number of Miss: %d " % len(df[df['Miss'] == 1])
    print "Number of Dr.: %d " % len(df[df['Dr'] == 1])
    print "Number of Master: %d " % len(df[df['Master'] == 1])
    print "Number of Reverends: %d " % len(df[df['Rev'] == 1])
    print "Number of Military Officers: %d " % len(df[df['Mil'] == 1])
    print "Total number of people: %d " % len(df)

    return df

def one_hot_encode_embarked_variables(df):
    embark_dummies  = pd.get_dummies(df['Embarked'])
    embark_dummies.drop(['S'], axis=1, inplace=True)
    df.drop(['Embarked'], axis=1,inplace=True)
    return df.join(embark_dummies)

def one_hot_encode_gender_variables(df):
    gender_dummies = pd.get_dummies(df['Sex'])
    gender_dummies.drop(['male'], axis=1, inplace=True)
    df = df.join(gender_dummies)
    df.drop(['Sex'], axis=1,inplace=True)
    return df

def one_hot_encode_class_variables(df):
    pclass_dummies  = pd.get_dummies(df['Pclass'])
    pclass_dummies.columns = ['Class_1','Class_2','Class_3']
    pclass_dummies.drop(['Class_3'], axis=1, inplace=True)
    df.drop(['Pclass'],axis=1,inplace=True)
    df = df.join(pclass_dummies)
    return df

def define_young_old_miss_features(df):
    df['Young_Miss'] = np.where(((df['Miss'] == 1) & (df['Parch'] > 0)),1,0)
    df['Old_Miss'] = np.where(((df['Miss'] == 1) & (df['Parch'] == 0)),1,0)
    df.drop(['Miss'],axis=1,inplace=True)
    return df

def make_family_size_feature(df):
    df['Family_Size'] = 1 + df['Parch'] + df['SibSp']
    df.drop(['SibSp'],axis=1,inplace=True)
    return df
