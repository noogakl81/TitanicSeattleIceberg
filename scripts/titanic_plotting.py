import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
sns.set_style('whitegrid')

def plot_age_survival_distribution(df):
    # peaks for survived/not survived passengers by their age
    facet = sns.FacetGrid(df, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,'Age',shade= True)
    facet.set(xlim=(0, df['Age'].max()))
    facet.add_legend()

def visualize_decision_tree_and_write_to_file(tree_to_see,features,classes,filename):
    from sklearn.externals.six import StringIO
    import pydotplus
    from sklearn import tree

    dot_data = StringIO()
    tree.export_graphviz(tree_to_see, out_file=dot_data,
                         feature_names=features,
                         class_names=classes,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename)
    return graph

def make_class_survival_plot(df):
    sns.factorplot('Pclass','Survived',order=[1,2,3], data=df,size=5)

def make_parch_sibsp_survival_plots(df):
    fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

    # sns.factorplot('Family',data=df,kind='count',ax=axis1)
    sns.countplot(x='Parch', data=df, order=[1,0], ax=axis1)

    # average of survived for those who had/didn't have any family member
    parch_perc = df[["Parch", "Survived"]].groupby(['Parch'],as_index=False).mean()
    sns.barplot(x='Parch', y='Survived', data=parch_perc, order=[1,0], ax=axis2)

    axis1.set_xticklabels(["With Parents or Children","Alone"], rotation=0)

    fig2, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

    # sns.factorplot('Family',data=df,kind='count',ax=axis1)
    sns.countplot(x='SibSp', data=df, order=[1,0], ax=axis1)

    # average of survived for those who had/didn't have any family member
    sibsp_perc = df[["SibSp", "Survived"]].groupby(['SibSp'],as_index=False).mean()
    sns.barplot(x='SibSp', y='Survived', data=sibsp_perc, order=[1,0], ax=axis2)

    axis1.set_xticklabels(["With Sibling or Spouse","Alone"], rotation=0)

def make_fare_plot(df):
    # get fare for survived & didn't survive passengers 
    fare_not_survived = df["Fare"][df["Survived"] == 0]
    fare_survived     = df["Fare"][df["Survived"] == 1]
    # get average and std for fare of survived/not survived passengers
    average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
    std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])
    # plot
    df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))
    average_fare.index.names = std_fare.index.names = ["Survived"]
    average_fare.plot(yerr=std_fare,kind='bar',legend=False)

def make_embarked_survival_plot(df):
    sns.factorplot('Embarked','Survived', data=df,size=4,aspect=3)
    fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
    sns.countplot(x='Embarked', data=df, ax=axis1)
    sns.countplot(x='Survived', hue="Embarked", data=df, order=[1,0], ax=axis2)
    # group by embarked, and get the mean for survived passengers for each value in Embarked
    embark_perc = df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
    sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

