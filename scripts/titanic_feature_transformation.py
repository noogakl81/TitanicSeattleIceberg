import pandas as pd

def min_max_scaling(X_train,X_test):
    X_total = pd.concat([X_train,X_test])

    from sklearn.preprocessing import MinMaxScaler

    min_max_scaler = MinMaxScaler()
    min_max_scaler.partial_fit(X_total)

    X_train = pd.DataFrame(min_max_scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(min_max_scaler.transform(X_test), columns=X_test.columns)
    return X_train,X_test,min_max_scaler

def transform_features_by_PCA(X_train,X_test):
    X_total = pd.concat([X_train,X_test])

    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(X_total)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

