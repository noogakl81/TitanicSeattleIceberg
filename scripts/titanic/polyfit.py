r"""polyfit.py

Routines for constructing a polynomial interpolations of missing pandas data.
"""

import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polyfit(df, xlabels, ylabel, deg=3, inplace=False):
    r"""Return a polynomial fit of `xdata` using the `df` DataFrame.

    Given a Pandas DataFrame `df` compute a degree `deg` polynomial fit of the
    values in the column `ylabel` as a function of the data contained in the
    columns `xlabels`. The unknown / NaN entries in the `ylabel` column will be
    filled with the predicted values.

    Note that all of the data under the `xlabels` columns must be filled. That
    is, no NaNs are allowed.

    Parameters
    ----------
    df : DataFrame
    xlabels : list
        A list of column headers of the data frame to use as the "feature set"
        or independent variables in the polynomial fit.
    ylabel : str
        A column header of the data frame to predict. The dependent variable in
        the polynomial fit.
    deg : int
        (Default: 3) The degree of the polynomial fit.
    inplace : bool
        (Default: False) If True, modifies `df` inplace.

    Returns
    -------
    DataFrame
        If `inplace = False`, returns a new dataframe with the predicted
        values.
    """
    # create a copy if not inplace and perform inplace operations on this copy
    if not inplace:
        df = df.copy()

    # separate the known and unknown data in the ylabel
    df_known = df[pd.notnull(df[ylabel])]
    df_unknown = df[pd.isnull(df[ylabel])]

    # extract the desired features and known values as a numpy arrays
    X_known = df_known.loc[:,xlabels]
    X_known = X_known.as_matrix()
    Y_known = df_known.loc[:,ylabel]
    Y_known = Y_known.as_matrix()
    Y_known = Y_known.reshape(-1,1)  # must be a column vector
    X_unknown = df_unknown.loc[:,xlabels]
    X_unknown = X_unknown.as_matrix()

    # construct a set of "polynomial features", all of the monomials in the
    # dependent variables, so that we can transform the problem into a linear
    # regression problem
    poly = PolynomialFeatures(deg)
    X_known_poly = poly.fit_transform(X_known)
    X_unknown_poly = poly.fit_transform(X_unknown)

    # fit a linear model and predict on the unknown features
    clf = LinearRegression()
    clf.fit(X_known_poly, Y_known)
    Y_unknown = clf.predict(X_unknown_poly)

    # insert predicted values into the data frame. this is done-inplace with
    # either (a) the copy made at the beginning if `inplace` is set to False or
    # (b) the original if `inplace` is set to True
    df.loc[pd.isnull(df[ylabel]), ylabel] = Y_unknown.flatten()

    # return the copy made above if not inplace
    if not inplace:
        return df
    return None

