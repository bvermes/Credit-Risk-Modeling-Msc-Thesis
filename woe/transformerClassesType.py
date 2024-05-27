import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import feature_selection
import sklearn
from sklearn.preprocessing import StandardScaler
import traceback
import re
import scipy.stats.stats as stats
from pandas import Series
import numpy as np


class clipTransformer(BaseEstimator, TransformerMixin):
    """Cutting the outliers respectively to lower/upper bounds"""

    def __init__(self, quantile=0.25):
        self.quantile = quantile
        self.lower = None
        self.upper = None
        self.cols_to_cut = None

    def fit(self, X, y=None):
        self.cols_to_cut = []

        for col in X.columns:
            if X[col].unique().shape[0] > 2:
                self.cols_to_cut.append(col)
        self.lower = X[self.cols_to_cut].quantile(self.quantile)
        self.upper = X[self.cols_to_cut].quantile(1 - self.quantile)
        self.iqr = self.upper - self.lower
        self.lower_outlier = self.lower - 1.5 * self.iqr
        self.upper_outlier = self.upper + 1.5 * self.iqr

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.cols_to_cut] = X_[self.cols_to_cut].clip(
            lower=self.lower, upper=self.upper, axis=1
        )
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)


class missingClassHandler(BaseEstimator, TransformerMixin):
    """[summary]

    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]
    """

    def __init__(self, missing_value_option):
        self.missing_value_option = missing_value_option

    def fit(self, X, y=None):
        self.base_columns = X.columns
        self.mean_values = X.mean()
        self.median_values = X.median()

    def transform(self, X, y=None):
        X_ = X.copy()
        if self.missing_value_option == "median":
            X_ = X_.fillna(self.median_values)
        elif self.missing_value_option == "mean":
            X_ = X_.fillna(self.mean_values)
        else:
            X_ = X_.dropna()
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)


class correlationTransformer(BaseEstimator, TransformerMixin):
    """[summary]

    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]
    """

    def __init__(self, correlation_treshold):
        self.correlation_treshold = correlation_treshold

    def fit(self, X, y=None):
        correlation_matrix = X.corr()
        correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > self.correlation_treshold:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
        self.correlated_features = correlated_features

    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = X_.drop(self.correlated_features, axis=1)

        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)


class transformedKbest(BaseEstimator, TransformerMixin):
    """[summary]

    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]
    """

    def __init__(self, p_value_treshold, feature_number):
        self.p_value_treshold = p_value_treshold
        self.feature_number = feature_number

    def fit(self, X, y):
        f_classif_results = sklearn.feature_selection.f_classif(X, y)
        p_values = pd.Series(f_classif_results[1])
        p_values.index = X.columns
        p_values.sort_values(ascending=True, inplace=True)
        p_values = p_values[p_values < self.p_value_treshold]
        if p_values.shape[0] < self.feature_number:
            self.final_variables = p_values.index.to_list()
        else:
            self.final_variables = p_values.iloc[: self.feature_number,].index.to_list()

    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = X_.loc[:, self.final_variables]
        return X_

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


# This is unneccesary


class removeConstantFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.constant_filter = feature_selection.VarianceThreshold(
            threshold=self.threshold
        )
        self.constant_filter.fit(X)

    def transform(self, X, y=None):
        X_ = X.copy()
        X_array = self.constant_filter.transform(X_)
        X_ = pd.DataFrame(
            data=X_array,
            columns=X_.columns[self.constant_filter.get_support()],
        )
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)


class transformedScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        self.scaler = StandardScaler()

        self.scaler.fit(X)

    def transform(self, X, y=None):
        X_ = X.copy()
        X_array = self.scaler.transform(X_)
        X_ = pd.DataFrame(data=X_array, columns=list(X_.columns))
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)

    # remove_correlated_features
    # RFE based selection might be removed
    # Anova based selection it is a must


# start writing woe based transformation


class transformDummyToCategorical(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def contains_digit(self, str):
        contains = False
        for char in str:
            if char.isdigit():
                contains = True

        return contains

    def find_dummy_columns(self, X, y=None):
        col_list = X.columns.to_list()
        variables_selected = []
        dummy_vars = []
        for var in col_list:
            short_var = var[0 : (len(var) - 3)]
            if any(short_var in s for s in variables_selected):
                continue
            matching = [s for s in col_list if short_var in s]
            contain_num_list = []
            for s in matching:
                if self.contains_digit(s):
                    contain_num_list.append(s)
            if len(matching) > 2 and len(matching) == len(contain_num_list):
                dummy_vars.append(matching)
                variables_selected.append(short_var)
        return dummy_vars

    def fit(self, X, y=None):
        self.dummy_columns = self.find_dummy_columns(X)

    def transform(self, X, y=None):
        X_ = X.copy()
        for var_list in self.dummy_columns:
            df_dummy_var = X_[var_list]
            first_var = var_list[0]
            short_var = first_var[0 : (len(first_var) - 3)]
            x = df_dummy_var.stack()
            result = pd.Series(pd.Categorical(x[x != 0].index.get_level_values(1)))
            X_ = X_.drop(var_list, axis=1)
            X_[short_var] = result.astype(str)
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)


class transformIntoWoeValues(BaseEstimator, TransformerMixin):
    def __init__(self, best_n_variables):
        self.best_n_variables = best_n_variables

    def mono_bin(self, Y, X, n=20):

        df1 = pd.DataFrame({"X": X, "Y": Y})
        justmiss = df1[["X", "Y"]][df1.X.isnull()]
        notmiss = df1[["X", "Y"]][df1.X.notnull()]
        r = 0
        while np.abs(r) < 1:
            try:
                d1 = pd.DataFrame(
                    {"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)}
                )
                d2 = d1.groupby("Bucket", as_index=True, observed=False)
                r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
                n = n - 1
            except Exception as e:
                n = n - 1

        if len(d2) == 1:
            n = 3
            bins = np.quantile(notmiss["X"], np.linspace(0, 1, n))
            if len(np.unique(bins)) == 2:
                bins = np.insert(bins, 0, 1)
                bins[1] = bins[1] - (bins[1] / 2)
            d1 = pd.DataFrame(
                {
                    "X": notmiss.X,
                    "Y": notmiss.Y,
                    "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True),
                }
            )
            d2 = d1.groupby("Bucket", as_index=True, observed=False)

        d3 = pd.DataFrame({}, index=[])
        d3["MIN_VALUE"] = d2.min().X
        d3["MAX_VALUE"] = d2.max().X
        d3["COUNT"] = d2.count().Y
        d3["EVENT"] = d2.sum().Y
        d3["NONEVENT"] = d2.count().Y - d2.sum().Y
        d3 = d3.reset_index(drop=True)

        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({"MIN_VALUE": np.nan}, index=[0])
            d4["MAX_VALUE"] = np.nan
            d4["COUNT"] = justmiss.count().Y
            d4["EVENT"] = justmiss.sum().Y
            d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
            d3 = pd.concat([d3, d4], ignore_index=True)

        d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(
            d3.DIST_EVENT / d3.DIST_NON_EVENT
        )
        d3["VAR_NAME"] = "VAR"
        d3 = d3[
            [
                "VAR_NAME",
                "MIN_VALUE",
                "MAX_VALUE",
                "COUNT",
                "EVENT",
                "EVENT_RATE",
                "NONEVENT",
                "NON_EVENT_RATE",
                "DIST_EVENT",
                "DIST_NON_EVENT",
                "WOE",
                "IV",
            ]
        ]
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3.IV = d3.IV.sum()

        return d3

    def char_bin(self, Y, X):

        df1 = pd.DataFrame({"X": X, "Y": Y})
        justmiss = df1[["X", "Y"]][df1.X.isnull()]
        notmiss = df1[["X", "Y"]][df1.X.notnull()]
        df2 = notmiss.groupby("X", as_index=True)

        d3 = pd.DataFrame({}, index=[])
        d3["COUNT"] = df2.count().Y
        d3["MIN_VALUE"] = df2.sum().Y.index
        d3["MAX_VALUE"] = d3["MIN_VALUE"]
        d3["EVENT"] = df2.sum().Y
        d3["NONEVENT"] = df2.count().Y - df2.sum().Y

        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({"MIN_VALUE": np.nan}, index=[0])
            d4["MAX_VALUE"] = np.nan
            d4["COUNT"] = justmiss.count().Y
            d4["EVENT"] = justmiss.sum().Y
            d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
            d3 = pd.concat([d3, d4], ignore_index=True)

        d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
        total_event = d3["EVENT"].sum()
        total_non_event = d3["NONEVENT"].sum()
        d3["DIST_EVENT"] = d3["EVENT"] / total_event
        d3["DIST_NON_EVENT"] = d3["NONEVENT"] / total_non_event
        d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(
            d3.DIST_EVENT / d3.DIST_NON_EVENT
        )
        d3["VAR_NAME"] = "VAR"
        d3 = d3[
            [
                "VAR_NAME",
                "MIN_VALUE",
                "MAX_VALUE",
                "COUNT",
                "EVENT",
                "EVENT_RATE",
                "NONEVENT",
                "NON_EVENT_RATE",
                "DIST_EVENT",
                "DIST_NON_EVENT",
                "WOE",
                "IV",
            ]
        ]
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3.IV = d3.IV.sum()
        d3 = d3.reset_index(drop=True)

        return d3

    def data_vars(self, df1, target):

        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]
        vars_name = re.compile(r"\((.*?)\).*$").search(code).groups()[0]
        final = (re.findall(r"[\w']+", vars_name))[-1]

        x = df1.dtypes.index
        count = -1

        for i in x:
            if i.upper() not in (final.upper()):
                if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                    conv = self.mono_bin(target, df1[i])
                    conv["VAR_NAME"] = i
                    count = count + 1
                else:
                    conv = self.char_bin(target, df1[i])
                    conv["VAR_NAME"] = i
                    count = count + 1

                if count == 0:
                    iv_df = conv
                else:
                    iv_df = pd.concat([iv_df, conv], ignore_index=True)

        iv = pd.DataFrame({"IV": iv_df.groupby("VAR_NAME").IV.max()})
        iv = iv.reset_index()
        return (iv_df, iv)

    def transform_one_var(self, df, var, transform_prefix=""):
        small_df = self.final_iv[self.final_iv["VAR_NAME"] == var]
        transform_dict = dict(zip(small_df.MAX_VALUE, small_df.WOE))
        replace_cmd = ""
        replace_cmd1 = ""
        try:
            for i in sorted(transform_dict.items()):
                if str(i[0]) == "nan":
                    replace_cmd = (
                        replace_cmd + str(i[1]) + str(" if math.isnan(x) else ")
                    )
                    replace_cmd1 = (
                        replace_cmd1 + str(i[1]) + str(" if math.isnan(x) else ")
                    )
                else:
                    replace_cmd = (
                        replace_cmd
                        + str(i[1])
                        + str(" if x <= ")
                        + str(i[0])
                        + " else "
                    )
                    replace_cmd1 = (
                        replace_cmd1
                        + str(i[1])
                        + str(' if x == "')
                        + str(i[0])
                        + '" else '
                    )
        except:
            for i in transform_dict.items():
                if str(i[0]) == "nan":
                    replace_cmd = (
                        replace_cmd + str(i[1]) + str(" if math.isnan(x) else ")
                    )
                    replace_cmd1 = (
                        replace_cmd1 + str(i[1]) + str(" if math.isnan(x) else ")
                    )
                else:
                    replace_cmd = (
                        replace_cmd
                        + str(i[1])
                        + str(" if x <= ")
                        + str(i[0])
                        + " else "
                    )
                    replace_cmd1 = (
                        replace_cmd1
                        + str(i[1])
                        + str(' if x == "')
                        + str(i[0])
                        + '" else '
                    )
        replace_cmd = replace_cmd + "0"
        replace_cmd1 = replace_cmd1 + "0"
        if replace_cmd != "0":
            try:
                df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd))
            except:
                df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd1))
        return df

    def fit(self, X, y):
        self.final_iv, self.IV = self.data_vars(X, y)
        self.var_list = (
            self.IV.sort_values("IV", ascending=False)
            .iloc[: self.best_n_variables, :]["VAR_NAME"]
            .to_list()
        )
        print("WoE fitting has been completed.")

    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = X_[self.var_list]
        for var in self.var_list:
            X_ = self.transform_one_var(X_, var)

        X_ = X_.astype(float)
        return X_

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
