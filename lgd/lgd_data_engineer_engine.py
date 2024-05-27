from parent_classes.data_engineer_engine import DataEngineerEngine
import sys
import pandas as pd

from woe.woebin import *
from woe.var_filter import *
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
class LGDDataEngineerEngine(DataEngineerEngine):
    def __init__(
        self,
        df,
        output_path,
        test_size,
        validation_size,
        random_state,
        visualization_weight,
        columns_to_drop,
    ):
        super().__init__(
            df,
            test_size,
            validation_size,
            random_state,
            visualization_weight,
            output_path + "data_preparation/",
            columns_to_drop,
            target="lgd",
        )
    
    def _clean_dataframe(self, df):
        return df

    def _feature_engineering(self):
        self.df["dnb_rating_failure_score_date"] = pd.to_datetime(
            self.df["dnb_rating_failure_score_date"]
        ).dt.year
        self.df.dropna(axis=1, how="all", inplace=True)

        self.df["date_month"] = self.df["date_month"].apply(lambda x: x.split("T")[0])
        self.df["date_month"] = pd.to_datetime(self.df["date_month"]).dt.strftime(
            "%Y-%m"
        )
        original_data = self.df.copy()
        outliers = self._detect_outliers()
        outliers.to_excel(
            self.output_path + "outliers.xlsx", index=False
        )
        #After reviewing the outliers, we can correct the rows that are probably incorrect
        self.df = self._clean_dataframe(self.df)

        for visualization_function in [
            self._visualizer._top5_corr,
            self._visualizer._top_missing_features,
        ]:
            weight = self._visualizer.visualization_weights[visualization_function]
            if self.visualization_weight >= weight:
                visualization_function()

    def _calculate_woe_iv(self):
        dt_s = self.df
        
        null_percentage = dt_s.isnull().mean(axis=1)
        dt_s = dt_s[null_percentage < 0.8]
        null_percentage = dt_s.isnull().mean()

        # Drop columns where the percentage of null values exceeds 90%
        columns_to_drop = null_percentage[null_percentage > 0.9].index
        dt_s.drop(columns_to_drop, axis=1, inplace=True)
        
        val = self._create_validation_data(dt_s, self.validation_size)
        self._visualizer.dt_s = dt_s
        self._visualizer.val = val
        for visualization_function in [
            self._visualizer._dist_unique_tax_ids_val,
            self._visualizer._dis_val_original_by_month,
        ]:
            weight = self._visualizer.visualization_weights[visualization_function]
            if self.visualization_weight >= weight:
                visualization_function()
        indices_to_remove = val.index
        dt_s = dt_s.drop(indices_to_remove)

        self.test_size = self.test_size / (1 - self.validation_size) 
        train_ids, test_ids = train_test_split(
            dt_s["tax_id"].unique(), test_size=self.test_size, random_state=42
        )

        train = dt_s[dt_s["tax_id"].isin(train_ids)]
        test = dt_s[dt_s["tax_id"].isin(test_ids)]

        train.reset_index(drop=True, inplace=True)
        tax_id_index_mapping = train.reset_index()[["index", "tax_id"]]
        test.reset_index(drop=True, inplace=True)

        self._visualizer.dt_s = dt_s
        self._visualizer.train = train
        self._visualizer.test = test

        for visualization_function in [
            self._visualizer._dist_unique_tax_ids,
            self._visualizer._dis_train_test_by_month,
            self._visualizer._dis_train_test_by_month_and_target,
        ]:
            weight = self._visualizer.visualization_weights[visualization_function]
            if self.visualization_weight >= weight:
                visualization_function()
        # all the time date_month, tax_id validation should be made before this step
        dt_s = dt_s.drop(
            ["tax_id", "date_month", "dnb_rating_failure_score_date"], axis=1
        )
        train = train.drop(
            ["tax_id", "date_month", "dnb_rating_failure_score_date"], axis=1
        )
        test = test.drop(
            ["tax_id", "date_month", "dnb_rating_failure_score_date"], axis=1
        )
        val = val.drop(
            ["tax_id", "date_month", "dnb_rating_failure_score_date"], axis=1
        )
        categorical_cols = [col for col in dt_s.columns if dt_s[col].dtype == 'object']
        dt_s = dt_s.drop(categorical_cols, axis=1)
        train = train.drop(categorical_cols, axis=1)
        test = test.drop(categorical_cols, axis=1)
        val = val.drop(categorical_cols, axis=1)
        
        
        X = dt_s.drop(["lgd"], axis=1)
        y = dt_s["lgd"]
        
        print("Number of NaN values in X before handling missing values:")
        print(X.isna().sum())

        X.fillna(method='ffill', inplace=True)
        X.interpolate(method='linear', inplace=True)
        print("Number of NaN values in X after handling missing values:")
        print(X.isna().sum())

        # RFE
        #categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        #preprocessor = ColumnTransformer(
        #    transformers=[
        #        ('cat', OneHotEncoder(), categorical_cols),
        #        ('num', SimpleImputer(strategy='mean'), ~X.columns.isin(categorical_cols))  # Impute numerical columns
        #    ],
        #    remainder='passthrough'
        #)
        #X = preprocessor.fit_transform(X)
        #estimator = LinearRegression()
        #rfe = RFE(estimator, n_features_to_select=5) 
        #rfe.fit(X, y)
        #selected_features = X.columns[rfe.support_]
        #selected_features = selected_features.insert(0, 'lgd')
        #
        #dt_s = dt_s[selected_features]
        #train = train[selected_features]
        #test = test[selected_features]
        #val = val[selected_features]

        bestfeatures = SelectKBest(score_func=f_regression, k=10)
        fit = bestfeatures.fit(X, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']
        print(featureScores.nlargest(10, 'Score'))
        X = pd.concat([X, dt_s["lgd"]], axis=1)
        
        selected_features = list(fit.scores_.argsort()[-10:][::-1])  # Indices of top 10 features
        
        columns_to_keep = X.columns[selected_features].copy()
        selected_features.append(X.columns.get_loc('lgd'))
        selected_columns = X.columns[selected_features]
        dt_s = dt_s[selected_columns]
        train = train[selected_columns]
        test = test[selected_columns]
        val = val[selected_columns]
        
        
        y_train = train.loc[:, "lgd"]
        X_train = train.drop(["lgd"], axis=1)
        y_test = test.loc[:, "lgd"]
        X_test = test.drop(["lgd"], axis=1)
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train, columns=columns_to_keep)
        X_test = pd.DataFrame(X_test, columns=columns_to_keep)
    
        
        # bins = woebin(
        #     dt_s,
        #     y="lgd",
        #     method="tree",
        # )
        # woebin_plot(bins, output=self.output_path)
        
        #train_woe = woebin_ply(train, bins)
        #test_woe = woebin_ply(test, bins)
        #woe_columns = columns_to_keep.copy()
        #for i in range(len(woe_columns)):
        #    woe_columns[i] += "_woe"
        #woe_columns.append("lgd")
        
        #train_woe = train_woe[woe_columns]
        #test_woe = test_woe[woe_columns]
        
        
        #X_train_woe = train_woe.drop(["lgd"], axis=1)
        #X_test_woe = test_woe.drop(["lgd"], axis=1)
        
        y_test = test.loc[:, "lgd"]
        y_train = train.loc[:, "lgd"]
        
        return (
            imputer,
            scaler,
            tax_id_index_mapping,
            columns_to_keep,
            val,
            X_train,
            X_test,
            y_train,
            y_test,
        )

    def run(self):

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self._clean_rows()
        self._remove_columns(self.columns_to_drop)
        self._feature_engineering()
        (
            imputer,
            scaler,
            tax_id_index_mapping,
            columns_to_keep,
            val,
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self._calculate_woe_iv()

        return (
            imputer,
            scaler,
            tax_id_index_mapping,
            columns_to_keep,
            val,
            X_train,
            X_test,
            y_train,
            y_test,
        )
