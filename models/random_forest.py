from sklearn.model_selection import GridSearchCV,GroupShuffleSplit
from imblearn.over_sampling import RandomOverSampler

from models.model import Model
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(Model):
    def __init__(self, param_grid, output_path, random_state, visualization_weight):
        super().__init__(
            param_grid=param_grid,
            output_path=output_path + "random_forest_model/",
            random_state=random_state,
            visualization_weight=visualization_weight,
            modelname="random_forest",
        )

    def _build_model_random_forest(self, X_train, X_test, y_train, y_test,tax_id_index_mapping):
        rf = RandomForestClassifier(random_state=self.random_state)
        X_train= X_train.merge(tax_id_index_mapping, left_index=True, right_on="index")
        # cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        #cv_method = TaxIDStratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_method = GroupShuffleSplit(n_splits=5,train_size=.8, random_state=self.random_state)
        oversample = RandomOverSampler(sampling_strategy="minority")
        
        #X_train.drop(columns=["index", "tax_id"], inplace=True)
        grid_search = GridSearchCV(rf, self.param_grid, 
                                   # cv=cv_method, 
                                   cv=None,
                                   scoring="roc_auc")

        # scaler = StandardScaler()  # Initialize StandardScaler
        # X_train_fold = scaler.fit_transform(X_train_fold)
        # X_val_fold = scaler.transform(X_val_fold)
        # X_test = scaler.transform(X_test)
        #
        # joblib.dump(scaler, f"{self.output_path}scaler.pkl")

        X_train_res, y_train_res = oversample.fit_resample(X_train, y_train)
        unique_tax_ids = np.unique(X_train_res["tax_id"])
        tax_id_to_group = {tax_id: group_num for group_num, tax_id in enumerate(unique_tax_ids)}
        groups = np.array([tax_id_to_group[tax_id] for tax_id in X_train_res["tax_id"]])
        X_train_res.drop(columns=["index", "tax_id"], inplace=True)
        grid_search = GridSearchCV(rf, self.param_grid, 
                                   cv=cv_method.split(X_train_res, y_train_res, groups), 
                                   scoring="roc_auc")
        
        categorical_columns = X_train_res.select_dtypes(include=["object"]).columns
        X_train_res = pd.get_dummies(
            X_train_res, columns=categorical_columns, dtype=bool
        )

        # One-hot encode categorical columns in X_test using the same set of categories as X_train
        categorical_columns = X_test.select_dtypes(include=["object"]).columns
        X_test = pd.get_dummies(X_test, columns=categorical_columns, dtype=bool)
        all_columns = X_train_res.columns.union(X_test.columns)

        missing_cols = set(all_columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = False

        missing_cols = set(all_columns) - set(X_train_res.columns)
        for col in missing_cols:
            X_train_res[col] = False

        X_test = X_test[X_train_res.columns]

        y_train_res = np.ravel(y_train_res)
        y_test = np.ravel(y_test)
        X_train_res.interpolate(method="linear", inplace=True)
        X_test.interpolate(method="linear", inplace=True)

        grid_search.fit(X_train_res, y_train_res)
        rf_best = grid_search.best_estimator_

        fraud_recall,threshold = self.draw_matrix(rf_best, X_test, y_test)

        if fraud_recall >= self.best_fraud_recall:
            self.best_fraud_recall = fraud_recall
            self.best_model_for_frauds = rf_best

        best_params = self._get_best_params(X_test, y_test)
        self.best_model = self.best_model_for_frauds
        
        score_df = pd.DataFrame(grid_search.cv_results_)
        score_df.to_excel(self.output_path + 'random_forest_results.xlsx')

        return grid_search.best_params_, best_params,threshold

    def run(self, X_train, X_test, y_train, y_test, tax_id_index_mapping):
        parameters, performance_metrics, threshold = self._build_model_random_forest(
            X_train, X_test, y_train, y_test,tax_id_index_mapping
        )
        self._save_model(
            best_model_to_export=self.best_model,
            parameters=parameters,
            modelname=self.modelname,
        )
        self._save_results(parameters, performance_metrics)
        return threshold
