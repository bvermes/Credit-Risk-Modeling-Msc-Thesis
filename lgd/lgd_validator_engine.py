import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)
import os
import time
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils.data_utils import fill_null_with_most_similar_row
import os
import time
import matplotlib.pyplot as plt


class LGDValidatorEngine:
    def __init__(self, output_path, visualization_weight):
        self.output_path = output_path
        self.visualization_weight = visualization_weight
        self.model = None
        self.model_output_path = None

    def _draw_matrix(self, X, y, threshold,name_id):
        test_pred = self.model.predict_proba(X)[:, 1]
        y_pred = np.where(test_pred >= threshold, 1, 0)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()
        image_dir = self.model_output_path + "/images"
        os.makedirs(image_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                image_dir, f"{int(time.time())}_confusion_matrix_{name_id}.png"
            )
        )
        plt.close()

    def _draw_roc_curve(self, X, y, name_id):
        test_pred = self.model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, test_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic Curve")
        plt.legend(loc="lower right")
        plt.savefig(
            os.path.join(
                self.model_output_path, f"{int(time.time())}_roc_curve{name_id}.png"
            )
        )
        plt.close()

    def _draw_pr_curve(self, X, y, name_id):
        test_pred = self.model.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, test_pred)
        plt.figure()
        plt.plot(recall, precision, color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title("Precision-Recall Curve")
        plt.savefig(
            os.path.join(
                self.model_output_path, f"{int(time.time())}_pr_curve{name_id}.png"
            )
        )
        plt.close()
        
    def _draw_tree_with_routes(self, X_train, y_train):
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(self.model, ax=ax)
        plt.savefig(f"{self.model_output_path}decision_tree.png")
        plt.close()
        # Logic to show routes for each row
        pass

    def _draw_curve_for_each_column(self, X, y):
        # Assuming self.model is a fitted logistic regression model
        if not isinstance(self.model, LogisticRegression):
            raise ValueError("Model is not a Logistic Regression model")

        fig, axes = plt.subplots(nrows=1, ncols=X.shape[1], figsize=(16, 4))

        for col_idx, ax in enumerate(axes):
            # Scatter plot of the variable against predicted probabilities
            ax.scatter(X[:, col_idx], self.model.predict_proba(X)[:, 1], alpha=0.5)
            ax.set_xlabel(f"Feature {col_idx}")
            ax.set_ylabel("Predicted Probability")
            ax.set_title(f"Probability vs Feature {col_idx}")

        plt.tight_layout()
        plt.savefig(f"{self.model_output_path}probability_vs_feature.png")
        plt.close()

    def _draw_probability_alignment(self, X, y, name_id):
        # Assuming self.model is a fitted logistic regression model
        if not isinstance(self.model, LogisticRegression):
            raise ValueError("Model is not a Logistic Regression model")

        # Predict probabilities
        probas = self.model.predict_proba(X)[:, 1]
        y_1d = np.ravel(y)
        # Sort by predicted probability
        #sorted_indices = np.argsort(probas)
        #sorted_probas = probas[sorted_indices]
        #sorted_y = y[sorted_indices]
        result_df = pd.DataFrame({'probas': probas, 'y': y_1d})

        # Sort by predicted probability
        sorted_result_df = result_df.sort_values(by='probas')

        # Access sorted probas and y
        sorted_probas = sorted_result_df['probas'].values
        sorted_y = sorted_result_df['y'].values

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            np.arange(len(sorted_y)),
            sorted_y,
            c=sorted_probas,
            cmap="coolwarm",
            alpha=0.5,
        )
        ax.plot(sorted_probas, color="red", label="Predicted Probability")
        ax.set_xlabel("Sample Index (sorted by predicted probability)")
        ax.set_ylabel("Actual Value of y")
        ax.set_title("Alignment of Actual y with Predicted Probability")
        ax.legend()
        plt.savefig(f"{self.model_output_path}probability_alignment{name_id}.png")
        plt.close()

    def _validate_train(self, X_train, y_train,threshold):
        self._draw_matrix(X_train, y_train, threshold, "_train")
        self._draw_roc_curve(X_train, y_train, "_train")
        self._draw_pr_curve(X_train, y_train, "_train")
        if isinstance(self.model, LogisticRegression):
            self._draw_probability_alignment(X_train, y_train, "_train")
        elif isinstance(self.model, DecisionTreeClassifier):
            self._draw_tree_with_routes(X_train, y_train)
        else:
            print("Unsupported model type for data visualization.")

    def _validate_test(self, X_test, y_test,threshold):
        self._draw_matrix(X_test, y_test, threshold, "_test")
        self._draw_roc_curve(X_test, y_test, "_test")
        self._draw_pr_curve(X_test, y_test, "_test")
        if isinstance(self.model, LogisticRegression):
            self._draw_probability_alignment(X_test, y_test, "_test")

    def _validate_val(self,X,y,threshold):
        self._draw_matrix(X, y,threshold, "_validation")
        self._draw_roc_curve(X, y, "_validation")
        self._draw_pr_curve(X, y, "_validation")
        if isinstance(self.model, LogisticRegression):
            self._draw_probability_alignment(X, y, "_validation")

    def run(
        self,
        imputer,
        scaler,
        columns_to_keep,
        train_columns_with_dummies,
        val,
        X_train,
        X_test,
        y_train,
        y_test,
        loaded_models,
        thresholds,
        ):
        
        print("Running LGD validation engine")
        self.scaler = scaler
        self.imputer = imputer
        
        for model_name, loaded_model, _bad_scaler in loaded_models:
            self.model_output_path = self.output_path + model_name + "/"
            os.makedirs(self.model_output_path, exist_ok=True)
            self.model = loaded_model
            X_train_model = None
            X_test_model = None
            y_train_model = None
            y_test_model = None
            X_val_model = None
            y_val_model = None
            if isinstance(self.model, LogisticRegression):
                threshold = thresholds['log_reg_treshold']
                X_train_model = X_train
                for index, row in X_train_model.iterrows():
                    if row.isnull().any():
                        fill_null_with_most_similar_row(X_train_model, index)
            
                X_test_model = X_test
                for index, row in X_test_model.iterrows():
                    if row.isnull().any():
                        fill_null_with_most_similar_row(X_test_model, index)
            
                y_train_model = np.ravel(y_train)
                y_test_model = np.ravel(y_test)
                X_val = val[columns_to_keep]
                
                X_val_model = X_val[X_train.columns]
                for index, row in X_val_model.iterrows():
                    if row.isnull().any():
                        fill_null_with_most_similar_row(X_val_model, index)
            
                
                y_val_model = val.loc[:, "lgd"]
                y_val_model = np.ravel(y_val_model)
            elif isinstance(self.model, DecisionTreeClassifier) or isinstance(
                self.model, RandomForestClassifier
            ):
                threshold = thresholds['dt_treshold'] if isinstance(self.model, DecisionTreeClassifier) else thresholds['rf_treshold']
                categorical_columns = X_train.select_dtypes(include=["object"]).columns
                X_train_model = pd.get_dummies(
                    X_train, columns=categorical_columns, dtype=bool
                )
                X_test_model = pd.get_dummies(
                    X_test, columns=categorical_columns, dtype=bool
                )
                missing_cols = set(train_columns_with_dummies) - set(
                    X_train_model.columns
                )
                for col in missing_cols:
                    X_train_model[col] = False

                missing_cols = set(train_columns_with_dummies) - set(
                    X_test_model.columns
                )
                for col in missing_cols:
                    X_test_model[col] = False

                X_train_model = X_train_model[train_columns_with_dummies]
                X_test_model = X_test_model[train_columns_with_dummies]

                # Good example for raw data evaluation
                X_val = val[columns_to_keep]
                y_val = val.loc[:, "lgd"]
                X_val_model = pd.get_dummies(
                    X_val, columns=categorical_columns, dtype=bool
                )
                missing_cols = set(train_columns_with_dummies) - set(
                    X_val_model.columns
                )
                for col in missing_cols:
                    X_val_model[col] = False
                X_val_model = X_val_model[train_columns_with_dummies]
                if isinstance(self.model, RandomForestClassifier):
                    X_train_model.interpolate(method="linear", inplace=True)
                    X_test_model.interpolate(method="linear", inplace=True)
                    X_val_model.interpolate(method="linear", inplace=True)
                    
                    total_rows_before = len(X_train_model)
                    X_train_model.dropna(inplace=True)
                    total_rows_after = len(X_train_model)
                    percentage_dropped = ((total_rows_before - total_rows_after) / total_rows_before) * 100
                    rows_to_drop = set(range(total_rows_before)) - set(X_train_model.index)
                    y_train_model = y_train
                    y_train_model.drop(rows_to_drop, inplace=True)
                    print(f"Percentage of rows dropped from train, because of null: {percentage_dropped:.2f}%")
                    
                    total_rows_before = len(X_test_model)
                    X_test_model.dropna(inplace=True)
                    total_rows_after = len(X_test_model)
                    percentage_dropped = ((total_rows_before - total_rows_after) / total_rows_before) * 100
                    rows_to_drop = set(range(total_rows_before)) - set(X_test_model.index)
                    y_test_model = y_test
                    y_test_model.drop(rows_to_drop, inplace=True)
                    print(f"Percentage of rows dropped from test, because of null: {percentage_dropped:.2f}%")
                    
                    total_rows_before = len(X_val_model)
                    X_val_model.dropna(inplace=True)
                    total_rows_after = len(X_val_model)
                    percentage_dropped = ((total_rows_before - total_rows_after) / total_rows_before) * 100
                    rows_to_drop = set(range(total_rows_before)) - set(X_val_model.index)
                    y_val_model = y_val
                    y_val_model.drop(rows_to_drop, inplace=True)
                    print(f"Percentage of rows dropped from val, because of null: {percentage_dropped:.2f}%")

                y_train_model = np.ravel(y_train)
                y_test_model = np.ravel(y_test)
                y_val_model = np.ravel(y_val)

            # elif isinstance(self.model, RandomForestClassifier):
            #    pass
            else:
                raise ValueError("Unsupported model type for validation.")
            self._validate_train(X_train_model, y_train_model,threshold)
            self._validate_test(X_test_model, y_test_model,threshold)
            self._validate_val(X_val_model, y_val_model,threshold)

