from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_error,
)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import pickle


class Model:
    def __init__(
        self, param_grid, output_path, random_state, visualization_weight, modelname
    ):
        self.param_grid = param_grid
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.random_state = random_state
        self.visualization_weight = visualization_weight
        self.modelname = modelname

        self.best_fraud_recall = float('-inf')
        self.best_model = None
        self.best_model_for_frauds = None
        
    def _find_best_threshold(self, lr, X_test, y_test):
        test_pred = lr.predict_proba(X_test)[:, 1]
        thresholds = np.linspace(0, 1, 100) 
        best_threshold = None
        best_score = float('-inf')

        for threshold in thresholds:
            y_pred = np.where(test_pred >= threshold, 1, 0)
            cm = confusion_matrix(y_test, y_pred)
            true_positive = cm[1, 1]
            true_negative = cm[0, 0]
            false_positive = cm[0, 1]
            false_negative = cm[1, 0]
            score = true_positive * 0 + true_negative * (0.05) + false_positive * (-0.05) + false_negative * (-0.32)
            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, score
    
    def draw_matrix(self, lr, X_test, y_test):
        test_pred = lr.predict_proba(X_test)[:, 1]
        threshold, score = self._find_best_threshold(lr, X_test, y_test)
        y_pred = np.where(test_pred >= threshold, 1, 0)
        cm = confusion_matrix(y_test, y_pred)
        fraud_recall = (
            cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        )
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()
        image_dir = self.output_path + "/images"
        os.makedirs(image_dir, exist_ok=True)
        plt.savefig(os.path.join(image_dir, f"{int(time.time())}_confusion_matrix.png"))
        plt.close()
        return fraud_recall, threshold

    def _save_results(self, parameters, performance_metrics):
        file_path = f"{self.output_path}/{self.modelname}parameter_results.xlsx"
        try:
            df = pd.read_excel(file_path)
            print("Existing file found. Reading data...")
        except FileNotFoundError:
            print("File not found. Creating a new file...")
            df = pd.DataFrame(
                columns=list(parameters.keys()) + list(performance_metrics.keys())
            )
        new_row = {**parameters, **performance_metrics}
        df.loc[len(df.index)] = new_row
        df.to_excel(file_path, index=False)

    def _save_model(self, best_model_to_export, parameters, modelname):
        formatted_parameters = "_".join(
            [f"{key}_{value}" for key, value in parameters.items()]
        )
        directory = self.output_path
        os.makedirs(directory, exist_ok=True)
        best_model_path = os.path.join(
            directory, f"{modelname}.pkl"
        )

        with open(best_model_path, "wb") as file:
            pickle.dump(best_model_to_export, file)

    def _get_best_params(self, X_test, y_test):
        test_accuracy = self.best_model_for_frauds.score(X_test, y_test)
        test_f1 = f1_score(y_test, self.best_model_for_frauds.predict(X_test))
        test_precision = precision_score(
            y_test, self.best_model_for_frauds.predict(X_test)
        )
        test_recall = recall_score(y_test, self.best_model_for_frauds.predict(X_test))
        test_roc_auc = roc_auc_score(
            y_test, self.best_model_for_frauds.predict_proba(X_test)[:, 1]
        )
        test_confusion_matrix = confusion_matrix(
            y_test, self.best_model_for_frauds.predict(X_test)
        )
        test_mcc = matthews_corrcoef(y_test, self.best_model_for_frauds.predict(X_test))
        test_log_loss = log_loss(
            y_test, self.best_model_for_frauds.predict_proba(X_test)
        )
        test_mae = mean_absolute_error(
            y_test, self.best_model_for_frauds.predict(X_test)
        )
        test_mse = mean_squared_error(
            y_test, self.best_model_for_frauds.predict(X_test)
        )
        test_rmse = mean_squared_error(
            y_test, self.best_model_for_frauds.predict(X_test), squared=False
        )
        return {
            "accuracy": test_accuracy,
            "f1_score": test_f1,
            "precision": test_precision,
            "recall": test_recall,
            "roc_auc": test_roc_auc,
            "confusion_matrix": test_confusion_matrix,
            "mcc": test_mcc,
            "log_loss": test_log_loss,
            "mae": test_mae,
            "mse": test_mse,
            "rmse": test_rmse,
        }
