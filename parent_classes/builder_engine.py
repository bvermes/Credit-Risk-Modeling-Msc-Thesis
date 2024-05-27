import sys

sys.path.append("../")

import pandas as pd
import joblib
import pickle
import os
import csv


class BuilderEngine:
    def __init__(self, config, param_grids):
        self.input_data = config["input_data"]
        self.output_path = config["output_path"]
        self.test_size = config["test_size"]
        self.random_state = config["random_state"]
        self.validation_size = config["validation_size"]
        self.visualization_weight = config["visualization_weight"]
        self.export_path = config["model_name"] + "_data/"
        self.model_name = config["model_name"].split('/')[-1]
        self.columns_to_drop = config["columns_to_drop"]
        self.param_grids = param_grids
        self._create_notes()

    def load_models(self, path, isLatest=False):
        if isLatest:
            result_dir = "results/"
            folders = [
                f
                for f in os.listdir(result_dir)
                if os.path.isdir(os.path.join(result_dir, f))
            ]
            latest_folder = max(
                folders, key=lambda f: os.path.getctime(os.path.join(result_dir, f))
            )
            path = os.path.join(result_dir, latest_folder)
            path = os.path.join(path, self.model_name)
        else:
            if not path:
                raise ValueError("Path must be provided when isLatest is False")

        models = []
        models_path = os.path.join(path, "models")
        for __, model_types, __ in os.walk(models_path):
            for model_type in model_types:
                scaler = None
                file_folder = os.path.join(models_path, model_type)
                try:
                    file_list = os.listdir(file_folder)
                except FileNotFoundError:
                    continue
                model_name = dir + "_" + model_type
                for file in file_list:
                    if file.endswith(".pkl") and not file.startswith(
                        "scaler"
                    ):
                        model_path = os.path.join(file_folder, file)
                        with open(model_path, "rb") as f:
                            loaded_model = pickle.load(f)
                    if file.endswith(".pkl") and file.startswith("scaler"):
                        scaler_path = os.path.join(file_folder, file)
                        with open(scaler_path, "rb") as f:
                            scaler = joblib.load(f)
                if loaded_model is not None:
                    models.append((model_name, loaded_model, scaler))
        return models

    def _save_tresholds(self, thresholds):
        os.makedirs(self.export_path, exist_ok=True)
        with open(f"{self.export_path}thresholds.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(thresholds.keys())  # Write header row
            writer.writerow(thresholds.values())  # Write values row
    
    def _save_output(self):
        os.makedirs(self.export_path, exist_ok=True)
        #PD
        if hasattr(self, 'bins') :
            joblib.dump(self.bins, f"{self.export_path}bins.pkl")
        if(hasattr(self, 'X_train_woe')):
            self.X_train_woe.to_csv(f"{self.export_path}X_train_woe.csv", index=False)
        if(hasattr(self, 'X_test_woe')):
            self.X_test_woe.to_csv(f"{self.export_path}X_test_woe.csv", index=False)
        #LGD
        if(hasattr(self, 'scaler')):
            joblib.dump(self.scaler, f"{self.export_path}scaler.pkl")
        if(hasattr(self, 'imputer')):
            joblib.dump(self.imputer, f"{self.export_path}imputer.pkl")
        self.val.to_csv(f"{self.export_path}val.csv", index=False)
        self.tax_id_index_mapping.to_csv(f"{self.export_path}tax_id_index_mapping.csv", index=False)
        self.X_train.to_csv(f"{self.export_path}X_train.csv", index=False)
        self.X_test.to_csv(f"{self.export_path}X_test.csv", index=False)
        self.y_train.to_csv(f"{self.export_path}y_train.csv", index=False)
        self.y_test.to_csv(f"{self.export_path}y_test.csv", index=False)
        with open(f"{self.export_path}columns_to_keep.csv", mode="w") as file:
           writer = csv.writer(file)
           writer.writerow(self.columns_to_keep)
           
    def _create_notes(self):
        os.makedirs(self.output_path, exist_ok=True)
        file_path = os.path.join(self.output_path, "notes.txt")
        formatted_content = "\n".join([f"{key}: {value}" for key, value in self.param_grids.items()])

        with open(file_path, "w") as file:
            file.write("Parameters:\n")
            file.write(f"{formatted_content}")
        
    def _read_output(self):
        #PD
        if os.path.exists(f"{self.export_path}X_train_woe.csv"):
            self.X_train_woe = pd.read_csv(f"{self.export_path}X_train_woe.csv")
        if os.path.exists(f"{self.export_path}X_test_woe.csv"):
            self.X_test_woe = pd.read_csv(f"{self.export_path}X_test_woe.csv")
        if os.path.exists(f"{self.export_path}bins.pkl"):
            self.bins = joblib.load(f"{self.export_path}bins.pkl")
        #LGD
        if os.path.exists(f"{self.export_path}scaler.pkl"):
            self.scaler = joblib.load(f"{self.export_path}scaler.pkl")
        if os.path.exists(f"{self.export_path}imputer.pkl"):
            self.imputer = joblib.load(f"{self.export_path}imputer.pkl")
            
        self.X_train = pd.read_csv(f"{self.export_path}X_train.csv")
        self.X_test = pd.read_csv(f"{self.export_path}X_test.csv")
        self.y_train = pd.read_csv(f"{self.export_path}y_train.csv")
        self.y_test = pd.read_csv(f"{self.export_path}y_test.csv")
        self.val = pd.read_csv(f"{self.export_path}val.csv")

        self.tax_id_index_mapping = pd.read_csv(f"{self.export_path}tax_id_index_mapping.csv")

        with open(f"{self.export_path}columns_to_keep.csv", mode="r") as file:
            reader = csv.reader(file)
            self.columns_to_keep = next(reader)
    
    def _read_columns_with_dummies(self):
        with open(
            f"{self.export_path}train_columns_with_dummies.csv", mode="r"
        ) as file:
            reader = csv.reader(file)
            self.train_columns_with_dummies = next(reader)
            
    def _read_thresholds(self):
        thresholds = {}
        with open(f"{self.export_path}thresholds.csv", mode="r", newline="") as file:
            reader = csv.reader(file)
            header = next(reader)  # Read the header row
            values = next(reader)  # Read the values row
            for key, value in zip(header, values):
                thresholds[key] = float(value)  # Assuming the thresholds are stored as strings
        return thresholds