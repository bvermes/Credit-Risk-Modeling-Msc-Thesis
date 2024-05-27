import joblib
import psycopg2
import os
from dotenv import load_dotenv
import pandas as pd
import pickle
from woe.woebin import *
from woe.var_filter import *
load_dotenv()

class ModelInterface():
    def __init__(self, config):
        self.path = config["path"]
        self.db_connection_string = os.environ.get('PROD_DB_CONNECTION_STR_BALAZS')
        
        
        self.pd_pre_behav_model = self._load_model(self.path, model = "pd_pre_behav")
        self.pd_post_behav_model = self._load_model(self.path , model =  "pd_post_behav")
        self.lgd_pre_behav_model = self._load_model(self.path , model =  "lgd_pre_behav")
        self.lgd_post_behav_model = self._load_model(self.path , model =  "lgd_post_behav")
    
    def _load_model(self, path, model):
        model_path = path + model
        if(model == "pd_pre_behav" or model == "pd_post_behav"):
            return {
                "decision_tree": pickle.load(model_path + "/models/decision_tree_model/decision_tree.pkl", "rb"),
                "random_forest": pickle.load(model_path + "/models/random_forest_model/random_forest.pkl", "rb"),
                "log_reg": pickle.load(model_path + "/models/log_reg_model/log_reg.pkl", "rb"),
                
                "bins": joblib.load(model_path + "_model_data/bins.pkl"),
                "columns_to_keep": pd.read_csv(model_path + "_model_data/columns_to_keep.csv"),
                "thresholds": pd.read_csv(model_path + "_model_data/thresholds.csv"),
                "train_columns_with_dummies": pd.read_csv(model_path + "_model_data/train_columns_with_dummies.csv")  
            }
        if(model == "lgd_pre_behav" or model == "lgd_post_behav"):
            return {
                "decision_tree": pickle.load(model_path + "/models/decision_tree_model/decision_tree.pkl", "rb"),
                "random_forest": pickle.load(model_path + "/models/random_forest_model/random_forest.pkl", "rb"),
                "log_reg": pickle.load(model_path + "/models/log_reg_model/log_reg.pkl", "rb"),
                
                "imputer": joblib.load(model_path + "_model_data/imputer.pkl"),
                "scaler": joblib.load(model_path + "_model_data/scaler.pkl"),
                "columns_to_keep": pd.read_csv(model_path + "_model_data/columns_to_keep.csv"),
                "thresholds": pd.read_csv(model_path + "_model_data/thresholds.csv"),
                "train_columns_with_dummies": pd.read_csv(model_path + "_model_data/train_columns_with_dummies.csv")  
                
            }
    def select_from_table(self, table_name, where_condition):
        conn = conn = psycopg2.connect(self.db_connection_string)
        cursor = conn.cursor()

        select_query = f"""
        SELECT * 
        FROM {table_name}
        WHERE {where_condition}
        """

        cursor.execute(select_query)

        data_rows = [row for row in cursor]
        col_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data_rows, columns=col_names)
        conn.close()
        return df
    
    def transform_pd_data(self, df):
        #log reg
        pd_pre_X_reg = df[self.pd_pre_behav_model["columns_to_keep"]]
        pd_pre_X_reg = woebin_ply(pd_pre_X_reg, self.pd_pre_behav_model["bins"])
        
        pd_post_X_reg = df[self.pd_post_behav_model["columns_to_keep"]]
        pd_post_X_reg = woebin_ply(pd_post_X_reg, self.pd_post_behav_model["bins"])       
        #fill null values
        
        #decision tree and random forest
        pd_pre_X_class = df[self.pd_pre_behav_model["columns_to_keep"]]
        categorical_columns = pd_pre_X_class.select_dtypes(include=['object']).columns
        pd_pre_X_class = pd.get_dummies(
                    pd_pre_X_class, columns=categorical_columns, dtype=bool
                )
        missing_cols = set(self.pd_pre_behav_model["train_columns_with_dummies"]) - set(
            pd_pre_X_class.columns
        )
        for col in missing_cols:
            pd_pre_X_class[col] = False
        pd_pre_X_class = pd_pre_X_class[self.pd_pre_behav_model["train_columns_with_dummies"]]
        pd_pre_X_class = pd_pre_X_class.interpolate(method="linear", inplace=True)
        
        pd_post_X_class = df[self.pd_post_behav_model["columns_to_keep"]]
        categorical_columns = pd_post_X_class.select_dtypes(include=['object']).columns
        pd_post_X_class = pd.get_dummies(
                    pd_post_X_class, columns=categorical_columns, dtype=bool
                )
        missing_cols = set(self.pd_post_behav_model["train_columns_with_dummies"]) - set(
            pd_post_X_class.columns
        )
        for col in missing_cols:
            pd_post_X_class[col] = False
        pd_post_X_class = pd_post_X_class[self.pd_post_behav_model["train_columns_with_dummies"]]
        pd_post_X_class = pd_post_X_class.interpolate(method="linear", inplace=True)
        
        return pd_pre_X_reg, pd_post_X_reg, pd_pre_X_class, pd_post_X_class
        
    def _process_pd_row(self, row):
        pd_pre_X_reg, pd_post_X_reg, pd_pre_X_class, pd_post_X_class = self.transform_pd_data(row)
        pd_pre_log_reg_proba = self.pd_pre_behav_model["log_reg"].predict_proba(pd_pre_X_reg)
        pd_pre_dc_proba = self.pd_pre_behav_model["decision_tree"].predict_proba(pd_pre_X_class)
        pd_pre_rf_proba = self.pd_pre_behav_model["random_forest"].predict_proba(pd_pre_X_class)
        
        pd_post_log_reg_proba =self.pd_post_behav_model["log_reg"].predict_proba(pd_post_X_reg)
        pd_post_dc_proba =self.pd_post_behav_model["decision_tree"].predict_proba(pd_post_X_class)
        pd_post_rf_proba = self.pd_post_behav_model["random_forest"].predict_proba(pd_post_X_class)
        
        return pd_pre_log_reg_proba, pd_pre_dc_proba, pd_pre_rf_proba, \
            pd_post_log_reg_proba, pd_post_dc_proba, pd_post_rf_proba
                
    def _evaluate_all_debtor(self):
        #pd
        pd_df = self.select_from_table("risk.mv_TODO", "true")
        probabilities_df = pd_df.apply(self._process_pd_row, axis=1, result_type='expand')
        result_df = pd.concat([pd_df, probabilities_df], axis=1)
        
        #lgd
        
    
    def _evaluate_specific_debtor(self, tax_id):
        pd_df = self.select_from_table("risk.mv_TODO", f"tax_id = '{tax_id}' LIMIT 1")
        probabilities_df = pd_df.apply(self._process_pd_row, axis=1, result_type='expand')
        result_df = pd.concat([pd_df, probabilities_df], axis=1)
        
        #lgd
    
    def menu(self):
        while True:
            print("MENU")
            print("1. Evaluate Models on all debtor")
            print("2. Evaluate Models on specific debtor")
            print("3. Exit")
            choice = input("Enter choice: ")
            if choice == "1":
                self._evaluate_all_debtor()
            elif choice == "2":
                input_tax_id = input("Enter tax id: ")
                self._evaluate_specific_debtor(input_tax_id)
            elif choice == "3":
                break
            else:
                print("Invalid choice")



if __name__ == "__main__":
    config = {
        "path": "deployed_models/"
    }
    modelInterface = ModelInterface(config)
    modelInterface.menu()