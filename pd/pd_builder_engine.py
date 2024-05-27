import sys

sys.path.append("../")

from pd.pd_data_engineer_engine import PDDataEngineerEngine
from pd.pd_validator_engine import PDValidatorEngine
from pd.pd_modeller_engine import PDModellerEngine
import pandas as pd
import os
from parent_classes.builder_engine import BuilderEngine

class PDBuilderEngine(BuilderEngine):
    def __init__(self, config, param_grids):
        super().__init__(config, param_grids)
        self.df = pd.read_csv(self.input_data)

        # engines
        self.pdDataEngineerEngine = PDDataEngineerEngine(
            self.df,
            self.output_path + "data_preparation/",
            self.test_size,
            self.validation_size,
            self.random_state,
            self.visualization_weight,
            self.columns_to_drop,
        )
        self.pdModellerEngine = PDModellerEngine(
            self.output_path + "models/",
            self.random_state,
            self.visualization_weight,
            self.export_path,
            self.param_grids
        )  # Type: PDModellerEngine
        self.pdValidatorEngine = PDValidatorEngine(
            self.output_path + "validation/", self.visualization_weight
        )  # Type: PDEvaluatorEngine

        
    def run(self):
        
        (
           self.tax_id_index_mapping,
           self.columns_to_keep,
           self.bins,
           self.val,
           self.X_train,
           self.X_train_woe,
           self.X_test,
           self.X_test_woe,
           self.y_train,
           self.y_test,
        ) = self.pdDataEngineerEngine.run()
        
        self._save_output()
        self._read_output()
        
        thresholds = self.pdModellerEngine.run(self.X_train, self.X_train_woe, self.X_test, self.X_test_woe, self.y_train, self.y_test, self.tax_id_index_mapping)
        self._save_tresholds(thresholds)
        self._read_columns_with_dummies()
        thresholds = self._read_thresholds()
#
        isLatest = True
        path = "results/2024_05_05_20_21_15/"
        loaded_models = self.load_models(path, isLatest=isLatest)
        self.pdValidatorEngine.run(
            self.columns_to_keep,
            self.train_columns_with_dummies,
            self.bins,
            self.val,
            self.X_train,
            self.X_train_woe,
            self.X_test,
            self.X_test_woe,
            self.y_train,
            self.y_test,
            loaded_models,
            thresholds,
        )
        
        print(f"{os.path.basename(self.output_path.rstrip('/'))} model development finished successfully!")

