import pandas as pd
import joblib
import pickle
import os

from lgd.lgd_data_engineer_engine import LGDDataEngineerEngine
from lgd.lgd_validator_engine import LGDValidatorEngine
from lgd.lgd_modeller_engine import LGDModellerEngine
from parent_classes.builder_engine import BuilderEngine


class LGDBuilderEngine(BuilderEngine):
    def __init__(self, config, param_grids):
        super().__init__(config, param_grids)
        self.df = pd.read_csv(self.input_data)

        # engines
        self.lgdDataEngineerEngine = LGDDataEngineerEngine(
            self.df,
            self.output_path + "data_preparation/",
            self.test_size,
            self.validation_size,
            self.random_state,
            self.visualization_weight,
            self.columns_to_drop,
        )
        self.lgdModellerEngine = LGDModellerEngine(
            self.output_path + "models/",
            self.random_state,
            self.visualization_weight,
            self.export_path,
            self.param_grids
        )  # Type: PDModellerEngine
        self.lgdValidatorEngine = LGDValidatorEngine(
            self.output_path + "validation/", self.visualization_weight
        )  # Type: PDEvaluatorEngine


    def run(self):
        (
        self.imputer,
        self.scaler,
        self.tax_id_index_mapping,
        self.columns_to_keep,
        self.val,
        self.X_train,
        self.X_test,
        self.y_train,
        self.y_test) = self.lgdDataEngineerEngine.run()

        self._save_output()
        self._read_output()
        #
        thresholds = self.lgdModellerEngine.run(self.X_train, self.X_test, self.y_train, self.y_test, self.tax_id_index_mapping)
        self._save_tresholds(thresholds)
        self._read_columns_with_dummies()
        thresholds = self._read_thresholds()
        
        isLatest = True
        path = "results/2024_05_15_17_40_47_LGD_models/"
        loaded_models = self.load_models(path, isLatest=isLatest)
        
        self.lgdValidatorEngine.run(
            self.imputer,
            self.scaler,
            self.columns_to_keep,
            self.train_columns_with_dummies,
            self.val,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            loaded_models,
            thresholds,
        )
