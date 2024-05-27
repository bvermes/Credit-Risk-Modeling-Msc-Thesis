from models.decision_tree import DecisionTreeModel
from models.logistic_regression import LogisticRegressionModel
from models.rnn import RNNModel
from models.random_forest import RandomForestModel


class LGDModellerEngine:
    def __init__(self, output_path, random_state, visualization_weight, export_path, param_grids):
        self.output_path = output_path
        self.random_state = random_state
        self.visualization_weight = visualization_weight
        self.export_path = export_path
        self.param_grids = param_grids

        self._logistic_regression = LogisticRegressionModel(
            param_grid=self.param_grids["log_reg_param_grid"],
            output_path=self.output_path,
            random_state=self.random_state,
            visualization_weight=self.visualization_weight,
        )
        self._decision_tree = DecisionTreeModel(
            param_grid=self.param_grids["decision_tree_param_grid"],
            output_path=self.output_path,
            random_state=self.random_state,
            visualization_weight=self.visualization_weight,
            export_path=self.export_path,
        )
        self._random_forest = RandomForestModel(
            param_grid=self.param_grids["random_forest_param_grid"],
            output_path=self.output_path,
            random_state=self.random_state,
            visualization_weight=self.visualization_weight,
        )
        self._rnn = RNNModel(
            param_grid={"TODO": True},
            output_path=self.output_path,
            random_state=self.random_state,
            visualization_weight=self.visualization_weight,
        )

    def run(self, X_train, X_test, y_train, y_test, tax_id_index_mapping):
        print("Running LGD Modeller Engine")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        #We need to prepare the X_train and X_test dataframes for the models, because of generalization
        

        log_reg_treshold = self._logistic_regression.run(self.X_train, self.X_test, self.y_train, self.y_test, tax_id_index_mapping)
        dt_treshold = self._decision_tree.run(self.X_train, self.X_test, self.y_train, self.y_test,tax_id_index_mapping)
        rf_treshold = self._random_forest.run(self.X_train, self.X_test, self.y_train, self.y_test,tax_id_index_mapping)
        
        return {
            "log_reg_treshold": log_reg_treshold,
            "dt_treshold": dt_treshold,
            "rf_treshold": rf_treshold,
        }
