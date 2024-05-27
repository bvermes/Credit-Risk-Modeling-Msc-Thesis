from datetime import datetime
# Model parameters for grid search
param_grids = {
"log_reg_param_grid":{
        "penalty": ["l1", "l2"],
        #"C": [0.001, 0.01, 0.1,0.5,0.75, 1.0,1.25, 1.5, 3, 10.0],
        "solver": ["liblinear", "saga"],
        #"max_iter": [100, 200, 500, 1000],
    },
"decision_tree_param_grid":{
        #"ccp_alpha": np.linspace(0, 0.2, 10),
        "max_depth": [1,2,4],
        #"criterion": ["gini", "entropy"],
        #"min_samples_leaf": [10, 20, 30],
    },
"random_forest_param_grid":{
        #"ccp_alpha": np.linspace(0, 0.2, 10),
        #"max_depth": [1,2],
        #"n_estimators":[5,10,100],
        "criterion": ["gini", "entropy"],
        #"min_samples_leaf": [30, 50, 100],
        "bootstrap": [True],
        'max_features': ['sqrt', 'log2']
    }
}

# Configurations settings for each model type
curr_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
pd_pre_behav_config = {
        "input_data": "data/pd_dataset_2024-05-09T22_58_14.252757Z.csv",
        "output_path": f"results/{curr_time}/pd_pre_behav/",
        "test_size": 0.2,
        "random_state": 42,
        "validation_size": 0.1,
        "visualization_weight": 1,  # 0 is without visualization, 0.5 is just modell results, 1 is with all the visualization
        "model_name": "model_files/pd_pre_behav_model",
        "columns_to_drop": [
            "avg_daily_outstanding_amount",
            "max_daily_outstanding_amount",
            "funding_count",
            "overdue_ratio",
            "avg_overdue_days",
            "dnb_rating_failure_score",
            "dnb_rating_standard_financial_strength",
        ],
    }

pd_post_behav_config = {
    "input_data": "data/pd_dataset_2024-05-09T22_58_14.252757Z.csv",
    "output_path": f"results/{curr_time}//pd_post_behav/",
    "test_size": 0.2,
    "random_state": 42,
    "validation_size": 0.1,
    "visualization_weight": 1,  # 0 is without visualization, 0.5 is just modell results, 1 is with all the visualization
    "model_name": "model_files/pd_post_behav_model",
    "columns_to_drop": ["dnb_rating_failure_score","dnb_rating_standard_financial_strength"],
}

lgd_post_behav_config = {
    "input_data": "data/lgd_dataset_2024-05-15T13_26_54.27598Z.csv",
    "output_path": f"results/{curr_time}/lgd_post_behav/",
    "test_size": 0.2,
    "random_state": 42,
    "validation_size": 0.1,
    "model_name": "model_files/lgd_post_behav_model",
    "columns_to_drop": ["default_value", "lgd_worthy",
                        "rnk","dnb_rating_standard_financial_strength",
                        "dnb_rating_failure_score",
                        "payout_month", 
                        "bnpl_debtor_tax_number",
                        "product","id","created_at", 
                        "dnb_rating_credit_recommendation_amount_eur",
                        "dnb_rating_credit_recommendation_currency",
                        "short_tax_number",
                        "duns_number",
                        "year",],
    "visualization_weight": 1,  # 0 is without visualization, 0.5 is just modell results, 1 is with all the visualization
}

lgd_pre_behav_config = {
    "input_data": "data/lgd_dataset_2024-05-15T13_26_54.27598Z.csv",
    "output_path": f"results/{curr_time}/lgd_pre_behav/",
    "test_size": 0.2,
    "random_state": 42,
    "validation_size": 0.1,
    "model_name": "model_files/lgd_pre_behav_model",
    "columns_to_drop": [
        "avg_daily_outstanding_amount",
        "max_daily_outstanding_amount",
        "default_value",
        "lgd_worthy",
        "rnk",
        "dnb_rating_standard_financial_strength",
        "dnb_rating_failure_score",
        "payout_month",
        "bnpl_debtor_tax_number",
        "product",
        "id","created_at",
        "dnb_rating_credit_recommendation_amount_eur",
        "dnb_rating_credit_recommendation_currency",
        "short_tax_number",
        "duns_number",
        "year",
    ],
    "visualization_weight": 1,  # 0 is without visualization, 0.5 is just modell results, 1 is with all the visualization
}