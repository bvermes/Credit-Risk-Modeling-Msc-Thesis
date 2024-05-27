from parent_classes.data_engineer_engine import DataEngineerEngine
import sys
import pandas as pd

sys.path.append("..")
from woe.woebin import *
from woe.var_filter import *
from sklearn.model_selection import train_test_split
import woe.transformerClassesType as transform

class PDDataEngineerEngine(DataEngineerEngine):
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
            output_path,
            columns_to_drop,
            target = 'bad'
        )
        

    
    def _clean_dataframe(self,df):
        actions = []
        
        if 'year' in df.columns:
            actions.append("Deleting rows where 'year' is lower than 2018")
            df = df[df['year'] >= 2018]
        
        if 'roe' in df.columns:
            roe_mean = df['roe'].mean()
            if roe_mean > 50:
                actions.append("Replacing 'roe' values bigger than 50 with the mean value")
                df.loc[df['roe'] > 50, 'roe'] = roe_mean
        
        if 'ros' in df.columns:
            ros_mean = df['ros'].mean()
            actions.append("Replacing 'ros' values not between -2 and 0.6 with the mean value")
            df.loc[~df['ros'].between(-2, 0.6), 'ros'] = ros_mean
        
        if 'after_tax_roa' in df.columns:
            after_tax_roa_mean = df['after_tax_roa'].mean()
            actions.append("Replacing 'after_tax_roa' values greater than 10 with the mean value")
            df.loc[df['after_tax_roa'] > 10, 'after_tax_roa'] = after_tax_roa_mean
        
        if 'current_assets_eur' in df.columns:
            current_assets_mean = df['current_assets_eur'].mean()
            actions.append("Replacing 'current_assets_eur' values greater than 10000000000 with the mean value")
            df.loc[df['current_assets_eur'] > 10000000000, 'current_assets_eur'] = current_assets_mean
        
        if 'current_liabilities_eur' in df.columns:
            current_liabilities_mean = df['current_liabilities_eur'].mean()
            actions.append("Replacing 'current_liabilities_eur' values greater than 10000000000 with the mean value")
            df.loc[df['current_liabilities_eur'] > 10000000000, 'current_liabilities_eur'] = current_liabilities_mean
        
        if 'net_income_eur' in df.columns:
            net_income_mean = df['net_income_eur'].mean()
            actions.append("Replacing 'net_income_eur' values less than 0 with the mean value")
            df.loc[df['net_income_eur'] < 0, 'net_income_eur'] = net_income_mean
        
        if 'net_profit_margin' in df.columns:
            actions.append("Replacing 'net_profit_margin' values greater than 0.5 with 0.5")
            df.loc[df['net_profit_margin'] > 0.5, 'net_profit_margin'] = 0.5
        
        for action in actions:
            print(action)
        
        return df

    def _feature_engineering(self):
        self.df.drop_duplicates(inplace=True)
        
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
        result = var_filter(self.df, y="bad", iv_limit=0.1, return_rm_reason=True)
        dt_s = result["dt"]
        reason = result["rm"]
        reason.to_excel(f"{self.output_path}/filter_reasons.xlsx", index=False)
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
        y_train = train.loc[:, "bad"]
        X_train = train.drop(["bad"], axis=1)
        y_test = test.loc[:, "bad"]
        X_test = test.drop(["bad"], axis=1)

        TransDummy = transform.transformDummyToCategorical()
        TransDummy.fit(X_train)
        TransformIntoWoeValues = transform.transformIntoWoeValues(best_n_variables=10)
        TransformIntoWoeValues.fit(X_train, y_train)
        X_train = X_train[TransformIntoWoeValues.var_list]
        X_test = X_test[TransformIntoWoeValues.var_list]
        
        columns_to_keep = TransformIntoWoeValues.var_list.copy()
        columns_to_dts = TransformIntoWoeValues.var_list.copy()
        columns_to_dts.append("bad")
        # dt_s = dt_s[columns_to_dts]

        TransformIntoWoeValues.final_iv.to_excel(
            self.output_path + "final_iv.xlsx", index=False
        )
        breaks_list = {}
        for var_name in TransformIntoWoeValues.final_iv["VAR_NAME"].unique():
            intervals = []
            var_df = TransformIntoWoeValues.final_iv[
                TransformIntoWoeValues.final_iv["VAR_NAME"] == var_name
            ]
            var_df = var_df.sort_values(by="MIN_VALUE")
            # Filter out non-numeric values and find the maximum value
            max_bin = (
                var_df.iloc[-2]["MIN_VALUE"]
                if var_df["MIN_VALUE"].isna().any()
                else var_df.iloc[-1]["MAX_VALUE"]
            )
            for i in range(len(var_df)):
                min_value = var_df.iloc[i]["MIN_VALUE"]
                if i == 0:
                    continue
                elif min_value == max_bin:
                    intervals.append(min_value)
                    # intervals.append("Inf")
                elif pd.notnull(min_value):
                    intervals.append(min_value)
            if pd.isnull(var_df.iloc[-1]["MAX_VALUE"]):
                # intervals.append("missing")
                pass
            
            breaks_list[var_name] = intervals
        # woe_encoder = ce.WOEEncoder()
        # X_train_encoded = woe_encoder.fit_transform(X_train, y_train)
        # X_test_encoded = woe_encoder.transform(test.drop(["bad"], axis=1))

        # print("Encoded Training Data:")
        # print(X_train_encoded)
        #
        # iv_values = woe_encoder.get_feature_names_out()
        #
        # print("\nInformation Value for each feature:")
        # print(iv_values)
        for key, value in breaks_list.items():
            if isinstance(value, list):
                rounded_values = []
                for item in value:
                    if isinstance(item, (int, float)):
                        rounded_values.append(round(item) if item > 1000 else round(item, 2))
                    else:
                        rounded_values.append(item)
                breaks_list[key] = rounded_values
        bins = woebin(
            dt_s,
            y="bad",
            #breaks_list=breaks_list,
            method="tree",
        )
        # breaks_adj = woebin_adj(dt=dt_s, y="bad", bins=bins)
        # bins = woebin(dt_s, y="bad", breaks_list=breaks_adj)
        woebin_plot(bins, output=self.output_path)
        
        #train_ids, test_ids = train_test_split(
        #    dt_s["tax_id"].unique(), test_size=0.25, random_state=42
        #)
#
        #train = dt_s[dt_s["tax_id"].isin(train_ids)]
        #test = dt_s[dt_s["tax_id"].isin(test_ids)]
        
        train_woe = woebin_ply(train, bins)
        test_woe = woebin_ply(test, bins)
        woe_columns = columns_to_keep.copy()
        for i in range(len(woe_columns)):
            woe_columns[i] += "_woe"
        woe_columns.append("bad")
        
        train_woe = train_woe[woe_columns]
        test_woe = test_woe[woe_columns]
        
        
        y_train = train_woe.loc[:, "bad"]
        X_train_woe = train_woe.drop(["bad"], axis=1)
        y_test = test_woe.loc[:, "bad"]
        X_test_woe = test_woe.drop(["bad"], axis=1)
        return (
            tax_id_index_mapping,
            columns_to_keep,
            bins,
            val,
            X_train,
            X_train_woe,
            X_test,
            X_test_woe,
            y_train,
            y_test,
        )
    
    def run(self):
        print("Running PD Data Engineer Engine")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self._clean_rows()
        columns_to_drop = [
            "bnpl_debtor_tax_number",
            "default_value",
            "product",
            "id",
            "created_at",
            "country",
            "currency",
            "dnb_rating_credit_recommendation_currency",
            "created_at",
            "country",
            "short_tax_number",
            "company_name",
            "dnb_rating_credit_recommendation_currency",
            "rnk",
            "duns_number",
        ]
        columns_to_drop += self.columns_to_drop
        self._remove_columns(columns_to_drop)
        self._feature_engineering()

        (
            tax_id_index_mapping,
            columns_to_keep,
            bins,
            val,
            X_train,
            X_train_woe,
            X_test,
            X_test_woe,
            y_train,
            y_test,
        ) = self._calculate_woe_iv()

        return (
            tax_id_index_mapping,
            columns_to_keep,
            bins,
            val,
            X_train,
            X_train_woe,
            X_test,
            X_test_woe,
            y_train,
            y_test,
        )
