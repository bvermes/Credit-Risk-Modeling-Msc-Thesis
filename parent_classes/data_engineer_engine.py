import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from woe.woebin import *
from woe.var_filter import *
from scipy import stats

class DataEngineerEngine:
    def __init__(
        self,
        df,
        test_size,
        validation_size,
        random_state,
        visualization_weight,
        output_path,
        columns_to_drop,
        target
    ):
        self.df = df
        self.output_path = output_path
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.visualization_weight = visualization_weight
        self.columns_to_drop = columns_to_drop
        self.target = target
        self._visualizer = self._Visualizer(self)

    class _Visualizer:
        def __init__(self, parent):
            self.parent = parent
            self.image_dir = self.parent.output_path + "images/"
            os.makedirs(self.image_dir, exist_ok=True)
            self.visualization_weights = {
                self._top5_corr: 1,
                self._top_missing_features: 1,
                self._dist_unique_tax_ids: 1,
                self._dis_train_test_by_month: 1,
                self._dis_train_test_by_month_and_target: 1,
                self._dist_unique_tax_ids_val: 1,
                self._dis_val_original_by_month: 1,
            }
            self.dt_s = None
            self.train = None
            self.test = None
            self.val = None

        def _top5_corr(self):
            numeric_df = self.parent.df.select_dtypes(include="number")
            correlations = numeric_df.corr()[self.parent.target].drop(self.parent.target)
            top_5_correlations = correlations.abs().nlargest(5)
            plt.figure(figsize=(10, 5))
            ax = sns.barplot(x=top_5_correlations.index, y=top_5_correlations.values)
            ax.set_title(f"Top 5 Correlations with '{self.parent.target}'")
            ax.set_xlabel("Variables")
            ax.set_ylabel("Correlation")

            # Rotate x-axis labels for readability and adjust padding
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )
            plt.tight_layout()
            plt.savefig(f"{self.image_dir}top5_corr.png")
            plt.close()

        def _top_missing_features(self):
            missing_percentage = (
                self.parent.df.isnull().sum() / len(self.parent.df)
            ) * 100
            top_missing_features = missing_percentage.sort_values(ascending=False).head(
                9
            )
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            for i, feature in enumerate(top_missing_features.index):
                non_null_values = self.parent.df[feature].dropna()
                if (
                    not non_null_values.empty
                ):  # Check if the feature has non-null values
                    row, col = divmod(i, 3)
                    ax = axes[row, col]
                    sns.histplot(
                        data=non_null_values, kde=True, ax=ax, bins=20
                    )  # Adjust bins as needed
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {feature}")
                    ax.tick_params(axis="x", labelrotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.image_dir}top_missing_features.png")
            plt.close()

        def _dist_unique_tax_ids_val(self):
            unique_tax_ids_dt_s = self.dt_s["tax_id"].nunique()
            unique_tax_ids_val = self.val["tax_id"].nunique()

            labels = ["original", "val"]

            data_dt_s = [unique_tax_ids_dt_s, 0]
            data_val = [0, unique_tax_ids_val]

            fig, ax = plt.subplots()
            bars_dt_s = ax.bar(
                labels,
                data_dt_s,
                label="Unique tax_ids in dataframe (original)",
                color="blue",
            )
            bars_val = ax.bar(
                labels,
                data_val,
                bottom=data_dt_s,
                label="Unique tax_ids in validation",
                color="orange",
            )

            for bars in [bars_dt_s, bars_val]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        "{}".format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 1),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

            ax.set_ylabel("Count")
            ax.set_title("Distribution of Unique tax_ids")
            ax.legend()
            plt.savefig(f"{self.image_dir}dist_unique_tax_ids_validation_original.png")
            plt.close()

        def _dis_val_original_by_month(self):
            val_counts = self.val["date_month"].value_counts().sort_index()
            original_counts = self.dt_s["date_month"].value_counts().sort_index()
            all_months = sorted(
                set(self.dt_s["date_month"]).union(set(self.val["date_month"]))
            )
            val_counts = val_counts.reindex(all_months, fill_value=0)
            original_counts = original_counts.reindex(all_months, fill_value=0)
            fig, ax = plt.subplots(figsize=(30, 6))
            ax.bar(all_months, val_counts, color="blue", alpha=0.5, label="Validation")
            ax.bar(
                all_months, original_counts, color="red", alpha=0.5, label="Original"
            )
            ax.set_xlabel("Month")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Validation and Original Data by Month")
            ax.legend()
            ax.set_xticklabels(all_months, rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.image_dir}dis_train_test_by_month.png")
            plt.close()

        def _dist_unique_tax_ids(self):
            unique_tax_ids_dt_s = self.dt_s["tax_id"].nunique()
            unique_tax_ids_train = self.train["tax_id"].nunique()
            unique_tax_ids_test = self.test["tax_id"].nunique()

            labels = ["dt_s", "train", "test"]

            data_dt_s = [unique_tax_ids_dt_s, 0, 0]
            data_train = [0, unique_tax_ids_train, 0]
            data_test = [0, 0, unique_tax_ids_test]

            fig, ax = plt.subplots()
            bars_dt_s = ax.bar(
                labels, data_dt_s, label="Unique tax_ids in dt_s", color="blue"
            )
            bars_train = ax.bar(
                labels,
                data_train,
                bottom=data_dt_s,
                label="Unique tax_ids in train",
                color="orange",
            )
            bars_test = ax.bar(
                labels,
                data_test,
                bottom=np.array(data_dt_s) + np.array(data_train),
                label="Unique tax_ids in test",
                color="green",
            )

            for bars in [bars_dt_s, bars_train, bars_test]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        "{}".format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

            ax.set_ylabel("Count")
            ax.set_title("Distribution of Unique tax_ids")
            ax.legend()
            plt.savefig(f"{self.image_dir}dist_unique_tax_ids.png")
            plt.close()

        def _dis_train_test_by_month(self):
            train_counts = self.train["date_month"].value_counts().sort_index()
            test_counts = self.test["date_month"].value_counts().sort_index()
            all_months = sorted(
                set(self.train["date_month"]).union(set(self.test["date_month"]))
            )
            train_counts = train_counts.reindex(all_months, fill_value=0)
            test_counts = test_counts.reindex(all_months, fill_value=0)
            fig, ax = plt.subplots(figsize=(30, 6))
            ax.bar(all_months, train_counts, color="blue", alpha=0.5, label="Train")
            ax.bar(all_months, test_counts, color="red", alpha=0.5, label="Test")
            ax.set_xlabel("Month")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Train and Test Data by Month")
            ax.legend()
            ax.set_xticklabels(all_months, rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.image_dir}dis_train_test_by_month.png")
            plt.close()

        def _dis_train_test_by_month_and_target(self):
            all_months = sorted(
                set(self.train["date_month"]).union(set(self.test["date_month"]))
            )
            train_target_counts = (
                self.train[self.train[self.parent.target] == 1]["date_month"]
                .value_counts()
                .sort_index()
            )
            test_target_counts = (
                self.test[self.test[self.parent.target] == 1]["date_month"]
                .value_counts()
                .sort_index()
            )
            train_good_counts = (
                self.train[self.train[self.parent.target] == 0]["date_month"]
                .value_counts()
                .sort_index()
            )
            test_good_counts = (
                self.test[self.test[self.parent.target] == 0]["date_month"]
                .value_counts()
                .sort_index()
            )
            train_target_counts = train_target_counts.reindex(all_months, fill_value=0)
            test_target_counts = test_target_counts.reindex(all_months, fill_value=0)
            train_good_counts = train_good_counts.reindex(all_months, fill_value=0)
            test_good_counts = test_good_counts.reindex(all_months, fill_value=0)
            fig, ax1 = plt.subplots(figsize=(30, 6))
            ax1.bar(
                all_months, train_target_counts, color="blue", alpha=0.5, label=f"Train {self.parent.target}"
            )
            ax1.bar(
                all_months,
                train_good_counts,
                color="blue",
                alpha=0.2,
                label="Train good",
            )
            ax1.bar(
                all_months, test_target_counts, color="red", alpha=0.5, label=f"Test {self.parent.target}"
            )
            ax1.bar(
                all_months, test_good_counts, color="red", alpha=0.2, label="Test good"
            )
            ax1.set_ylabel("Count")
            ax1.set_xlabel("Month")
            ax1.set_title("Distribution of Train and Test Data by Month")
            ax1.legend(loc="upper left")
            ax1.set_xticklabels(all_months, rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.image_dir}dis_train_test_by_month_and_{self.parent.target}.png")
            plt.close()

    def _remove_columns(self, columns_to_drop):
        eur_columns_stripped = [
            col[:-4] for col in self.df.columns if col.endswith("_eur")
        ]
        filtered_columns = [
            col for col in self.df.columns if col not in eur_columns_stripped
        ]
        self.df = self.df[filtered_columns]
        self.df.drop(columns_to_drop, axis=1, inplace=True)

    def _detect_outliers(self):
        outlier_summary = pd.DataFrame(columns=['Column', 'Min', 'Max', 'Count', 'Mean (without outliers)', 'Max (without outliers)', 'Min (without outliers)'])
        
        for column in self.df.columns:
            if self.df[column].dtype in [np.int64, np.float64]:
                z = np.abs(stats.zscore(self.df[column]))
                threshold = 3
                outliers = self.df[z > threshold]

                if not outliers.empty:
                    min_value = outliers[column].min()
                    max_value = outliers[column].max()
                    count = outliers.shape[0]
                    clean_values = self.df[~self.df.index.isin(outliers.index)][column]
                    mean_without_outliers = clean_values.mean()
                    max_without_outliers = clean_values.max()
                    min_without_outliers = clean_values.min()
                    
                    outlier_summary = pd.concat([outlier_summary, pd.DataFrame([{
                        'Column': column,
                        'Min': min_value,
                        'Max': max_value,
                        'Count': count,
                        'Mean (without outliers)': mean_without_outliers,
                        'Max (without outliers)': max_without_outliers,
                        'Min (without outliers)': min_without_outliers
                    }])], ignore_index=True)

        return outlier_summary
    def _clean_rows(self):
        self.df.dropna(subset=["year"], inplace=True)

    def _create_validation_data(self, dt_s, validation_size):
        sorted_dt_s = dt_s.sort_values(by="date_month")
        grouped_sizes = sorted_dt_s.groupby("tax_id").size()
        cumulative_sizes = grouped_sizes.cumsum()
        total_size = cumulative_sizes.max()
        cutoff_index = int(total_size - validation_size * total_size)
        validation_tax_ids = cumulative_sizes[cumulative_sizes > cutoff_index].index
        validation_set = dt_s[dt_s["tax_id"].isin(validation_tax_ids)]
        return validation_set

    def _create_test_data(self, dt_s, test_size):
        grouped_sizes = dt_s.groupby("tax_id").size()
        cumulative_sizes = grouped_sizes.cumsum()
        total_size = cumulative_sizes.max()
        cutoff_index = int(total_size - test_size * total_size)
        test_tax_ids = cumulative_sizes[cumulative_sizes > cutoff_index].index
        test_tax_ids = dt_s[dt_s["tax_id"].isin(test_tax_ids)]
        return test_tax_ids

    def run(self):
        pass
