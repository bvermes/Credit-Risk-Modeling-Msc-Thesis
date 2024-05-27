def correlation_matrix(df, title="Correlation matrix", **kwargs):
    pass


visualization_weights = {
    correlation_matrix: 1,
}


def run_visualizations(visualization_weight):
    for task, weight in visualization_weights.items():
        if weight >= visualization_weight:
            task()


# if weight >= config.visualization_weight:
#     task()
