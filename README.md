# Credit-Risk-Modeling-Msc-Thesis
This repository is created to present my masters thesis

## Goal
The goal of my thesis is to design and implement predictive models that can
effectively calculate expected losses for a financial institution, thus helping to provision
and avoid loans with negative profit. The two main pillars of expected loss are the
probability of default (PD) and the loss given default (LGD). The thesis will present the
complete development process of both models, providing insights into research in the
field and use the lessons learned for the design and the implementation of my own
modeling process.


To achieve this, I defined the tasks to be solved and the methodology for the solution process. For each task, I researched related work in the field and applied the lessons learned to my own solution. I emphasized presenting the considerations comprehensively to shed light on this lesser-known area of modeling.

Particular attention was paid to ensuring data quality. I conducted several data cleaning processes specific to each modeling procedure. Then, for different algorithms, I created separate datasets for the corresponding training sets with appropriate transformations.

My goal was to use interpretable algorithms. I examined the results from several perspectives and conducted new iterations as necessary. The end result is a system where the start-up of modeling procedures is fast, easy to configure, and reusable. The result is stable, well-explained, marketable models, with the positive financial implications of their implementation supported by cost-benefit calculations.

For me, going through the whole process was very instructive, and I feel that I have expanded my knowledge in various data-related fields.

## Results

### PD with behavior variables
We can see that the slope isn't to steep, which is a good sign when validating probability of defaults. Further more we can see that the higher PD scores directly correlates with the amount of actual defaults.
The confusion matrix threshold is set to maximize profit and minimize loss. The False negatives are the ones that we need to avoid.
<img src="https://github.com/bvermes/Credit-Risk-Modeling-Msc-Thesis/blob/main/images/lr_matrix_and_slope.png" alt="PD with behavior variables">

The reason behind the large difference in the validation and test set, is the amount of default values. Validation has a lot more, thus harder to get a high roc curve. Train and test set has pretty much the same ROC curve.
<img src="https://github.com/bvermes/Credit-Risk-Modeling-Msc-Thesis/blob/main/images/lr_roc.png" alt="PD with behavior variables">

### PD without behavior variables
<img src="https://github.com/bvermes/Credit-Risk-Modeling-Msc-Thesis/blob/main/images/lgd_post_overall.png" alt="LGD with behavior variables">

### LGD with behavior variables
The LGD validation set doesn't have many records, therefore the results have a large variance.
<img src="https://github.com/bvermes/Credit-Risk-Modeling-Msc-Thesis/blob/main/images/lgd_post_overall.png" alt="LGD with behavior variables">

### LGD without behavior variables
The LGD validation set doesn't have many records, therefore the results have a large variance.
<img src="https://github.com/bvermes/Credit-Risk-Modeling-Msc-Thesis/blob/main/images/lgd_pre_overall.png" alt="LGD without behavior variables">



## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* GitHub: [bvermes](https://github.com/bvermes)

* Email: [Balázs Vermes](bvermes1999@gmail.com)


## License

The content developed by Balázs Vermes is distributed under the following license:

    Copyright 2024 Balázs Vermes
