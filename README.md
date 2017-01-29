# Benchmarking RFs for ilastik

Comparing different Random Forest implementation for ilastik pixel classification.
Using random forests:
* Vigra Random Forest 2 (https://github.com/ukoethe/vigra)
* Vigra Random Forest 3 (check out https://github.com/constantinpape/vigra for pybindings)
* Sklearn Random Forest (https://github.com/scikit-learn/scikit-learn) (v 0.18.1)

Training and prediction on cutout of FlyEM FIB25 data.

TODOs:
* Issue to sklearn about RAM consumption during training
* For Pipeline: Try single block parallelisation -> appears to be faster!
* Also compare to xgb
* Use results from grid search in pipeline

Note:
* The Vigra RF2 is not inherently parallelised, so I have used concurrent.futures to parallelize it from
python, by training / predicting multiple forests with the corresponding number of sub-trees. (This is also what ilastik does).
* The Sklearn RF needs insane amounts of RAM during prediction, for the feature matrix used here (~ 500 MB), it eats up all the 
RAM of my laptop (16 GB), even in single threaded prediction. See github issue: 
* The benchmarks and gridsearch were done on a workstation (20 cores, 256 GB RAM), the pipeline results on a laptob (4 cores, 16 GB RAM).
* For all experiments 17 features from filter responses per instance were used.

## Benchmark Training

Benchmarking the training for the 11,500 instances extracted from the ilastik project.

| Num Threads | Vigra RF2 | Vigra RF3 | Sklearn RF | 
| ----------- | --------: | --------: | ---------: | 
| 1           | 4.424 +- 0.072  | 5.220 +- 0.056  | 3.347 +- 0.059   | 
| 2           | 2.249 +- 0.054  | 2.648 +- 0.028  | 1.957 +- 0.045   | 
| 4           | 1.167 +- 0.037  | 1.363 +- 0.020  | 1.157 +- 0.048   | 
| 6           | 0.832 +- 0.018  | 0.973 +- 0.013  | 0.890 +- 0.012   | 
| 8           | 0.658 +- 0.010  | 0.763 +- 0.007  | 0.666 +- 0.024   | 
| 10           | 0.523 +- 0.007  | 0.622 +- 0.006  | 0.665 +- 0.045   | 
| 20           | 0.311 +- 0.023  | 0.372 +- 0.015  | 0.487 +- 0.032   | 
| 30           | 0.265 +- 0.009  | 0.295 +- 0.010  | 0.496 +- 0.011   | 
| 40           | 0.233 +- 0.014  | 0.262 +- 0.009  | 0.495 +- 0.014   | 

![alt text][plottrain]

## Benchmark Prediction

## Gridsearch

## Pipeline Results

[plottrain]: https://github.com/constantinpape/rf_benchmarks/blob/master/evaluation/plot_train.png  
