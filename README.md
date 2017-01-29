# Benchmarking RFs for ilastik

Comparing different Random Forest implementation for ilastik pixel classification.
Using random forests:
* Vigra Random Forest 2 (https://github.com/ukoethe/vigra)
* Vigra Random Forest 3 (check out https://github.com/constantinpape/vigra for pybindings)
* Sklearn Random Forest (https://github.com/scikit-learn/scikit-learn) (v 0.18.1)

Training and prediction on cutout of FlyEM FIB25 data.

Note:
* The Vigra RF2 is not internally parallelised, so I have used concurrent.futures to parallelize it from
python, by training / predicting multiple forests with the corresponding number of sub-trees. (This is also what ilastik does).
* The benchmarks and gridsearch were done on a workstation (20 cores, 256 GB RAM), the pipeline results on a laptob (4 cores, 16 GB RAM).
* All experiments were done with 17 features from filter responses for each instance.
* All experiments were done with 100 trees (ilastik default).

## Benchmark Training

Benchmarking the training for the 11,500 instances extracted from the ilastik project.

| Num Threads | Vigra RF2 [s] | Vigra RF3 [s] | Sklearn RF [s] | 
| ----------- | ------------: | ------------: | -------------: | 
| 1           | 4.424 +- 0.072 | 5.220 +- 0.056 | 3.347 +- 0.059  | 
| 2           | 2.249 +- 0.054 | 2.648 +- 0.028 | 1.957 +- 0.045  | 
| 4           | 1.167 +- 0.037 | 1.363 +- 0.020 | 1.157 +- 0.048  | 
| 6           | 0.832 +- 0.018 | 0.973 +- 0.013 | 0.890 +- 0.012  | 
| 8           | 0.658 +- 0.010 | 0.763 +- 0.007 | 0.666 +- 0.024  | 
| 10          | 0.523 +- 0.007 | 0.622 +- 0.006 | 0.665 +- 0.045  | 
| 20          | 0.311 +- 0.023 | 0.372 +- 0.015 | 0.487 +- 0.032  | 
| 30          | 0.265 +- 0.009 | 0.295 +- 0.010 | 0.496 +- 0.011  | 
| 40          | 0.233 +- 0.014 | 0.262 +- 0.009 | 0.495 +- 0.014  | 

![alt text][plottrain]

## Benchmark Prediction

Benchmarking the prediction for a 200^3 cutout.

| Num Threads | Vigra RF2 [s] | Vigra RF3 [s] | Sklearn RF [s] | 
| ----------- | ------------: | ------------: | -------------: | 
| 1           | 110.446 +- 4.239 | 116.642 +- 0.333 | 72.378 +- 0.133 | 
| 2           | 45.617 +- 0.474  | 55.341 +- 0.376  | 41.643 +- 0.272 | 
| 4           | 21.655 +- 0.169  | 28.441 +- 0.292  | 24.650 +- 0.335 | 
| 6           | 14.908 +- 0.126  | 19.621 +- 0.184  | 18.706 +- 0.502 | 
| 8           | 11.303 +- 0.100  | 15.112 +- 0.114  | 15.619 +- 0.307 | 
| 10          | 8.890 +- 0.030   | 12.379 +- 0.051  | 13.600 +- 0.453 | 
| 20          | 5.846 +- 0.111   | 6.636 +- 0.139   | 10.715 +- 0.496 | 
| 30          | 7.030 +- 0.075   | 5.876 +- 0.057   | 9.410 +- 0.399 | 
| 40          | 7.531 +- 0.097   | 4.871 +- 0.056   | 9.400 +- 0.178 | 

![alt text][plotprediction]

## Sklear RAM Issues

* The Sklearn RF needs insane amounts of RAM during prediction, for the feature matrix used here (~ 500 MB), it eats up all the 
RAM of my laptop (16 GB), even in single threaded prediction. See github issue: 
* Issue to sklearn about RAM consumption during training

## Gridsearch

## Pipeline Results

* For Pipeline: Try single block parallelisation -> appears to be faster!
* Use results from grid search in pipeline

##TODO

* Compare to more implementations / algourithms
** XGB
** GPU RandomForest


[plottrain]: https://github.com/constantinpape/rf_benchmarks/blob/master/evaluation/plot_train.png  
[plotprediction]: https://github.com/constantinpape/rf_benchmarks/blob/master/evaluation/plot_prediction.png  
