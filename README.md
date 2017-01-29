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
* All experiments were done with 100 trees (ilastik default). The random forests are trained until purity for the first benchmarks.

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

## Sklearn RAM Issues

The Sklearn RF needs insane amounts of RAM during prediction. For the feature matrix used here (~ 500 MB), it eats up all the 
RAM of my laptop (16 GB). Hence I have profiled the maximal RAM consumption. Apparently it copies the input for every tree during prediction (see table).
The number of threads does not affect the RAM usage.

See also github issue: TODO

| Num Threads   | 1   | 2   | 4   | 8   | 10  | 20  | 
| ------------- | --: | --: | --: | --: | --: | --: | 
| **Num Trees** |     |     |     |     |     |     | 
| 5             | 1.94 GB | 1.94 GB | 1.94 GB | 2.19 GB | 2.19 GB | 2.19 GB |
| 10            | 3.16 GB | 3.23 GB | 3.23 GB | 3.23 GB | 3.72 GB | 3.72 GB |
| 25            | 6.83 GB | 6.84 GB | 6.84 GB | 6.93 GB | 7.08 GB | 6.97 GB |
| 50            | 12.94 GB | 12.94 GB | 13.00 GB | 13.00 GB | 13.43 GB | 13.02 GB |
| 100           | 25.15 GB | 25.16 GB | 25.28 GB | 25.28 GB | 25.48 GB | 25.82 GB |
| 200           | 49.58 GB | 49.59 GB | 49.73 GB | 49.78 GB | 49.77 GB | 49.85 GB |

## Gridsearch

To see if reducing the tree complexity while keeping can speed up the prediction while keeping high enough accuracy, I have run a grid-search 
over the maximal tree depth and the minimal number of instances per leaf.
To check for high enough accuracy, I compare the prediction to a reference probability map from the random forest trained until purity. 
Note that this corresponds to the **bold** entry, hence all accuracies around 0.966 are equivalent to the RF trained to purity.

| MinLeafSize   | 1   | 2   | 5   | 10  | 15  | 20  | 
| ------------- | --: | --: | --: | --: | --: | --: | 
| **MaxDepth**  |     |     |     |     |     |     | 
| 8             | *15.839 +- 0.271 s* | 15.640 +- 0.241 s | 15.687 +- 0.233 s | 15.644 +- 0.255 s | 15.689 +- 0.247 s | 15.608 +- 0.274 s |
| 10            | 19.072 +- 0.295 s | 19.045 +- 0.274 s | 18.742 +- 0.223 s | 18.668 +- 0.231 s | 18.634 +- 0.237 s | 18.575 +- 0.261 s |
| 12            | 20.845 +- 0.706 s | 20.660 +- 0.353 s | 20.535 +- 0.230 s | 20.332 +- 0.236 s | 19.957 +- 0.223 s | 20.034 +- 0.349 s |
| 15            | 21.575 +- 0.300 s | 21.678 +- 0.366 s | 21.291 +- 0.258 s | 20.902 +- 0.641 s | 20.729 +- 0.381 s | 20.436 +- 0.261 s |
| None          | **21.740 +- 0.455 s** | 21.588 +- 0.258 s | 21.415 +- 0.306 s | 20.953 +- 0.177 s | 20.818 +- 0.387 s | 20.442 +- 0.444 s |

Grid search: prediction time (4 Threads)

| MinLeafSize   | 1   | 2   | 5   | 10  | 15  | 20  | 
| ------------- | --: | --: | --: | --: | --: | --: | 
| **MaxDepth**  |     |     |     |     |     |     | 
| 8             | *0.954 +- 0.002* | 0.954 +- 0.003 | 0.955 +- 0.003 | 0.954 +- 0.002 | 0.955 +- 0.002 | 0.954 +- 0.002 |
| 10            | 0.965 +- 0.001 | 0.966 +- 0.001 | 0.966 +- 0.001 | 0.966 +- 0.001 | 0.965 +- 0.001 | 0.965 +- 0.001 |
| 12            | 0.967 +- 0.001 | 0.967 +- 0.001 | 0.966 +- 0.001 | 0.967 +- 0.001 | 0.967 +- 0.001 | 0.967 +- 0.001 |
| 15            | 0.966 +- 0.001 | 0.965 +- 0.001 | 0.967 +- 0.001 | 0.967 +- 0.001 | 0.968 +- 0.001 | 0.967 +- 0.001 |
| None          | **0.966 +- 0.002** | 0.966 +- 0.001 | 0.967 +- 0.001 | 0.967 +- 0.001 | 0.967 +- 0.001 | 0.967 +- 0.001 |

Grid search: accuracy

As we can see the min leaf size does not affect training times or accuracies significantly.
All trees with a max depth of 10 and are equivalent to the one trained to purity.
They also don't offer a significant speedup.
The forests trained only to depth 8 are not equivalent, but also faster in prediction.
Optically, the differences between this reduced and the full model appear not to be significant.
See images below, showing the results of **full** and *reduced* model.

![alt text][rawortho1] ![alt text][fullortho1] ![alt text][reducedortho1]

Images: raw data (left) and prediction for background class of **full** (middle) and *reduced* (right) model.


## Pipeline Results

Runtimes for (300,300,200) cutout.

|           | In-Block-Parallelisation | Over-Blocks-Parallelisation |
| --------- | -----------------------: | --------------------------: |
| Full RF   | 159 s                    | 86 s                        |
| Reduced RF| 154 s                    | 96 s                        |

WIP


## TODO

Compare to more implementations / algourithms
* XGB
* GPU RandomForest


[plottrain]: https://github.com/constantinpape/rf_benchmarks/blob/master/evaluation/plot_train.png  
[plotprediction]: https://github.com/constantinpape/rf_benchmarks/blob/master/evaluation/plot_prediction.png  

[rawortho1]: https://github.com/constantinpape/rf_benchmarks/blob/master/evaluation/grid_raw_ortho1.png
[fullortho1]: https://github.com/constantinpape/rf_benchmarks/blob/master/evaluation/grid_full_ortho1.png
[reducedortho1]: https://github.com/constantinpape/rf_benchmarks/blob/master/evaluation/grid_reduced_ortho1.png
