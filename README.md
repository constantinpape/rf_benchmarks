# Benchmarking RFs for ilastik

Comparing different Random Forest implementation for ilastik pixel classification.
Using random forests:
* Vigra Random Forest 2 (https://github.com/ukoethe/vigra)
* Vigra Random Forest 3 (check out https://github.com/constantinpape/vigra for pybindings)
* sklearn Random Forest (https://github.com/scikit-learn/scikit-learn) (v 0.18.1)

Training and prediction on cutout of FlyEM FIB25 data.

TODOs:
* Issue to sklearn about RAM consumption during training
* Run evals
* For Pipeline: Try single block parallelisation -> appears to be faster!
* Also compare to xgb
* Use results from grid search in pipeline

## Benchmark Training

## Benchmark Prediction

## Gridsearch

## Pipeline Results
