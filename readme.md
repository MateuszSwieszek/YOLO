#  AI pipeline for object detection and profiling on CPU

This python script (main.py) generates release reports.

## Modules
DataGenerator - module contains `DataLoaderGenerator` class for downloading example coco vaildation dataset and wraps it with dataloader


SharedFunctions - module contains functions from `https://github.com/ultralytics/yolov5` used dof download, prepare dataset and  for pre-,post-pocessing. 


PerfMonitor - module contains `PerformaceMonitor` class that generates performance reports.


PerfMonitor - module contains `PerformaceMonitor` class that generates performance reports.


YOLOPipeline - module contains `Pipeline` class that runs inference, and stores example image results and performance results in `runs/` directory.

## How to run?

```
Prepare virtual enviroment `python -m venv venv` and install all required packages `pip uninstall -r requirements.txt`

```
```
run main.py file 
```

## Dataset

In this code, for inference is used COCO validation dataset. It's downloaded automatically by DataGenerator module, but you can create other dataloader with different dataset and pass it to `Pipeline` input.

## Profiling
Profiling profiling has been performed with `Pytorch Profiler` described here: `https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html`

## Results
Running pipeline will create `runs/nun<number>` with example images and folder `perf_reports` that contains:
```
 .html file that contains table with accumulated runtime for each inference segment (pre-processing, inference, metrics, post-processing)
```
```
`memory_usage_batch_<NUM>.txt` -contains memory usage of each operator
```
```
`runtime_batch_<NUM>.txt` -contains runtime of each operator
```
```
`trace_batch_<NUM>.txt` -contains json file with profiling results of each operator
```
