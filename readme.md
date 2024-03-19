
# Code of the paper "Benchmarking Aggregation-Disaggregation Pipelines for Smart Charging"

This repository contains all the implementations of the EV charging event aggregation and disaggregation methods and the benchmarking setup used in the paper "Benchmarking Aggregation-Disaggregation Pipelines for Smart Charging."

## Running the benchmark

Step-by-step guide:
1. You need access to the dataset "Mobilität in Deutschland 2017" (https://www.mobilitaet-in-deutschland.de/archive/index.html).
2. Change the path at the bottom of *benchmark.py* to refer to your copy of the "Wege.csv" of the "Mobilität in Deutschland 2017".
3. Download price, electricity generation, and electricity demand data for Germany 2022 in 15 min resolution from www.smard.de and change the paths at the bottom *benchmark.py* accordingly. (Alternative: Only use the SINE price signal in *run()*)
4. Run *benchmark.py* 

If you can't or don't want to obtain the "Mobilität in Deutschland 2017" dataset,
you can have a look at the *Agent* class in *ev_model.py*.
If you can fit your data into this interface,
all you need to do is swap the function *prepare_run()* in *utils.py*, and 
the benchmark should also run for your data set.

## Run the parameter tuning

1. You need to obtain the same data sets as for *benchmark.py*.
2. Change the file names at the bottom of *parameter_tuning.py*.
3. Run *parameter_tuning.py*
   
## Check the results

Once you run the two Python scripts,
you can use the notebook *Visualize.ipynb* to recreate the figures from the paper.

# Acknowledgement
This work is created as part of the ESM-Regio project (https://www.bayern-innovativ.de/de/seite/esm-regio-en) and is made possible through funding from the German Federal Ministry for Economic Affairs and Climate Action.
