
# Code of the paper "Benchmarking Aggregation-Disaggregation Pipelines for Smart Charging"

This repository contains all the implementations of the EV charging event aggregation and disaggregation methods and the benchmarking setup used in the paper "Benchmarking Aggregation-Disaggregation Pipelines for Smart Charging."

## Prerequisites

- Python 
- rustup (https://www.rust-lang.org/tools/install)
- maturin (```pip install maturin```)
- gurobi (https://www.gurobi.com/downloads/)
- Dataset "Mobilität in Deutschland 2017" (https://www.mobilitaet-in-deutschland.de/archive/index.html)
  - Alternatively, you can supply any data that can be converted into the *Agent* class in *python_code/profile_generator.py*.
    In that case, you need to replace the function *prepare_run()* in *python_code/utils.py* with a version that takes your data as input.
    Feel free to contact me if there are any issues with this process.
- Electricity price, generation, and demand data for Germany 2022 in 15 min resolution from https://www.smard.de/en/downloadcenter/download-market-data/ (Warning: Use the english version of the website!).
  - Alternative: Only use the SINE price signal in *benchmark.run()*

## Running the benchmark

1. Change the paths in *python_code/config.py* to direct to the MID and price data.
2. Run ```maturin develop --release```
3. Run ```python parameter_tuning.py``` for parameter tuning.
4. Run ```python benchmark.py``` for benchmark results.

The results will be located in *results/*.

Once you run the two Python scripts,
you can use the notebook *Visualize.ipynb* to recreate the figures from the paper.

## Project Structure

- The Rust implementations are found under *src/*.

- The raw Python implementations are found under *python_code/raw_python_impls/*.

- All benchmark along with all helper scripts is contained in *python_code/*. 

