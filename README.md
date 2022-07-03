# teneva_research_optima_tt


## Description

Numerical experiments for **optima_tt** method from [teneva](https://github.com/AndreiChertkov/teneva) python package.


## Installation

1. Install [python](https://www.python.org) (version >= 3.7; you may use [anaconda](https://www.anaconda.com) package manager);

2. Install dependencies:
    ```bash
    pip install numpy teneva==0.9.8
    ```


## Usage

1. Run `python calc.py random_small`. The results will be presented in the text files `result/logs/random_small.txt` and `result/logs/random_small_show.txt`. All calculation results will be also saved in the file `result/data/random_small.npz`;

2. Run `python calc.py random_big`. The results will be presented in the text files `result/logs/random_big.txt` and `result/logs/random_big_show.txt`. All calculation results will be also saved in the file `result/data/random_big.npz`;

3. Run `python calc.py function_small`. The results will be presented in the text files `result/logs/function_small.txt` and `result/logs/function_small_show.txt`. All calculation results will be also saved in the file `result/data/function_small.npz`;

4. Run `python calc.py function_big` (TODO). The results will be presented in the text files `result/logs/function_big.txt` and `result/logs/function_big_show.txt`. All calculation results will be also saved in the file `result/data/function_big.npz`;

5. Run `python calc.py random_stat` (TODO). The results will be presented in the text files `result/logs/random_stat.txt` and `result/logs/random_stat_show.txt`. All calculation results will be also saved in the file `result/data/random_stat.npz`.

> You can also run `python demo.py MODE`, where argument `MODE` is `random_small`, `random_big`, `function_small`, `function_big` (TODO) or `random_stat` (TODO) for demonstration of the corresponding calculation for one fixed set of parameters with detailed log output.


## Author

- [Andrei Chertkov](https://github.com/AndreiChertkov) (a.chertkov@skoltech.ru).
