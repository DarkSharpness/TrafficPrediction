# How to compile the project to install

First of all, you must have a basic environment with Python 3.6 or higher. Also, since our program depends on some data processing front-end written in C++, you must have a C++ compiler installed on your machine. We recommend using the GNU Compiler Collection (GCC) with version no less than 13.1.0.

You may check your environment by running the following commands:

```bash
python --version
g++ --version
```

If all the basic requirements are met, you can proceed to the next step.

```bash
make clean
make -j
```

It will automatically process the data in a way that is suitable for our model. The input data should be put in the `data/` directory. We need at least 4 files:

- data/geo_reference.csv
- data/loop_sensor_test_baseline.csv
- data/loop_sensor_test_x.csv
- data/loop_sensor_train.csv
