# DSL
Discrimination of Similar Languages


## Environment

```shell
pip install -r requirements.txt
```

## Dataset & Preprocessing
Download the dataset from [DSLLCC](http://ttg.uni-saarland.de/resources/DSLCC/)
We use [DSLCC v4.0](http://scholar.harvard.edu/files/malmasi/files/dslcc4.zip)

You should change the directory in the file
```
python preprocessing.py 
```

## Usage

SVM baselines
```shell
python -m svm.svm [charngram/2step]
```

Neural baselines
```shell
python -m charcnn.main --save_dir [checkpoints/temp] --model [bow/cnn/multicnn]
```
