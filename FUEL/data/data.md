## The data is downloaded as follows:
```wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data```
```wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test```

## Then run dataset_generate.py to process the dataset and the data is split into two clients in 
data/train/mytrain.json
data/test/mytest.json
(We follow exactly the same data processing procedures described in [the paper](https://arxiv.org/abs/1902.00146 and ) we are comparing with. See ```dataset_generate.py``` for the details.)
