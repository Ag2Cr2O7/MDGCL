# MDGCL
This is the PyTorch implementation of the paper that we submitted for under review to NeurIPS2023.
>MDGCL: Multi-Level Dual-Space Enhanced Graph Contrastive Learning for Recommendation.

>Submitted to 37th Conference on Neural Information Processing Systems (NeurIPS 2023).

>Anonymous Author(s)

## Environment Requirement

The code runs well under python 3.8.0. The required packages are as follows:
- torch == 1.13.1
- numpy == 1.22.4
- scipy == 1.7.0
- pandas == 1.1.5
- pynvml == 11.5.0

## How to run the codes
* Tmall
```
python main.py --data tmall 
```

* Gowalla
```
python main.py --data gowalla 
```

* Amazon
```
python main.py --data amazon 
```

* Yelp
```
python main.py --data yelp
```

* ML-10M
```
python main.py --data ml10m 
```

