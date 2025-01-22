# Lipschitz Estimation

Repository for the *Theretical fundamentals of Deep Learning* project in 3rd year at CentraleSupélec.
The code for SeqLip is from https://github.com/avirmaux/lipEstimation/tree/master

We kept only the utils components of the original code, and adapted the rest for our use cases : the sudy of PINNs

The code used to train the PINNs (Spring system and Gravitational Field) is from https://github.com/amigourou/PINNS

### Code organisation

* `lip_comparisons.ipynb` : the main code for comparing the results on our different use cases
* `lipschitz_approximations.py`: many estimators
* `lipschitz_utils.py`: toolbox for the different estimators
* `seqlip.py`: SeqLip and GreedySeqLip
* `utils.py`: utility functions

