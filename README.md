# Speech-based TBI detection using Parametrized Sinc Filters and a Cascading GRU

This repository contains python codes for "Continuous TBI monitoring from Spontaneous Speech using Parametrized Sinc Filters and a Cascading GRU" paper.

After setting data path in Coelho.py, the model can be trained by running the following command:
* python Coelho.py --cfg Coelho.cfg --fold 0

## Dependency
* Librosa=0.7.2
* pytorch=1.3.1
* scikit-learn