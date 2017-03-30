# Deep CCA for Neuroimaging

## TO-DO
- finalize+test hyper-parameter optimization code
- finalize deep+conv multivariate regression code
- multi-objective to make hidden layers correlated also?
- better example for conv cca (and deep cca too)
- include fancy_impute algorithms
- include confounds code (residuals of linear model)
- include batch effects (ComBat) code

## Overview

This repository contains the following models:
	
	- Sparse CCA (sparseDecom2.py)
		- Batch method (mycoption=0)
			- learn all components at once, with orthogonality constraints
		- Deflation method (mycoption=1)
			- learn components one at a time, with matrix deflation after each
	- Nonlinear Sparse CCA (kernelDecom2.py)
		- Batch or Deflation method
		- Passes the projections through some nonlinearity (e.g. sigmoid or relu)
	- Deep Feedforward CCA (ffDecom2.py)
		- Use feed-forward neural networks to learn components
		- Can be hybrid (e.g. deep layers on X and only one sparse layer on Y)
	- Deep Convolutional CCA (convDecom2.py)
		- Use convolutional neural networks to learn components

There is also a hyper-optimization algorithm using the Tree of Parzen Estimator algorithm for determining optimal hyper-parameters, which can then be passed into any cca function. It is in `hyperInit.py`.


## Installation Steps
1. Download zipped repository
2. Unpack
3. cd deepSCCAN-master
4. run `python setup.py install`
5. To use sparseDecom2 function, for instance:
	- `from neuroCombat.sparseDecom2 import sparseDecom2`
	- Now you can use the function.. e.g. result = sparseDecom2(..)
