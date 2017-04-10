# Deep CCA for Neuroimaging


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


## Wall Clock Timing

Sparse CCA:

MNIST (54k samples w/ 374 Features)
-----------------------------------
CPU: 


## Installation Steps
1. Download zipped repository
2. Unpack
3. cd deepSCCAN-master
4. run `python setup.py install`
5. To use sparseDecom2 function, for instance:
	- `from neuroCombat.sparseDecom2 import sparseDecom2`
	- Now you can use the function.. e.g. result = sparseDecom2(..)
