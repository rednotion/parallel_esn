<br/><br/>

Final Project for Harvard CS205: Computing Foundations for Computational Science ([course website](http://iacs-courses.seas.harvard.edu/courses/cs205/index.html))
**Contributors:** Zachary Blanks, Cedric Flamant, Elizabeth Lim, Zhai Yi

## Package Instructions
Github Repo: [link](https://github.com/zblanks/parallel_esn)

(Technical description of the software design, code baseline, dependencies, how to use the code, and system and environment needed to reproduce your tests)

## Project Overview
Echo State Networks (ESN) are recurrent neural networks making use of a single layer of sparsely connected nodes ('reservoir'). They are often used for time series tasks, and can be less computationally intensive other than deep learning methods. However, ESNs require fine tuning of many parameters, including the input weights, the reservoir (e.g. how many nodes in the reservoir, what is the spectral radius, etc). This has usually been done through either (a) sequential testing and optimization; or (b) instantiating many random instances, and then picking the best performing set of parameters. Depending on the length of the input data and the size of the reservoir, ESNs can thus be computationally intensive to train. In addition, we have to repeat this training many times before arriving at a good set of parameters. 

We propose to make use of parallel computing architectures to not only make this process **faster**, but also **smarter**. We do this through:
1. Setting the reservoir to be a _small world network_ with key properties to be defined 
2. Using _bayesian optimization_ to iteratively find the best set of parameters
3. Training the network faster through distributed computing with multiple nodes and multiple threads (_OpenMP_ and _MPI_)

- Description of problem and the need for HPC and/or Big Data
- Description of solution and comparison with existing work on the problem

## Echo State Networks
- Description of your model and/or data in detail: where did it come from, how did you acquire it, what does it mean, etc.
Technical description of the parallel application and programming models used

![Screenshot 2019-04-30 at 6.34.15 PM.png]

## Bayesian Optimization

## Architecture & Features
Technical description of the platform and infrastructure 
Description of advanced features like models/platforms not explained in class, advanced functions of modules, techniques to mitigate overheads, challenging parallelization or implementation aspects...


## Empirical Testing & Results
- Performance evaluation (speed-up, throughput, weak and strong scaling) and discussion about overheads and optimizations done


## Conclusions
Discussion about goals achieved, improvements suggested, lessons learnt, future work, interesting insights…

## Citations
Citations
