<br/><br/>

Final Project for Harvard CS205: Computing Foundations for Computational Science ([course website](http://iacs-courses.seas.harvard.edu/courses/cs205/index.html))
**Contributors:** Zachary Blanks, Cedric Flamant, Elizabeth Lim, Zhai Yi

## Package Instructions
- Technical description of the software design, code baseline, dependencies, how to use the code, and system and environment needed to reproduce your tests

Github Repo: [link](https://github.com/zblanks/parallel_esn)

**Notable & Advanced Features**
- Source code is a distributable Python package with auto-generated documentation with continuous integration
- C compiled code (Cython) for additional speed-up: When installing the package, the Cythonized functions will be compiled during the process. When using the package, users can set the flag `use_cython = True` to take advantage of this speed-up during training. 
- Unit testing with `pytest`
- MPI broadcasting when using parallel architecture systems (e.g. multi-node clusters). Sequential version is also available. 

**Dependencies & Installation Notes**
- NumPy must be installed before installing the _Parallel ESN_ package
- User must have GCC (or an equivalent compiler) in order to install the package
- Package Dependencies: NumPy, Cython, scikit-learn, networkx, mpi4py

**Running a simple example**
??? Or put on github repo

## **Project Overview**
- _Description of problem and the need for HPC and/or Big Data_
- _Description of solution and comparison with existing work on the problem_

Echo State Networks (ESN) are recurrent neural networks making use of a single layer of sparsely connected nodes ('reservoir'). They are often used for time series tasks, and can be less computationally intensive other than deep learning methods. However, ESNs require fine tuning of many parameters, including the input weights, the reservoir (e.g. how many nodes in the reservoir, what is the spectral radius, etc). This has usually been done through either (a) sequential testing and optimization; or (b) instantiating many random instances, and then picking the best performing set of parameters. Depending on the length of the input data and the size of the reservoir, ESNs can thus be computationally intensive to train. In addition, we have to repeat this training many times before arriving at a good set of parameters. 

We propose to make use of parallel computing architectures to not only make this process **faster**, but also **smarter**. We do this through:
1. Setting the reservoir to be a _small world network_ with key properties to be defined 
2. Using _bayesian optimization_ to iteratively find the best set of parameters
3. Training the network faster through distributed computing with multiple nodes and multiple threads (_OpenMP_ and _MPI_)

## **Echo State Networks**
- Description of your model and/or data in detail: where did it come from, how did you acquire it, what does it mean, etc.
- Technical description of the parallel application and programming models used

### Training an ESN
<center>
<img src="https://github.com/rednotion/parallel_esn_web/blob/master/Screenshot%202019-04-30%20at%206.34.15%20PM.png?raw=true" width="500">
<figcaption>Source: Kawai, Y., Tokuno, T., Park, J., & Asada, M. (2017) [1]</figcaption>
</center>
<br>
The classical method of training an ESN involves:
1. Generating the reservoir RNN with an input weight matrix $$\mathbf{W}_{in}$$ and reservoir matrix $$\mathbf{W}$$. The bigger the size of the reservoir (number of nodes), the more computational power needed. In both these matrices, the non-zero elements should follow a given distribution _(e.g. symmetrical uniform, gaussian, normal with mean 0)_. In our model, the matrices correspond to an adjacency matrix of a small world network (see below).
2. Train the network using the input $$\mathbf{u}(t)$$ and the activation states of the resevoir $$\mathbf{x}(n)$$. The update rule using some leaking rate $$\alpha$$ and a sigmoid wrapper such as $$tanh$$. 
3. Compute the linear readout weights $$\mathbf{W}_{out}$$ from the reservoir using linear regression that seeks to minimize the MSE between the estimated $$y(n)$$ and the true $$y^{target}(n)$$. We use the regularization coefficient $$\beta$$ during this process. 
4. Evaluate the performance of the model on either the training or validation set, using the inputs $$\mathbf{u}(t)$$ and the $$\mathbf{W}_{out}$$ obtained. Retune parameters if necessary.

Although it may seem simplistic to use a simple linear combination of weights to create the final prediction $$y(n)$$, the ESN capitalizes on the reservoir that both helps create non-linearity of the input, as well as retains memory of the input, to provide complex and rich information. 

### Small World Networks
Small world networks are a family of graphs which are characterized by a small shortest path length (average distance between two nodes) and a large clustering coefficient (density of closed triangles, which are three nodes that are all connected). In layman's terms, one can think of this is a social graph, whereby any one individual is connected to a stranger through a series of connections ('_six degrees of separation_'). 

Kawai et. al (2017) [1] show that using small world networks can produce high performance even when the number of input and output nodes were reduced, unlike standard random or fully connected ESNs. 

We use the Watts and Strogatz method of generating small world networks, which first involves choosing the number of nodes $$N$$, then setting $$k$$ number of neighbors for each node, and then rewiring each of these connections with a probability $$p$$ to a randomly selected node. The final graph is then converted to an adjacency matrix, and connection weights are sampled from  a chosen distribution (e.g. normal, uniform...) to form a reservoir. This can be easily done by making use of existing packages such as `networkx` in Python.

The last important parameter for the reservoir matrix is the spectral radius $$\rho$$, which scales the matrix. The spectral radius $$\rho$$ should be tuned according to how much memory the output depends on (smaller values for short memory).




## **Bayesian Optimization**
Bayesian Optimization is often used in instances where we aim to evaluate some non-analytic and unknown function $$f(\mathbf{x})$$. In this scenario, our goal is to find the best $$\mathbf{x^*}$$ that maximizes/minimizes our function (e.g. lowest validation error) in the shortest amount of time, through sampling. The strategy works like this: we place some prior belief on what the random function could look like. Then, we sample different $$\mathbf{x}_i$$s and the associated value $$f(\mathbf{x}_i)$$. With these evaluations, we update the prior to form the posterior distribution. From this new distribution, we can approximate an '_acquisition function_' that tells us where in the sample space to search and evaluate next. 

### Thompson Sampling
There exist different algorithms for choosing the next best point to evaluate the function at. For this project, we use Thompson Sampling, which generates a sample from the posterior distribution, and then selects a point from that sample that maximizes the expected return. 

### Asynchronous Bayesian Optimization
Asynchronous Bayesian Optimization refers to the practice whereby we might have many workers simultaneously trying out evaluating $$\mathbf{x}_i$$, but we do not need to wait for all of them to finish before updating the posterior and generating new points to search. Instead, we simply do it whenever at least one worker has completed the process of training the ESN.

Some studies have shown that the results obtained from sequential bayesian optimization is equivalent to doing these tasks in parallel, among multiple workers. In addition, under time constraints, doing the bayesian optimization in parallel might lead to less regret (less error) than performing it in a sequential fashion. (See [5])


## **Architecture & Features**
- Technical description of the platform and infrastructure 
- Description of advanced features like models/platforms not explained in class, advanced functions of modules, techniques to mitigate overheads, challenging parallelization or implementation aspects...

### Parallel ESN Package
We developed the Parallel ESN package from scratch in Python. We leveraged on a few key libraries, such as `networkx` to create the small world network graphs, `numpy` for arrays and matrices, and `scikit-learn` for evaluating the validation/testing error. In addition, we used `Cython` to compile parts of the code in C, and `mpi4py` to set up the communications between nodes in the computing cluster. In addition, we also built helper functions that run examples, or help the user split the data into using training and validation sets. (More information about the package can be found at the top of this page under _Package Instructions_)

### Computing Architecture
The set-up of the Parallel ESN is depicted in the figure below: There is one leader node that manages the bayesian optimization. It distributes a set of parameters to each worker node to try, and upon completion of the ESN training, the worker node will report back the validation error associated with those parameters. The leader node will then update its posterior belief before distributing new parameters. The computing architecture of this process represents **coarse-grained parallelism**.
<center>
<img src="https://github.com/rednotion/parallel_esn_web/blob/master/Screenshot%202019-04-30%20at%206.35.07%20PM.png?raw=true" width="600">
</center>

In addition to coarse-grained parallelism, we also attempt to optimize the training of each individual ESN for **fine-grained parallelism**. In addition to Cythonizing parts of the function, we also use **multi-threading** for the matrix multiplication operations, since those account for a large proportion of computation.

### Technical Specifications
- AWS nodes (m4xlarge)
- Set-up instructions

### Overheads and Mitigations
**Communication**: In order to minimize overhead caused by communication, we kept the number and size of messages to the minimum. In particular, a leader node will only send out a _dictionary_ of parameters to try, and a worker node will send back the _same dictionary_ and the _validation error_. These are simple and small variables that are quick to send. 

**Synchronization**: The process of updating the bayesian belief and generating new samples to try is generally quick, and indeed much faster than the training of a single ESN. Thus, it is unlikely that the leader node will cause a lag in the system. In addition, since we are doing _asynchronous bayesian optimization_ (rather than batch/synchronous), there is no need to wait for certain worker nodes to finish trying their parameters, before new ones can be issued. 

**Sequential Sections**: Although computing X matrix within the reservoir is sequential (due to the memory/time-dependent property), the matrix multiplications that make up each one of these time-steps can be parallelized/threaded through NumPy or OpenMP. 

**Load Balancing**: In general, there is no worry about load-balancing since each worker node is actually handling the _same amount/set of data_, just using different parameters in the training process.

## **Data**
- _Description of your model and/or data in detail: where did it come from, how did you acquire it, what does it mean, etc._

**Historical Hourly Weather Data 2012-2017** ([Dataset on Kaggle](https://www.kaggle.com/selfishgene/historical-hourly-weather-data)): The main dataset that we are testing for this project is historical hourly weather data. In particular, we subset the data to focus on a few key and continguous cities along the West Coast, and use 3 variables: `Temperature`, `Humidity` and `Air Pressure`. Weather patterns are a common example of time series data, and by using records from different (but contiguous cities), we hope to capture any time-lag effects (e.g. occurence on rain in a city upstate 1 hour earlier could predict rain now). A cleaned version of the dataset can be accessed [here](https://raw.githubusercontent.com/rednotion/parallel_esn_web/master/west_coast_weather.csv).

**Hourly Energy Consumption** ([Dataset on Kaggle](https://www.kaggle.com/robikscube/hourly-energy-consumption#EKPC_hourly.csv)): This dataset is included in the examples built into the package. It is much smaller and runs quickly, but shows the performance of using ESNs in time series settings.

### Train-Validation Split
(if Cedric wants to write anything)

## **Empirical Testing & Results**
- Performance evaluation (speed-up, throughput, weak and strong scaling) and discussion about overheads and optimizations done

### Fine-grained (Number of cores)

### Coarse-grained (Number of nodes)

## **Conclusions**
Discussion about goals achieved, improvements suggested, lessons learnt, future work, interesting insights…

## **Citations**
[1] Kawai, Y., Tokuno, T., Park, J., & Asada, M. (2017). Echo in a small-world reservoir: Time-series prediction using an economical recurrent neural network. 2017 Joint IEEE International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob). [doi:10.1109/devlrn.2017.8329797](doi:10.1109/devlrn.2017.8329797)

[2] Lukoševičius, M. (2012). A Practical Guide to Applying Echo State Networks. Lecture Notes in Computer Science Neural Networks: Tricks of the Trade, 659-686. [doi:10.1007/978-3-642-35289-8_36](doi:10.1007/978-3-642-35289-8_36)

[3] H. Jaeger (2002): Tutorial on training recurrent neural networks, covering BPPT, RTRL, EKF and the "echo state network" approach. GMD Report 159, German National Research Center for Information Technology, 2002 (48 pp.) [https://www.pdx.edu/sites/www.pdx.edu.sysc/files/Jaeger_TrainingRNNsTutorial.2005.pdf](https://www.pdx.edu/sites/www.pdx.edu.sysc/files/Jaeger_TrainingRNNsTutorial.2005.pdf)

[4] Yperman, Becker, & Thijs. (2017, June 14). Bayesian optimization of hyper-parameters in reservoir computing. Retrieved from [https://arxiv.org/abs/1611.05193](https://arxiv.org/abs/1611.05193)

[5] Kandasamy, K., Krishnamurthy, A., Schneider, J. & Poczos, B.. (2018). Parallelised Bayesian Optimisation via Thompson Sampling. Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, in PMLR 84:133-142 [http://proceedings.mlr.press/v84/kandasamy18a/kandasamy18a.pdf](http://proceedings.mlr.press/v84/kandasamy18a/kandasamy18a.pdf)
