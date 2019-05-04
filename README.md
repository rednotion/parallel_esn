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
- Dependencies: NumPy, Cython **COMPLETE THIS SECTION**

## **Project Overview**
Echo State Networks (ESN) are recurrent neural networks making use of a single layer of sparsely connected nodes ('reservoir'). They are often used for time series tasks, and can be less computationally intensive other than deep learning methods. However, ESNs require fine tuning of many parameters, including the input weights, the reservoir (e.g. how many nodes in the reservoir, what is the spectral radius, etc). This has usually been done through either (a) sequential testing and optimization; or (b) instantiating many random instances, and then picking the best performing set of parameters. Depending on the length of the input data and the size of the reservoir, ESNs can thus be computationally intensive to train. In addition, we have to repeat this training many times before arriving at a good set of parameters. 

We propose to make use of parallel computing architectures to not only make this process **faster**, but also **smarter**. We do this through:
1. Setting the reservoir to be a _small world network_ with key properties to be defined 
2. Using _bayesian optimization_ to iteratively find the best set of parameters
3. Training the network faster through distributed computing with multiple nodes and multiple threads (_OpenMP_ and _MPI_)

- Description of problem and the need for HPC and/or Big Data
- Description of solution and comparison with existing work on the problem

## **Echo State Networks**
- Description of your model and/or data in detail: where did it come from, how did you acquire it, what does it mean, etc.
- Technical description of the parallel application and programming models used

### Training an ESN
<center>
<img src="https://github.com/rednotion/parallel_esn_web/blob/master/Screenshot%202019-04-30%20at%206.34.15%20PM.png?raw=true" width="500">
</center>

The classical method of training an ESN involves:
1. Generating the reservoir RNN with an input weight matrix $$\mathbf{W}_{in}$$ and reservoir matrix $$\mathbf{W}$$. The bigger the size of the reservoir (number of nodes), the more computational power needed. In both these matrices, the non-zero elements should follow a given distribution _(e.g. symmetrical uniform, gaussian, normal with mean 0)_. In our model, the matrices correspond to an adjacency matrix of a small world network (see below).
2. Train the network using the input $$\mathbf{u}(t)$$ and the activation states of the resevoir $$\mathbf{x}(n)$$. The update rule using some leaking rate $$\alpha$$ and a sigmoid wrapper such as $$tanh$$. 
3. Compute the linear readout weights $$\mathbf{W}_{out}$$ from the reservoir using linear regression that seeks to minimize the MSE between the estimated $$y(n)$$ and the true $$y^{target}(n)$$. We use the regularization coefficient $$\beta$$ during this process. 
4. Evaluate the performance of the model on either the training or validation set, using the inputs $$\mathbf{u}(t)$$ and the $$\mathbf{W}_{out}$$ obtained. Retune parameters if necessary.

Although it may seem simplistic to use a simple linear combination of weights to create the final prediction $$y(n)$$, the ESN capitalizes on the reservoir that both helps create non-linearity of the input, as well as retains memory of the input, to provide complex and rich information. 

### Small World Networks

A _small world network_ that can be defined by (a) the number of nodes and (b) the spectral radius $$\rho$$. The spectral radius $$\rho$$ should be tuned according to how much memory the output depends on (smaller values for short memory). Similarly, all non-zero nodes follow the same distribution as $$\mathbf{W}_{in}$$. 

N, p, $\rho$, k (number of nearest neighbors)
Watts and Strogatz

A small-world network lies between regular and random networks. regular network where nodes were connected only with their neighbors (see Fig. 1 (a)). Next, these connections were rewired with a probability p to a randomly selected node

small-world network and can spread information with a minimum number of long-range short cuts, resulting in low wiring-costs. This type of network is characterized by two factors: the shortest path length, L, and the clustering coefficient, C. L is defined as the average number of connections in the shortest path between two nodes. C is defined as the density of closed triangles, or triplets, consisting of three connected nodes. A regular network is characterized by a large C as well as a large L, while in a random network both L and C are small. In contrast, the small-world network has a small L and a large C.

In order to leverage the potential of the small-world topology, we limit the number of the reservoir nodes that receive external input (i.e., input nodes) or omit their signals to the output layer (i.e., output nodes). In addition, we segregate the input nodes from the output nodes, thereby necessitating the propagation of the input signals to the output nodes through the small-world reservoir. 

## **Bayesian Optimization**
Bayesian Optimization is often used in instances where we aim to evaluate some non-analytic function $f(x)$ 

### Asynchronous Bayesian Optimization
Some studies have shown that the results obtained from sequential bayesian optimization is equivalent to doing these tasks in parallel, among multiple workers. In addition, under time constraints, doing the bayesian optimization in parallel might lead to less regret (less error) than performing it in a sequential fashion. 



## **Architecture & Features**
- Technical description of the platform and infrastructure 
- Description of advanced features like models/platforms not explained in class, advanced functions of modules, techniques to mitigate overheads, challenging parallelization or implementation aspects...

### Computing Architecture
The set-up of the Parallel ESN is depicted in the figure below: There will be one leader node that manages the bayesian optimization. It distributes a set of paramater to each worker node to try, and upon completion of the ESN training, the worker node will report back the validation error associated with those parameters, for the leader node to update it's posterior belief before distributing new parameters. The computing architecture of this process represents **coarse-grained parallelism**.
<center>
<img src="https://github.com/rednotion/parallel_esn_web/blob/master/Screenshot%202019-04-30%20at%206.35.07%20PM.png?raw=true" width="600">
</center>

In addition to coarse-grained parallelism, we also attempt to optimize the training of each individual ESN for **fine-grained parallelism**. In addition to Cythonizing parts of the function, we also use **multi-threading** for the matrix multiplication operations, since those account for a large proportion of computation.

### Technical Specifications
- AWS nodes (m4xlarge)
- Set-up instructions

### Overheads and Mitigations
- **Communication**: In order to minimize overhead caused by communication, we kept the number and size of messages to the minimum. In particular, a leader node will only send out a _dictionary_ of parameters to try, and a worker node will send back the _same dictionary_ and the _validation error_. These are simple and small variables that are quick to send. 
- **Synchronization**: The process of updating the bayesian belief and generating new samples to try is generally quick, and indeed much faster than the training of a single ESN. Thus, it is unlikely that the leader node will cause a lag in the system. In addition, since we are doing _asynchronous bayesian optimization_ (rather than batch/synchronous), there is no need to wait for certain worker nodes to finish trying their parameters, before new ones can be issued. 
- **Sequential Sections**: Although computing X matrix within the reservoir is sequential (due to the memory/time-dependent property), the matrix multiplications that make up each one of these time-steps can be parallelized/threaded through NumPy or OpenMP. 
- **Load Balancing**: In general, there is no worry about load-balancing since each worker node is actually handling the _same amount/set of data_, just using different parameters in the training process.

## **Data**
- _Description of your model and/or data in detail: where did it come from, how did you acquire it, what does it mean, etc._

**Historical Hourly Weather Data 2012-2017** ([Dataset on Kaggle](https://www.kaggle.com/selfishgene/historical-hourly-weather-data)): The main dataset that we are testing for this project is historical hourly weather data. In particular, we subset the data to focus on a few key and continguous cities along the West Coast, and use 3 variables: Temperature, Humidity and Air Pressure. Weather patterns are a common example of time series data, and by using records from different (but contiguous cities), we hope to capture any time-lag effects (e.g. occurence on rain in a city upstate 1 hour earlier could predict rain now). A cleaned version of the dataset can be accessed **INCLUDE LINK HERE**.

**Hourly Energy Consumption** ([Dataset on Kaggle](https://www.kaggle.com/robikscube/hourly-energy-consumption#EKPC_hourly.csv)): This dataset is included in the examples built into the package. 

### Train-Validation Split
(if Cedric wants to write anything)

## **Empirical Testing & Results**
- Performance evaluation (speed-up, throughput, weak and strong scaling) and discussion about overheads and optimizations done

### Fine-grained (Number of cores)

### Coarse-grained (Number of nodes)

## **Conclusions**
Discussion about goals achieved, improvements suggested, lessons learnt, future work, interesting insights…

## **Citations**
Kawai, Y., Tokuno, T., Park, J., & Asada, M. (2017). Echo in a small-world reservoir: Time-series prediction using an economical recurrent neural network. 2017 Joint IEEE International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob). doi:10.1109/devlrn.2017.8329797

Lukoševičius, M. (2012). A Practical Guide to Applying Echo State Networks. Lecture Notes in Computer Science Neural Networks: Tricks of the Trade, 659-686. doi:10.1007/978-3-642-35289-8_36

https://www.pdx.edu/sites/www.pdx.edu.sysc/files/Jaeger_TrainingRNNsTutorial.2005.pdf
https://arxiv.org/pdf/1611.05193.pdf
http://proceedings.mlr.press/v84/kandasamy18a/kandasamy18a.pdf
