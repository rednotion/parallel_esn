<br/><br/>

Final Project for Harvard CS205: Computing Foundations for Computational Science ([course website](http://iacs-courses.seas.harvard.edu/courses/cs205/index.html))
**Contributors:** Zachary Blanks, Cedric Flamant, Elizabeth Lim, Zhai Yi

## Package Instructions
Technical description of the software design, code baseline, dependencies, instruction for usage, and test examples can be found on the **Github Repo**: [link](https://github.com/zblanks/parallel_esn)

**Notable & Advanced Features**
- Source code is a distributable Python package with auto-generated documentation with continuous integration
- C compiled code (Cython) for additional speed-up: When installing the package, the Cythonized functions will be compiled during the process. When using the package, users can set the flag `use_cython = True` to take advantage of this speed-up during training. 
- Unit testing with `pytest`
- Uses `mpi4py` in parallel architecture systems (e.g. multi-node clusters). Sequential version is also available. 

## **Project Overview**
Echo State Networks (ESN) are recurrent neural networks making use of a single layer of sparsely connected nodes ('reservoir'). They are often used for time series tasks, and can be less computationally intensive other than deep learning methods. However, ESNs require fine tuning of many parameters, including the input weights, the reservoir (e.g. how many nodes in the reservoir, what is the spectral radius, etc). This has usually been done through either (a) sequential testing and optimization; or (b) instantiating many random instances, and then picking the best performing set of parameters. Depending on the length of the input data and the size of the reservoir, ESNs can thus be computationally intensive to train. In addition, we have to repeat this training many times before arriving at a good set of parameters. 

We propose to make use of parallel computing architectures to not only make this process **faster**, but also **smarter**. We do this through:
1. Setting the reservoir to be a _small world network_ with key properties to be defined 
2. Using _bayesian optimization_ to iteratively find the best set of parameters
3. Training the network faster through distributed computing with multiple nodes and multiple threads (_OpenMP_ and _MPI_)

### Existing Work
A popular algorithm for time series data is Long Short-Term Memory (LSTMs), which have a computational complexity of $$O(ELW)$$ where $$E$$ is the number of training epochs, $$L$$ is the length of the time series, and $$W$$ is the number of weights. In contrast, a single ESN is much more efficient with a complexity of $$O(LW)$$. In both cases, parameters would have to be tuned and the network trained multiple times, which makes the process time and computation-intensive. Using ESNs thus already saves significant time. 

In terms of existing implementations of ESN, there exist some practical guides as to how to apply ESNs (see [2]), but these rely largely on rule-of-thumbs and heuristics. Another option is to do a grid search (such as this paper [here](https://phy.duke.edu/sites/phy.duke.edu/files/file-attachments/2015_Thesis_JennySu.pdf)). However, this is computationally inefficient, and would be better served by using intelligent search, which is what we propose. 

## **Echo State Networks**
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
### Parallel ESN Package
We developed the Parallel ESN package from scratch in Python. We leveraged on a few key libraries, such as `networkx` to create the small world network graphs, `numpy` for arrays and matrices, and `scikit-learn` for evaluating the validation/testing error. In addition, we used `Cython` to compile parts of the code in C, and `mpi4py` to set up the communications between nodes in the computing cluster. In addition, we also built helper functions that run examples, or help the user split the data into using training and validation sets. (More information about the package can be found at the top of this page under _Package Instructions_ or at the github repo)

### Computing Architecture
The set-up of the Parallel ESN is depicted in the figure below: There is one leader node that manages the bayesian optimization. It distributes a set of parameters to each worker node to try, and upon completion of the ESN training, the worker node will report back the validation error associated with those parameters. The leader node will then update its posterior belief before distributing new parameters. The computing architecture of this process represents **coarse-grained parallelism**.
<center>
<img src="https://github.com/rednotion/parallel_esn_web/blob/master/Screenshot%202019-04-30%20at%206.35.07%20PM.png?raw=true" width="600">
</center>

In addition to coarse-grained parallelism, we also attempt to optimize the training of each individual ESN for **fine-grained parallelism**. In addition to Cythonizing parts of the function, we also use **multi-threading** for the matrix multiplication operations, since those account for a large proportion of computation.

### Technical Specifications
The experiment was run on 9 AWS **m4.2xlarge** instances, with the following hardware specs:
- CPUs: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
- Cache Memory: L1d (32K), L1i (32K), L2 (256K), L3 (46080K)
- Memory: 16Gb in bank 0 and bank 1
- Network Bandwidth: Minimum 1Gigabit/s
- Storage: 7.7Gb

and the following software specs:
- OS: Ubuntu 16.04.5 LTS
- Linux Kernel: 4.4.0-1079-aws
- For a full list of dependencies, see [here](https://raw.githubusercontent.com/rednotion/parallel_esn_web/master/dependencies.txt)

The instructions for setting up the cluster and running the package and experiments can be found on the GitHub repo [here](https://github.com/zblanks/parallel_esn/blob/master/cluster_instructions.md).

### Overheads and Mitigations
**Communication**: In order to minimize overhead caused by communication, we kept the number and size of messages to the minimum. In particular, a leader node will only send out a _dictionary_ of parameters to try, and a worker node will send back the _same dictionary_ and the _validation error_. These are simple and small variables that are quick to send. 

**Synchronization**: The process of updating the bayesian belief and generating new samples to try is generally quick, and indeed much faster than the training of a single ESN. Thus, it is unlikely that the leader node will cause a lag in the system. In addition, since we are doing _asynchronous bayesian optimization_ (rather than batch/synchronous), there is no need to wait for certain worker nodes to finish trying their parameters, before new ones can be issued. 

**Sequential Sections**: Although computing the X matrix within the reservoir is sequential (due to the memory/time-dependent property), the matrix multiplications that make up each one of these time-steps can be parallelized and multi-threaded with OpenMP on the backend. 

**Load Balancing**: In general, there is no worry about load-balancing since each worker node only gets assigned a new task whenever it completes the previous one.

## **Data**
**Historical Hourly Weather Data 2012-2017** ([Dataset on Kaggle](https://www.kaggle.com/selfishgene/historical-hourly-weather-data)): The main dataset that we are testing for this project is historical hourly weather data. In particular, we subset the data to focus on a few key and continguous cities along the West Coast, and use 3 variables: `Temperature`, `Humidity` and `Air Pressure`. Weather patterns are a common example of time series data, and by using records from different (but contiguous cities), we hope to capture any time-lag effects (e.g. occurence on rain in a city upstate 1 hour earlier could predict rain now). A cleaned version of the dataset can be accessed [here](https://raw.githubusercontent.com/rednotion/parallel_esn_web/master/west_coast_weather.csv).
- 44345 x 15 input matrix 
- 90% of data used for training, 8.75% for validation, 1.25% for testing


**Hourly Energy Consumption** ([Dataset on Kaggle](https://www.kaggle.com/robikscube/hourly-energy-consumption#EKPC_hourly.csv)): A small portion of this dataset is included in some of the examples built into the package. It is much smaller and runs quickly, but shows the basics of using ESNs to predict time series.

### Train-Validation Split / Data Cleaning
For the Hourly Energy Consumption data learning demonstrated in the `example` folder of `parallel_esn`, the last 1/10th of the timeseries data is saved to use as a validation set. No other data cleaning was necessary, as seen in  `parallel_esn.example.power_consumption_recursive`.

The Historical Hourly Weather Data 2012-2017 dataset required the extraction of the cities and measured quantities we wished to train with, and also necessitated some cleaning. This was done using the script [extract_final.py](https://raw.githubusercontent.com/rednotion/parallel_esn_web/master/extract_final.py) which was run in the same directory as the unzipped Kaggle dataset. The cleaning entailed removing extreme outliers in some of the measurements, which were due to sensor malfunctions or data-collection mistakes in the original dataset. The comments in the extraction script detail how outliers were determined. Missing data was linearly interpolated when possible (as opposed to dropping the row of data) to ensure continuity of the time series. We truncated rows at the beginning and the end of the dataset where some feature columns could not be interpolated.

For the experiment runs in `parallel_esn.experiments.seq_west_coast_weather` and `parallel_esn.experiments.mpi_west_coast_weather` we saved the last 10% of the data for validation (8.75%) and testing (1.25%). 


## **Empirical Testing & Results**
### Sequential baseline
The following table contains the results of running the baseline sequential code for different problem sizes.

| N Iterations | Time (seconds) |
| ------------ | -------------- |
| 200          | 2024.9139      |
| 400          | 4038.2928      |
| 800          | 8248.6406      |
| 1600         | 19237.6378     |

### Fine-grained (Number of threads)
<center><img src="https://github.com/rednotion/parallel_esn_web/blob/master/Finegrained.png?raw=true" width="400"></center>
We started the code parallelization process by using fine-grained changes via OpenMP. To see the resulting speed-up of the code we ran the previous test on a single shared-memory machine and varied the number of threads between one and four (the number of cores on the AWS m4.2xlarge instance). The speed-up is shown in the above figure. The primary takeaway from our results is that by using OpenMP we are indeed able to get a speed-up; however, it is not close to linear. We believe this result can be attributed to two causes:

1. OpenMP introduces parallel overheads by synchronizing the threads during the matrix multiplication operations
2. There are sequential parts of the code that cannot be performed in parallel and thus form a bottleneck during the training process.

Therefore to achieve the goal of further minimizing the training time and potentially enabling us to solve larger problems, we felt it was necessary to then employ a coarse-grained parallelization via the message passing interface (MPI).

### Coarse-grained (Number of nodes)
<center><img src="https://github.com/rednotion/parallel_esn_web/blob/master/Speedup.png?raw=true" width="400"></center>
Coarse-grained parallelism was achieved by implementing the parallel algorithm detailed previously in this report. Namely, a leader node controls the Bayesian updates and worker nodes train an ESN. To test the scalability of our algorithm, we investigated its strong and weak scaling properties, as well as performing code optimizations by tuning the MPI and OpenMP tasks and threads.
 
#### Strong Scaling
For each of the strong scaling experiments, we fixed the number of Bayesian search iterations to 800 and tracked the run time when there were one, two, four, and eight worker nodes and one leader node. To be concrete, the term node denotes a single m4.2xlarge instance in the MPI cluster. For these experiments, we went with the default of providing each node with four threads for OpenMP operations. 

For the experiment with one leader and one worker node, the algorithms performs worse than the optimized sequential benchmark which is a consequence of the asynchronous Bayesian optimization procedure. By having only one worker node which communicates with a leader node, we are introducing more parallel overheads with no benefit because the search is not being distributed to other workers. However, in the case with two, four, and eight worker nodes, our algorithm does provide a significant speed-up. Notice though that there is a near constant optimality gap between a linear speed-up and the strong-scaling speed-up displayed in the figure. This gap can be primarily attributed to the fact that the algorithm is providing four threads to the leader node even though this may not be beneficial. This topic further explored during code optimizations section.

#### Weak Scaling
For the weak scaling experiments, we grew the size of the problem proportional to the provided parallel resources. In particular we investigated the following values:

| Bayesian Iterations | Number of Worker Nodes |
|-------------------- |------------------------|
| 200                 | 1                      |
| 400                 | 2                      |
| 800                 | 4                      |
| 1600                | 8                      |

Similar to the strong scaling experiments we went with the default of providing four threads to each of the nodes. 

The weak-scaling may seem a bit strange at first glance. For the cases of one, two, and four worker nodes, they all take approximately 2100 seconds. Nevertheless, when the number of iterations is increased to 1600, the compute time jumps to roughly 3500. Upon further investigation, we believe the cause of this spike is the Bayesian search process started considering larger matrices which increases the computational burden on the system. An additional point is that Bayesian optimization, as we have coded it, is a stochastic algorithm and thus these variances in speed-up can sometimes be attributed to the process search more or less computationally expensive spaces.

## Code Optimizations

| No. MPI tasks   | No. Threads     | Speed-up    |
| --------------- | --------------- | ----------- |
| 9               | 4               | 7.38        |
| 18              | 2               | **8.18**    |
| 36              | 1               | 6.57        |

For the final portions of our experiments, we wanted to tune the number of MPI tasks and OpenMP threads to ensure that we were getting as much performance out of the system as possible. The values and results of this experiment are shown in the table above. The upper-bound for MPI tasks is 36 because there was a total of 36 cores in the cluster.

We found the best performing set-up was the case where there was 18 MPI tasks and each task was given two threads. We believe the best explanation for this outcome is that the 18-2 split finds the sweet spot in terms of distributing the work of training ESNs while still providing the speed-up that we demonstrated during the multi-threaded experiment. Remember the primary inhibitor to the scalability in the strong-scaling experiments was the fact that the leader node was getting four threads even though for most of the operations it performs, it may not need that many. This raises a reasonable objection though: why did the pure MPI implementation of 36 tasks and one thread not do the best if the previous claim is true? We believe the answer to this question is two-fold. One, there is a benefit to having multiple threads on a shared-memory system primarily for matrix multiplication operations. Two, when there is too many tasks a bottleneck is created where nodes are waiting for the Gaussian process to be updated because the leader node can only receive one update at a time.

Ultimately the results of this experiment demonstrate the potential speed-ups one can see when combining the strengths of the two programming platforms. Indeed, we went from a speed-up of 7.38 to 8.18 -- much closer to a linear value.

## **Conclusions**
Over the course of this project, we felt like we achieved a substantial amount: we managed to pull together some cutting-edge concepts in the field of reservoir computing (asynchronous bayesian optimization and small world networks), and create a beta implementation from scratch.

Future work could look at extending the algorithm to use GPUs, which could help with larger matrices. Alternatively, we could also look at support for sparse matrices. Certain parts of the documentation and implementation could also be cleaned up, such as removing Cython as a dependency and distributing the C code to compile instead. 

We only applied the parallel ESN framework to weather time series data, but it would be interesting to see how it performs in other domains and applications.

Finally, an important takeaway is the value in hybrid computing: because of the differences in computation between an ESN training phase and a bayesian updating phase, there is a balance that can be struck so as to reduce as much idle time as possible. In fact, when applying the ESN to other domains, perhaps with larger matrices, the best combination of MPI tasks and threads could change based on the computing time. Understanding the concepts that drive or hinder performance can thus be very helpful in practical implementation. 


## **Citations**
[1] Kawai, Y., Tokuno, T., Park, J., & Asada, M. (2017). Echo in a small-world reservoir: Time-series prediction using an economical recurrent neural network. 2017 Joint IEEE International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob). [doi:10.1109/devlrn.2017.8329797](doi:10.1109/devlrn.2017.8329797)

[2] Lukoševičius, M. (2012). A Practical Guide to Applying Echo State Networks. Lecture Notes in Computer Science Neural Networks: Tricks of the Trade, 659-686. [doi:10.1007/978-3-642-35289-8_36](doi:10.1007/978-3-642-35289-8_36)

[3] H. Jaeger (2002): Tutorial on training recurrent neural networks, covering BPPT, RTRL, EKF and the "echo state network" approach. GMD Report 159, German National Research Center for Information Technology, 2002 (48 pp.) [https://www.pdx.edu/sites/www.pdx.edu.sysc/files/Jaeger_TrainingRNNsTutorial.2005.pdf](https://www.pdx.edu/sites/www.pdx.edu.sysc/files/Jaeger_TrainingRNNsTutorial.2005.pdf)

[4] Yperman, Becker, & Thijs. (2017, June 14). Bayesian optimization of hyper-parameters in reservoir computing. Retrieved from [https://arxiv.org/abs/1611.05193](https://arxiv.org/abs/1611.05193)

[5] Kandasamy, K., Krishnamurthy, A., Schneider, J. & Poczos, B.. (2018). Parallelised Bayesian Optimisation via Thompson Sampling. Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, in PMLR 84:133-142 [http://proceedings.mlr.press/v84/kandasamy18a/kandasamy18a.pdf](http://proceedings.mlr.press/v84/kandasamy18a/kandasamy18a.pdf)
