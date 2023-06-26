# **Unmasking the Unseen: Exploring Hidden Markov Models in Time Series Analysis**

### Authors: Huang Yuli, Lee Weng Keong, Ng Huai Ling, Ong Zhi Hao

<br> 

## **Introduction**

A **time series** is a **sequence** of data points **ordered in time**, such as stock prices, population growth, and sensor readings. Analysing time series data can reveal useful patterns, trends, and relationships that can help us understand the underlying dynamics of the system and make predictions or decisions.

However, time series data can also be noisy, complex, or exhibit time-varying behavior, meaning that the underlying patterns of the data may change over time. This makes it challenging to apply traditional methods such as linear regression or ARIMA models to time series data.

One way to overcome these challenges is to use **Hidden Markov Models (HMMs)**, which are a type of **probabilistic graphical model** that assumes that the data depends on some hidden factors or states that change over time (or whatever the index of the data is). In this article, we will explore the history of HMMs and time series analysis, how HMMs can be used for time series analysis, and how they are different from other time series analysis methods.


## **History**

The concept of HMMs was first introduced in the late 1960s by Leonard E. Baum and other researchers as a statistical model to capture the dynamics of sequential data with hidden states. Initially applied in the field of speech recognition, HMMs have since found applications in diverse domains, such as **bioinformatics**, **finance**, **weather forecasting**, and **natural language processing.**

While newer models like recurrent neural networks (RNNs), convolutional neural networks (CNNs), and transformer networks have gained popularity in recent years, HMMs still have their unique strengths and continue to be used in various domains.

HMMs are particularly effective in scenarios where the underlying system exhibits **discrete latent states** and the **observed data has sequential dependencies**. They are widely used in fields such as speech recognition, natural language processing, bioinformatics, and finance. HMMs have proven to be valuable in tasks such as part-of-speech tagging, gene sequencing, gesture recognition, and speech synthesis.

Moreover, HMMs have a solid theoretical foundation, are computationally efficient, and provide **interpretability** through the explicit modeling of hidden states and their transitions. They can handle missing data and accommodate different probability distributions for observations and transitions.


## **Components of a HMM**

One of the challenges of time series analysis is that the data may not follow a simple or stationary distribution, but rather depend on some hidden factors or states that change over time. For example, the stock market may switch between bullish and bearish regimes, or the weather may change from sunny to rainy. These hidden factors or states may not always be directly observable, but they can affect the observable data in some way.

A **hidden markov model (HMM)** is a statistical model that can capture this kind of dependency between the **observable data** and the **hidden states**. 

An HMM consists of a visible process and a hidden process. The visible process is a sequence of random variables that represent the observable data, such as the unemployment rate or air conditioning usage. The hidden process is a sequence of random variables that represent the hidden states, such as the market regime or the weather condition. The hidden processes is assumed to be a Markov chain, which means that **the current state depends only on the previous state**. 

To illustrate how a HMM might work for a time series problem, let us consider a toy example. Suppose our observable data is the weather condition, which can either be sunny or rainy. Assuming that rain is caused primarily by high humidity, we can model the system to include two hidden states: high humidity (humid), and low humidity (dry). 

### **Emission Matrix $B$**

When the humidity is high, there is a high chance of rain, say 90%, and when the humidity is low, the chance of rain is also low, say 20%. Since the weather can be either rainy or sunny, the chance of it being sunny given some humidity level will just be 100% minus the chance of it raining given that humidity level. \
These conditional probabilities $P(\textnormal{Sunny}|\textnormal{Dry})=0.8$, $P(\textnormal{Rainy}|\textnormal{Dry})=0.2$, $P(\textnormal{Sunny}|\textnormal{Humid})=0.1$, and $P(\textnormal{Rainy}|\textnormal{Humid})=0.9$ form what is known as the **emission matrix** $B = \begin{pmatrix}
0.8 & 0.2 \\
0.1 & 0.9
\end{pmatrix}$, which describes how likely an observable state is to occur given the current hidden state: 
|   |$\textnormal{Sunny}$|$\textnormal{Rainy}$|
|---|---|---|
|$\textnormal{Dry}$|$0.8$|$0.2$|
|$\textnormal{Humid}$|$0.1$|$0.9$|

Formally, $B = (b_{ij}) = (P(o_t = v_j| x_t = q_i))$ is a $N \times M$ matrix, where $o_t$ and $x_t$ are the observed and hidden states at time $t$, $v_j$ is the $j^{th}$ observable state, $q_i$ is the $i^{th}$ hidden state, and $N$ and $M$ are the numbers of hidden states and observable states respectively. \
We also have $\sum _{j=1} ^N B_{ij} = \sum _{j=1} ^N P(o_t = v_j| x_t = q_i) = 1$. 

### **Transition Matrix $A$**

Since the hidden sequence is a Markov chain, each random variable in the sequence depends only on the previous variable (except for the first variable). In our example, this means that the chance of it being humid now depends only on whether or not it was humid in the previous time step. Let us suppose that the chance of it being humid now when it was already humid in the previous time step is 70%, and the chance of it being humid now when it was dry is 40%. Then we have $P(\textnormal{Dry}_\textnormal{current}|\textnormal{Dry}_\textnormal{previous}) = 0.6$, $P(\textnormal{Humid}_\textnormal{current}|\textnormal{Dry}_\textnormal{previous}) = 0.4$, $P(\textnormal{Dry}_\textnormal{current}|\textnormal{Humid}_\textnormal{previous}) = 0.3$, and $P(\textnormal{Humid}_\textnormal{current}|\textnormal{Humid}_\textnormal{previous}) = 0.7$. \
These conditional probabilities form another matrix $A = \begin{pmatrix}
0.6 & 0.4 \\
0.3 & 0.7
\end{pmatrix}$ known as the **transition matrix**, which describes how likely a hidden state is to occur given the previous hidden state: 
|   |$\textnormal{Dry}_\textnormal{current}$|$\textnormal{Humid}_\textnormal{current}$|
|---|---|---|
|$\textnormal{Dry}_\textnormal{previous}$|$0.6$|$0.4$|
|$\textnormal{Humid}_\textnormal{previous}$|$0.3$|$0.7$|

Formally, $A = (a_{ij}) = (P(x_t = q_j|x_{t-1} = q_i))$ is a $N \times N$ matrix, where $q_i$ is the $i^{th}$ hidden state, and $x_t$ is the $t^{th}$ element in the hidden sequence. \
We also have $\sum _{j=1} ^N A_{ij} = \sum _{j=1} ^N P(x_t = q_j|x_{t-1} = q_i) = 1$. 


### **Initial Probability Distribution $\pi$**

Finally, since the hidden sequence must start with some initial variable, we also need the probabililities of the hidden sequence starting in each hidden state. In our example, these will be the chance that it starts out humid and the chance that it starts out dry, say 0.5 for both. These probabilities form the **initial probability distribution** $\pi = (0.5, 0.5)$, which describes the chances of the hidden sequence starting in each hidden state. 

Formally, $\pi = (\pi_1, \pi_2, ... , \pi_N)$, where $\pi_i = P(x_1 = q_i)$ is the probability that the first element in the hidden sequence is the $i^{th}$ hidden state. 


### **Hidden Markov Model $\lambda = (A, B, \pi)$**

The HMM for our example can be summarised in the following diagram:

<img src="images\toy_example.jpg" width="500"/>


## **Generating the Sequences**

### **Hidden Sequence $X$**

Now that we have defined all the probabilities required in the model, we can generate the sequences. We shall start with the hidden sequence. The first element in the hidden sequence is obtained by randomly choosing a hidden state following the initial probability distribution $\pi$. In our example, this means that the initial humidity level is randomly decided, with a 50% chance of being either high or low (since $\pi=(0.5, 0.5)$). \
Suppose that the initial humidity level is low. Then the first element in our hidden sequence $x_1 = q_0  = \textnormal{Dry}$. \
The second element in the hidden sequence is a random variable following the probability distribution represented in the $i^{th}$ row of the transition matrix $A$, where $q_i$ is the hidden state of the first element. In our example, we have $P(\textnormal{Dry}_\textnormal{current}|\textnormal{Dry}_\textnormal{previous}) = 0.6$, $P(\textnormal{Humid}_\textnormal{current}|\textnormal{Dry}_\textnormal{previous}) = 0.4$, so there is a slightly higher chance that the second element in the hidden sequence will still be $\textnormal{Dry}$. For each time step $t$, we can repeat this probability sampling for $x_t$ based on the transition probabilities associated with its previous hidden state $x_{t-1}$. \
The **hidden sequence** $X = (x_1, x_2, ... , x_T)$, where $T$ is the length of the sequence or the total number of time steps, is thus obtained from this repeated process of random sampling.

### **Observation sequence $O$**

The generation of the observation sequence is very similar to that of the hidden sequence, just that we now randomly sample using the probabilities found in the emission matrix $B$ instead of the transition matrix $A$. \
The first element in the observation sequence $o_1$ is a random variable following the probability distribution represented in the $i^{th}$ row of the emission matrix $B$, where $q_i = x_1$ is the hidden state of the first element. \
In our example, the probability of the weather starting out sunny is 80%, since the initial humidity level is low. \
The subsequent elements in the observation sequence are obtained in the same fashion by considering the conditional probabilities of generating each observable state given the current hidden state.  
The **observation sequence** $O = (o_1, o_2, ... , o_T)$, where $T$ is the length of the sequence or the total number of time steps, is thus obtained from this repeated process of random sampling.

The diagram below summarises how the hidden and obseravtion sequences are generated:

<img src="images\sequences_toy_example.jpg" width="500"/>

## **The Three Basic Problems of HMMs**

From the earlier exposition, we know that a HMM $\lambda = (A, B, \pi)$ is defined by its transition, emission, and initial probabilities. Depending on our available information and goals, there are three basic problems related to HMMs:

1. **Decoding Problem:** \
Given the model $\lambda = (A, B, \pi)$ and observation sequence $O$, \
what is the optimal hidden sequence $X$ that best explains the observation sequence $O$?
2. **Evaluation Problem:** \
Given the model $\lambda = (A, B, \pi)$ and observation sequence $O$, \
what is the probability of the observation sequence $P(O | \lambda)$?
3. **Learning Problem:** \
Given an observation sequence $O$, \
what is the optimal model $\lambda = (A, B, \pi)$ that maximises $P(O | \lambda)$?

The algorithms that are customarily used to solve each of these three problems are:

1. Solution to Decoding Problem: \
the **Viterbi algorithm**.
2. Solution to Evaluation Problem: \
the **Forward-Backward algorithm**.
3. Solution to Learning Problem: \
the **Baum-Welch algorithm**.

In the following sections, we will give a brief overview of how each of these algorithms work.

### **1. Viterbi Algorithm For Decoding Problem**

#### **Decoding Problem**

Going back to our toy example, suppose we know from past statistical research the conditional probabilities associated with the weather condition and humidity level, i.e. we know the transition, emission, and initial probabilities described above. Although we know that the system we are interested in now should be very similar to what we have studied before, we do not have any knowledge of the current or past humidity levels for our current case study. \
Even so, we have plenty of observations over time about the weather condition. Perhaps we are interested in finding out the humidity levels for some other research project. Using our sequence of observations, we can try to infer the corresponding sequence of hidden states, which are the humidity levels in our case. This problem of finding the most likely hidden sequence given our model and observation sequence is known as the decoding problem.

#### **Viterbi Algorithm**

Given $N$ hidden states and $T$ total time steps, the number of probabilities of all transitions over time to calculate would be $N^T$. The Viterbi algorithm addresses the computational challenge of calculating probabilities for all possible state transitions, which can be a daunting task when dealing with a large number of states and time points.
The key insight of the Viterbi algorithm is that, at any given time point, there is only one most probable path to reach a particular state. Therefore, instead of recalculating all possible paths when transitioning from one time point to the next, the algorithm discards less likely paths and retains only the most probable one for further calculations. By applying this approach at each time step, the number of computations is significantly reduced from exponential $N^T$ to quadratic $TN^2$, which is much more manageable.

(Cautionary Note: While the Viterbi algorithm is not the only solution to the decoding problem, it is often chosen because of its efficiency. However, since it is a greedy algorithm, it is not guaranteed to return the global optimal solution.)

Suppose that we observe the following sequence of weather conditions: $(\textnormal{Rainy}, \textnormal{Sunny}, \textnormal{Rainy})$.
Then the probability of producing the first observation $\textnormal{Rainy}$ is 
either $P(\textnormal{Rainy}_{t=1}|\textnormal{Dry}_{t=1})\times P(\textnormal{Dry}_{t=1}) = 0.2\times 0.5 = 0.25$ 
or $P(\textnormal{Rainy}_{t=1}|\textnormal{Humid}_{t=1})\times P(\textnormal{Humid}_{t=1}) = 0.9\times 0.5 = 0.45$ 
depending on whether the first hidden state is $\textnormal{Dry}$ or $\textnormal{Humid}$.
Let us denote these probabilities as $\delta_1(\textnormal{Dry}) = 0.25$ and $\delta_1(\textnormal{Humid}) = 0.45$.

For the second observation $\textnormal{Sunny}$, there are 4 possible hidden sequences that can generate the observation sequence $(\textnormal{Rainy}, \textnormal{Sunny})$: $(\textnormal{Dry}, \textnormal{Dry}), (\textnormal{Dry}, \textnormal{Humid}), (\textnormal{Humid}, \textnormal{Dry}), (\textnormal{Humid}, \textnormal{Humid})$.
The first probability will be
$P((\textnormal{Rainy}, \textnormal{Sunny})|(\textnormal{Dry}, \textnormal{Dry})) = \delta_1(\textnormal{Rainy}) \times P(\textnormal{Dry}_{t=2}|\textnormal{Dry}_{t=1}) \times P(\textnormal{Sunny}|\textnormal{Dry}_{t=2}) = 0.25 \times 0.6 \times 0.8 = 0.12$.
Similarly, we have
$P((\textnormal{Rainy}, \textnormal{Sunny})|(\textnormal{Dry}, \textnormal{Humid})) = 0.25 \times 0.4 \times 0.1 = 0.01$
$P((\textnormal{Rainy}, \textnormal{Sunny})|(\textnormal{Humid}, \textnormal{Dry})) = 0.45 \times 0.3 \times 0.8 = 0.108$
$P((\textnormal{Rainy}, \textnormal{Sunny})|(\textnormal{Humid}, \textnormal{Humid})) = 0.45 \times 0.7 \times 0.1 = 0.0315$
To compute $\delta_2(\textnormal{Dry})$, we take the maximum of the two probabilities $P((\textnormal{Rainy}, \textnormal{Sunny})|(\textnormal{Dry}, \textnormal{Dry}))$ and $P((\textnormal{Rainy}, \textnormal{Sunny})|(\textnormal{Humid}, \textnormal{Dry}))$, i.e. we pick the most likely probability path:
$\delta_2(\textnormal{Dry}) = \max (0.12, 0.108) = 0.12$. 
Similarly, $\delta_2(\textnormal{Humid} )= \max (0.01, 0.0315) = 0.0315$.
We can repeat this for $t = 3$ to obtain $\delta_t(q_i)$ across all time steps $t$ and hidden states $q_i$:

|   |$o_1 = \textnormal{Rainy}$|$o_2 = \textnormal{Sunny}$|$o_3 = \textnormal{Rainy}$|
|---|---|---|---|
|$\delta_t(\textnormal{Dry})$|$0.2\times 0.5 = 0.25$|$\max \{(0.25 \times 0.6 \times 0.8), (0.45 \times 0.3 \times 0.8)\} = \max \{0.12, 0.108\} = 0.12$|$\max \{(0.12 \times 0.6 \times 0.2), (0.0315 \times 0.3 \times 0.2)\} = \max \{0.0144, 0.00189\} = 0.0144$|
|$\delta_t(\textnormal{Humid})$|$0.9\times 0.5 = 0.45$|$\max \{(0.25 \times 0.4 \times 0.1), (0.45 \times 0.7 \times 0.1)\} = \max \{0.01, 0.0315\} = 0.0315$|$\max \{(0.12 \times 0.4 \times 0.9), (0.0315 \times 0.7 \times 0.9)\} = \max \{0.0432, 0.108\} = 0.0432$|

Since $\delta_3(\textnormal{Humid}) = 0.0432 > 0.0144 = \delta_3(\textnormal{Dry})$, the hidden sequence corresponding to this probability will be output as the optimal hidden sequence by the Viterbi algorithm. By tracing back the path from the calculations, we have the hidden sequence $(\textnormal{Dry}, \textnormal{Dry}, \textnormal{Humid})$. 

In general, we have the following steps for the Viterbi algorithm:
1. Initialisation:
$\delta_1(q_j) = \pi_j b_{ji} $, where $o_1 = v_i$.
2. Recursion:
$\delta_t(q_j) = \underset{1 \leq i \leq N}{\max} \delta_{t-1}(i) b_{ji} a_{ij}$, where $o_t = v_i$.
3. Termination:
Choose the best path ending at $t=T$ with $\underset{1 \leq i \leq N}{\max} \delta_T(q_i)$ by backtracking.


### **2. Forwards-Backwards Algorithm For Evaluation Problem**

#### **Evaluation Problem**

The evaluation problem is motivated by the learning problem, which is the problem of finding the optimal model $\lambda$ that maximises the probability of the observation sequence $O$. When we do not know the underlying probabilities of the system, solving the learning problem allows us to fit the best model to our observed data to form future predictions.
The evaluation problem provides the metric by which we can assess the goodness-of-fit of the training model $\lambda = (A, B, \pi)$ to the observations sequence $O$. 


#### **Forward-Backward Algorithm**

The forward-backward algorithm is a dynamic programming algorithm that can be used to solve the evaluation problem. The algorithm works by calculating two sequence of probabilities, the forward and backward probabilities, which are then used to compute the likelihood $P(O|\lambda)$.

##### **Forward Algorithm**

The forward algorithm is actually almost identical to the Viterbi algorithm, except that we replace all the maximum operators $\max$ with the summation operator $\sum$:

1. Initialisation:
$\alpha_1(q_j) = \pi_j b_{ji}$, where $o_1 = v_i$.
2. Forward Recursion:
$\alpha_t(q_j) = \underset{i = 1} {\overset{N}{\sum}} \alpha_{t-1}(q_i) b_{ji} a_{ij}$, where $o_t = v_i$.
3. Termination:
Ends when $t=T$ with $\alpha_T(q_i)$.

For our toy example, we have:

|   |$o_1 = \textnormal{Rainy}$|$o_2 = \textnormal{Sunny}$|$o_3 = \textnormal{Rainy}$|
|---|---|---|---|
|$\alpha_t(\textnormal{Dry})$|$0.2\times 0.5 = 0.25$|$ (0.25 \times 0.6 \times 0.8) + (0.45 \times 0.3 \times 0.8) = 0.12 + 0.108 = 0.228$|$(0.228 \times 0.6 \times 0.2) + (0.0415 \times 0.3 \times 0.2) = 0.02736 + 0.00249 = 0.02985$|
|$\alpha_t(\textnormal{Humid})$|$0.9\times 0.5 = 0.45$|$(0.25 \times 0.4 \times 0.1) + (0.45 \times 0.7 \times 0.1) = 0.01 + 0.0315 = 0.0415$|$(0.228 \times 0.4 \times 0.9) + (0.0415 \times 0.7 \times 0.9) = 0.08208 + 0.026145 = 0.108225$|

The computations in the following sections for our toy example will be left as an exercise for the reader.

##### **Backward Algorithm**

The backward algorithm is identical to the forward algorithm, except that we now perform the recursion backwards starting from the current observation:

1. Initialisation:
$\beta_T(q_i) = 1 $.
2. Backward Recursion:
$\beta_t(q_i) = \underset{j = 1} {\overset{N}{\sum}} \beta_{t+1}(q_j) b_{ji} a_{ij}$ where $o_{t+1} = v_i$.
3. Termination:
Ends when $t=1$ with $\beta_1(q_i)$.


##### **Forward-Backward Algorithm**

Returning to the Forward-Backward algorithm, the likelihood $P(O|\lambda)$ is defined as $\sum_{i=1}^N \alpha_T(q_i)\beta_T(q_i)$.


### **3. Baum-Welch Algorithm For Learning Problem**

The Baum-Welch algorithm is an iterative algorithm that updates the model $\lambda$ based on the observed data $O$ until convergence. 

Cautionary Note: The Baum-Welch algorithm is guaranteed to converge to a local maximum of the likelihood function. However, it is possible that the algorithm will converge to a suboptimal solution.

The Baum-Welch algorithm is defined by the following steps:

1. Initialisation:
Start with an initial estimate of the model parameters, such as random values or prior knowledge.
$\lambda_0 = (A_0, B_0, \pi_0)$
    <br>
2. Expectation Step:
Using the current model parameters, perform the forward-backward algorithm (or the forward procedure and backward procedure) to calculate the forward and backward probabilities for each time step. From these probabilities, we can calculate the expected state, emission and transition probabilities.
We define the probability of being in state $q_j$ at time $t$ to be
$\gamma_t(q_i) = \dfrac {\alpha_t(q_i)\beta_t(q_i)} {\sum_{i=1}^N \alpha_t(q_i)\beta_t(q_i)},$
and the probability of being in state $q_i$ at time $t$ and state $q_j$ at time $t+1$ to be
$\xi_t(q_i, q_j) = \dfrac {\alpha_t(q_i) a_{ij} b_{jk} \beta_t(q_j)} {\sum_{i=1}^N \sum_{j=1}^N  \alpha_t(q_i) a_{ij} b_{jk} \beta_t(q_j)}$,
where $o_{t+1} = k$, and $\alpha_t(q_i)$ and $\beta_t(q_i)$ are as defined earlier.
Then we have
Expected number of transitions from state $q_i$ to state $q_j$
$ = \underset{t = 1} {\overset{T}{\sum}} \xi_t(q_i, q_j)$
Expected number of transitions out of state $q_i$
$ = \underset{t = 1} {\overset{T}{\sum}} \gamma_t(q_i)$
Expected number of times observation $v_j$ occurs in state $q_i$
$ =\underset{t = 1, o_t=v_j} {\overset{T}{\sum}} \gamma_t(q_i)$
Expected frequency in state $q_i$ at time $t=1$
$ = \gamma_1(q_i)$

    <br>
3. Maximisation Step:
Based on the calculated expected values from the expectation step, update the model. This involves re-estimating the transition probabilities, emission probabilities, and initial state probabilities to maximise the likelihood of the observed data. <br>
$\begin{aligned}
\hat a_{ij} &= \dfrac{\textnormal{Expected number of transitions from state } q_i \textnormal{ to state } q_j}{\textnormal{Expected number of transitions out of state } q_j} \\
&= \dfrac{\sum _{t=1}^T \xi_t(q_i, q_j)}{\sum _{t=1}^T \gamma_t(q_i)}  
\end{aligned}$ <br>
$\begin{aligned}
\hat b_{ij} &= \dfrac{\textnormal{Expected number of  times observation } v_j \textnormal{ occurs in state } q_i}{\textnormal{Expected number of times in state } q_i} \\
&= \dfrac{\underset{t = 1, o_t=v_j} {\overset{T}{\sum}} \gamma_t(q_i)}{\sum _{t=1}^T \gamma_t(q_i)}  
\end{aligned}$ <br>
$\begin{aligned}
\hat \pi_i &= \textnormal{Expected frequency in state } q_i \textnormal{ at time } t = 1 \\
&= \gamma_1(q_i)  
\end{aligned}$ <br>

    <br>
4. Iteration:
Repeat the expectation and maximisation steps until the model parameters converge or reach a predefined stopping criterion.


The diagram below illustrates the Baum-Welch algorithm:

<img src="images\forward-backward-algo.png" width="500"/>

![Click here to view the Jupyter Notebook](HMM_code.html)

## **Comparison Of HMM With Other Models**

### **Model Results**

For our experiment, we chose to compare the HMN with the LSTM and 1D-CNN models.
We fitted the models on the Beijing PM2.5 Data from https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
to using only the PM2.5 values for forecasting.

### **Model Results**

#### **HMM**
```
Number of Hidden States: 9
Number of Iterations: 10
```
<img src="images\plot_HMM.png" width="700"/>

#### **LSTM**
```
Number of LSTM Layers: 1
Number of Units: 32
Number of Epochs: 10
```
<img src="images\plot_LSTM.png" width="700"/>

#### **1D-CNN**
```
Number of 1D-CNN Layers: 1
Number of Units: 32
Kernel size: 3
Number of Epochs: 10
```
<img src="images\plot_1DCNN.png" width="700"/>


#### **Evaluation Metrics**

<img src="images\plot_radar.png" width="700"/>


### **Time Complexity Analysis**

| Model        | Time Complexity for Training | Time Complexity for Inference |
|--------------|------------------------------|-------------------------------|
|ARIMA| $O(T^2)$|$O(T)$|
|HMM| $O(T N^2)$|$O(T N^2)$|
|RNN| $O(T N^2)$|$O(T N)$|
|CNN| $O(T K^2 N^2)$|$O(T K^2 N)$|
|Transformer|$O(T^2 N)$|$O(T^2 N)$|

#### **ARIMA**

The time complexity for training an ARIMA model is $O(T^2)$, where $T$ is the number of time steps. This is because in order to train the model, we need to compute the autocorrelation function for lags up to $T$, which involves a double summation over the data points, resulting in quadratic complexity. The time complexity for inference or making predictions with an ARIMA model is $O(T)$, as once the model is trained, generating a prediction for the next step only involves a fixed number of operations.

#### **Hidden Markov Model (HMM)**

The time complexity for training an HMM model is $O(TN^2)$, where $T$ is the sequence length and $N$ is the number of hidden states. This is due to the Baum-Welch algorithm (a special case of the Expectation-Maximisation algorithm) used for training HMMs, which iterates over the sequence and for each position recalculates estimates for the transition and emission probabilities, leading to a time complexity of $O(TN^2)$. The same time complexity applies for inference using the Viterbi algorithm to find the most likely state sequence.

#### **Recurrent Neural Networks (RNNs) Networks**

The time complexity for training an RNN is $O(TN^2)$, where $T$ is the sequence length and $N$ is the dimension of the hidden state. This is because at each time step we need to update the hidden state using a function of the previous hidden state and the current input. This function usually involves a matrix-vector multiplication, the complexity of which is proportional to the square of the dimension of the hidden state. The time complexity for inference is $O(TN)$ as once the model is trained, generating a prediction involves iterating over the sequence and performing a fixed number of operations for each step.

#### **Convolutional Neural Networks (CNNs)**

The time complexity for training a CNN is $O(TK^2N^2)$, where $T$ is the number of time steps, $K$ is the size of the filter, and $N$ is the number of filters. This is due to the need to perform a convolution operation between the filter and the input at each time step for each filter, and for each position of the filter, which leads to this time complexity. The time complexity for inference is $O(TK^2N)$ as once the model is trained, generating a prediction involves convolving the input with the learned filters.

#### **Transformers**

The time complexity for training a Transformer model is $O(T^2 N)$, where $T$ is the sequence length and $N$ is the feature dimension. This is due to the self-attention mechanism of Transformers which computes pairwise interactions between all inputs. The same time complexity applies during inference as the self-attention mechanism also applies then.

## **Limitations Of HMM**

While Hidden Markov Models (HMMs) have been widely used and proven effective in various applications, they do have certain limitations that should be considered when applying them to real-world problems. Some of the key limitations of HMMs include:

### **Reliance on Markovian Property of the Times Series**

One major assumption in HMMs is that the current state/observation only depends on the previous state/observation. This assumption, known as the Markov property, implies that the state transitions and emissions are independent of each other given the current state. However, in many real-world scenarios, this assumption may not hold.
For example, in natural language processing tasks, the next word in a sentence may depend on multiple previous words rather than just the immediate previous word. Similarly, in some speech recognition applications, the acoustic features of a current frame may be influenced by past frames beyond the immediate preceding frame.
The independence assumption can limit the modeling capabilities of HMMs when capturing complex dependencies and relationships between states and observations.

### **Difficulty in modeling long-term dependencies**
HMMs are often limited in their ability to model long-term dependencies in sequences. Since the transition probabilities in HMMs are based on the current state and do not explicitly consider distant past states.
Long-term dependencies are prevalent in various real-world sequences, such as natural language sentences, genomic sequences, and financial time series. Failing to model these dependencies accurately can lead to suboptimal performance in tasks such as language modeling, speech recognition, and sequence prediction.


## **Variants Of HMM**

| Model Variant | Description |
|---------------|-------------|
| Profile HMM | Adds match, insert, and delete states to more effectively model alignment to a sequence profile. This is particularly useful in bioinformatics where it's used to represent homologous positions in multiple aligned sequences of proteins or nucleic acids, thus aiding in the identification of new members of a protein family. |
| Pair HMM | Extends HMMs to model two sequences concurrently, with transitions and emissions depending on both sequences. This is particularly useful in the field of bioinformatics for sequence alignment, where it can be used to find the optimal alignment of two biological sequences, such as DNA, RNA, or proteins. |
| Context-Sensitive HMM | Allows transition and emission probabilities to depend on previous states or outputs. This is particularly useful in advanced speech recognition, where the pronunciation of a phoneme can be influenced by the preceding and following phonemes, improving the recognition accuracy. |
| Continuous HMM | Allows for continuous rather than discrete output distributions, usually via a mixture of Gaussians. This is particularly useful in speech recognition, where it is used to model the continuous spectral envelope of a speech signal, resulting in a more accurate representation of the spoken words. |
| Hidden Semi-Markov Model | Extends HMMs by allowing state durations to follow an explicit distribution. This is particularly useful in human activity recognition, where the duration of certain activities (like sleeping or working) can vary greatly, and modeling these durations can improve the accuracy of the recognition system. |
| Hierarchical HMM | Models with a hierarchy of HMMs, where transitions at one level correspond to entire sub-models at a lower level. This is particularly useful in complex behavior modeling, such as daily routine recognition, where high-level states might represent different activities (like sleeping or working), and lower-level states represent sub-activities within those activities. |
| Switching HMM | HMMs where the transition and emission probabilities can switch between different regimes or modes. This is particularly useful in financial modeling, where it can be used to model the behavior of financial markets under different economic conditions, aiding in the prediction of future market behavior. |
| Multi-layer HMM | Allows for multiple parallel sequences of states, with dependencies between layers modeled through transition probabilities between states in different layers. This is particularly useful in speech recognition, where it can be used to model the parallel processes of articulatory gestures and acoustic signals, providing a more comprehensive and accurate model of the speech production process. |

### **Continuous HMM**

<img src="images\CHMM.jpg" width="500"/>

Traditional Hidden Markov Models (HMMs) are designed to handle sequences of discrete observations. However, in many fields, observations are continuous and may exhibit complex statistical properties that cannot be adequately captured using discrete states. For example, in speech recognition, the audio signal is a continuous waveform, and treating it as a sequence of discrete symbols can lead to a loss of information.

The Continuous Hidden Markov Model (CHMM) addresses this problem by modeling the observations as continuous variables. Each hidden state is associated with a probability density function (pdf) over the observation space, instead of a discrete emission probability. The Gaussian distribution is commonly used as it well represents the majority of real-life probability distribution. The Gaussian Mixture Model (GMM), a weighted sum of multiple Gaussian distributions, is also often used for more complex distributions.

### **Hidden Semi-Markov Model**

<img src="images\HSMM.png" width="500"/>

Traditional HMMs model sequences where the time spent in each state is geometrically distributed, which can be a poor approximation for many real-world sequences. For example, in activity recognition, certain activities (like sleeping or working) may have drastically different durations. Within the duration, both the state and observation remains the same. Instead of following a fixed time steps size of minutes or hours, we can have variable time steps size suitable for each activity, greatly reduces the sequence length and hence computation.

The Hidden Semi-Markov Model (HSMM) addresses this problem by allowing for explicit duration modeling in each state. In an HSMM, the probability of transitioning to a new state depends not only on the current state but also on the time spent in the current state, i.e. the duration spent in one time step is considered as another hidden state.

### **Multi-layer HMM**

Standard Hidden Markov Models (HMMs) typically operate on a single layer, dealing with sequences of observations and transitions between a fixed set of hidden states. However, in some complex scenarios, such as the analysis of hierarchical or multi-level data, a single layer may not adequately capture the underlying dynamics. For instance, in natural language processing, we may want to consider the hierarchical nature of language, where letters form words, words form sentences, and sentences form paragraphs.

The Multi-layer HMM addresses this issue by extending the standard HMM to a hierarchical structure with multiple layers of hidden states. In the first layer, the model could capture low-level dynamics, such as transitions between individual words in a sentence. The second layer might represent higher-level transitions, such as the flow of sentences in a paragraph. Each layer's states encapsulate the behavior of the entire lower-level HMM, allowing the model to capture complex, multi-scale phenomena more effectively.

The effectiveness of Multi-layer HMMs can be seen in applications like advanced natural language processing tasks. For example, they can be used in machine translation where capturing the hierarchical structure of language can improve the quality of translations. The multi-layer structure helps the model to understand context better by considering the relationship between words, sentences, and paragraphs, leading to more accurate and natural-sounding translations.

### **Hierarchical HMM**

While standard Hidden Markov Models (HMMs) have been extensively applied in various fields, they often struggle to capture complex hierarchical processes. Hierarchical processes exist in various domains, such as speech recognition where sounds form phonemes, phonemes form words, and words form sentences.

The Hierarchical Hidden Markov Model (HHMM) is an extension of the standard HMM designed to handle these hierarchical structures. In an HHMM, states of the model are organized in a hierarchical manner, where a state at a higher level can expand into a sequence of lower-level states, forming a sub-model of its own. This hierarchy of states allows the model to capture the hierarchical dependencies in the data. The transitions within each sub-model are governed by their own set of transition probabilities, providing a way to model different stages or levels of the process separately.

One key difference between Hierarchical HMMs and Multi-layer HMMs lies in the way they model dependencies. In a Multi-layer HMM, each layer of states encapsulates the behavior of the entire lower-level HMM, creating a direct dependency between the layers. In contrast, in an HHMM, a state at a higher level expands into a sequence of lower-level states, modeling dependencies within a hierarchical structure.

An example of the Hierarchical HMM in action is in advanced speech recognition systems. In these systems, an HHMM can represent the hierarchical structure of language, with higher-level states representing words or sentences and lower-level states representing the phonemes or sounds that make up those words. This allows the model to better understand the hierarchy and dependencies in human speech, leading to improved recognition accuracy.

### **Hybrid-HMM**

While HMMs are good at modeling sequence dynamics and handling temporal dependencies of short range, they can struggle with modeling longer-term dependencies and complex feature interactions in the input data. On the other hand, RNNs and LSTMs excel at modeling complex dependencies and feature interactions, but they lack explicit modeling of state transitions that HMMs provide. For example, in speech recognition, the sequence of spoken words (modeled by the HMM) can depend on long-term contextual information such as the topic of conversation (modeled by the RNN/LSTM).

The RNN-HMM or LSTM-HMM hybrids combine the strengths of both models: they use the RNN/LSTM to model complex dependencies and feature interactions, and the HMM to model the sequence dynamics and state transitions. In such hybrid, the RNN/LSTM processes the input sequence to extract a high-level feature representation. This feature representation is then used as the observation sequence for the HMM. The RNN/LSTM and HMM are trained jointly to optimize the overall sequence modeling performance.


## **Future Directions**

The future of Hidden Markov Models (HMMs) lies in the integration of deep learning techniques, the development of hybrid models. These advancements aim to enhance the capabilities of HMMs and enable them to address more complex modeling tasks.

### **Integration of Deep Learning Techniques**

Deep learning has revolutionized various areas of machine learning and has shown significant advancements in sequence modeling tasks. The integration of deep learning techniques with HMMs holds promise for improving the modeling capabilities of HMMs.

Hybrid models that combine the strengths of different probabilistic models are another direction for enhancing the capabilities of HMMs. These models aim to overcome the limitations of HMMs by incorporating complementary models that can capture specific aspects of the data.

For example, combining Hidden Markov Models with Recurrent Neural Networks (RNNs) has been explored to capture both the local dependencies modeled by the HMMs and the long-term dependencies captured by the RNNs. This hybrid approach, known as the Hidden Markov Model-Deep Neural Network (HMM-DNN), has achieved impressive results in speech recognition and other sequence modeling tasks.
The combination of HMMs with Gaussian Mixture Models (GMMs) or Deep Belief Networks (DBNs) has been explored to model the emission probabilities in a more expressive manner. This allows for better modeling of the data distribution, particularly in scenarios with complex or multimodal observations.

By leveraging the powerful representation learning capabilities of deep neural networks, the integration of deep learning techniques with HMMs can enable more accurate modeling of complex patterns, better handling of long-term dependencies, and improved performance in challenging tasks.


### **HMM As An Inspiration To Other Domains**

The Markovian property is foundational to HMM as well as many other areas of artificial intelligence. The memory-less state transition greatly simplifies the calculation, which is an essential prerequiste to the application of Dynamic Programming and Bellman backup.

Reinforcement Learning (RL) is one such area where the Markovian Properties are heavily exploited. RL problems often involve making sequences of decisions, where the optimal decision depends on the current "state" (or "state-action" paris) of the environment.

The transition between the "states", and the emssion from the "state" to observation are analogous to the environment dynamics and agent's policy in RL. The concept of Viterbi algorithm, which tries to identify the most likely hidden state sequence based on a known observation sequence, can be directly related to the concept of Partially Observable Markov Decision Process (POMDP), i.e., the true state may not be directly observable, and instead, we only have access to observations that are influenced by the state, we will establish a 'belief' of the true state sequence, and correct our belief in every subsequent steps based on new observations using Bayesian Inference.

The Baum-Welch algorithm, which is to learn the model, can find analogy in all variations of majoy RL approaches, including Value-iteration, Policy-iteration or Model-based RL.  This setup is very similar to an HMM, where the underlying state needs to be estimated based on the observations. Many RL algorithms use methods similar to those used in HMMs (like the Baum-Welch or Viterbi algorithms) to estimate these underlying states.

The use of HMMs is also found in newer areas like stable diffusion processes, which are used for generative modeling tasks. The concept of a diffusion process can be thought of as a continuous-time version of an HMM. In a diffusion process, the system transitions between an infinite number of possible states, and these transitions happen continuously over time. The "stable" in stable diffusion process refers to the property that the distribution of states at any time point is a stable distribution. Similar to HMMs, these processes involve an underlying latent process that generates observations. However, unlike HMMs, the transitions in diffusion processes are modeled as continuous-time stochastic processes.

Furthermore, the notion of hidden states and the probabilistic modeling approach of HMMs have influenced the design of many modern deep learning architectures. For instance, Recurrent Neural Networks (RNNs), and particularly their gated variants such as LSTMs and GRUs, model sequential data by maintaining hidden states across time steps, akin to the hidden states in HMMs.


## **References**
- https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e
- https://homepages.ecs.vuw.ac.nz/~marslast/MLbook.html
- https://www.cs.hmc.edu/~yjw/teaching/cs158/lectures/17_19_HMMs.pdf
- https://web.stanford.edu/~jurafsky/slp3/A.pdf
