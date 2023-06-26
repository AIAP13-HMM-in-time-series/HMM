# **Unmasking the Unseen: Exploring Hidden Markov Models in Time Series Analysis**

### Authors: Huang Yuli, Lee Weng Keong, Ng Huai Ling, Ong Zhi Hao

<br> 

## **Introduction**

A **time series** is a sequence of data points ordered in time, such as stock prices, population growth, and sensor readings. Analysing time series data can reveal useful patterns, trends, and relationships that can help us understand the underlying dynamics of the system and make predictions or decisions.

However, time series data can also be noisy, complex, or non-stationary, meaning that the statistical properties of the data change over time. This makes it challenging to apply traditional methods such as linear regression or ARIMA models to time series data.

One way to overcome these challenges is to use **Hidden Markov Models (HMMs)**, which are a type of probabilistic graphical model that assumes that the data depends on some hidden factors or states that change over time (or whatever the index of the data is). In this article, we will explore the history of HMMs and time series analysis, how they can be used for time series analysis, and how they are different from other time series analysis methods.


## **History**

The concept of HMMs was first introduced in the late 1960s by Leonard E. Baum and other researchers as a statistical model to capture the dynamics of sequential data with hidden states. Initially applied in the field of speech recognition, HMMs have since found applications in diverse domains, such as bioinformatics, finance, weather forecasting, and natural language processing.

While newer models like recurrent neural networks (RNNs), convolutional neural networks (CNNs), and transformer networks have gained popularity in recent years, HMMs still have their unique strengths and continue to be used in various domains.

HMMs are particularly effective in scenarios where the underlying system exhibits discrete latent states and the **observed data has sequential dependencies**. They are widely used in fields such as speech recognition, natural language processing, bioinformatics, and finance. HMMs have proven to be valuable in tasks such as part-of-speech tagging, gene sequencing, gesture recognition, and speech synthesis.

Moreover, HMMs have a solid theoretical foundation, are computationally efficient, and provide interpretability through the explicit modeling of hidden states and their transitions. They can handle missing data and accommodate different probability distributions for observations and transitions.


## **Components**

One of the challenges of time series analysis is that the data may not follow a simple or stationary distribution, but rather depend on some hidden factors or states that change over time. For example, the stock market may switch between bullish and bearish regimes, or the weather may change from sunny to rainy to cloudy. These hidden factors or states may not always be directly observable, but they can affect the observable data in some way.

A **hidden markov model (HMM)** is a statistical model that can capture this kind of dependency between the observable data and the hidden states. 

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

Formally, $B_{ij} = P(v_j|q_i)$ is a $N \times M$ matrix, where $v_j$ is the $j^{th}$ observable state, $q_i$ is the $i^{th}$ hidden state, and $N$ and $M$ are the numbers of hidden states and observable states respectively. \
We also have $\sum _{j=1} ^N B_{ij} = \sum _{j=1} ^N P(v_j|q_i) = 1$. 

### **Transition Matrix $A$**

Since the hidden sequence is a Markov chain, each random variable in the sequence depends only on the previous variable (except for the first variable). In our example, this means that the chance of it being humid now depends only on whether or not it was humid in the previous time step. Let us suppose that the chance of it being humid now when it was already humid in the previous time step is 70%, and the chance of it being humid now when it was dry is 40%. \
Then we have $P(\textnormal{Dry}_\textnormal{current}|\textnormal{Dry}_\textnormal{previous}) = 0.6$, $P(\textnormal{Humid}_\textnormal{current}|\textnormal{Dry}_\textnormal{previous}) = 0.4$, $P(\textnormal{Dry}_\textnormal{current}|\textnormal{Humid}_\textnormal{previous}) = 0.3$, and $P(\textnormal{Humid}_\textnormal{current}|\textnormal{Humid}_\textnormal{previous}) = 0.7$. \
These conditional probabilities form another matrix $A = \begin{pmatrix}
0.6 & 0.4 \\
0.3 & 0.7
\end{pmatrix}$ known as the **transition matrix**, which describes how likely a hidden state is to occur given the previous hidden state: 
|   |$\textnormal{Dry}_\textnormal{current}$|$\textnormal{Humid}_\textnormal{current}$|
|---|---|---|
|$\textnormal{Dry}_\textnormal{previous}$|$0.6$|$0.4$|
|$\textnormal{Humid}_\textnormal{previous}$|$0.3$|$0.7$|

Formally, $A_{ij} = P({q_j}_t|{q_i}_{t-1})$ is a $N \times N$ matrix, where $q_i$ and $q_j$ are the $i^{th}$ and $j^{th}$ hidden states respectively, and the subscript $t$ denotes the index of the sequence, in this case the time. \
We also have $\sum _{j=1} ^N A_{ij} = \sum _{j=1} ^N P({q_j}_t|{q_i}_{t-1}) = 1$, where $N$ is the number of possible hidden states. 

### **Initial Probability Distribution $\pi$**

Finally, since the hidden sequence must start with some initial variable, we also need the probabililities of the hidden sequence starting in each hidden state. In our example, these will be the chance that it starts out humid and the chance that it starts out dry, say 0.5 for both. These probabilities form the **initial probability distribution** $\pi = (0.5, 0.5)$, which describes the chances of the hidden sequence starting in each hidden state. 

Formally, $\pi = (\pi_1, \pi_2, ... , \pi_N)$, where $\pi_i = P({q_i}_1)$ is the probability that the first entry in the hidden sequence is the $i^{th}$ hidden state. 

### **Hidden Sequence $X$**

Now that we have defined all the probabilities required in the model, we can generate the sequences. We shall start with the hidden sequence. The first entry in the hidden sequence is obtained by randomly choosing a hidden state following the initial probability distribution $\pi$. In our example, this means that the initial humidity level is randomly decided, with a 50% chance of being either high or low (since $\pi=(0.5, 0.5)$). \
Suppose that it starts out being dry. Then the first element in our hidden sequence $x_1 = 0$ since $q_0  = \textnormal{Dry}$. \
The second element in the hidden sequence is a random variable following the probability distribution represented in the $i^{th}$ row of the transition matrix $A$, where $q_i = x_1$ is the hidden state of the first element. In our example, we have $P(\textnormal{Dry}_\textnormal{current}|\textnormal{Dry}_\textnormal{previous}) = 0.6$, $P(\textnormal{Humid}_\textnormal{current}|\textnormal{Dry}_\textnormal{previous}) = 0.4$, so there is a slightly higher chance that the second element in the hidden sequence will still be $0$. For each time step $t$, we can repeat this probability sampling for $x_t$ based on the transition probabilities associated with its previous hidden state $x_{t-1}$. \
The hidden sequence $X = (x_1, x_2, ... , x_T)$, where $T$ is the length of the sequence or the total number of time steps, is thus obtained from this repeated process of random sampling.

### **Observable Sequence $O$**

The generation of the observable sequence is very similar to that of the hidden sequence, just that we now randomly sample using the probabilities found in the emission matrix $B$ instead of the transition matrix $A$. \
The first element in the observable sequence $o_1$ is a random variable following the probability distribution represented in the $i^{th}$ row of the emission matrix $B$, where $q_i = x_1$ is the hidden state of the first element. \
In our example, the probability of the weather starting out sunny is 80%, since the initial humidity level is low. \
The subsequent elements in the observable sequence are obtained in the same fashion by considering the conditional probabilities of generating each observable state given the current hidden state.  


### 
