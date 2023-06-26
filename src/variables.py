from typing import List
import numpy as np
from itertools import product
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from functools import reduce

class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs  = probabilities.values()

        assert len(states) == len(probs), \
            "The probabilities must match the states."
        assert len(states) == len(set(states)), \
            "The states must be unique."
        assert abs(sum(probs) - 1.0) < 1e-12, \
            "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."

        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x:
            probabilities[x], self.states))).reshape(1, -1)

    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, states: list):
        return cls(dict(zip(states, list(array))))

    @property
    def dict(self):
        return {k:v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]
    
class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):

        assert len(prob_vec_dict) > 1, \
            "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."

        self.states      = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values      = np.stack([prob_vec_dict[x].values \
                           for x in self.states]).squeeze()

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) \
             / (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array:
                  np.ndarray,
                  states: list,
                  observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x))) \
                  for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values,
               columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)

class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        """
        Initialize a HiddenMarkovChain object.
        
        Args:
            T: The transmission matrix (A) representing state transition probabilities.
            E: The emission matrix (B) representing observation (emission) probabilities.
            pi: The initial probability vector representing the initial distribution of states.
        """
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables

    def __repr__(self):
        """
        Return a string representation of the HiddenMarkovChain object.
        """
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        """
        Initialize a HiddenMarkovChain object with given states and observables.
        
        Args:
            states: List of states.
            observables: List of observables.
        
        Returns:
            An initialized HiddenMarkovChain object.
        """

        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)

    # def _create_all_chains(self, chain_length):
    #     """
    #     Create all possible state chains of a given length.
        
    #     Args:
    #         chain_length: Length of the state chain.
            
    #     Returns:
    #         List of all possible state chains.
    #     """
    #     return list(product(*(self.states,) * chain_length))


    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)  # Shape: (n_observations, n_states)
        return float(alphas[-1].sum())  # Sum over the last set of alphas to get the final score

    def _alphas(self, observations: list) -> np.ndarray:
        """
        Compute the alpha values (forward probabilities) for a sequence of observations.
        
        Args:
            observations: List of observations.
            
        Returns:
            Numpy array of alpha values.
        """
        alphas = np.zeros((len(observations), len(self.states)))  # Shape: (n_observations, n_states)
        alphas[0, :] = self.pi.values * self.E[observations[0]].T  # Initialization based on initial distribution and emission probabilities
        for t in range(1, len(observations)):
            # Iteratively compute alpha for each state and each observation
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1)  # Shape: (1, n_states)
                         @ self.T.values) * self.E[observations[t]].T  # Shape: (n_states, n_states), then (n_states,), then broadcasted to (1, n_states)
        return alphas  # Shape: (n_observations, n_states)

    def _betas(self, observations: list) -> np.ndarray:
        """
        Compute the beta values (backward probabilities) for a sequence of observations.
        
        Args:
            observations: List of observations.
            
        Returns:
            Numpy array of beta values.
        """
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] \
                        * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
        return betas

    def score(self, observations: list) -> float:
        """
        Compute the probability score of a sequence of observations given the model.
        
        Args:
            observations: List of observations.
            
        Returns:
            Probability score of the observations.
        """
        alphas = self._alphas(observations)  # Shape: (n_observations, n_states)
        return float(alphas[-1].sum())  # Sum over the last set of alphas to get the final score

    def uncover(self, observations: list) -> list:
        """
        Find the most likely sequence of hidden states given a sequence of observations.
        
        Args:
            observations: List of observations.
            
        Returns:
            List of the most likely sequence of hidden states.
        """
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))

    def run(self, length: int) -> (list, list):
        """
        Generate a sequence of observations and the corresponding sequence of hidden states.
        
        Args:
            length: Length of the generated sequence.
            
        Returns:
            Tuple containing the generated sequence of observations and hidden states.
        """
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)

        prb = self.pi.values
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())

        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())

        return o_history, s_history
    
    def _digammas(self, observations: list) -> np.ndarray:
        """
        Compute the digammas (pairwise state probabilities) for a sequence of observations.
        
        Args:
            observations: List of observations.
            
        Returns:
            Numpy array of digammas.
        """
        L, N = len(observations), len(self.states)
        digammas = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2 / score
        return digammas