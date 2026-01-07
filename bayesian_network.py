""" Bayesian networks """

from probability import BayesNet, enumeration_ask, elimination_ask, rejection_sampling, likelihood_weighting, gibbs_ask
from timeit import timeit, repeat
import pickle
import numpy as np

T, F = True, False

class DataPoint:
    """
    Represents a single datapoint gathered from one lap.
    Attributes are exactly the same as described in the project spec.
    """
    def __init__(self, muchfaster, early, overtake, crash, win):
        self.muchfaster = muchfaster
        self.early = early
        self.overtake = overtake
        self.crash = crash
        self.win = win

def generate_bayesnet():
    """
    Generates a BayesNet object representing the Bayesian network in Part 2
    returns the BayesNet object
    """
    bayes_net = BayesNet()

    data = pickle.load(open("data/bn_data.p", "rb"))
    
    # BEGIN_YOUR_CODE ######################################################
    
    def probability(query, given=None):
        if given is None:
            count = sum(1 for d in data if all(getattr(d, var) == val for var, val in query.items()))
        else:
            count = sum(1 for d in data if all(getattr(d, var) == val for var, val in {**query, **given}.items()))
        total = sum(1 for d in data if all(getattr(d, var) == val for var, val in given.items())) if given else len(data)
        return count / total if total > 0 else 0

    bayes_net.add(('MuchFaster', '', probability({'muchfaster': T})))
    bayes_net.add(('Early', '', probability({'early': T})))

    bayes_net.add(('Overtake', 'MuchFaster Early', {
        (T, T): probability({'overtake': T}, {'muchfaster': T, 'early': T}),
        (T, F): probability({'overtake': T}, {'muchfaster': T, 'early': F}),
        (F, T): probability({'overtake': T}, {'muchfaster': F, 'early': T}),
        (F, F): probability({'overtake': T}, {'muchfaster': F, 'early': F}),
    }))

    bayes_net.add(('Crash', 'MuchFaster Early', {
        (T, T): probability({'crash': T}, {'muchfaster': T, 'early': T}),
        (T, F): probability({'crash': T}, {'muchfaster': T, 'early': F}),
        (F, T): probability({'crash': T}, {'muchfaster': F, 'early': T}),
        (F, F): probability({'crash': T}, {'muchfaster': F, 'early': F}),
    }))

    bayes_net.add(('Win', 'Overtake Crash', {
        (T, T): probability({'win': T}, {'overtake': T, 'crash': T}),
        (T, F): probability({'win': T}, {'overtake': T, 'crash': F}),
        (F, T): probability({'win': T}, {'overtake': F, 'crash': T}),
        (F, F): probability({'win': T}, {'overtake': F, 'crash': F}),
    }))
    # END_YOUR_CODE ########################################################

    return bayes_net


def find_best_overtake_condition(bayes_net):
    """
    Finds the optimal condition for overtaking the car, as described in Part 3.
    Returns the optimal values for (MuchFaster, Early).
    """
    # BEGIN_YOUR_CODE ######################################################

    T, F = True, False

    conditions = [(T, T), (T, F), (F, T), (F, F)]

    probabilities = {}

    for much_faster, early in conditions:
        result = elimination_ask(
            'Win',
            dict(Crash=False, MuchFaster=much_faster, Early=early),
            bayes_net
        )
        probabilities[(much_faster, early)] = result[T]

    optimal_condition = max(probabilities, key=probabilities.get)

    return optimal_condition
    # END_YOUR_CODE ########################################################



def main():
    bayes_net = generate_bayesnet()
    cond = find_best_overtake_condition(bayes_net)
    print("Best overtaking condition: MuchFaster={}, Early={}".format(cond[0],cond[1]))

if __name__ == "__main__":
    main()


Q2.3 Submit the code, show the conditions and math formula
10 Points
Grading comment:
Submit the code for calculating conditions:

 bayesian_network.py
 Download
""" Bayesian networks """

from probability import BayesNet, enumeration_ask, elimination_ask, rejection_sampling, likelihood_weighting, gibbs_ask
from timeit import timeit, repeat
import pickle
import numpy as np

T, F = True, False

class DataPoint:
    """
    Represents a single datapoint gathered from one lap.
    Attributes are exactly the same as described in the project spec.
    """
    def __init__(self, muchfaster, early, overtake, crash, win):
        self.muchfaster = muchfaster
        self.early = early
        self.overtake = overtake
        self.crash = crash
        self.win = win

def generate_bayesnet():
    """
    Generates a BayesNet object representing the Bayesian network in Part 2
    returns the BayesNet object
    """
    bayes_net = BayesNet()

    data = pickle.load(open("data/bn_data.p", "rb"))
    
    # BEGIN_YOUR_CODE ######################################################
    
    def probability(query, given=None):
        if given is None:
            count = sum(1 for d in data if all(getattr(d, var) == val for var, val in query.items()))
        else:
            count = sum(1 for d in data if all(getattr(d, var) == val for var, val in {**query, **given}.items()))
        total = sum(1 for d in data if all(getattr(d, var) == val for var, val in given.items())) if given else len(data)
        return count / total if total > 0 else 0

    bayes_net.add(('MuchFaster', '', probability({'muchfaster': T})))
    bayes_net.add(('Early', '', probability({'early': T})))

    bayes_net.add(('Overtake', 'MuchFaster Early', {
        (T, T): probability({'overtake': T}, {'muchfaster': T, 'early': T}),
        (T, F): probability({'overtake': T}, {'muchfaster': T, 'early': F}),
        (F, T): probability({'overtake': T}, {'muchfaster': F, 'early': T}),
        (F, F): probability({'overtake': T}, {'muchfaster': F, 'early': F}),
    }))

    bayes_net.add(('Crash', 'MuchFaster Early', {
        (T, T): probability({'crash': T}, {'muchfaster': T, 'early': T}),
        (T, F): probability({'crash': T}, {'muchfaster': T, 'early': F}),
        (F, T): probability({'crash': T}, {'muchfaster': F, 'early': T}),
        (F, F): probability({'crash': T}, {'muchfaster': F, 'early': F}),
    }))

    bayes_net.add(('Win', 'Overtake Crash', {
        (T, T): probability({'win': T}, {'overtake': T, 'crash': T}),
        (T, F): probability({'win': T}, {'overtake': T, 'crash': F}),
        (F, T): probability({'win': T}, {'overtake': F, 'crash': T}),
        (F, F): probability({'win': T}, {'overtake': F, 'crash': F}),
    }))
    # END_YOUR_CODE ########################################################

    return bayes_net


def find_best_overtake_condition(bayes_net):
    """
    Finds the optimal condition for overtaking the car, as described in Part 3.
    Returns the optimal values for (MuchFaster, Early).
    """
    # BEGIN_YOUR_CODE ######################################################

    T, F = True, False

    conditions = [(T, T), (T, F), (F, T), (F, F)]

    probabilities = {}

    for much_faster, early in conditions:
        result = elimination_ask(
            'Win',
            dict(Crash=False, MuchFaster=much_faster, Early=early),
            bayes_net
        )
        probabilities[(much_faster, early)] = result[T]

    optimal_condition = max(probabilities, key=probabilities.get)

    return optimal_condition
    # END_YOUR_CODE ########################################################



def main():
    bayes_net = generate_bayesnet()
    cond = find_best_overtake_condition(bayes_net)
    print("Best overtaking condition: MuchFaster={}, Early={}".format(cond[0],cond[1]))

if __name__ == "__main__":
    main()
