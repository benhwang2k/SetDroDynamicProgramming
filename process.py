import numpy as np
from ambiguity import AmbiguityBall
from basis import FranklinBasis
import time
import threading


class Mdp:
    """
    This class defines a robust Markov Decision Process
    with a set of cost functions and an ambiguity set of
    transition kernels.

    """
    def __init__(self, S=[0,1], U=[0,1], G=None, discount=0.7, A:AmbiguityBall = None, tol=1e-6):
        """
        :param S: State space of the process
        :param U: control space of the process
        :param G: A set of cost functions
        :param gamma: Discount factor
        :param P: A set of transition kernels
        """
        self.S = S
        self.U = U
        self.G = G
        self.discount = discount
        self.A = A
        self.tol = tol

    def eval(self, x, u, g, J: np.ndarray):
        # calculate g(x,u) + sup_{p \in A} ( integral_{S} J(y)p(y) dy )
        if self.A is None:
            raise(AttributeError(f"None ambiguity set"))
        exp, coef = self.A[(x, u)].eval_dro_expectation(J)
        return g(x, u) + self.discount * exp


    def Belmman_iteration(self, g, J, new_values, index):
        new_value = np.array([])
        for i in range(len(self.S)):
            # Calculate min_{u \in U} g(x,
            x = self.S[i]
            minimizer = self.U[0]
            minimal = self.eval(x, minimizer, g, J)
            for u in self.U:
                tic = time.time()
                v = self.eval(x, u, g, J)
                if v < minimal:
                    minimizer = u
                    minimal = v
            # print(f"minimizer = {minimizer} \n minimal = {minimal} \n ----------")
            new_value = np.append(new_value, minimal)
        # new_value is a list of constant values
        # translate this into a franklin function
        add_val = True
        for val in new_values:
            add_val = add_val and not (np.linalg.norm(val - new_value) < self.tol)
        if add_val:
            new_values[index] = new_value

    def BellmanMT(self, values: np.ndarray):
        """
        :param Values: A set of value functions (specified as np.arrays) that
            are piecewise constant on the segments of the approximation.
        :return: A set of value functions, and corresponding policies for one
            iteration of the Bellman operator.
        """
        if self.S is None or self.U is None or self.G is None or self.A is None:
            raise(AttributeError(f"The problem has a none parameter."))
        # compute the policy and the value function
        new_values = [np.array([])] * (len(values) * len(self.G))
        count = 0
        elapsed = 0
        tic = time.time()
        index = 0
        threads = []
        for J in values:
            for g in self.G:
                # multithread this
                t = threading.Thread(target=self.Belmman_iteration, args=(g,J,new_values,index))
                threads.append(t)
                t.start()
                index += 1
                print(f"starting {index} out of {len(values)*len(self.G)}", end="\r")
        count = 1
        for t in threads:
            t.join()
            print(f"ended {count} out of {len(values) * len(self.G)}", end="\r")
            count += 1
        return new_values
    def Bellman(self, values: np.ndarray):
        """
        :param Values: A set of value functions (specified as np.arrays) that
            are piecewise constant on the segments of the approximation.
        :return: A set of value functions, and corresponding policies for one
            iteration of the Bellman operator.
        """
        if self.S is None or self.U is None or self.G is None or self.A is None:
            raise(AttributeError(f"The problem has a none parameter."))
        # compute the policy and the value function
        new_values = [np.array([])] * (len(values) * len(self.G))
        count = 0
        elapsed = 0
        tic = time.time()
        for J in values:
            for g in self.G:
                # multithread this
                new_value = np.array([])
                for i in range(len(self.S)):
                    # Calculate min_{u \in U} g(x,
                    x = self.S[i]
                    minimizer = self.U[0]
                    minimal = self.eval(x, minimizer, g, J)
                    for u in self.U:
                        elapsed = (elapsed * count / (count + 1)) + ((time.time() - tic)/(count + 1))
                        print(f"Bellman: {round((count / (len(values) * len(self.G)* len(self.U)* len(self.S))) * 100, 4)}% complete, estimated time remaining = {(len(values) * len(self.G)* len(self.U)* len(self.S) - count)*elapsed}", end="\r")
                        count += 1
                        tic = time.time()
                        v = self.eval(x, u, g, J)
                        if v < minimal:
                            minimizer = u
                            minimal = v
                    # print(f"minimizer = {minimizer} \n minimal = {minimal} \n ----------")
                    new_value = np.append(new_value, minimal)
                # new_value is a list of constant values
                # translate this into a franklin function
                add_val = True
                for val in new_values:
                    add_val = add_val and not(np.linalg.norm(val - new_value) < self.tol)
                if add_val:
                    new_values.append(new_value)
        return new_values

    def mcts(self):
        # create tree object
        pass


class Tree:
    def __init__(self, baseMDP: Mdp, value, state_dist, cost, action, parent):

        # this gives the state and action space as well as the transition ambiguity balls and costs
        self.baseMDP = baseMDP

        # value of this node
        self.value = value

        # The cost that was incurred by getting to this node
        self.cost = cost

        # The action take to get to this  node
        self.action = action

        # track the number of visits
        self.num_visits = 0

        self.parent = parent
        self.children = []

    def select(self):
        # select which branch to explore
        pass

    def expand(self, cost, action, v):
        pass
        # # if this node has node has no children, then we will add one
        # for child in self.children:
        #     if child.cost == cost and child.action == action:
        #         return child
        # _, worst_case_coef = self.baseMDP.A.eval_dro_expectation(v)
        # transition = self.baseMDP.A.basis.lin_comb(worst_case_coef)
        # next_dist = np.zeros(v.shape)
        # for state in self.baseMDP.S
        #
        # new_child = Tree(self.baseMDP, 0, child_distribution)

    def simulate(self):
        # go down the tree choosing random costs and actions
        pass

    def backup(self):
        # repropogate
        pass

    def visit(self, value_estimate: np.ndarray, prev_state_distribution: np.ndarray):
        """
        apply one iteration of the MCTS algorithm to the current node

        :param value_estimate:
        :param prev_state_distribution:
        :return: The new value of this node
        """

        # apply the process to this node
        _, worst_coeff = self.baseMDP.A.eval_dro_expectation(value_estimate)
        worst_P = self.baseMDP.A.basis.lin_comb(worst_coeff)
        state_distribution = worst_P.convolve(prev_state_distribution)

        pass


