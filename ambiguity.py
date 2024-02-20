import numpy as np
import cvxpy as cp
import gurobipy
from basis import FranklinBasis
from func import Func
import math
"""
This file contains the classes required to create ambiguity sets
for pdfs of empirical distributions created from data.
"""

class AmbiguityBall:
    """
    Ambiguity Ball is a set of vectors of coefficients.
    Each vector of coefficients alpha defines a function as
        f(x) = sum(alpha_i * basis_i)
    where the basis functions are Franklin funcitons of the
    space.

    Formally, the ambiguity ball will be  A := {gamma s.t. ||gamma - alpha||_2 < R}
    for a nominal (center) alpha, and radius R.
    """
    def __init__(self, basis: FranklinBasis, data, M, delta):
        """
        inputs:
            data: data set from which the center of the ball will be calculated
            M: resolution of the grid = 2 ** M
            delta: confidence parameters s.t. we are 1-delta confident that the true
                    is inside the ball
        """
        self.basis = basis  #FranklinBasis(len(data[0]), M)
        max_2 = sum([v**2 for v in self.basis.max_values])
        self.r = ((64*math.log(1/delta)*(max_2**2))/(2*len(data)))**0.5 + (1.0/len(data))*max_2
        self.center_f, self.center_c = self.basis.approximate_emp(data)
        # set up the cvx dro problem with parameters J
        n = len(self.center_c)
        self.alpha = cp.Variable(n)
        self.cent_alpha = cp.Parameter(n)
        self.cent_alpha.value = self.center_c

        self.f_integrals = cp.Parameter(n)
        self.f_integrals.value = self.basis.get_integrals()

        self.j_integrals = cp.Parameter(n)
        self.j_integrals.value = self.basis.get_integrals()

        self.constraints = [cp.sum_squares(self.alpha - self.cent_alpha) <= self.r]
        self.constraints += [self.alpha @ self.f_integrals - 1 == 0]
        self.objective = self.alpha @ self.j_integrals

        self.prob = cp.Problem(cp.Maximize(self.objective), self.constraints)



    def eval_dro_expectation(self, J: np.ndarray):
        """
        :param J: J is a Franklin approximation of a value function
        :return: supremum over ambiguity set of expectation of J
        """
        jparam = []
        for base in self.basis.basis:
            jparam.append(base.inner_prod(J))
        self.j_integrals.value = np.array(jparam)
        self.prob.solve(solver=cp.GUROBI)
        return self.prob.value, self.alpha.value



    def __str__(self):
        return f"center coefficients = {self.center_c} \n radius = {self.r}"




