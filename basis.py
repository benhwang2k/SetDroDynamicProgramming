from func import Func
import math
from scipy.integrate import quad, dblquad
import numpy as np
import threading


class FranklinBasis:

    def __init__(self, dim, M):
        self.M = M
        self.dim = dim
        self.segments = 2**M

        self.basis = []
        self.max_values = []

        n = [-1] * self.dim
        d = [1] * self.dim
        for i in range((self.segments + 1)**self.dim):
            print(f"Forming Basis: {round(i * 100 /(self.segments + 1)**self.dim, 3)}%", end="\r")
            index = [int(i / ((self.segments + 1)**(c))) % (self.segments + 1) for c in range(self.dim)]
            a = [n[c]/d[c] for c in range(len(n))]
            update_index = 0
            while update_index < self.dim - 1 and index[update_index] == self.segments:
                update_index += 1
                # reset the previous counters
                for reset_index in range(update_index):
                    n[reset_index] = -1
                    d[reset_index] = 1
            if n[update_index] == -1:
                n[update_index] = 0
            elif n[update_index] + 1 == d[update_index]:
                n[update_index] = 1
                d[update_index] = d[update_index] * 2
            else:
                n[update_index] += 2


            def v(a, x):
                tot = 1
                for c in range(self.dim):
                    if a[c] == -1:
                        tot *= 1
                    else:
                        tot *= 0 if x[c] < a[c] else (x[c] - a[c])
                return tot
            Fnm = Func(self.dim, (self.segments,) * self.dim, lambda x: v(a, x))
            for e in self.basis:
                # project the new function on to the old and subtract out
                # print(f'inner prod at i = {i} = {Fnm.inner_prod(e)}')
                Fnm = Fnm - (e*(Fnm.inner_prod(e)))
                if abs(Fnm.inner_prod(e)) > 1e-14:
                    print(f" inner after sub: {Fnm.inner_prod(e)}")
                if abs(e.inner_prod(Fnm)) > 1e-14:
                    print(f" inner after sub (reversed) : {e.inner_prod(Fnm)}")
                # print(f"reversed : {e.inner_prod(Fnm)}")
                # normalize
            # print(f"normalization factor = {1.0/math.sqrt(Fnm.inner_prod(Fnm))}")
            Fnm = Fnm*(1.0/math.sqrt(Fnm.inner_prod(Fnm)))
            self.basis.append(Fnm)
            self.max_values.append(np.amax(Fnm.mesh))

    def lin_comb(self, coef):
        F = Func(self.dim, (self.segments,) * self.dim)
        for i in range(len(self.basis)):
            F += self.basis[i]*coef[i]
        return F

    def get_approx_coeff(self, base, f, coefficients, index):
        if self.dim == 2:
            alpha, err = dblquad(lambda x, y: base.eval((x, y)) * f(x, y), 0, 1, 0, 1)
        elif self.dim == 1:
            alpha, err = quad(lambda x: base.eval((x,)) * f(x), 0, 1)

        coefficients[index] = alpha

    def approximate_funcMT(self, f):
        f_approx = Func(self.dim, (self.segments,)*self.dim, lambda x: 0)
        max = len(self.basis)
        coefficients = [0.0] * max
        # can multithread here
        threads = []
        index = 0
        for base in self.basis:
            print(f"Computing Approximation: {round((index/max)*100, 3)}%", end="\r")
            t = threading.Thread(target=self.get_approx_coeff, args=(base, f, coefficients, index))
            t.start()
            threads.append(t)
            index += 1
        index = 0
        for base in self.basis:
            threads[index].join()
            f_approx = f_approx + (base*coefficients[index])
            index += 1
        return f_approx, np.array([c for c in coefficients])


    def approximate_func(self, f):
        f_approx = Func(self.dim, (self.segments,)*self.dim, lambda x: 0)
        count = 0
        max = len(self.basis)
        coefficients = [0.0] * max
        # can multithread here
        for base in self.basis:
            print(f"Computing Approximation: {round((count/max)*100, 3)}%", end="\r")
            count += 1
            if self.dim == 2:
                alpha, err = dblquad(lambda x, y: base.eval((x,y)) * f(x, y), 0,1,0,1)
            elif self.dim == 1:
                alpha, err = quad(lambda x: base.eval((x,)) * f(x), 0,1)

            coefficients.append(alpha)
            f_approx = f_approx + (base*alpha)
        return f_approx, np.array([c for c in coefficients])

    def get_integrals(self):
        integrals = []
        for base in self.basis:
            integrals.append(base.integral())
        return np.array(integrals)

    def approximate_emp(self, data):
        D = len(data)
        coefficients = []
        f_approx = Func(self.dim, (self.segments,) * self.dim, lambda x: 0)
        # data is the list of data points collected from empirical measurement
        count = 0
        max = len(self.basis)
        for base in self.basis:
            print(f"Computing Empirical Approximation: {round((count / max) * 100, 3)}%", end="\r")
            count += 1
            alpha = 0
            for xi in data:
                alpha += base.eval(xi) / D
            coefficients.append(alpha)
            f_approx = f_approx + (base * alpha)
        if self.dim == 1:
            normalization = f_approx.integral()
        elif self.dim == 2:
            normalization = f_approx.integral()
        else:
            raise(AttributeError(f"empirical approximation only supported for dim 1, 2. Your dim = {self.dim}"))

        f_approx = f_approx * (1.0 / normalization)
        return f_approx, np.array([c/normalization for c in coefficients])











