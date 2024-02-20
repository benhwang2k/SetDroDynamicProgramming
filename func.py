import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad


class Func:

    def __init__(self, dim, segments, f=None):
        # dim is the dimension of the space
        # segments is the number of segements in the space
        self.dim = dim
        self.segments = segments
        if not(dim == len(segments)):
            raise AttributeError(f"dimension of function {dim} not equal to dim of spacing {len(segments)}. ")
        self.mesh = np.zeros([i + 1 for i in segments])
        dx = [1.0 / i for i in self.segments]
        if f:
            for i in np.ndenumerate(self.mesh):
                self.mesh[i[0]] = f(tuple([dx[j[0]]*j[1] for j in enumerate(i[0])]))

    def __add__(self, other):
        f = Func(self.dim, self.segments)
        f.mesh = self.mesh + other.mesh
        return f

    def __sub__(self, other):
        f = Func(self.dim, self.segments)
        f.mesh = self.mesh - other.mesh
        return f

    def __mul__(self, other: float):
        f = Func(self.dim, self.segments)
        f.mesh = self.mesh * other
        return f

    def convolve(self, state_dist: np.ndarray):
        assert(state_dist.shape == self.dim)
        next_dist = np.zeros(state_dist.shape)
        for index, value in np.ndenumerate(state_dist):
            init = np.zeros(state_dist.shape)
            init[index] = value
            next_dist[index] = self.inner_prod(init)
        return next_dist


    def eval(self, x):
        # eval function at x
        total = 0
        if self.dim == 2:
            coordinate = [int(x[i]*self.segments[i]) for i in range(len(x))]
            if x[0] == 1:
                print(f"1 in 0")
                coordinate[0] = self.segments[0] - 1
            if x[1] == 1:
                print(f"1 in 1")
                coordinate[1] = self.segments[1] - 1
            dx = [(x[i] - coordinate[i]/self.segments[i])/(1.0/self.segments[i]) for i in range(self.dim)]
            c0 = self.mesh[tuple(coordinate)]
            c1 = self.mesh[tuple([coordinate[i] + 1 if i == 0 else coordinate[i] for i in range(self.dim)])]
            c2 = self.mesh[tuple([coordinate[i] + 1 if i == 1 else coordinate[i] for i in range(self.dim)])]
            c3 = self.mesh[tuple([c + 1 for c in coordinate])]
            total = c0 + dx[0]*(c1-c0) + dx[1]*(c2-c0) + dx[0]*dx[1]*(c0 + c3 - c1 -c2)
        elif self.dim == 1:
            coordinate = int(x[0] * self.segments[0])
            dx = (x[0] - coordinate / self.segments[0]) / (1.0 / self.segments[0])
            c0 = self.mesh[coordinate]
            c1 = self.mesh[coordinate + 1]
            total = c0 + dx * (c1 - c0)
        else:
            raise(AttributeError("dim not equal 1 or 2"))
        return total

    def change_segments(self, newsegs):
        return Func(self.dim, newsegs, self.eval)

    def integral(self):
        one = Func(self.dim, self.segments, lambda x: 1)
        return self.inner_prod(one)

    def inner_prod(self, other):
        if type(other) == np.ndarray:
            if not(len(other) == self.segments[0]):
                raise(AttributeError(f"Incorrect dimensions in inner product: segments = {self.segments}, len ofother = {len(other)}"))
            dx = 1.0 / self.segments[0]
            total = 0.0
            if self.dim == 1:
                for i in range(self.segments[0]):
                    m1 = (self.mesh[i + 1] - self.mesh[i]) / dx
                    m2 = 0
                    a = self.mesh[i]
                    c = 1
                    total += other[i] * (a * c * dx + 0.5 * (a * m2 + c * m1) * (dx ** 2) + (1 / 3.) * (m1 * m2) * (dx ** 3))
            elif self.dim == 2:
                for cell in np.ndenumerate(self.mesh):
                    on_edge = False
                    for ind in enumerate(cell[0]):
                        if ind[1] == self.segments[ind[0]]:
                            on_edge = True

                            break
                    if not on_edge:
                        # index of fist corner
                        corner = list(cell[0])
                        # define all constants
                        c0 = self.mesh[tuple(corner)]
                        k0 = 1
                        c1 = self.mesh[tuple([sum(i) for i in zip(corner, [0, 1])])]
                        k1 = 1
                        c2 = self.mesh[tuple([sum(i) for i in zip(corner, [1, 0])])]
                        k2 = 1
                        c3 = self.mesh[tuple([sum(i) for i in zip(corner, [1, 1])])]
                        k3 = 1

                        pre = 0
                        pre += c0 * k0
                        pre += 0.5 * (k0 * c1 + k0 * c2 + c0 * k1 + c0 * k2 - 4.0 * c0 * k0)
                        pre += (1.0 / 3.0) * (2 * c0 * k0 + c1 * k1 + c2 * k2 - c1 * k0 - c0 * k1 - c0 * k2 - c2 * k0)
                        pre += (1.0 / 4.0) * (
                                    4 * c0 * k0 + c1 * k2 + c2 * k1 + c0 * k3 + c3 * k0 - 2 * c1 * k0 - 2 * c2 * k0 - 2 * c0 * k1 - 2 * c0 * k2)
                        pre += (1.0 / (6.0)) * (
                                    (c1 + c2 - 2 * c0) * (k0 + k3 - k1 - k2) + (k1 + k2 - 2 * k0) * (c0 + c3 - c1 - c2))
                        pre += (1.0 / (9.0)) * ((c0 + c3 - c1 - c2) * (k0 + k3 - k1 - k2))
                        pre *= dx * dx
                        # print(f"pre is {pre} before multiplying by {other[tuple(corner)]}")
                        total += pre * other[tuple(corner)]
            return total

        # only do for 2D, lots of calculations
        if not(self.segments == other.segments):
            raise AttributeError("grids must align")
        if self.dim > 2:
            raise AttributeError("more than 2D not implemented")
        if len(self.segments) > 1 and not(self.segments[0] == self.segments[1]):
            raise AttributeError("resolution should be the same in all dimensions")
        dx = 1.0/self.segments[0]
        total = 0.0
        if self.dim == 1:
            for i in range(self.segments[0]):
                m1 = (self.mesh[i + 1] - self.mesh[i]) / dx
                m2 = (other.mesh[i + 1] - other.mesh[i]) / dx
                a = self.mesh[i]
                c = other.mesh[i]
                total += a * c * dx + 0.5 * (a * m2 + c * m1) * (dx ** 2) + (1 / 3.) * (m1 * m2) * (dx ** 3)
        elif self.dim == 2:
            for cell in np.ndenumerate(self.mesh):
                on_edge = False
                for ind in enumerate(cell[0]):
                    if ind[1] == self.segments[ind[0]]:
                        on_edge = True

                        break
                if not on_edge:
                    # index of fist corner
                    corner = list(cell[0])
                    # define all constants
                    c0 = self.mesh[tuple(corner)]
                    k0 = other.mesh[tuple(corner)]
                    c1 = self.mesh[tuple([sum(i) for i in zip(corner, [0, 1])])]
                    k1 = other.mesh[tuple([sum(i) for i in zip(corner, [0, 1])])]
                    c2 = self.mesh[tuple([sum(i) for i in zip(corner, [1, 0])])]
                    k2 = other.mesh[tuple([sum(i) for i in zip(corner, [1, 0])])]
                    c3 = self.mesh[tuple([sum(i) for i in zip(corner, [1, 1])])]
                    k3 = other.mesh[tuple([sum(i) for i in zip(corner, [1, 1])])]


                    pre = 0
                    pre += c0*k0
                    pre += 0.5*(k0*c1 + k0*c2 + c0*k1 + c0*k2 - 4.0*c0*k0)
                    pre += (1.0/3.0)*(2*c0*k0 + c1*k1 + c2*k2 -c1*k0 - c0*k1 - c0*k2 - c2*k0)
                    pre += (1.0/4.0)*(4*c0*k0 + c1*k2 + c2*k1 + c0*k3 + c3*k0 - 2*c1*k0 - 2*c2*k0 - 2*c0*k1 - 2*c0*k2)
                    pre += (1.0/(6.0))*((c1 + c2 -2*c0)*(k0 + k3 - k1 - k2) + (k1 + k2 - 2*k0)*(c0 + c3 - c1 - c2))
                    pre += (1.0/(9.0))*((c0+c3-c1-c2)*(k0+k3-k1-k2))
                    pre *= dx*dx
                    total += pre
        return total

    def plot(self, newplot=True):
        if newplot:
            plt.figure()

        if self.dim == 1:
            x = np.linspace(0, 1, self.segments[0] + 1)
            plt.plot(x, self.mesh, "-")
        elif self.dim == 2:
            x = np.linspace(0, 1, self.segments[0] + 1)
            y = np.linspace(0, 1, self.segments[1] + 1)
            x, y = np.meshgrid(x, y)
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot_surface(x, y, self.mesh)
        else:
            raise AttributeError(f"plotting only available for 1D and 2D mesh. Your dimension = {self.dim}")














