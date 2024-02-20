import ambiguity
from func import Func
from basis import FranklinBasis
from ambiguity import AmbiguityBall
from process import Mdp
from scipy.integrate import quad, dblquad
import math
import matplotlib.pyplot as plt
import time
import random
from functools import partial
import numpy as np

def data_barchart(data, segments):
    numInSeg = [0] * segments
    index = 0
    for seg in range(segments):
        for d in data:
            if (seg/segments) <= d < ((seg + 1)/segments):
                numInSeg[index] += 1.0/(len(data)*(1.0/segments))
        index += 1
    return [((2*seg + 1)/(2*segments)) for seg in range(segments)], numInSeg

def data_barchart_2d(data, segments):
    numInSeg = [[0] * segments for i in range(segments)]
    for i, l in enumerate(numInSeg):
        for j, v in enumerate(l):
            for (d1, d2) in data:
                if (i/segments) <= d1 <= ((i+1)/segments) and (j/segments) <= d2 <= ((j+1)/segments):
                    numInSeg[i][j] += 1.0/(len(data)*(1.0/segments)*(1.0/segments))
    x = np.linspace(0, 1, segments)
    y = np.linspace(0, 1, segments)
    x, y = np.meshgrid(x, y)
    return x, y, np.array(numInSeg)



# data = [random.random() for i in range(1000)]
# x_bar, y_bar = data_barchart(data, 2**5)
# plt.bar(x_bar, y_bar, width=0.4*(1.0/(2**4)))

# data = [(random.random(), random.random()) for i in range(1000)]
# print(f'data = {data}')
# x,y,z = data_barchart_2d(data, 2**3)
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot_surface(x, y, z)

# # 1 D approx
# basis = FranklinBasis(1,4)
#
# pdf_approx, coeff = basis.approximate_emp(data)
# pdf_approx.plot(newplot=False)




# # 2 D approx
# basis = FranklinBasis(2,3)
#
# pdf_approx, coeff = basis.approximate_emp(data)
# pdf_approx.plot(newplot=False)



# basis = FranklinBasis(1, 3)

# F = lambda x, y: math.sin(2*math.pi*x) * math.cos(math.pi*2*y)
# F = lambda x: math.sin(2*math.pi*x)
#
# F_approx, coeff = basis.approximate_func(F)
# F_approx.plot(newplot=False)
# x = np.linspace(0,1,100)
# y = []
# for dx in x:
#     y.append(F(dx))
# plt.figure()
# plt.plot(x, y)
# plt.show()



"""
basis = FranklinBasis(2, 2)
data = [(random.random(), random.random()) for i in range(10000)]
M = 4
delta = 0.1
ball = AmbiguityBall(basis, data, M, delta)

J = np.zeros((4,4))
value, coeffs = ball.eval_dro_expectation(J)
print(f'the optimal value is {value}')
print(f'the Franklin coeffs of the worst case pdf are {coeffs}')


basis = FranklinBasis(1, 1)
S = [(0.25,), (0.75,)]
U = [(0.25,), (0.75,)]
discount = 0.1
g0 = lambda x, u: 100*x[0] + u[0]
g1 = lambda x, u: 100*(1-x[0]) + u[0]
G = [g0, g1]
D = 10000
data = [[]]*4
data[0] = ([(0.25,)] * (int(0.9 * D))) + ([(0.75,)] * (D - int(0.9 * D)))
data[1] = ([(0.75,)] * (int(0.7 * D))) + ([(0.25,)] * (D - int(0.7 * D)))
data[2] = ([(0.25,)] * (int(0.7 * D))) + ([(0.75,)] * (D - int(0.7 * D)))
data[3] = ([(0.75,)] * (int(0.9 * D))) + ([(0.25,)] * (D - int(0.9 * D)))


M = 4
delta = 0.1
balls = {}
balls[((0.25,),(0.25,))] = AmbiguityBall(basis, data[0], M, delta)
balls[((0.25,),(0.75,))] = AmbiguityBall(basis, data[1], M, delta)
balls[((0.75,),(0.25,))] = AmbiguityBall(basis, data[2], M, delta)
balls[((0.75,),(0.75,))] = AmbiguityBall(basis, data[3], M, delta)


proc = Mdp(S, U, G, discount, balls, tol=0.1)
values = [np.zeros(2)]

newValues = proc.Bellman(values=values)
print(f"new Values = {newValues}")
values = newValues
for i in range(8):
    newValues = proc.Bellman(values=values)
    print(f"new Values = {newValues}")
    values = newValues


"""




x = np.genfromtxt("x_train.csv", delimiter=",", usemask=True)
#  map this to [0,1]
def state_map(u):
    """
    maps a control vector to a point in [0,1]
    :param u:
    :return:
    """
    result = 0
    for i in range(5):
        if u[i] > 0.6:
            result += 1.0/(2**(5 - i))
    return result
def reverse_state_map(state):
    """
    converts point in [0,1] back to a row in the data table
    :param state:
    :return:
    """

    row = 31
    for i in range(5):
        if state >= (1/(2**(i + 1))):
            row -= 2**(4-i)
            state -= (1/(2**(i + 1)))
    return row


#create basis
basis = FranklinBasis(1, 5)

#choose confidence parameter s.t. we are (1-delta) confident that the true lies in A
delta = 0.9

# create the data set
M = 5
D = 1000000 # 1000000
f_correct = 0.9
data={}
ball_map = {}
for u in range(2**M):
    data[u] = []
    for d in range(D):
        if d < f_correct*D:
            data[u].append((u / (2 ** M),))
    u_bin = x[reverse_state_map(u / (2 ** M))]
    u_bin = [0 if val < 0.6 else 1 for val in u_bin]

    per = int((D * (1-f_correct)) / 5)
    for i in range(5):
        for d in range(per):
            u_new = u_bin
            u_new[i] = int(not(u_new[i]))
            data[u].append((state_map(u_new),))
    print(f"ball numer {u} out of {2**M}")
    ball = AmbiguityBall(basis, data[u], M, delta)
    for s in range(2**M):
        st = s/(2**M)
        ac = u/(2**M)
        ball_map[((st,), (ac,))] = ball

S = [(i/(2**M),) for i in range(2**M)] #[(0.25,), (0.75,)]
U = [(i/(2**M),) for i in range(2**M)] #[(0.25,), (0.75,)]
discount = 0.5


# cost functions still need work
y = np.genfromtxt('y_train.csv', delimiter=",", usemask=True)


def g(sp, x, u):
    state_row = reverse_state_map(x[0])
    powersetpoint = abs(y[(sp, 27)] - y[(state_row, 27)])
    volt_vio = 0
    for col in range(27):
        volt_vio += abs(1 - y[(state_row, col)])
    return powersetpoint + volt_vio
G = []
for row in range(32):
    G.append(partial(g,row))


proc = Mdp(S, U, G, discount, ball_map, tol=0.01)
values = [np.zeros(32)]

newValues = proc.Bellman(values=values)
values = newValues
value_iter = 1
diff = 2
while diff > 1:
    print(f"value iteration number {value_iter} \n -----------------------")
    newValues = proc.BellmanMT(values=values)
    diff = abs(len(values)-len(newValues))
    if diff == 0:
        for indv in range(len(values)):
            diff += np.linalg.norm(values[indv] - newValues[indv])
    else:
        diff = 2
    print(f"diff = {diff}")
    value_iter += 1
    values = newValues
    for num, value_function in enumerate(values):
        np.savetxt(f"value_function_{num}.csv", fmt='%.18e', delimiter=',', newline="\n")


