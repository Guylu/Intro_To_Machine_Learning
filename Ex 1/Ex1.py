import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr
import math

NUM_TOSSES = 1000
NUM_SEQ = 100000
P = 0.25
NUM_BOUNDS = 5
MAX_PROB = 1
Z_AXIS_PARAM = 2
LOWER_BOUND_Q15 = -0.4
UPPER_BOUND_Q15 = 0.1

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    """

    :param dim:
    :return:
    """
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    """
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    """
    plot points in 2D
    :param x_y the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def partial_averages(iterable):
    """
    creates partial averages of an iterable object
    each call returns the next partial average of the sequence
    :param iterable: iterable item
    :return:partial average so far
    """
    total = 0
    num = 0
    for i in iterable:
        total += i
        num += 1
        yield total / num


def qualified_percentage(my_averages, e, p):
    """
    plots percentage of sequences that meet the condition with the given bound
    :param my_averages: array of already averaged out data set
    as to not calculate it each time - its constant independently of e,p
    :param e: epsilon
    :param p: probability - 0.25 in this exercise
    :return:nothing
    """
    res = np.empty(NUM_TOSSES)
    for i in range(NUM_TOSSES):
        # summing in each cell the percentage of element that meet the criteria:
        res[i] = np.count_nonzero(np.abs(my_averages[:, i] - p) >= e) / NUM_SEQ
    plt.plot(res, label="Percentage of sequences that satisfy condition")


# q11:
plot_3d(x_y_z)
plt.title("Q11 random dots, from normal distribution")
plt.show()

# q12:
s = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2.0]])
scaling_trans_analytical = s.dot(cov.dot(s.T))
x_y_z_scaled = s.dot(x_y_z)
plot_3d(x_y_z_scaled)
plt.title("Q12 transformed dots by matrix S")
plt.show()

scaling_trans_numerical = np.cov(x_y_z_scaled)
print("S analytical:")
print(scaling_trans_analytical)
print("S numerical:")
print(scaling_trans_numerical)
print()

# q13:
rand_ortho = get_orthogonal_matrix(3)
rotated_trans_analytical = rand_ortho.dot(
    scaling_trans_analytical.dot(rand_ortho.T))
x_y_z_rotated = rand_ortho.dot(x_y_z_scaled)
plot_3d(x_y_z_rotated)
plt.title("Q13 rotated dots with random orthogonal matrix")
plt.show()

rotated_trans_numerical = np.cov(x_y_z_rotated)
print("analytical:")
print(rotated_trans_analytical)
print("numerical:")
print(rotated_trans_numerical)

# q14:
# making a projection onto XY plane using projection matrix
proj = np.array([[1, 0, 0], [0, 1, 0]])
proj_trans = proj.dot(rotated_trans_analytical)
plot_2d(proj_trans.dot(x_y_z_rotated))
plt.title("Q14 projection of previous dots onto XY plane")
plt.show()

# q15:
filtered_z_values = np.array(
    [column for column in rotated_trans_analytical.dot(x_y_z).T if
     LOWER_BOUND_Q15 < column[Z_AXIS_PARAM] < UPPER_BOUND_Q15])
plot_2d(proj_trans.dot(filtered_z_values.T))
plt.title("Q15 projection of filtered dots with -0.4<Z<0.1 dots onto XY plane")
plt.show()

# q16a
data = np.random.binomial(1, P, (NUM_SEQ, NUM_TOSSES))
epsilon = np.array([0.5, 0.25, 0.1, 0.01, 0.001])
plt.title("Q16a average of coin tosses convergence")
plt.xlabel("Number of coin tosses")
plt.ylabel("Average of sequence")
# creating partial averages array for each epsilon:
averages = [list(partial_averages(data[i])) for i in range(5)]
for i in range(NUM_BOUNDS):
    plt.plot(list(range(NUM_TOSSES)), averages[i],
             label="Sequence number: " + str(i))
    plt.legend()
plt.show()

# q16b+c
# bounds:
chevi_bounds = [[min(1 / (4 * j * epsilon[i] * epsilon[i]), MAX_PROB)
                 for j in range(1, NUM_TOSSES)] for i in range(NUM_BOUNDS)]
hoff_bounds = [[min(2 * math.exp((-2 * j * epsilon[i] * epsilon[i])), MAX_PROB)
                for j in range(1, NUM_TOSSES)] for i in range(NUM_BOUNDS)]

for i in range(NUM_BOUNDS):
    plt.plot(chevi_bounds[i], label="Using Chebyshev's inequality bound")
    plt.plot(hoff_bounds[i], label="Using Hoeffding's inequality bound")
    plt.title("Q16b probability to deviate from expectancy with epsilon = " +
              str(epsilon[i]))
    plt.xlabel("Number of coin tosses")
    plt.ylabel("Probability")
    # creating averages that will be used for each epsilon
    # no need to calculate it each time - once before the func is enough :)
    average = np.cumsum(data, axis=1) / np.array(range(1, 1001))
    # func to calculate the percentages for q 16c
    qualified_percentage(average, epsilon[i], P)
    plt.legend()
    plt.show()
