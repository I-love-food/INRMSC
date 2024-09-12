import numpy as np

def fun1(points):
    x = points[:, 0]
    y = points[:, 1]
    return np.sin(x) + np.cos(y)

def grad_fun1(points):
    x = points[:, 0]
    y = points[:, 1]
    dfdx = np.cos(x).reshape(-1, 1)
    dfdy = -np.sin(y).reshape(-1, 1)
    return np.concatenate((dfdx, dfdy), axis=1)

def ackley(points, a=20, b=0.2, c=2 * np.pi):
    x = points[:, 0]
    y = points[:, 1]
    exp1 = np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    exp2 = np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return -a * exp1 - exp2 + a + np.exp(1)

def grad_ackley(points, a=20, b=0.2, c=2 * np.pi):
    x = points[:, 0]
    y = points[:, 1]
    dfdx = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2))) * (-b * x / np.sqrt(2 * (x**2 + y**2))) \
        + np.exp(0.5 * (np.cos(c * x) + np.cos(c * y))) * 0.5 * c * np.sin(c * x)
    dfdy = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2))) * (-b * y / np.sqrt(2 * (x**2 + y**2))) \
        + np.exp(0.5 * (np.cos(c * x) + np.cos(c * y))) * 0.5 * c * np.sin(c * y)
    dfdx = dfdx.reshape(-1, 1)
    dfdy = dfdy.reshape(-1, 1)
    return np.concatenate((dfdx, dfdy), axis=1)

def test_fun(points):
    points = points.reshape(-1, 2)
    return ackley(points)

def grad_test_fun(points):
    points = points.reshape(-1, 2)
    return grad_ackley(points)





