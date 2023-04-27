import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

y = cp.Variable(1)
x = cp.Parameter(1)
x_2 = cp.power(x, 2)
constraints = [x + y >= 0, y >= cp.power(x, 2)]
objective = cp.Minimize(y)
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[x], variables=[y])
# x_num = torch.tensor([-1.5], requires_grad=True)
# # solve the problem
# solution, = cvxpylayer(x)
# solution.sum().backward()
# compute the gradient of the sum of the solution with respect to A, b


x = torch.tensor([-1.0001], requires_grad=True)
solution, = cvxpylayer(x)
solution.sum().backward()
print(f"x.grad at [-1.0001]: {x.grad}")

x = torch.tensor([-1.0], requires_grad=True)
solution, = cvxpylayer(x)
solution.sum().backward()
print(f"x.grad at [-1.0]: {x.grad}")

x = torch.tensor([-0.9999], requires_grad=True)
solution, = cvxpylayer(x)
solution.sum().backward()
print(f"x.grad at [-0.9999]: {x.grad}")

x = torch.tensor([-0.5], requires_grad=True)
solution, = cvxpylayer(x)
solution.sum().backward()
print(f"x.grad at [-0.5]: {x.grad}")

x = torch.tensor([0.0], requires_grad=True)
solution, = cvxpylayer(x)
solution.sum().backward()
print(f"x.grad at [0.0]: {x.grad}")

x = torch.tensor([0.0001], requires_grad=True)
solution, = cvxpylayer(x)
solution.sum().backward()
print(f"x.grad at [0.0001]: {x.grad}")

x = torch.tensor([-0.0001], requires_grad=True)
solution, = cvxpylayer(x)
solution.sum().backward()
print(f"x.grad at [-0.0001]: {x.grad}")

# print(f"x_num.grad: {x_num.grad}")