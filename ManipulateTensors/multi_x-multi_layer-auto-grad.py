import torch

x = torch.autograd.Variable(torch.tensor([0.0]), requires_grad=False)
y = torch.autograd.Variable(torch.tensor([0.0]), requires_grad=False)

w1 = torch.autograd.Variable(torch.tensor(torch.rand(5, 2)), requires_grad=True)
b1 = torch.autograd.Variable(torch.tensor(torch.rand(5, 2)), requires_grad=True)
w2 = torch.autograd.Variable(torch.tensor(torch.rand(10, 5)), requires_grad=True)
b2 = torch.autograd.Variable(torch.tensor(torch.rand(10, 1)), requires_grad=True)
w3 = torch.autograd.Variable(torch.tensor(torch.rand(1, 10)), requires_grad=True)

w1.data /= 10.0
b1.data /= 10.0
w2.data /= 10.0
b2.data /= 10.0
w3.data /= 10.0

data_x = [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5], [5.0, 5.5]]
data_y = []

for d in data_x:
    data_y.append(-2.0 * d[0] - 3.0 * d[1])

for i in range(100):
    for d in range(len(data_x)):
        x.data = torch.tensor([data_x[d]])
        y.data = torch.tensor([data_y[d]])
        o1 = torch.matmul(w1, x.T) + b1
        o2 = torch.matmul(w2, o1) + b2
        o3 = torch.matmul(w3, o2)
        o = torch.sum(o3)
        loss = (o - y) ** 2
        loss.backward()
        w1.data = w1.data - 0.001 * w1.grad.data
        w2.data = w2.data - 0.001 * w2.grad.data
        b1.data = b1.data - 0.001 * b1.grad.data
        b2.data = b2.data - 0.001 * b2.grad.data
        w3.data = w3.data - 0.001 * w3.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b1.grad.data.zero_()
        b2.grad.data.zero_()
        w3.grad.data.zero_()
        print("loss: ", loss.data.tolist())