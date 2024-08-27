import torch

x = torch.autograd.Variable(torch.tensor([0.0]), requires_grad=False)
y = torch.autograd.Variable(torch.tensor([0.0]), requires_grad=False)

w = torch.autograd.Variable(torch.tensor(torch.rand(2, 5)), requires_grad=True)

data_x = [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5], [5.0, 5.5]]
data_y = []

for d in data_x:
    data_y.append(-2.0 * d[0] - 3.0 * d[1])

for i in range(100):
    for d in range(len(data_x)):
        x.data = torch.tensor([data_x[d]])
        y.data = torch.tensor([data_y[d]])
        o = torch.sum(torch.matmul(w.T, x.T))
        loss = (o - y) ** 2
        loss.backward()
        w.data = w.data - 0.001 * w.grad.data
        w.grad.data.zero_()
        print("loss: ", loss.data.tolist())