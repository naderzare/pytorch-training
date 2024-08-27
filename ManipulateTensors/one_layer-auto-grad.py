import torch

x = torch.autograd.Variable(torch.tensor([0.0]), requires_grad=False)
y = torch.autograd.Variable(torch.tensor([0.0]), requires_grad=False)

w = torch.autograd.Variable(torch.tensor(torch.rand(1, 5)), requires_grad=True)

data_x = [1.0, 2.0, 3.0, 4.0, 5.0]
data_y = [-2.0, -4.0, -6.0, -8.0, -10.0]

for i in range(100):
    for d in range(len(data_x)):
        x.data = torch.tensor([data_x[d]])
        y.data = torch.tensor([data_y[d]])
        o = torch.sum(w * x)
        loss = (o - y) ** 2
        loss.backward()
        w.data = w.data - 0.001 * w.grad.data
        w.grad.data.zero_()
        print("loss: ", loss.data.tolist(), "w: ", w.data.tolist())