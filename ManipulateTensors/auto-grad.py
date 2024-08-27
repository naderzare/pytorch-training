import torch

x = torch.autograd.Variable(torch.tensor([2.0]), requires_grad=False)
y = torch.autograd.Variable(torch.tensor([1.0]), requires_grad=True)
z = torch.autograd.Variable(torch.tensor([5.0]), requires_grad=True)

a = x * y
f = z * a

print("f: ", f)

f.backward()
print("x.grad: ", x.grad)
print("y.grad: ", y.grad)
print("z.grad: ", z.grad)
