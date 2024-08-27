import torch

sample_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

print(sample_tensor[3])
print(sample_tensor[3].item())
print(sample_tensor[1:3])
print(sample_tensor[1:3].tolist())

sample_tensor = torch.tensor([[1, 2, 3], [4, 5, 2]], dtype=torch.float32)

print(sample_tensor[1, 2])
print(sample_tensor[1, 2].item())
print(sample_tensor[0, 0:2])
print(sample_tensor[0, 0:2].tolist())
print(sample_tensor[0:2, 0:2].tolist())
print(sample_tensor[sample_tensor < 3].tolist())

sample_tensor1 = torch.tensor([[1, 2, 3, 4, 5], [6,7,8,9,10]], dtype=torch.float32)
sample_tensor2 = torch.tensor([[11,12,13,14,15], [16,17,18,19,20]], dtype=torch.float32)

combined_tensor1 = torch.cat((sample_tensor1, sample_tensor2), dim=0)
print(combined_tensor1.tolist())

combined_tensor = torch.stack((sample_tensor1, sample_tensor2), dim=0)
print(combined_tensor.tolist())

unbinding_tensor1, unbinding_tensor2 = combined_tensor.unbind(dim=0)
print(unbinding_tensor1.tolist())
print(unbinding_tensor2.tolist())

# Pointwise operations
sample_tensor1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
print("sample_tensor1: ", sample_tensor1.tolist())

sample_tensor2 = torch.tensor([6, 7, 8, 9, 10], dtype=torch.float32)
print("sample_tensor2: ", sample_tensor2.tolist())


print("sample_tensor1.add(sample_tensor2): ", sample_tensor1.add(sample_tensor2).tolist())
print("sample_tensor1.sub(sample_tensor2): ", sample_tensor1.sub(sample_tensor2).tolist())
print("sample_tensor1.mul(sample_tensor2): ", sample_tensor1.mul(sample_tensor2).tolist())
print("sample_tensor1.div(sample_tensor2): ", sample_tensor1.div(sample_tensor2).tolist())
print("sample_tensor1.pow(sample_tensor2): ", sample_tensor1.pow(sample_tensor2).tolist())
print("sample_tensor1.sqrt(): ", sample_tensor1.sqrt().tolist())
print("sample_tensor1.exp(): ", sample_tensor1.exp().tolist())
print("sample_tensor1.log(): ", sample_tensor1.log().tolist())
print("sample_tensor1.abs(): ", sample_tensor1.abs().tolist())
print("sample_tensor1.neg(): ", sample_tensor1.neg().tolist())
print("sample_tensor1.reciprocal(): ", sample_tensor1.reciprocal().tolist())

# Reduction operations
sample_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("sample_tensor: ", sample_tensor.tolist())

print("sample_tensor.sum(): ", sample_tensor.sum().tolist())
print("sample_tensor.prod(): ", sample_tensor.prod().tolist())
print("sample_tensor.mean(): ", sample_tensor.mean().tolist())
print("sample_tensor.std(): ", sample_tensor.std().tolist())
print("sample_tensor.var(): ", sample_tensor.var().tolist())
print("sample_tensor.min(): ", sample_tensor.min().tolist())
print("sample_tensor.max(): ", sample_tensor.max().tolist())
print("sample_tensor.argmax(): ", sample_tensor.argmax().tolist())
print("sample_tensor.argmin(): ", sample_tensor.argmin().tolist())

print("sample_tensor.sum(dim=0): ", sample_tensor.sum(dim=0).tolist())
print("sample_tensor.prod(dim=0): ", sample_tensor.prod(dim=0).tolist())
print("sample_tensor.mean(dim=0): ", sample_tensor.mean(dim=0).tolist())

print("sample_tensor.sum(dim=1): ", sample_tensor.sum(dim=1).tolist())
print("sample_tensor.prod(dim=1): ", sample_tensor.prod(dim=1).tolist())
print("sample_tensor.mean(dim=1): ", sample_tensor.mean(dim=1).tolist())

# Comparison operations
sample_tensor1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
print("sample_tensor1: ", sample_tensor1.tolist())

sample_tensor2 = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32)
print("sample_tensor2: ", sample_tensor2.tolist())

print("sample_tensor1.eq(sample_tensor2): ", sample_tensor1.eq(sample_tensor2).tolist())
print("sample_tensor1.ge(sample_tensor2): ", sample_tensor1.ge(sample_tensor2).tolist())
print("sample_tensor1.le(sample_tensor2): ", sample_tensor1.le(sample_tensor2).tolist())
print("sample_tensor1.gt(sample_tensor2): ", sample_tensor1.gt(sample_tensor2).tolist())
print("sample_tensor1.lt(sample_tensor2): ", sample_tensor1.lt(sample_tensor2).tolist())
print("sample_tensor1.ne(sample_tensor2): ", sample_tensor1.ne(sample_tensor2).tolist())

# Linear algebra operations matmul
sample_tensor1 = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float32)
print("sample_tensor1: ", sample_tensor1.tolist())

sample_tensor2 = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float32)
print("sample_tensor2: ", sample_tensor2.tolist())

print("sample_tensor1.matmul(sample_tensor2.T): ", sample_tensor1.matmul(sample_tensor2.T).tolist())
print("torch.matmul(sample_tensor1, sample_tensor2.T): ", torch.matmul(sample_tensor1, sample_tensor2.T).tolist())

sample_tensor_2_to_3 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("sample_tensor_2_to_3: ", sample_tensor_2_to_3.tolist())

sample_tensor_3_to_2 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
print("sample_tensor_3_to_2: ", sample_tensor_3_to_2.tolist())

print("sample_tensor_2_to_3.matmul(sample_tensor_3_to_2): ", sample_tensor_2_to_3.matmul(sample_tensor_3_to_2).tolist())
print("torch.matmul(sample_tensor_2_to_3, sample_tensor_3_to_2): ", torch.matmul(sample_tensor_2_to_3, sample_tensor_3_to_2).tolist())

# Linear algebra operations dot
first_tensor = torch.randn(2, 3)
second_tensor = torch.randn(3, 4)
third_tensor = torch.randn(4, 5)
fourth_tensor = torch.randn(5, 6)
fifth_tensor = torch.randn(6, 7)
print("torch.linalg.multi_dot([first_tensor, second_tensor, third_tensor, fourth_tensor, fifth_tensor]).tolist()")
print(torch.linalg.multi_dot([first_tensor, second_tensor, third_tensor, fourth_tensor, fifth_tensor]).tolist())

# Linear algebra operations eigenvectors and eigenvalues
sample_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
print("sample_tensor: ", sample_tensor.tolist())
eigenvalues, eigenvectors = torch.linalg.eig(sample_tensor)
print("eigenvalues: ", eigenvalues.tolist())
print("eigenvectors: ", eigenvectors.tolist())