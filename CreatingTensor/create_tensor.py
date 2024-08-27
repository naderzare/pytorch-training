import torch
import numpy as np

tensor_from_list = torch.tensor([1, 2, 3, 4])
tensor_from_tuple = torch.tensor((1, 2, 3, 4))
tensor_from_array = torch.tensor(np.array([1, 2, 3, 4]))

print("tensor_from_list:", tensor_from_list)
print("tensor_from_tuple:", tensor_from_tuple)
print("tensor_from_array:", tensor_from_array)

tensor_zero = torch.zeros(2, 3)
tensor_ones = torch.ones(2, 3)
tensor_rand = torch.rand(2, 3)

print("tensor_zero:", tensor_zero)
print("tensor_ones:", tensor_ones)
print("tensor_rand:", tensor_rand)

tensor_like = torch.zeros_like(tensor_rand)

print("tensor_like:", tensor_like)

tensor_full_5 = torch.full((2, 3), 5)
