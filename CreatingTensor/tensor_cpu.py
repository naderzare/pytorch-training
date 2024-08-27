import torch

first_tensor = torch.tensor([[12, 10, 11, 9], [13, 15, 14, 16]])
second_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

added_tensor = first_tensor + second_tensor

print(added_tensor, type(added_tensor), added_tensor.size())

sub_tensor = first_tensor - second_tensor

print(sub_tensor, type(sub_tensor), sub_tensor.size())