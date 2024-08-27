import torch

print(torch.__version__)
device = 'cpu'
if torch.cuda.is_available():
    print("CUDA is available")
    device = 'cuda'

first_tensor = torch.tensor([[12, 10, 11, 9], [13, 15, 14, 16]], device=device)
second_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)

added_tensor = first_tensor + second_tensor

print(added_tensor, type(added_tensor), added_tensor.size())

sub_tensor = first_tensor - second_tensor

print(sub_tensor, type(sub_tensor), sub_tensor.size())

multi_tensor = first_tensor * second_tensor

print(multi_tensor, type(multi_tensor), multi_tensor.size())

# moving tensor to cpu
added_tensor = added_tensor.cpu()
sub_tensor = sub_tensor.cpu()
multi_tensor = multi_tensor.cpu()

print(added_tensor, type(added_tensor), added_tensor.size())
print(sub_tensor, type(sub_tensor), sub_tensor.size())
print(multi_tensor, type(multi_tensor), multi_tensor.size())