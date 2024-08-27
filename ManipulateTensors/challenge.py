# Python code​​​​​​‌​‌‌‌​‌​​​‌‌‌‌​​‌​‌‌​‌‌​​ below
# Use print("Debug messages...") to debug your solution.

# import MockTensor, since torch library is not supported in coderpad
import torch
show_expected_result = False
show_hints = False


def split_x_into_chunks(x):
    # split x into 4 chunks along its first dimension
    x = torch.chunk(x, 4, 0)
    return x

def split_y_into_chunks(y):
    # Your code goes here
    y = torch.chunk(y, 4, 0)
    return y

def split_x_custom(x):
    # Your code goes here
    # split x to chunks with the following number of rows: 5 and 3 along its first dimension
    x = torch.split(x, [5, 3], 0)
    return x

def split_y_custom(y):
    # Your code goes here
    y = torch.split(y, [4, 6, 6], 0)
    return y

x = torch.rand(8, 5)
y = torch.rand(16)