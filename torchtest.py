import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
#print(torch.__version__)
#print(!nvidia-smi)


###Tensors
#Creating Tensors
#1) Scalar
#Scalar = torch.tensor(7)
#print(Scalar)
#print(Scalar.ndim)
#Scalar.item()

#Vector
#vector = torch.tensor([7,7])
#print(vector.ndim)

#MATRIX
#MATRIX = torch.tensor([[7,8],[9,10]])

#TENSORS #Random
#TENSOR = torch.tensor([[[1,2,3],
#                       [4,5,6],[7,8,9]]])

#print(TENSOR.shape)
#print(TENSOR[0])
#print(TENSOR.ndim)

#Making Random TENSORS
#random_tensor = torch.rand(10,3,4)
#print(random_tensor)
#print(random_tensor.ndim)
#print(random_tensor.shape)
#ZERO and ONES Tensors
#zeros = torch.zeros(size=(3,4))
#print(zeros)
#ran_tensor = torch.rand(3,4)
#print(ran_tensor)
#print(zeros * ran_tensor)

#ones = torch.ones(3,4)
#print(ones)
#print(ones.dtype)

# Check if GPU is available
#if torch.cuda.is_available():
#    device = torch.device("cuda")  # Use GPU
#    print("Using GPU:", torch.cuda.get_device_name(0))
#else:
#    device = torch.device("cpu")  # Use CPU
#    print("Using CPU")




#Range of Tensors and Tensors Like!

#Check Range
#one_to_ten = torch.arange(start = 1, end=11, step =1)
#print(one_to_ten)

#Tensor Like
#get_ten_zeroes = torch.zeros_like(input= one_to_ten)
#print(get_ten_zeroes)
#print(torch.arange(0,10))

#Dealing With tensor Data Types
#float_32_tensor = torch.tensor([3.0,6.0,9.0],
#dtype= None,#What data tyoe the tensor amounts to
 #device=None, # What device the tensor runs on (e.g. CPU , GPU)
  #requires_grad=False # Does it require to track gradients
  #)

#The following Parameters are considered very important for creation of a Tensor i.e. (dtype, device, requires_grad) Always Keep in Mind!

#print (float_32_tensor)
#print(float_32_tensor.dtype) # To check the data type use tensor_name.dtype
#print(float_32_tensor.ndim) # To check the number of dimensions use tensor_name.ndim

#float_16_tensor = float_32_tensor.type(torch.float16)
#print(float_16_tensor)

#print(float_16_tensor * float_32_tensor)




#Getting Tensor Attributes

#int_32_tensor = torch.tensor([3,6,9], dtype=torch.int32)
#print(int_32_tensor)

#print(float_32_tensor * int_32_tensor)

###MOST COMMON PROBLEMS TO RUN INTO###
##Getting Information from the Tensors
##1. Information about the dataType? ---> tensor.dtype
##2. Information about Tensor shape ---> tensor.shape
##3. Information about the Device Type i.e. if the tensor is on a GPU or CPU  --> tensor.device


##TESTING THE ABOVE ONES OUT

#some_tensor = torch.rand(3,4)
#print(some_tensor)
#print(f"Data Type of the Tensor: {some_tensor.dtype}")
#print(f"Shape of the Tensor: {some_tensor.shape}")
#print(f"Device of the Tensor: {some_tensor.device}")




#some_new_tensor = torch.rand((3,4),dtype=torch.float16)
#print(some_new_tensor)
#print(f"Data Type of the Tensor: {some_new_tensor.dtype}")
#print(f"Shape of the Tensor: {some_new_tensor.shape}")
#print(f"Device of the Tensor: {some_new_tensor.device}")


###THERE ARE TWO MAIN WAYS OF USING MATRIX MULTIPLICATION IN NEURAL NETWORKS AND DEEP LEARNING
#1) Multiplication by element wise operation
#2) Matrix multiplication by DOT Product

 #Matrix Transpose Method

Tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [5,6]])


Tensor_B = torch.tensor([[7,8],
                         [9,10],
                         [11,12]])



#print(Tensor_A)
#print(Tensor_B)



#print(torch.matmul(Tensor_A, Tensor_B))
#print(Tensor_A.shape, Tensor_B.shape)

#Here Clearly the shape of Tensor_A & Tensor_B match to perform operations so we can use "Matrix Transpose" method to change the shape of a tensor.

#Transpose Switches the Axes or dimensions of a given tensor


#print(Tensor_B, Tensor_B.shape)
#print(Tensor_B.T, Tensor_B.T.shape)



#print(torch.matmul(Tensor_A, Tensor_B.T).shape)



#MIN MAX AGR Functions

x = torch.arange(0,100,10)


print(x)
#To print out the miminum position
print(x.min())

#To print out the maximum position
print(x.max())
