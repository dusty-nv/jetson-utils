import numpy as np
import ctypes as C


class cudaArrayInterface():
    """
    Exposes __cuda_array_interface__ - typically used as a temporary view into a larger buffer
    https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """
    def __init__(self, data, shape, dtype=np.float32):
        self.__cuda_array_interface__ = {
            'data': (data, False),  # R/W
            'shape': shape,
            'typestr': np.dtype(dtype).str,
            'version': 3,
        }  


def cudaToNumpy(ptr, shape, dtype):
    """
    Map a shared CUDA pointer into np.ndarray with the given shape and datatype.
    The pointer should have been allocated with cudaMallocManaged() or using cudaHostAllocMapped,
    and the user is responsible for any CPU/GPU synchronization (i.e. by using cudaStreams)
    """
    array = np.ctypeslib.as_array(C.cast(ptr, C.POINTER(dtype_to_ctype(dtype))), shape=shape)
    
    if dtype == np.float16:
        array.dtype = np.float16
       
    return array
 

def cudaToTorch(ptr, shape, dtype):
    """
    Map a shared CUDA pointer into np.ndarray with the given shape and datatype.
    The pointer should have been allocated with cudaMallocManaged() or using cudaHostAllocMapped,
    and the user is responsible for any CPU/GPU synchronization (i.e. by using cudaStreams)
    """
    import torch
    return torch.as_tensor(cudaArrayInterface(ptr, shape, dtype), device='cuda')
    

__all__ = ['cudaArrayInterface', 'cudaToNumpy', 'cudaToTorch']