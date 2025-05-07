
import numpy as np

from jetson_utils import getLogger, NamedDict
from . import cudaToNumpy, cudaToTorch, assert_cuda


log = getLogger(__name__)


def cudaAllocMapped(shape, dtype, map_numpy=True, map_torch=True, return_dict=True):
    """
    Allocate cudaMallocManaged() memory and map it to a numpy array and PyTorch tensor
    If return dict is true, these will be returned in a dict-like DictAttr object
    with keys for 'ptr', 'array' (if map_numpy is True), and 'tensor' (if map_torch is True).
    Otherwise, a tuple will be returned with (ptr, array, tensor)
    """
    dsize = np.dtype(dtype).itemsize

    if isinstance(shape, int):
        size = shape * dsize
        shape = [shape]
    else:
        size = math.prod(shape) * dsize
    
    log.debug(f"Allocating {size} bytes ({size/(1024*1024):.2f} MB) of shared memory with cudaAllocMapped()")
    
    # TODO update for cuda driver API
    #err, ptr = cudaMallocManaged(size, cudaMemAttachGlobal) 
    err, ptr = cudaHostAlloc(size, cudaHostAllocMapped)
    assert_cuda(err)
    err, ptr = cudaHostGetDevicePointer(ptr, 0)
    
    if map_numpy:
        array = cudaToNumpy(ptr, shape, dtype)
        
    if map_torch:
        tensor = cudaToTorch(ptr, shape, dtype)
        
    if return_dict:
        d = NamedDict()
        d.ptr = ptr
        d.shape = shape
        d.dtype = dtype
        
        if map_numpy:
            d.array = array
            
        if map_torch:
            d.tensor = tensor

        return d
    else:
        if map_numpy and map_torch:
            return ptr, array, tensor
        elif map_numpy:
            return ptr, array
        elif map_torch:
            return ptr, tensor
        else:
            return ptr


__all__ = ['cudaAllocMapped']
