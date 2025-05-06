import numpy as np
import ctypes as C


def as_dtype(dtype, to='np'):
    """
    Convert a numpy or torch dtype to one of the following formats:

      * np (np.dtype)
      * pt (torch.dtype)
      * py (native type)
      * c  (ctypes)

    It will attempt to convert other known primitive types native to Python.
    """
    if dtype is None or to is None:
        return None

    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
        
    #elif isinstance(dtype, torch.dtype):
    #    return np.dtype(str(dtype).split('.')[-1]) # remove the torch.* prefix

    if to == 'np':
        return dtype
    elif to == 'pt':
        return torch_dtype(dtype)
    elif to == 'c' or to == 'ctypes':
        return as_ctype(dtype)
    else:
        raise ValueError(f"Unrecognized argument to='{to}' (expected one of:  np, pt, py, c)")


def as_ctype(dtype):
    """
    Convert a numpy or torch dtype to a native C type
    """
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    if dtype == np.float16:
        return C.c_ushort
    else:
        return np.ctypeslib.as_ctypes_type(dtype)


def torch_dtype(dtype):
    """
    Convert numpy.dtype or str to torch.dtype
    """
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    elif not isinstance(dtype, type):
        # from np.dtype() (not a built-in np.float32, ect)
        torch_dtypes = {
            'bool'       : torch.bool,
            'uint8'      : torch.uint8,
            'int8'       : torch.int8,
            'int16'      : torch.int16,
            'int32'      : torch.int32,
            'int64'      : torch.int64,
            'float16'    : torch.float16,
            'float32'    : torch.float32,
            'float64'    : torch.float64,
            'complex64'  : torch.complex64,
            'complex128' : torch.complex128
        }

        torch_dtype = torch_dtypes.get(str(dtype))
        
        if torch_dtype is None:
            raise ValueError("unknown dtype {dtype}  (type={type(dtype)}")
            
        return torch_dtype

    if dtype == np.float32:      return torch.float32
    elif dtype == np.float64:    return torch.float64
    elif dtype == np.int8:       return torch.int8
    elif dtype == np.int16:      return torch.int16
    elif dtype == np.int32:      return torch.int32
    elif dtype == np.int64:      return torch.int64
    elif dtype == np.uint8:      return torch.uint8
    elif dtype == np.uint16:     return torch.uint16
    elif dtype == np.uint32:     return torch.uint32
    elif dtype == np.uint64:     return torch.uint64
    elif dtype == np.complex64:  return torch.complex64
    elif dtype == np.complex128: return torch.complex128
    elif dtype == np.bool_:      return torch.bool
    
    raise ValueError("unknown dtype {dtype}  (type={type(dtype)}")


__all__ = ['as_dtype', 'as_ctype', 'torch_dtype']