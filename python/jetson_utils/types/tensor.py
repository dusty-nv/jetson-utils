import numpy as np
import ctypes as C

from . import as_dtype


def as_tensor(tensor, to='np', device=None, dtype=None, **kwargs):
    """
    Convert tensors between numpy/torch/python types.
    
    TODO: add options for zero-copy mapping when possible
          like with cudaToNumpy() and cudaToTorch()
    """
    if tensor is None:
        return None

    try: # optionally enable torch.Tensor
        import torch
        HAS_TORCH=True
    except Exception:
        HAS_TORCH=False

    if HAS_TORCH and isinstance(to, str) and to.startswith('cuda'):
        device = to
        to = 'pt'
    
    if isinstance(to, type):
        if to == np.ndarray:
            to = 'np'
        elif HAS_TORCH and to == torch.Tensor:
            to = 'pt'
        elif to == list:
            to = 'list'
        else:
            raise TypeError(f"expected 'to' as np.ndarray, torch.Tensor, or list (was {to})")
            
    if to == 'pt' and not HAS_TORCH:
        raise ImportError(f"Could not 'import torch' - install PyTorch to convert to torch.Tensor format")

    dtype = as_dtype(dtype, to)
    
    if isinstance(tensor, np.ndarray):
        if to == 'np':   # np->np
            if dtype:
                tensor = tensor.astype(dtype=convert_dtype(dtype, to='np'), copy=False)
            return tensor
        elif HAS_TORCH and to == 'pt': # np->pt
            return torch.from_numpy(tensor).to(device=device, dtype=convert_dtype(dtype, to='pt'), **kwargs)
        elif to == 'list': # np->list
            return tensor.tolist()
    elif HAS_TORCH and isinstance(tensor, torch.Tensor):
        if to == 'np':   # pt->np
            if dtype:
                tensor = tensor.type(dtype=convert_dtype(dtype, to='pt'))
            return tensor.detach().cpu().numpy()
        elif to == 'pt': # pt->pt
            if device is not None or dtype is not None:
                return tensor.to(device=device, dtype=convert_dtype(dtype, to='pt'), **kwargs)
            else:
                return tensor
        elif to == 'list':
            return tensor.tolist()
    elif isinstance(tensor, list):
        if to == 'np':
            return np.asarray(tensor, dtype=dtype)
        elif HAS_TORCH and to == 'pt':
            return torch.as_tensor(tensor, dtype=dtype, device=device)
        elif to == 'list':
            return tensor
                       
    raise ValueError(f"unsupported tensor input/output type (in={type(tensor)} out={to})")

    

__all__ = ['as_tensor']