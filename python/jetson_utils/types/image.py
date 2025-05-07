import io
import numpy as np

from jetson_utils import HAS_JETSON_UTILS_EXT, getLogger

ImageTypes = (np.ndarray) # available image tensor types/loaders 
ImageExtensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

try: # check for Pillow
    import PIL
    ImageTypes = (*ImageTypes, PIL.Image.Image)
    HAS_PIL=True   
except:
    HAS_PIL=False

try: # check for torchvision
    import torch
    import torchvision.transforms.functional as F
    ImageTypes = (*ImageTypes, torch.Tensor)
    HAS_TORCHVISION=True
except:
    HAS_TORCHVISION=False


log = getLogger(__name__)


def is_image(image):
    """
    Returns true if the object is a PIL.Image, np.ndarray, torch.Tensor, or jetson_utils.cudaImage
    """
    return isinstance(image, ImageTypes)
    
 
def image_size(image):
    """
    Returns the dimensions of the image as a ``(height, width, channels)`` tuple.
    """
    if HAS_JETSON_UTILS and isinstance(image, cudaImage):
        return image.shape
    if isinstance(image, (np.ndarray, torch.Tensor)):
        return image.shape
    elif isinstance(image, PIL.Image.Image):
        return image.size
    else:
        raise TypeError(f"expected an image of type {ImageTypes} (was {type(image)})")
        
    
def load_image(path):
    """
    Load an image from a local path or URL that will be downloaded.
    
    Args:
      path (str): either a path or URL to the image.
      
    Returns:
      ``PIL.Image`` instance
    """
    if path.startswith('http') or path.startswith('https'):
        logging.debug(f'downloading {path}')
        response = requests.get(path)
        image = PIL.Image.open(io.BytesIO(response.content)).convert('RGB')
    else:
        logging.debug(f'loading {path}')
        image = PIL.Image.open(path).convert('RGB')
        
    return image


def cuda_image(image):
    """
    Convert an image from `PIL.Image`, `np.ndarray`, `torch.Tensor`, or `__gpu_array_interface__`
    to a jetson_utils.cudaImage on the GPU (without using memory copies when possible)
    """   
    if not HAS_JETSON_UTILS_EXT:
        raise RuntimeError(f"jetson-utils cuda extension should be installed to use cudaImage")
        
    # TODO implement __gpu_array_interface__
    # TODO torch image formats https://github.com/dusty-nv/jetson-utils/blob/f0bff5c502f9ac6b10aa2912f1324797df94bc2d/python/examples/cuda-from-pytorch.py#L47
    if not is_image(image):
        raise TypeError(f"expected an image of type {ImageTypes} (was {type(image)})")
        
    if isinstance(image, cudaImage):
        return image
        
    if isinstance(image, PIL.Image.Image):
        image = np.asarray(image)  # no copy
        
    if isinstance(image, np.ndarray):
        return cudaFromNumpy(image)
        
    if isinstance(image, torch.Tensor):
        input = input.to(memory_format=torch.channels_last)   # or tensor.permute(0, 3, 2, 1)
        
        return cudaImage(
            ptr=input.data_ptr(), 
            width=input.shape[-1], 
            height=input.shape[-2], 
            format=torch_image_format(input)
        )
        
 
def torch_image(image, dtype=None, device=None):
    """
    Convert the image to a type that is compatible with PyTorch ``(torch.Tensor, ndarray, PIL.Image)``
    """
    if not isinstance(image, ImageTypes):
        raise TypeError(f"expected an image of type {ImageTypes} (was {type(image)})")
        
    if HAS_JETSON_UTILS and isinstance(image, cudaImage):
        image = torch.as_tensor(image, dtype=dtype, device=device).permute(2,0,1)
        if dtype == torch.float16 or dtype == torch.float32:
            image = image / 255.0
    elif isinstance(image, (PIL.Image.Image, np.ndarray)):
        image = F.to_tensor(image)

    return image.to(dtype=dtype, device=device)
        
        
def torch_image_format(tensor):
    """
    Determine the cudaImage format string (eg 'rgb32f', 'rgba32f', ect) from a PyTorch tensor.
    Only float and uint8 tensors are supported because those datatypes are supported by cudaImage.
    """
    if tensor.dtype != torch.float32 and tensor.dtype != torch.uint8:
        raise ValueError(f"PyTorch tensor datatype should be torch.float32 or torch.uint8 (was {tensor.dtype})")
        
    if len(tensor.shape)>= 4:     # NCHW layout
        channels = tensor.shape[1]
    elif len(tensor.shape) == 3:   # CHW layout
        channels = tensor.shape[0]
    elif len(tensor.shape) == 2:   # HW layout
        channels = 1
    else:
        raise ValueError(f"PyTorch tensor should have at least 2 image dimensions (has {tensor.shape.length})")
        
    if channels == 1:   return 'gray32f' if tensor.dtype == torch.float32 else 'gray8'
    elif channels == 3: return 'rgb32f'  if tensor.dtype == torch.float32 else 'rgb8'
    elif channels == 4: return 'rgba32f' if tensor.dtype == torch.float32 else 'rgba8'
    
    raise ValueError(f"PyTorch tensor should have 1, 3, or 4 image channels (has {channels})")
    
  
__all__ = [
    'ImageTypes', 'ImageExtensions', 'is_image', 'image_size',
     'load_image', 'cuda_image', 'torch_image', 'torch_image_format', 
]