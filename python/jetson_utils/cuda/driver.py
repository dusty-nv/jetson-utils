import ctypes

from .error import assert_cuda


# lazy-loaded driver API
libcuda = None  


def cuInit(cached=True):
    """
    Load libcuda.so if not already. If `cached=False`, the driver
    will only be loaded temporarily (for example device enumeration)
    """
    global libcuda

    if libcuda is not None:
        return libcuda

    driver = ctypes.CDLL('libcuda.so')
    assert_cuda(driver.cuInit(0))

    if cached:
        libcuda = driver

    return driver
    

__all__ = ['libcuda', 'cuInit']
