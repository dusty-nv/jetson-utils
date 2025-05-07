import ctypes
import logging

from jetson_utils import NamedDict, getenv
from . import assert_cuda, cuInit, cudaDeviceFamily, cudaCoresPerSM, CUDA_SUCCESS


log = logging.getLogger(__name__)


def cudaDeviceQuery():
    """
    Get GPU device info by loading/calling libcuda directly.
    """
    try:
        return _cudaDeviceQuery()
    except Exception as error:
        log.warning(f'cudaDeviceQuery() failed:  {error}')
        raise error
    

def _cudaDeviceQuery():
    """
    https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
    """
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36

    cuda = cuInit(cached=False)

    nGpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    freeMem = ctypes.c_size_t()
    totalMem = ctypes.c_size_t()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()

    output = []

    assert_cuda(cuda.cuDeviceGetCount(ctypes.byref(nGpus)))
    log.debug("Found %d CUDA devices" % nGpus.value)

    for i in range(nGpus.value):
        assert_cuda(cuda.cuDeviceGet(ctypes.byref(device), i))
        info = NamedDict()

        # decode GPU device name
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            info.name = name.split(b'\0', 1)[0].decode()

        # get CUDA compute capability (SM)
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            cc = (cc_major.value, cc_minor.value)
            info.family = cudaDeviceFamily(*cc)
            info.cc = cc_major.value * 10 + cc_minor.value

        # calculate number of CUDA cores
        if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS:
            info.mp = cores.value
            info.cores = (cores.value * cudaCoresPerSM(cc_major.value, cc_minor.value) or 0)
            if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS:
                info.threads = (cores.value * threads_per_core.value)

        #if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS:
        #    info.gpu_clock = clockrate.value / 1000.
        #if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) == CUDA_SUCCESS:
        #    info.mem_clock = clockrate.value / 1000.

        try:
            result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
        except AttributeError:
            result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)

        if result == CUDA_SUCCESS:
            try:
                result = cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem))
            except AttributeError:
                result = cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))

            if result == CUDA_SUCCESS:
                info.mem_total = int(totalMem.value / 1024**2)
                info.mem_free = int(freeMem.value / 1024**2)
            else:
                assert_cuda(result, 'cuMemGetInfo failed')

            cuda.cuCtxDetach(context)
        else:
            assert_cuda(result, 'cuCtxCreate failed')
            
        if not info:
            continue

        # extended board name for Jetson's
        if info.name and info.mem_total:
            if info.name.lower() == 'orin':
                if info.mem_total > 50000:
                    info.name = 'AGX Orin 64GB'
                elif info.mem_total > 24000:
                    info.name = 'AGX Orin 32GB'
                elif info.mem_total > 12000:
                    info.name = 'Orin NX 16GB' 
                elif info.mem_total > 6000:
                    info.name = 'Orin Nano 8GB'  
                elif info.mem_total > 2500:
                    info.name = 'Orin Nano 4GB'

        info.name = info.name.replace('NVIDIA', '').replace('Generation', '').strip()
        info.short_name = cudaShortName(info.name)

        output.append(info)
        
    return output


def cudaShortName(name):
    """
    Get board identifier and name
    """
    if not name:
      return ''

    if name == 'AGX Orin 64GB': return 'agx-orin'
    elif name == 'AGX Orin 32GB': return 'agx-orin-32gb'
    elif name == 'Orin NX 16GB': return 'orin-nx'
    elif name == 'Orin NX 8GB': return 'orin-nx-8gb'
    elif name == 'Orin Nano 8GB': return 'orin-nano'
    elif name == 'Orin Nano 4GB': return 'orin-nano-4gb'

    return name.lower().replace(' ', '-')


def cudaShortVersion(version: str=None):
    """
    Return CUDA version tag (like cu126 for CUDA 12.6)
    """
    if not version:
        version = getenv('CUDA_VERSION')

    return f"cu{version.replace('.','')}"


__all__ = ['cudaDeviceQuery', 'cudaShortName', 'cudaShortVersion']
