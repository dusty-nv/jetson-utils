
def cudaCoresPerSM(major, minor):
    """
    Returns the number of CUDA cores per multiprocessor for a given Compute Capability version. 
    There is no way to retrieve that via the API, so it needs to be hard-coded.
    See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    """
    return {(1, 0): 8,    # Tesla
            (1, 1): 8,
            (1, 2): 8,
            (1, 3): 8,
            (2, 0): 32,   # Fermi
            (2, 1): 48,
            (3, 0): 192,  # Kepler
            (3, 2): 192,
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,  # Maxwell
            (5, 2): 128,
            (5, 3): 128,
            (6, 0): 64,   # Pascal
            (6, 1): 128,
            (6, 2): 128,
            (7, 0): 64,   # Volta
            (7, 2): 64,
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere
            (8, 6): 128,
            (8, 7): 128,
            (8, 9): 128,  # Ada
            (9, 0): 128,  # Hopper
            }.get((major, minor), 0)


def cudaDeviceFamily(major, minor):
    """
    Map CUDA compute capability to GPU family names.
    """
    if major == 1:      return "Tesla"
    elif major == 2:    return "Fermi"
    elif major == 3:    return "Kepler"
    elif major == 5:    return "Maxwell"
    elif major == 6:    return "Pascal"
    elif major == 7:
        if minor < 5:   return "Volta"
        else:           return "Turing"
    elif major == 8: 
        if minor < 9:   return "Ampere"
        else:           return "Ada"
    elif major == 9:    return "Hopper"
    elif major == 10:   return "Blackwell"
    else:               return "Unknown"


__all__ = ['cudaCoresPerSM', 'cudaDeviceFamily']
