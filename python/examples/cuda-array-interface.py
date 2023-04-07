#!/usr/bin/env python3
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import argparse
import numpy as np

from jetson_utils import cudaAllocMapped, cudaDeviceSynchronize


def test_numpy(x, y):
    print('testing numpy.add()')
    result = np.add(x, y)
    print('numpy.add() results:')
    print(result)
    
def test_cupy(x, y):
    try:
        import cupy
    except ImportError:
        print("failed to import cupy - if you wish to test cupy, please install it")
        return
        
    print('testing cupy.add()')
    result = cupy.add(x, cupy.array(y))
    print('cupy.add() results:')
    print(result)
    
def test_pycuda(x, y):
    try:
        import pycuda
        import pycuda.driver as cuda
        import pycuda.autoinit

        from pycuda.compiler import SourceModule
    except ImportError:
        print("failed to import pycuda - if you wish to test pycuda, please install it")
        return
    
    result = np.empty_like(y)
    
    print('testing pycuda kernel...')
    
    if y.dtype == np.float32:
        type = 'float'
    else:
        type = 'unsigned char'

    module = SourceModule(f"""
        __global__ void cuda_add( {type}* a, {type}* b, {type}* c )
        {{
            int idx = threadIdx.y * blockDim.x * blockDim.z + threadIdx.x * blockDim.z + threadIdx.z;
            c[idx] = a[idx] + b[idx];
        }}
        """)
    
    func = module.get_function('cuda_add')
    func(x, cuda.In(y), cuda.Out(result), block=(x.shape[0], x.shape[1], x.shape[2]))
    cudaDeviceSynchronize()
    
    print('pycuda kernel result:')
    print(result)
        
def test_numba(x, y):
    try:
        from numba import guvectorize
    except ImportError:
        print("failed to import numba - if you wish to test numba, please install it")
        return
        
    print('testing cuda guvectorized ufunc...')
    
    @guvectorize(['uint8[:], uint8[:], uint8[:]',
                  'float32[:], float32[:], float32[:]'], 
                  '(n),(n)->(n)',
                 target='cuda')
    def numba_add_arrays(x, y, res):
        for i in range(x.shape[0]):  # number of channels (3)
            res[i] = x[i] + y[i]
    
    result = numba_add_arrays(x, y)

    print('numba guvectorize ufunc results:')
    print(result.copy_to_host())
    
    
if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser('Demonstrate usage of cudaImage __array_interface__')

    parser.add_argument("--width", type=int, default=4, help="width of the array (in pixels)")
    parser.add_argument("--height", type=int, default=2, help="height of the array (in pixels)")
    parser.add_argument("--format", type=str, default="rgb32f", help="format of the array (default rgb32f)")

    args = parser.parse_args()
    print(args)

    # allocate cuda memory
    cuda_img = cudaAllocMapped(width=args.width, height=args.height, format=args.format)
    
    print(cuda_img)
    print(cuda_img.__cuda_array_interface__)
    
    # fill with monotonically increasing pattern
    for y in range(cuda_img.shape[0]):
        for x in range(cuda_img.shape[1]):
            for z in range(cuda_img.shape[2]):
                cuda_img[y,x,z] = y * cuda_img.shape[1] * cuda_img.shape[2] + x * cuda_img.shape[2] + z
        
    # parse numpy datatype
    if args.format.find('32f') >= 0:
        dtype = np.float32
    else:
        dtype = np.uint8
        
    # create another ndarray and do some ops with it
    array2 = np.full(cuda_img.shape, 1, dtype)

    # run tests (add 1 to arrays)
    test_numpy(cuda_img, array2)
    test_cupy(cuda_img, array2)
    test_numba(cuda_img, array2)
    test_pycuda(cuda_img, array2)

    #
    # the results should always be:
    #
    # [[[ 1  2  3  4]
    # [ 5  6  7  8]
    # [ 9 10 11 12]
    # [13 14 15 16]]
    # 
    # [[17 18 19 20]
    # [21 22 23 24]
    # [25 26 27 28]
    # [29 30 31 32]]]
    #
