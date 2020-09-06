#ifndef __CUDA_SPLIT_H__
#define __CUDA_SPLIT_H__


#include "cudaUtility.h"
#include "cudaVector.h"
#include "imageFormat.h"
#include <vector>


/**
 * Split an image on the GPU (supports RGB/BGR, RGBA/BGRA)
 * @ingroup split
 */
cudaError_t cudaSplit(void *input, void **output, size_t width, size_t height, imageFormat format);

#endif
