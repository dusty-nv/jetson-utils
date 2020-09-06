#ifndef __CUDA_MERGE_H__
#define __CUDA_MERGE_H__


#include "cudaUtility.h"
#include "cudaVector.h"
#include "imageFormat.h"
#include <vector>


/**
 * Merge an image on the GPU (supports RGB/BGR, RGBA/BGRA)
 * @ingroup merge
 */
cudaError_t cudaMerge(void **input, void *output, size_t width, size_t height, imageFormat format);

#endif
