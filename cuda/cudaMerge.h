#ifndef __CUDA_MERGE_H__
#define __CUDA_MERGE_H__


#include "cudaUtility.h"
#include "cudaVector.h"
#include "imageFormat.h"
#include <vector>


/**
 * Merge an image on the GPU (supports RGB/BGR, RGBA/BGRA from some single color planes(using GRAY format))
 * @ingroup merge
 */
cudaError_t cudaMerge(void **input, void *output, size_t width, size_t height, imageFormat format);

/**
 * Merge an image on the GPU (supports RGB/BGR, RGBA/BGRA from 3 colors and alpha plane)
 * @ingroup merge
 */
cudaError_t cudaMerge(void *input_color, void *input_alpha, void *output, size_t width, size_t height, imageFormat format);

#endif
