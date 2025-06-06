#ifndef __CUDA_SPLIT_H__
#define __CUDA_SPLIT_H__


#include "cudaUtility.h"
#include "cudaVector.h"
#include "imageFormat.h"
#include <vector>


/**
 * Split an image on the GPU (supports RGB/BGR, RGBA/BGRA to some single color planes(using GRAY format))
 * @ingroup split
 */
cudaError_t cudaSplit(void *input, void **output, size_t width, size_t height, imageFormat format);

/**
 * Split an image on the GPU (supports RGBA/BGRA to 3 colors and alpha plane)
 * @ingroup split
 */
cudaError_t cudaSplit(void *input, void *output_color, void *output_alpha, size_t width, size_t height, imageFormat format);

#endif
