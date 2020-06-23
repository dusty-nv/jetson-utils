/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#ifndef __IMAGE_IO_H_
#define __IMAGE_IO_H_


#include "cudaUtility.h"
#include "imageFormat.h"


/**
 * Load a color image from disk into CUDA memory, in uchar3/uchar4/float3/float4 formats with pixel values 0-255.
 *
 * Supported image file formats by loadImage() include:
 * 
 *   - JPG
 *   - PNG
 *   - TGA
 *   - BMP
 *   - GIF
 *   - PSD
 *   - HDR
 *   - PIC
 *   - PNM (PPM/PGM binary)
 *
 * This function loads the image into shared CPU/GPU memory, using the functions from cudaMappedMemory.h
 *
 * @param[in] filename Path to the image file to load from disk.
 * @param[out] ptr Reference to pointer that will be set to the shared CPU/GPU buffer containing the image that will be allocated.
 *                 This buffer will be allocated by loadImage() in CUDA mapped memory, so it is shared between CPU/GPU.
 *                 gpu will be the pointer to this shared buffer in the GPU's address space.  There is physically one buffer in memory.
 * @param[in,out] width Pointer to int variable that gets set to the width of the image in pixels.
 *                      If the width variable contains a non-zero value when it's passed in, the image is resized to this desired width.
 *                      Otherwise if the value of width is 0, the image will be loaded with it's dimensions from the file on disk.
 * @param[in,out] height Pointer to int variable that gets set to the height of the image in pixels.
 *                       If the height variable contains a non-zero value when it's passed in, the image is resized to this desired height.
 *                       Otherwise if the value of height is 0, the image will be loaded with it's dimensions from the file on disk.
 * @ingroup image
 */
template<typename T> bool loadImage( const char* filename, T** ptr, int* width, int* height )		{ return loadImage(filename, (void**)ptr, width, height, imageFormatFromType<T>()); }
	
/**
 * Load a color image from disk into CUDA memory, in uchar3/uchar4/float3/float4 formats with pixel values 0-255.
 *
 * Supported image file formats by loadImage() include:
 * 
 *   - JPG
 *   - PNG
 *   - TGA
 *   - BMP
 *   - GIF
 *   - PSD
 *   - HDR
 *   - PIC
 *   - PNM (PPM/PGM binary)
 *
 * This function loads the image into shared CPU/GPU memory, using the functions from cudaMappedMemory.h
 *
 * @param[in] filename Path to the image file to load from disk.
 * @param[out] ptr Reference to pointer that will be set to the shared CPU/GPU buffer containing the image that will be allocated.
 *                 This buffer will be allocated by loadImage() in CUDA mapped memory, so it is shared between CPU/GPU.
 *                 gpu will be the pointer to this shared buffer in the GPU's address space.  There is physically one buffer in memory.
 * @param[in,out] width Pointer to int variable that gets set to the width of the image in pixels.
 *                      If the width variable contains a non-zero value when it's passed in, the image is resized to this desired width.
 *                      Otherwise if the value of width is 0, the image will be loaded with it's dimensions from the file on disk.
 * @param[in,out] height Pointer to int variable that gets set to the height of the image in pixels.
 *                       If the height variable contains a non-zero value when it's passed in, the image is resized to this desired height.
 *                       Otherwise if the value of height is 0, the image will be loaded with it's dimensions from the file on disk.
 * @ingroup image
 */
bool loadImage( const char* filename, void** output, int* width, int* height, imageFormat format );

/**
 * Load a color image from disk into CUDA memory with alpha, in float4 RGBA format with pixel values 0-255.
 * @see loadImage() for more details about parameters and supported image formats.
 * @deprecated this overload of loadImageRGBA() is deprecated and provided for legacy compatbility.
 *             it is recommended to use loadImage() instead, which supports multiple image formats.
 * @ingroup image
 */
bool loadImageRGBA( const char* filename, float4** ptr, int* width, int* height );

/**
 * Load a color image from disk into CUDA memory with alpha, in float4 RGBA format with pixel values 0-255.
 * @see loadImage() for more details about parameters and supported image formats.
 * @deprecated this overload of loadImageRGBA() is deprecated and provided for legacy compatbility.
 *             having separate CPU and GPU pointers for shared memory is no longer needed, as they are the same.
 *             it is recommended to use loadImage() instead, which supports multiple image formats.
 * @ingroup image
 */
bool loadImageRGBA( const char* filename, float4** cpu, float4** gpu, int* width, int* height );


/**
 * Save an image in CPU/GPU shared memory to disk.
 *
 * Supported image file formats by saveImage() include:  
 *
 *   - JPG
 *   - PNG
 *   - TGA
 *   - BMP
 *
 * @param filename Desired path of the image file to save to disk.
 * @param ptr Pointer to the buffer containing the image in shared CPU/GPU zero-copy memory.
 * @param width Width of the image in pixels.
 * @param height Height of the image in pixels.
 * @param max_pixel The maximum pixel value of this image, by default it's 255 for images in the range of 0-255.
 *                  If your image is in the range of 0-1, pass 1.0 as this value.  Then the pixel values of the
 *                  image will be rescaled appropriately to be stored on disk (which expects a range of 0-255).
 * @param quality Indicates the compression quality level (between 1 and 100) to be applied for JPEG and PNG images.
 *                A level of 1 correponds to reduced quality and maximum compression.
 *                A level of 100 corresponds to maximum quality and reduced compression.
 *                By default a level of 95 is used for high quality and moderate compression. 
 *                Note that this quality parameter only applies to JPEG and PNG, other formats will ignore it.
 * @ingroup image
 */
template<typename T> bool saveImage( const char* filename, T* ptr, int width, int height,
							  int quality=95, const float2& pixel_range=make_float2(0,255) )		{ return saveImage(filename, (void*)ptr, width, height, imageFormatFromType<T>(), quality, pixel_range); }
	
/**
 * Save an image in CPU/GPU shared memory to disk.
 *
 * Supported image file formats by saveImage() include:  
 *
 *   - JPG
 *   - PNG
 *   - TGA
 *   - BMP
 *
 * @param filename Desired path of the image file to save to disk.
 * @param ptr Pointer to the buffer containing the image in shared CPU/GPU zero-copy memory.
 * @param width Width of the image in pixels.
 * @param height Height of the image in pixels.
 * @param max_pixel The maximum pixel value of this image, by default it's 255 for images in the range of 0-255.
 *                  If your image is in the range of 0-1, pass 1.0 as this value.  Then the pixel values of the
 *                  image will be rescaled appropriately to be stored on disk (which expects a range of 0-255).
 * @param quality Indicates the compression quality level (between 1 and 100) to be applied for JPEG and PNG images.
 *                A level of 1 correponds to reduced quality and maximum compression.
 *                A level of 100 corresponds to maximum quality and reduced compression.
 *                By default a level of 95 is used for high quality and moderate compression. 
 *                Note that this quality parameter only applies to JPEG and PNG, other formats will ignore it.
 * @ingroup image
 */
bool saveImage( const char* filename, void* ptr, int width, int height, imageFormat format,
			 int quality=95, const float2& pixel_range=make_float2(0,255) );

/**
 * Save a float4 image in CPU/GPU shared memory to disk.
 * @see saveImage() for more details about parameters and supported image formats.
 * @deprecated saveImageRGBA() is deprecated and provided for legacy compatbility.
 *             it is recommended to use saveImage() instead, which supports multiple image formats.
 * @ingroup image
 */
bool saveImageRGBA( const char* filename, float4* ptr, int width, int height, float max_pixel=255.0f, int quality=100 );


/**
 * @internal
 * @ingroup image
 */
#define LOG_IMAGE "[image] "


#endif

