/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "cudaColorspace.h"

#include "cudaRGB.h"
#include "cudaYUV.h"
#include "cudaBayer.h"
#include "cudaGrayscale.h"

#include "logging.h"


// cudaConvertColor
cudaError_t cudaConvertColor( void* input, imageFormat inputFormat,
					     void* output, imageFormat outputFormat,
					     size_t width, size_t height,
						 const float2& pixel_range)
{
	if( inputFormat == IMAGE_NV12 )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaNV12ToRGB(input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaNV12ToRGB(input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaNV12ToRGBA(input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaNV12ToRGBA(input, (float4*)output, width, height));
	}
	else if( inputFormat == IMAGE_I420 )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaI420ToRGB(input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaI420ToRGB(input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaI420ToRGBA(input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaI420ToRGBA(input, (float4*)output, width, height));
	}
	else if( inputFormat == IMAGE_YV12 )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaYV12ToRGB(input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaYV12ToRGB(input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaYV12ToRGBA(input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaYV12ToRGBA(input, (float4*)output, width, height));
	}
	else if( inputFormat == IMAGE_YUYV )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaYUYVToRGB(input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaYUYVToRGB(input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaYUYVToRGBA(input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaYUYVToRGBA(input, (float4*)output, width, height));
	}
	else if( inputFormat == IMAGE_YVYU )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaYVYUToRGB(input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaYVYUToRGB(input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaYVYUToRGBA(input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaYVYUToRGBA(input, (float4*)output, width, height));
	}
	else if( inputFormat == IMAGE_UYVY )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaUYVYToRGB(input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaUYVYToRGB(input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaUYVYToRGBA(input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaUYVYToRGBA(input, (float4*)output, width, height));
	}
	else if( inputFormat == IMAGE_RGB8 )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaRGB8ToRGBA8((uchar3*)input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaRGB8ToRGB32((uchar3*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaRGB8ToRGBA32((uchar3*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_BGR8 )
			return CUDA(cudaRGB8ToBGR8((uchar3*)input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_BGRA8 )
			return CUDA(cudaRGB8ToRGBA8((uchar3*)input, (uchar4*)output, width, height, true));
		else if( outputFormat == IMAGE_BGR32F )
			return CUDA(cudaRGB8ToRGB32((uchar3*)input, (float3*)output, width, height, true));
		else if( outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaRGB8ToRGBA32((uchar3*)input, (float4*)output, width, height, true));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaRGB8ToGray8((uchar3*)input, (uint8_t*)output, width, height));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaRGB8ToGray32((uchar3*)input, (float*)output, width, height));
		else if( outputFormat == IMAGE_I420 )
			return CUDA(cudaRGBToI420((uchar3*)input, output, width, height));
		else if( outputFormat == IMAGE_YV12 )
			return CUDA(cudaRGBToYV12((uchar3*)input, output, width, height));
	}
	else if( inputFormat == IMAGE_RGBA8 )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaRGBA8ToRGB8((uchar4*)input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaRGBA8ToRGB32((uchar4*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaRGBA8ToRGBA32((uchar4*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_BGR8 )
			return CUDA(cudaRGBA8ToRGB8((uchar4*)input, (uchar3*)output, width, height, true));
		else if( outputFormat == IMAGE_BGRA8 )
			return CUDA(cudaRGBA8ToBGRA8((uchar4*)input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_BGR32F )
			return CUDA(cudaRGBA8ToRGB32((uchar4*)input, (float3*)output, width, height, true));
		else if( outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaRGBA8ToRGBA32((uchar4*)input, (float4*)output, width, height, true));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaRGBA8ToGray8((uchar4*)input, (uint8_t*)output, width, height));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaRGBA8ToGray32((uchar4*)input, (float*)output, width, height));
		else if( outputFormat == IMAGE_I420 )
			return CUDA(cudaRGBAToI420((uchar4*)input, output, width, height));
		else if( outputFormat == IMAGE_YV12 )
			return CUDA(cudaRGBAToYV12((uchar4*)input, output, width, height));
	}
	else if( inputFormat == IMAGE_RGB32F )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaRGB32ToRGB8((float3*)input, (uchar3*)output, width, height, false, pixel_range));	
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaRGB32ToRGBA8((float3*)input, (uchar4*)output, width, height, false, pixel_range));	
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaRGB32ToRGBA32((float3*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_BGR8 )
			return CUDA(cudaRGB32ToRGB8((float3*)input, (uchar3*)output, width, height, true, pixel_range));	
		else if( outputFormat == IMAGE_BGRA8 )
			return CUDA(cudaRGB32ToRGBA8((float3*)input, (uchar4*)output, width, height, true, pixel_range));	
		else if( outputFormat == IMAGE_BGR32F )
			return CUDA(cudaRGB32ToBGR32((float3*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaRGB32ToRGBA32((float3*)input, (float4*)output, width, height, true));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaRGB32ToGray8((float3*)input, (uint8_t*)output, width, height, false, pixel_range));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaRGB32ToGray32((float3*)input, (float*)output, width, height));
		else if( outputFormat == IMAGE_I420 )
			return CUDA(cudaRGBToI420((float3*)input, output, width, height));
		else if( outputFormat == IMAGE_YV12 )
			return CUDA(cudaRGBToYV12((float3*)input, output, width, height));
	}
	else if( inputFormat == IMAGE_RGBA32F )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaRGBA32ToRGB8((float4*)input, (uchar3*)output, width, height, false, pixel_range));	
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaRGBA32ToRGBA8((float4*)input, (uchar4*)output, width, height, false, pixel_range));	
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaRGBA32ToRGB32((float4*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
		else if( outputFormat == IMAGE_BGR8 )
			return CUDA(cudaRGBA32ToRGB8((float4*)input, (uchar3*)output, width, height, true, pixel_range));	
		else if( outputFormat == IMAGE_BGRA8 )
			return CUDA(cudaRGBA32ToRGBA8((float4*)input, (uchar4*)output, width, height, true, pixel_range));	
		else if( outputFormat == IMAGE_BGR32F )
			return CUDA(cudaRGBA32ToRGB32((float4*)input, (float3*)output, width, height, true));
		else if( outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaRGBA32ToBGRA32((float4*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaRGBA32ToGray8((float4*)input, (uint8_t*)output, width, height, false, pixel_range));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaRGBA32ToGray32((float4*)input, (float*)output, width, height));
		else if( outputFormat == IMAGE_I420 )
			return CUDA(cudaRGBAToI420((float4*)input, output, width, height));
		else if( outputFormat == IMAGE_YV12 )
			return CUDA(cudaRGBAToYV12((float4*)input, output, width, height));
		
	}
	else if( inputFormat == IMAGE_BGR8 )
	{
		if( outputFormat == IMAGE_BGR8 )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
		else if( outputFormat == IMAGE_BGR8 )
			return CUDA(cudaRGB8ToRGBA8((uchar3*)input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_BGR32F )
			return CUDA(cudaRGB8ToRGB32((uchar3*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaRGB8ToRGBA32((uchar3*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaRGB8ToBGR8((uchar3*)input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaRGB8ToRGBA8((uchar3*)input, (uchar4*)output, width, height, true));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaRGB8ToRGB32((uchar3*)input, (float3*)output, width, height, true));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaRGB8ToRGBA32((uchar3*)input, (float4*)output, width, height, true));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaRGB8ToGray8((uchar3*)input, (uint8_t*)output, width, height, true));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaRGB8ToGray32((uchar3*)input, (float*)output, width, height, true));
	}
	else if( inputFormat == IMAGE_BGRA8 )
	{
		if( outputFormat == IMAGE_BGR8 )
			return CUDA(cudaRGBA8ToRGB8((uchar4*)input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_BGRA8 )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
		else if( outputFormat == IMAGE_BGR32F )
			return CUDA(cudaRGBA8ToRGB32((uchar4*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaRGBA8ToRGBA32((uchar4*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaRGBA8ToRGB8((uchar4*)input, (uchar3*)output, width, height, true));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaRGBA8ToBGRA8((uchar4*)input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaRGBA8ToRGB32((uchar4*)input, (float3*)output, width, height, true));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaRGBA8ToRGBA32((uchar4*)input, (float4*)output, width, height, true));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaRGBA8ToGray8((uchar4*)input, (uint8_t*)output, width, height, true));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaRGBA8ToGray32((uchar4*)input, (float*)output, width, height, true));
	}
	else if( inputFormat == IMAGE_BGR32F )
	{
		if( outputFormat == IMAGE_BGR8 )
			return CUDA(cudaRGB32ToRGB8((float3*)input, (uchar3*)output, width, height, false, pixel_range));	
		else if( outputFormat == IMAGE_BGRA8 )
			return CUDA(cudaRGB32ToRGBA8((float3*)input, (uchar4*)output, width, height, false, pixel_range));	
		else if( outputFormat == IMAGE_BGR32F )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
		else if( outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaRGB32ToRGBA32((float3*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaRGB32ToRGB8((float3*)input, (uchar3*)output, width, height, true, pixel_range));	
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaRGB32ToRGBA8((float3*)input, (uchar4*)output, width, height, true, pixel_range));	
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaRGB32ToBGR32((float3*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaRGB32ToRGBA32((float3*)input, (float4*)output, width, height, true));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaRGB32ToGray8((float3*)input, (uint8_t*)output, width, height, true, pixel_range));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaRGB32ToGray32((float3*)input, (float*)output, width, height, true));
	}
	else if( inputFormat == IMAGE_BGRA32F )
	{
		if( outputFormat == IMAGE_BGR8 )
			return CUDA(cudaRGBA32ToRGB8((float4*)input, (uchar3*)output, width, height, false, pixel_range));	
		else if( outputFormat == IMAGE_BGRA8 )
			return CUDA(cudaRGBA32ToRGBA8((float4*)input, (uchar4*)output, width, height, false, pixel_range));	
		else if( outputFormat == IMAGE_BGR32F )
			return CUDA(cudaRGBA32ToRGB32((float4*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
		else if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaRGBA32ToRGB8((float4*)input, (uchar3*)output, width, height, true, pixel_range));	
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaRGBA32ToRGBA8((float4*)input, (uchar4*)output, width, height, true, pixel_range));	
		else if( outputFormat == IMAGE_RGB32F )
			return CUDA(cudaRGBA32ToRGB32((float4*)input, (float3*)output, width, height, true));
		else if( outputFormat == IMAGE_RGBA32F )
			return CUDA(cudaRGBA32ToBGRA32((float4*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaRGBA32ToGray8((float4*)input, (uint8_t*)output, width, height, true, pixel_range));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaRGBA32ToGray32((float4*)input, (float*)output, width, height, true));		
	}
	else if( inputFormat == IMAGE_GRAY8 )
	{
		if( outputFormat == IMAGE_RGB8 || outputFormat == IMAGE_BGR8 )
			return CUDA(cudaGray8ToRGB8((uint8_t*)input, (uchar3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA8 || outputFormat == IMAGE_BGRA8 )
			return CUDA(cudaGray8ToRGBA8((uint8_t*)input, (uchar4*)output, width, height));
		else if( outputFormat == IMAGE_RGB32F || outputFormat == IMAGE_BGR32F )
			return CUDA(cudaGray8ToRGB32((uint8_t*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F || outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaGray8ToRGBA32((uint8_t*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaGray8ToGray32((uint8_t*)input, (float*)output, width, height));
	}
	else if( inputFormat == IMAGE_GRAY32F )
	{
		if( outputFormat == IMAGE_RGB8 || outputFormat == IMAGE_BGR8 )
			return CUDA(cudaGray32ToRGB8((float*)input, (uchar3*)output, width, height, pixel_range));
		else if( outputFormat == IMAGE_RGBA8 || outputFormat == IMAGE_BGRA8 )
			return CUDA(cudaGray32ToRGBA8((float*)input, (uchar4*)output, width, height, pixel_range));
		else if( outputFormat == IMAGE_RGB32F || outputFormat == IMAGE_BGR32F )
			return CUDA(cudaGray32ToRGB32((float*)input, (float3*)output, width, height));
		else if( outputFormat == IMAGE_RGBA32F || outputFormat == IMAGE_BGRA32F )
			return CUDA(cudaGray32ToRGBA32((float*)input, (float4*)output, width, height));
		else if( outputFormat == IMAGE_GRAY8 )
			return CUDA(cudaGray32ToGray8((float*)input, (uint8_t*)output, width, height, pixel_range));
		else if( outputFormat == IMAGE_GRAY32F )
			return CUDA(cudaMemcpy(output, input, imageFormatSize(inputFormat, width, height), cudaMemcpyDeviceToDevice));
	}
	else if( imageFormatIsBayer(inputFormat) )
	{
		if( outputFormat == IMAGE_RGB8 )
			return CUDA(cudaBayerToRGB((uint8_t*)input, (uchar3*)output, width, height, inputFormat));
		else if( outputFormat == IMAGE_RGBA8 )
			return CUDA(cudaBayerToRGBA((uint8_t*)input, (uchar3*)output, width, height, inputFormat));
	}

	LogError(LOG_CUDA "cudaColorConvert() -- invalid input/output format combination (%s -> %s)\n", imageFormatToStr(inputFormat), imageFormatToStr(outputFormat));
	return cudaErrorInvalidValue;
}


