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
 
#include "imageIO.h"
#include "cudaMappedMemory.h"
#include "filesystem.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

#define LOG_IMAGE "[image] "


// limit_pixel
static inline unsigned char limit_pixel( float pixel, float max_pixel )
{
	if( pixel < 0 )
		pixel = 0;

	if( pixel > max_pixel )
		pixel = max_pixel;

	return (unsigned char)pixel;
}


// saveImageRGBA
bool saveImageRGBA( const char* filename, float4* cpu, int width, int height, float max_pixel, int quality )
{
	// validate parameters
	if( !filename || !cpu || width <= 0 || height <= 0 )
	{
		printf(LOG_IMAGE "saveImageRGBA() - invalid parameter\n");
		return false;
	}
	
	if( quality < 1 )
		quality = 1;

	if( quality > 100 )
		quality = 100;
	
	// allocate memory for the uint8 image
	const size_t stride = width * sizeof(unsigned char) * 4;
	const size_t size   = stride * height;
	unsigned char* img  = (unsigned char*)malloc(size);

	if( !img )
	{
		printf(LOG_IMAGE "failed to allocate %zu bytes to save %ix%i image '%s'\n", size, width, height, filename);
		return false;
	}

	// convert image from float to uint8
	const float scale = 255.0f / max_pixel;

	for( int y=0; y < height; y++ )
	{
		const size_t yOffset = y * stride;

		for( int x=0; x < width; x++ )
		{
			const size_t offset = yOffset + x * sizeof(unsigned char) * 4;
			const float4 pixel  = cpu[y * width + x];

			img[offset + 0] = limit_pixel(pixel.x * scale, max_pixel);
			img[offset + 1] = limit_pixel(pixel.y * scale, max_pixel);
			img[offset + 2] = limit_pixel(pixel.z * scale, max_pixel);
			img[offset + 3] = limit_pixel(pixel.w * scale, max_pixel);
		}
	}

	// determine the file extension
	const std::string ext = fileExtension(filename);
	const char* extension = ext.c_str();

	if( ext.size() == 0 )
	{
		printf(LOG_IMAGE "invalid filename or extension, '%s'\n", filename);
		free(img);
		return false;
	}

	// save the image
	int save_result = 0;

	if( strcasecmp(extension, "jpg") == 0 || strcasecmp(extension, "jpeg") == 0 )
	{
		save_result = stbi_write_jpg(filename, width, height, 4, img, quality);
	}
	else if( strcasecmp(extension, "png") == 0 )
	{
		// convert quality from 1-100 to 0-9 (where 0 is high quality)
		quality = (100 - quality) / 10;

		if( quality < 0 )
			quality = 0;
		
		if( quality > 9 )
			quality = 9;

		stbi_write_png_compression_level = quality;

		// write the PNG file
		save_result = stbi_write_png(filename, width, height, 4, img, stride);
	}
	else if( strcasecmp(extension, "tga") == 0 )
	{
		save_result = stbi_write_tga(filename, width, height, 4, img);
	}
	else if( strcasecmp(extension, "bmp") == 0 )
	{
		save_result = stbi_write_bmp(filename, width, height, 4, img);
	}
	else if( strcasecmp(extension, "hdr") == 0 )
	{
		save_result = stbi_write_hdr(filename, width, height, 4, (float*)cpu);
	}
	else
	{
		printf(LOG_IMAGE "invalid extension format '.%s' saving image '%s'\n", extension, filename);
		printf(LOG_IMAGE "valid extensions are:  JPG/JPEG, PNG, TGA, BMP, and HDR.\n");

		free(img);
		return false;
	}

	// check the return code
	if( !save_result )
	{
		printf(LOG_IMAGE "failed to save %ix%i image to '%s'\n", width, height, filename);
		free(img);
		return false;
	}

	free(img);
	return true;
}


// loadImageIO (internal)
static unsigned char* loadImageIO( const char* filename, int* width, int* height, int* channels )
{
	// validate parameters
	if( !filename || !width || !height )
	{
		printf(LOG_IMAGE "loadImageIO() - invalid parameter(s)\n");
		return NULL;
	}
	
	// verify file path
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		printf(LOG_IMAGE "failed to find file '%s'\n", filename);
		return NULL;
	}

	// load original image
	int imgWidth = 0;
	int imgHeight = 0;
	int imgChannels = 0;

	unsigned char* img = stbi_load(path.c_str(), &imgWidth, &imgHeight, &imgChannels, 0);

	if( !img )
	{
		printf(LOG_IMAGE "failed to load '%s'\n", path.c_str());
		printf(LOG_IMAGE "(error:  %s)\n", stbi_failure_reason());
		return NULL;
	}

	// validate dimensions for sanity
	printf(LOG_IMAGE "loaded '%s'  (%i x %i, %i channels)\n", filename, imgWidth, imgHeight, imgChannels);

	if( imgWidth < 0 || imgHeight < 0 || imgChannels < 0 || imgChannels > 4 )
	{
		printf(LOG_IMAGE "'%s' has invalid dimensions\n", filename);
		return NULL;
	}

	// if the user provided a desired size, resize the image if necessary
	const int resizeWidth  = *width;
	const int resizeHeight = *height;

	if( resizeWidth > 0 && resizeHeight > 0 && resizeWidth != imgWidth && resizeHeight != imgHeight )
	{
		unsigned char* img_org = img;

		printf(LOG_IMAGE "resizing '%s' to %ix%i\n", filename, resizeWidth, resizeHeight);

		// allocate memory for the resized image
		img = (unsigned char*)malloc(resizeWidth * resizeHeight * imgChannels * sizeof(unsigned char));

		if( !img )
		{
			printf(LOG_IMAGE "failed to allocated memory to resize '%s' to %ix%i\n", filename, resizeWidth, resizeHeight);
			free(img_org);		
			return NULL;
		}

		// resize the original image
		if( !stbir_resize_uint8(img_org, imgWidth, imgHeight, 0,
						    img, resizeWidth, resizeHeight, 0, imgChannels) )
		{
			printf(LOG_IMAGE "failed to resize '%s' to %ix%i\n", filename, resizeWidth, resizeHeight);
			free(img_org);
			return NULL;
		}

		// update resized dimensions
		imgWidth  = resizeWidth;
		imgHeight = resizeHeight;

		free(img_org);
	}	

	*width = imgWidth;
	*height = imgHeight;
	*channels = imgChannels;

	return img;
}


// loadImageRGBA
bool loadImageRGBA( const char* filename, float4** cpu, float4** gpu, int* width, int* height, const float4& mean )
{
	// validate parameters
	if( !filename || !cpu || !gpu || !width || !height )
	{
		printf(LOG_IMAGE "loadImageRGBA() - invalid parameter(s)\n");
		return NULL;
	}

	// attempt to load the data from disk
	int imgWidth = *width;
	int imgHeight = *height;
	int imgChannels = 0;

	unsigned char* img = loadImageIO(filename, &imgWidth, &imgHeight, &imgChannels);
	
	if( !img )
		return false;
	

	// allocate CUDA buffer for the image
	const size_t imgSize = imgWidth * imgHeight * sizeof(float) * 4;

	if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
	{
		printf(LOG_CUDA "failed to allocate %zu bytes for image '%s'\n", imgSize, filename);
		return false;
	}


	// convert uint8 image to float4
	float4* cpuPtr = *cpu;
	
	for( int y=0; y < imgHeight; y++ )
	{
		const size_t yOffset = y * imgWidth * imgChannels * sizeof(unsigned char);

		for( int x=0; x < imgWidth; x++ )
		{
			#define GET_PIXEL(channel)	    float(img[offset + channel])
			#define SET_PIXEL_FLOAT4(r,g,b,a) cpuPtr[y*imgWidth+x] = make_float4(r,g,b,a)

			const size_t offset = yOffset + x * imgChannels * sizeof(unsigned char);
					
			switch(imgChannels)
			{
				case 1:	
				{
					const float grey = GET_PIXEL(0);
					SET_PIXEL_FLOAT4(grey - mean.x, grey - mean.y, grey - mean.z, 255.0f - mean.w); 
					break;
				}
				case 2:	
				{
					const float grey = GET_PIXEL(0);
					SET_PIXEL_FLOAT4(grey - mean.x, grey - mean.y, grey - mean.z, GET_PIXEL(1) - mean.w);
					break;
				}
				case 3:
				{
					SET_PIXEL_FLOAT4(GET_PIXEL(0) - mean.x, GET_PIXEL(1) - mean.y, GET_PIXEL(2) - mean.z, 255.0f - mean.w);
					break;
				}
				case 4:
				{
					SET_PIXEL_FLOAT4(GET_PIXEL(0) - mean.x, GET_PIXEL(1) - mean.y, GET_PIXEL(2) - mean.z, GET_PIXEL(3) - mean.w);
					break;
				}
			}
		}
	}
	
	*width  = imgWidth;
	*height = imgHeight;
	
	free(img);
	return true;
}


// loadImageRGB
bool loadImageRGB( const char* filename, float3** cpu, float3** gpu, int* width, int* height, const float3& mean )
{
	// validate parameters
	if( !filename || !cpu || !gpu || !width || !height )
	{
		printf(LOG_IMAGE "loadImageRGB() - invalid parameter(s)\n");
		return NULL;
	}

	// attempt to load the data from disk
	int imgWidth = *width;
	int imgHeight = *height;
	int imgChannels = 0;

	unsigned char* img = loadImageIO(filename, &imgWidth, &imgHeight, &imgChannels);
	
	if( !img )
		return false;
	

	// allocate CUDA buffer for the image
	const size_t imgSize = imgWidth * imgHeight * sizeof(float) * 3;

	if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
	{
		printf(LOG_CUDA "failed to allocate %zu bytes for image '%s'\n", imgSize, filename);
		return false;
	}


	// convert uint8 image to float4
	float3* cpuPtr = *cpu;
	
	for( int y=0; y < imgHeight; y++ )
	{
		const size_t yOffset = y * imgWidth * imgChannels * sizeof(unsigned char);

		for( int x=0; x < imgWidth; x++ )
		{
			#define SET_PIXEL_FLOAT3(r,g,b) cpuPtr[y*imgWidth+x] = make_float3(r,g,b)

			const size_t offset = yOffset + x * imgChannels * sizeof(unsigned char);
					
			switch(imgChannels)
			{
				case 1:	
				{
					const float grey = GET_PIXEL(0);
					SET_PIXEL_FLOAT3(grey - mean.x, grey - mean.y, grey - mean.z); 
					break;
				}
				case 2:	
				{
					const float grey = GET_PIXEL(0);
					SET_PIXEL_FLOAT3(grey - mean.x, grey - mean.y, grey - mean.z);
					break;
				}
				case 3:
				case 4:
				{
					SET_PIXEL_FLOAT3(GET_PIXEL(0) - mean.x, GET_PIXEL(1) - mean.y, GET_PIXEL(2) - mean.z);
					break;
				}
			}
		}
	}
	
	*width  = imgWidth;
	*height = imgHeight;
	
	free(img);
	return true;
}


// loadImageBGR
bool loadImageBGR( const char* filename, float3** cpu, float3** gpu, int* width, int* height, const float3& mean )
{
	// validate parameters
	if( !filename || !cpu || !gpu || !width || !height )
	{
		printf(LOG_IMAGE "loadImageRGB() - invalid parameter(s)\n");
		return NULL;
	}

	// attempt to load the data from disk
	int imgWidth = *width;
	int imgHeight = *height;
	int imgChannels = 0;

	unsigned char* img = loadImageIO(filename, &imgWidth, &imgHeight, &imgChannels);
	
	if( !img )
		return false;
	

	// allocate CUDA buffer for the image
	const size_t imgSize = imgWidth * imgHeight * sizeof(float) * 3;

	if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
	{
		printf(LOG_CUDA "failed to allocate %zu bytes for image '%s'\n", imgSize, filename);
		return false;
	}


	// convert uint8 image to float4
	float3* cpuPtr = *cpu;
	
	for( int y=0; y < imgHeight; y++ )
	{
		const size_t yOffset = y * imgWidth * imgChannels * sizeof(unsigned char);

		for( int x=0; x < imgWidth; x++ )
		{
			#define SET_PIXEL_FLOAT3(r,g,b) cpuPtr[y*imgWidth+x] = make_float3(r,g,b)

			const size_t offset = yOffset + x * imgChannels * sizeof(unsigned char);
					
			switch(imgChannels)
			{
				case 1:	
				{
					const float grey = GET_PIXEL(0);
					SET_PIXEL_FLOAT3(grey - mean.x, grey - mean.y, grey - mean.z); 
					break;
				}
				case 2:	
				{
					const float grey = GET_PIXEL(0);
					SET_PIXEL_FLOAT3(grey - mean.x, grey - mean.y, grey - mean.z);
					break;
				}
				case 3:
				case 4:
				{
					SET_PIXEL_FLOAT3(GET_PIXEL(2) - mean.x, GET_PIXEL(1) - mean.y, GET_PIXEL(0) - mean.z);
					break;
				}
			}
		}
	}
	
	*width  = imgWidth;
	*height = imgHeight;
	
	free(img);
	return true;
}

/*
  TODO:  implement band-sequential mode

	// note:  caffe/GIE is band-sequential (as opposed to the typical Band Interleaved by Pixel)
	const uint32_t imgPixels = imgWidth * imgHeight;

	cpuPtr[imgPixels * 0 + y * imgWidth + x] = px.x; 
	cpuPtr[imgPixels * 1 + y * imgWidth + x] = px.y; 
	cpuPtr[imgPixels * 2 + y * imgWidth + x] = px.z; 
*/

