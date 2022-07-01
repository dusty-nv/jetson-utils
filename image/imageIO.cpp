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
#include "cudaColorspace.h"

#include "filesystem.h"
#include "logging.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

#include <memory>


namespace {
    struct Deleter {
        void operator()(unsigned char* data) const noexcept(noexcept(stbi_image_free)) { stbi_image_free(data); }
    };
    using StbBuffer = std::unique_ptr<unsigned char[], Deleter>;
}

// loadImageIO (internal)
static StbBuffer loadImageIO( const char* filename, int* width, int* height, int* channels )
{
	// validate parameters
	if( !filename || !width || !height || !channels )
	{
		LogError(LOG_IMAGE "loadImageIO() - invalid parameter(s)\n");
		return NULL;
	}
	
	// verify file path
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		LogError(LOG_IMAGE "failed to find file '%s'\n", filename);
		return NULL;
	}

	// load original image
	int imgWidth = 0;
	int imgHeight = 0;
	int imgChannels = 0;

	auto img = StbBuffer(stbi_load(path.c_str(), &imgWidth, &imgHeight, &imgChannels, *channels));

	if( !img )
	{
		LogError(LOG_IMAGE "failed to load '%s'\n", path.c_str());
		LogError(LOG_IMAGE "(error:  %s)\n", stbi_failure_reason());
		return NULL;
	}

	if( *channels != 0 )
		imgChannels = *channels;

	// validate dimensions for sanity
	LogVerbose(LOG_IMAGE "loaded '%s'  (%ix%i, %i channels)\n", filename, imgWidth, imgHeight, imgChannels);

	if( imgWidth < 0 || imgHeight < 0 || imgChannels < 0 || imgChannels > 4 )
	{
		LogError(LOG_IMAGE "'%s' has invalid dimensions\n", filename);
		return NULL;
	}

	// if the user provided a desired size, resize the image if necessary
	const int resizeWidth  = *width;
	const int resizeHeight = *height;

	if( resizeWidth > 0 && resizeHeight > 0 && resizeWidth != imgWidth && resizeHeight != imgHeight )
	{
		const auto img_org = std::move(img);

		LogVerbose(LOG_IMAGE "resizing '%s' to %ix%i\n", filename, resizeWidth, resizeHeight);

		// allocate memory for the resized image
		img.reset(new unsigned char[resizeWidth * resizeHeight * imgChannels * sizeof(unsigned char)]);

		if( !img )
		{
			LogError(LOG_IMAGE "failed to allocated memory to resize '%s' to %ix%i\n", filename, resizeWidth, resizeHeight);
			return NULL;
		}

		// resize the original image
		if( !stbir_resize_uint8(img_org.get(), imgWidth, imgHeight, 0,
						    img.get(), resizeWidth, resizeHeight, 0, imgChannels) )
		{
			LogError(LOG_IMAGE "failed to resize '%s' to %ix%i\n", filename, resizeWidth, resizeHeight);
			return NULL;
		}

		// update resized dimensions
		imgWidth  = resizeWidth;
		imgHeight = resizeHeight;

	}	

	*width = imgWidth;
	*height = imgHeight;
	*channels = imgChannels;

	return img;
}


// loadImage
bool loadImage( const char* filename, void** output, int* width, int* height, imageFormat format )
{
	// validate parameters
	if( !filename || !output || !width || !height )
	{
		LogError(LOG_IMAGE "loadImage() - invalid parameter(s)\n");
		return NULL;
	}

	// check that the requested format is supported
	if( !imageFormatIsRGB(format) )
	{
		LogError(LOG_IMAGE "loadImage() -- unsupported output image format requested (%s)\n", imageFormatToStr(format));
		LogError(LOG_IMAGE "               supported output formats are:\n");
		LogError(LOG_IMAGE "                   * rgb8\n");		
		LogError(LOG_IMAGE "                   * rgba8\n");		
		LogError(LOG_IMAGE "                   * rgb32\n");		
		LogError(LOG_IMAGE "                   * rgba32\n");

		return NULL;
	}

	// attempt to load the data from disk
	int imgWidth = *width;
	int imgHeight = *height;
	int imgChannels = imageFormatChannels(format);

	auto img = loadImageIO(filename, &imgWidth, &imgHeight, &imgChannels);
	
	if( !img )
		return false;	

	// allocate CUDA buffer for the image
	const size_t imgSize = imageFormatSize(format, imgWidth, imgHeight);

	if( !cudaAllocMapped((void**)output, imgSize) )
	{
		LogError(LOG_IMAGE "loadImage() -- failed to allocate %zu bytes for image '%s'\n", imgSize, filename);
		return false;
	}

	// convert from uint8 to float
	if( format == IMAGE_RGB32F || format == IMAGE_RGBA32F )
	{
		const imageFormat inputFormat = (imgChannels == 3) ? IMAGE_RGB8 : IMAGE_RGBA8;
		const size_t inputImageSize = imageFormatSize(inputFormat, imgWidth, imgHeight);

		void* inputImgGPU = NULL;

		if( !cudaAllocMapped(&inputImgGPU, inputImageSize) )
		{
			LogError(LOG_IMAGE "loadImage() -- failed to allocate %zu bytes for image '%s'\n", inputImageSize, filename);
			return false;
		}

		memcpy(inputImgGPU, img.get(), imageFormatSize(inputFormat, imgWidth, imgHeight));

		if( CUDA_FAILED(cudaConvertColor(inputImgGPU, inputFormat, *output, format, imgWidth, imgHeight)) )
		{
			printf(LOG_IMAGE "loadImage() -- failed to convert image from %s to %s ('%s')\n", imageFormatToStr(inputFormat), imageFormatToStr(format), filename);
			return false;
		}

		CUDA(cudaFreeHost(inputImgGPU));
	}
	else
	{
		// uint8 output can be straight copied to GPU memory
		memcpy(*output, img.get(), imgSize);
	}

	*width  = imgWidth;
	*height = imgHeight;
	
	return true;
}


// loadImageRGBA
bool loadImageRGBA( const char* filename, float4** output, int* width, int* height )
{
	return loadImage(filename, (void**)output, width, height, IMAGE_RGBA32F);
}

// loadImageRGBA
bool loadImageRGBA( const char* filename, float4** cpu, float4** gpu, int* width, int* height )
{
	const bool result = loadImageRGBA(filename, gpu, width, height);

	if( !result )
		return false;

	if( cpu != NULL )
		*cpu = *gpu;

	return true;
}


// limit_pixel
/*static inline unsigned char limit_pixel( float pixel, float max_pixel )
{
	if( pixel < 0 )
		pixel = 0;

	if( pixel > max_pixel )
		pixel = max_pixel;

	return (unsigned char)pixel;
}*/


// saveImage
bool saveImage( const char* filename, void* ptr, int width, int height, imageFormat format, int quality, const float2& pixel_range, bool sync )
{
	// validate parameters
	if( !filename || !ptr || width <= 0 || height <= 0 )
	{
		LogError(LOG_IMAGE "saveImageRGBA() - invalid parameter\n");
		return false;
	}
	
	if( quality < 1 )
		quality = 1;

	if( quality > 100 )
		quality = 100;
	
	// check that the requested format is supported
	if( !imageFormatIsRGB(format) && !imageFormatIsGray(format) )
	{
		LogError(LOG_IMAGE "saveImage() -- unsupported input image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_IMAGE "               supported input image formats are:\n");
		LogError(LOG_IMAGE "                   * rgb8\n");		
		LogError(LOG_IMAGE "                   * rgba8\n");		
		LogError(LOG_IMAGE "                   * rgb32f\n");		
		LogError(LOG_IMAGE "                   * rgba32f\n");
		LogError(LOG_IMAGE "                   * gray8\n");
		LogError(LOG_IMAGE "                   * gray32\n");

		return false;
	}
	
	// allocate memory for the uint8 image
	const size_t channels = imageFormatChannels(format);
	const size_t stride   = width * sizeof(unsigned char) * channels;
	const size_t size     = stride * height;
	unsigned char* img    = (unsigned char*)ptr;

	// if needed, convert from float to uint8
	const imageBaseType baseType = imageFormatBaseType(format);

	if( baseType == IMAGE_FLOAT )
	{
		imageFormat outputFormat = IMAGE_UNKNOWN;

		if( channels == 1 )
			outputFormat = IMAGE_GRAY8;
		else if( channels == 3 )
			outputFormat = IMAGE_RGB8;
		else if( channels == 4 )
			outputFormat = IMAGE_RGBA8;

		if( !cudaAllocMapped((void**)&img, size) )
		{
			LogError(LOG_IMAGE "saveImage() -- failed to allocate %zu bytes for image '%s'\n", size, filename);
			return false;
		}

		if( CUDA_FAILED(cudaConvertColor(ptr, format, img, outputFormat, width, height, pixel_range)) )  // TODO limit pixel
		{
			LogError(LOG_IMAGE "saveImage() -- failed to convert image from %s to %s ('%s')\n", imageFormatToStr(format), imageFormatToStr(outputFormat), filename);
			return false;
		}
		
		sync = true;
	}
	
	if( sync )
		CUDA(cudaDeviceSynchronize());
	
	#define release_return(x) 	\
		if( baseType == IMAGE_FLOAT ) \
			CUDA(cudaFreeHost(img)); \
		return x;
	
	// determine the file extension
	const std::string ext = fileExtension(filename);
	const char* extension = ext.c_str();

	if( ext.size() == 0 )
	{
		LogError(LOG_IMAGE "invalid filename or extension, '%s'\n", filename);
		release_return(false);
	}

	// save the image
	int save_result = 0;

	if( strcasecmp(extension, "jpg") == 0 || strcasecmp(extension, "jpeg") == 0 )
	{
		save_result = stbi_write_jpg(filename, width, height, channels, img, quality);
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
		save_result = stbi_write_png(filename, width, height, channels, img, stride);
	}
	else if( strcasecmp(extension, "tga") == 0 )
	{
		save_result = stbi_write_tga(filename, width, height, channels, img);
	}
	else if( strcasecmp(extension, "bmp") == 0 )
	{
		save_result = stbi_write_bmp(filename, width, height, channels, img);
	}
	/*else if( strcasecmp(extension, "hdr") == 0 )
	{
		save_result = stbi_write_hdr(filename, width, height, channels, (float*)cpu);
	}*/
	else
	{
		LogError(LOG_IMAGE "invalid extension format '.%s' saving image '%s'\n", extension, filename);
		LogError(LOG_IMAGE "valid extensions are:  JPG/JPEG, PNG, TGA, BMP.\n");
		
		release_return(false);
	}

	// check the return code
	if( !save_result )
	{
		LogError(LOG_IMAGE "failed to save %ix%i image to '%s'\n", width, height, filename);
		release_return(false);
	}

	LogVerbose(LOG_IMAGE "saved '%s'  (%ix%i, %zu channels)\n", filename, width, height, channels);

	release_return(true);
}


// saveImageRGBA
bool saveImageRGBA( const char* filename, float4* ptr, int width, int height, float max_pixel, int quality )
{
	return saveImage(filename, ptr, width, height, IMAGE_RGBA32F, quality, make_float2(0, max_pixel));
}
