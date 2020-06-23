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
 
#ifndef __IMAGE_WRITER_H_
#define __IMAGE_WRITER_H_


#include "videoOutput.h"


/**
 * Save an image or set of images to disk.
 *
 * Supported image formats for saving are JPG, PNG, TGA, and BMP. Internally, 
 * imageLoader uses the saveImage() function to save the images, so the 
 * supported formats are the same.
 *
 * imageWriter has the ability to write a sequence of images to a directory,
 * for example `images/%i.jpg` (where `%i` becomes the image number), or 
 * just a single image with a static filename (e.g. `images/my_image.jpg`).
 * When given just the path of a directory as output, it will default to
 * incremental `%i.jpg` sequencing and save in JPG format.
 *
 * @note imageWriter implements the videoOutput interface and is intended to
 * be used through that as opposed to directly.  videoOutput implements
 * additional command-line parsing of videoOptions to construct instances.
 *
 * @see videoOutput
 * @ingroup image
 */
class imageWriter : public videoOutput
{
public:
	/**
	 * Create an imageWriter instance from a path and optional videoOptions.
	 */
	static imageWriter* Create( const char* path, const videoOptions& options=videoOptions() );

	/**
	 * Create an imageWriter instance from the provided video options.
	 */
	static imageWriter* Create( const videoOptions& options );

	/**
	 * Destructor
	 */
	virtual ~imageWriter();

	/**
	 * Save the next frame.
	 * @see videoOutput::Render()
	 */
	template<typename T> bool Render( T* image, uint32_t width, uint32_t height )		{ return Render((void**)image, width, height, imageFormatFromType<T>()); }
	
	/**
	 * Save the next frame.
	 * @see videoOutput::Render()
	 */
	virtual bool Render( void* image, uint32_t width, uint32_t height, imageFormat format );

	/**
	 * Return the interface type (imageWriter::Type)
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 * Unique type identifier of imageWriter class.
	 */
	static const uint32_t Type = (1 << 5);

	/**
	 * String array of supported image file extensions, terminated
	 * with a NULL sentinel value.  The supported extension are:
	 *
	 *    - JPG / JPEG
	 *    - PNG
	 *    - TGA / TARGA
	 *    - BMP
	 *
	 * @see IsSupportedExtension() to check a string against this list.
	 */
	static const char* SupportedExtensions[];

	/**
	 * Return true if the extension is in the list of SupportedExtensions.
	 * @param ext string containing the extension to be checked (should not contain leading dot)
	 * @see SupportedExtensions for the list of supported video file extensions.
	 */
	static bool IsSupportedExtension( const char* ext );

protected:
	imageWriter( const videoOptions& options );

	uint32_t mFileCount;
	char     mFileOut[1024];
};

#endif
