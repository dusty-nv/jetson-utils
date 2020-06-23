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
 
#ifndef __IMAGE_LOADER_H_
#define __IMAGE_LOADER_H_


#include "videoSource.h"

#include <string>
#include <vector>


/**
 * Load an image or set of images from disk into GPU memory.
 *
 * Supported image formats for loading are JPG, PNG, TGA, BMP, GIF, PSD, HDR,
 * PIC, and PNM (PPM/PGM binary). Internally, imageLoader uses the loadImage() 
 * function to load the images, so the supported formats are the same.
 *
 * imageLoader has the ability to load an sequence of images from a directory,
 * including wildcard characters (e.g. `images/*.jpg`), or just a single image.
 * When given just the path to a directory, it will load all valid images from
 * that directory.
 *
 * @note imageLoader implements the videoSource interface and is intended to
 * be used through that as opposed to directly.  videoSource implements
 * additional command-line parsing of videoOptions to construct instances.
 *
 * @see videoSource
 * @ingroup image
 */
class imageLoader : public videoSource
{
public:
	/**
	 * Create an imageLoader instance from a path and optional videoOptions.
	 */
	static imageLoader* Create( const char* path, const videoOptions& options=videoOptions() );
	
	/**
	 * Create an imageLoader instance from the provided video options.
	 */
	static imageLoader* Create( const videoOptions& options );

	/**
	 * Destructor
	 */
	virtual ~imageLoader();

	/**
	 * Load the next frame.
	 * @see videoSource::Capture()
	 */
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX )		{ return Capture((void**)image, imageFormatFromType<T>(), timeout); }
	
	/**
	 * Load the next frame.
	 * @see videoSource::Capture()
	 */
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX );

	/**
	 * Open the stream.
	 * @see videoSource::Open()
	 */
	virtual bool Open();

	/**
	 * Close the stream.
	 * @see videoSource::Close()
	 */
	virtual void Close();

	/**
	 * Return true if End Of Stream (EOS) has been reached.
	 * In the context of imageLoader, EOS means that all images
	 * in the sequence have been loaded, and looping is either
	 * disabled or all loops have already been run.
	 */
	inline bool IsEOS() const				{ return mEOS; }

	/**
	 * Return the interface type (imageLoader::Type)
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 * Unique type identifier of imageLoader class.
	 */
	static const uint32_t Type = (1 << 4);

	/**
	 * String array of supported image file extensions, terminated
	 * with a NULL sentinel value.  The supported extension are:
	 *
	 *    - JPG / JPEG
	 *    - PNG
	 *    - TGA / TARGA
	 *    - BMP
	 *    - GIF
	 * 	 - PSD
	 *    - HDR
	 *    - PIC
	 *    - PNM / PBM / PPM / PGM
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
	imageLoader( const videoOptions& options );

	inline bool isLooping() const { return (mOptions.loop < 0) || ((mOptions.loop > 0) && (mLoopCount < mOptions.loop)); }

	bool mEOS;
	size_t mLoopCount;
	size_t mNextFile;
	
	std::vector<std::string> mFiles;
	std::vector<void*> mBuffers;
};

#endif
