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
 
#ifndef __GL_TEXTURE_H__
#define __GL_TEXTURE_H__


#include "glBuffer.h"


/**
 * Alias for GL_RGB32F
 * @ingroup OpenGL
 */
#ifndef GL_RGB32F
#define GL_RGB32F GL_RGB32F_ARB
#endif

/**
 * Alias for GL_RGBA32F
 * @ingroup OpenGL
 */
#ifndef GL_RGBA32F
#define GL_RGBA32F GL_RGBA32F_ARB
#endif


/**
 * OpenGL texture with CUDA interoperability.
 * @ingroup OpenGL
 */
class glTexture
{
public:
	/**
	 * Allocate an OpenGL texture
	 * @param width the width of the texture in pixels
	 * @param height the height of the texture in pixels
	 * @param format GL_RGBA8, GL_RGBA32F, ect.
	 * @param data initialize the texture's memory with this CPU pointer, size is width*height*bpp
	 */
	static glTexture* Create( uint32_t width, uint32_t height, uint32_t format, void* data=NULL );
	
	/**
	 * Free the texture
	 */
	~glTexture();
	
	/**
	 * Activate using the texture
	 */
	bool Bind();

	/**
	 * Deactivate using the texture
	 */
	void Unbind();

	/**
	 * Render the texture at the specified window coordinates.
	 */
	void Render( float x, float y );

	/**
	 * Render the texture with the specified position and size.
	 */
	void Render( float x, float y, float width, float height );

	/**
	 * Render the texture to the specific screen rectangle.
	 */
	void Render( const float4& rect );
	
	/**
	 * Retrieve the OpenGL resource handle of the texture.
	 */
	inline uint32_t GetID() const		{ return mID; }

	/**
	 * Retrieve the width in pixels of the texture.
	 */
	inline uint32_t GetWidth() const	{ return mWidth; }

	/**
	 * Retrieve the height in pixels of the texture.
	 */
	inline uint32_t GetHeight() const	{ return mHeight; }

	/**
	 * Retrieve the texture's format (e.g. GL_RGBA8, GL_RGBA32F, ect.)
	 */
	inline uint32_t GetFormat() const	{ return mFormat; }

	/**
	 * Retrieve the size in bytes of the texture
	 */
	inline uint32_t GetSize() const	{ return mSize; }
	
	/**
	 * Map the texture for accessing from the CPU or CUDA.
	 *
	 * @param device either GL_MAP_CPU or GL_MAP_CUDA
	 *
	 * @param flags should be one of the following:
	 *                 - GL_READ_WRITE
	 *                 - GL_READ_ONLY
	 *                 - GL_WRITE_ONLY
	 *                 - GL_WRITE_DISCARD
	 *
	 * @returns CPU pointer to buffer if GL_MAP_CPU was specified,
	 *          CUDA device pointer to buffer if GL_MAP_CUDA was specified,
	 *          or NULL if an error occurred mapping the buffer.                  
	 */
	void* Map( uint32_t device, uint32_t flags );

	/**
	 * Unmap the texture from CPU/CUDA access.
	 * @note the texture will be unbound after calling Unmap()
	 */
	void Unmap();

	/**
	 * Copy entire contents of the texture to/from CPU or CUDA memory.
	 *
	 * @param ptr the memory pointer to copy to/from, either in
	 *            CPU or CUDA address space depending on flags.
	 *            It's assumed that the size of the memory from
	 *            this pointer is equal to GetSize(), and the
	 *            entire contents of the buffer will be copied.
	 *
	 * @param flags should be one of the following:
	 *           - GL_FROM_CPU  (copy from CPU->OpenGL)
	 *           - GL_FROM_CUDA (copy from CUDA->OpenGL)
	 *           - GL_TO_CPU    (copy from OpenGL->CPU)
	 *           - GL_TO_CUDA   (copy from OpenGL->CUDA)
	 *
	 * @returns true on success, false on failure
	 */
	bool Copy( void* ptr, uint32_t flags );

	/**
	 * Copy contents of the texture to/from CPU or CUDA memory.
	 *
	 * @param ptr the memory pointer to copy to/from, either in
	 *            CPU or CUDA address space depending on flags.
	 *
	 * @param size the number of bytes to copy
	 * 
	 * @param flags should be one of the following:
	 *           - GL_FROM_CPU  (copy from CPU->OpenGL)
	 *           - GL_FROM_CUDA (copy from CUDA->OpenGL)
	 *           - GL_TO_CPU    (copy from OpenGL->CPU)
	 *           - GL_TO_CUDA   (copy from OpenGL->CUDA)
	 *
	 * @returns true on success, false on failure
	 */
	bool Copy( void* ptr, uint32_t size, uint32_t flags );

	/**
	 * Copy contents of the texture to/from CPU or CUDA memory.
	 *
	 * @param ptr the memory buffer to copy to/from, either in
	 *            CPU or CUDA address space depending on flags
	 *
	 * @param offset the offset into the OpenGL buffer to copy.
	 *               It is assumed any offset to the CPU/CUDA 
	 *               pointer argument has already been applied.
	 *
	 * @param size the number of bytes to copy
	 * 
	 * @param flags should be one of the following:
	 *           - GL_FROM_CPU  (copy from CPU->OpenGL)
	 *           - GL_FROM_CUDA (copy from CUDA->OpenGL)
	 *           - GL_TO_CPU    (copy from OpenGL->CPU)
	 *           - GL_TO_CUDA   (copy from OpenGL->CUDA)
	 *
	 * @returns true on success, false on failure
	 */
	bool Copy( void* ptr, uint32_t offset, uint32_t size, uint32_t flags );

	
private:
	glTexture();

	bool init( uint32_t width, uint32_t height, uint32_t format, void* data );

	uint32_t allocDMA( uint32_t type );
	cudaGraphicsResource* allocInterop( uint32_t type, uint32_t flags );

	uint32_t mID;
	uint32_t mPackDMA;
	uint32_t mUnpackDMA;

	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mFormat;
	uint32_t mSize;

	uint32_t mMapDevice;
	uint32_t mMapFlags;

	cudaGraphicsResource* mInteropPack;
	cudaGraphicsResource* mInteropUnpack;
};


#endif
