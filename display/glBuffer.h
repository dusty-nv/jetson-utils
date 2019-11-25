/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __GL_BUFFER_H__
#define __GL_BUFFER_H__


#include "cudaUtility.h"
#include "cuda_gl_interop.h"


/**
 * Alias for vertex buffers
 * @ingroup OpenGL
 */
#define GL_VERTEX_BUFFER 	GL_ARRAY_BUFFER

/**
 * Alias for index buffers
 * @ingroup OpenGL
 */
#define GL_INDEX_BUFFER 		GL_ELEMENT_ARRAY_BUFFER

/**
 * Map the buffer to CPU address space
 * @ingroup OpenGL
 */
#define GL_MAP_CPU 			0x1

/**
 * Map the buffer to CUDA address space
 * @ingroup OpenGL
 */
#define GL_MAP_CUDA 		0x2

/**
 * Copy the buffer from CPU to OpenGL
 * @ingroup OpenGL
 */
#define GL_FROM_CPU			0x3

/**
 * Copy the buffer from CUDA to OpenGL
 * @ingroup OpenGL
 */
#define GL_FROM_CUDA		0x4

/**
 * Copy the buffer to CPU from OpenGL
 * @ingroup OpenGL
 */
#define GL_TO_CPU			0x5

/**
 * Copy the buffer to CUDA from OpenGL
 * @ingroup OpenGL
 */
#define GL_TO_CUDA			0x6

/**
 * Map the buffer as write-only and discard previous contents
 * @ingroup OpenGL
 */
#define GL_WRITE_DISCARD 	(GL_READ_WRITE + 0xff) 

/**
 * Map the buffer with write-only access
 * @ingroup OpenGL
 */
#ifndef GL_WRITE_ONLY
#define GL_WRITE_ONLY 		GL_WRITE_ONLY_ARB
#endif

/**
 * Map the buffer with read-only access
 * @ingroup OpenGL
 */
#ifndef GL_READ_ONLY
#define GL_READ_ONLY 		GL_READ_ONLY_ARB
#endif

/**
 * Map the buffer with read/write access
 * @ingroup OpenGL
 */
#ifndef GL_READ_WRITE
#define GL_READ_WRITE 		GL_READ_WRITE_ARB
#endif


/**
 * OpenGL buffer with CUDA interoperability.
 * @ingroup OpenGL
 */
class glBuffer
{
public:
	/**
	 * Allocate an OpenGL buffer.
	 * @param type either GL_VERTEX_BUFFER for a vertex buffer, or GL_INDEX_BUFFER for an index buffer
	 * @param size the size in bytes to allocated for the buffer
	 * @param data pointer to the initial memory of this buffer
	 * @param usage GL_STATIC_DRAW (never updated), GL_STREAM_DRAW (occasional updates), or GL_DYNAMIC_DRAW (per-frame updates)
	 */
	static glBuffer* Create( uint32_t type, uint32_t size, void* data=NULL, uint32_t usage=GL_STATIC_DRAW );
	
	/**
	 * Allocate an OpenGL buffer.
	 * @param type either GL_VERTEX_BUFFER for a vertex buffer, or GL_INDEX_BUFFER for an index buffer
	 * @param numElements the number of elements (i.e. vertices or indices) in the buffer
	 * @param elementSize the size in bytes of each element
	 * @param data pointer to the initial memory of this buffer
	 * @param usage GL_STATIC_DRAW (never updated), GL_STREAM_DRAW (occasional updates), or GL_DYNAMIC_DRAW (per-frame updates)
	 */
	static glBuffer* Create( uint32_t type, uint32_t numElements, uint32_t elementSize, void* data=NULL, uint32_t usage=GL_STATIC_DRAW );
	
	/**
	 * Free the buffer
	 */
	~glBuffer();

	/**
	 * Activate using the buffer
	 */
	bool Bind();

	/**
	 * Deactivate using the buffer
	 */
	void Unbind();
	
	/**
	 * Retrieve the OpenGL resource handle of the buffer.
	 */
	inline uint32_t GetID() const			{ return mID; }

	/**
	 * Retrieve the buffer type (GL_VERTEX_BUFFER or GL_INDEX_BUFFER)
	 */
	inline uint32_t GetType() const		{ return mType; }

	/**
	 * Retrieve the total size in bytes of the buffer
	 */
	inline uint32_t GetSize() const		{ return mSize; }

	/**
	 * Retrieve the number of elements (i.e. vertices or indices)
	 *
	 * @note if the number of elements weren't specified while creating 
	 *       the buffer, GetNumElements() will be equal to GetSize()
	 */
	inline uint32_t GetNumElements() const	{ return mNumElements; }

	/**
	 * Retrieve the size in bytes of each element
	 *
	 * @note if the element size wasn't specified while creating 
	 *       the buffer, GetElementSize() will be equal to 1
	 */
	inline uint32_t GetElementSize() const	{ return mElementSize; }

	/**
	 * Map the buffer for accessing from the CPU or CUDA.
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
	 * Unmap the buffer from CPU/CUDA access.
	 * @note the buffer will be unbound after calling Unmap()
	 */
	void Unmap();

	/**
	 * Copy entire contents of the buffer to/from CPU or CUDA memory.
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
	 * Copy contents of the buffer to/from CPU or CUDA memory.
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
	 * Copy contents of the buffer to/from CPU or CUDA memory.
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
	glBuffer();

	bool init( uint32_t type, uint32_t size, void* data, uint32_t usage);
	
	uint32_t mID;
	uint32_t mSize;
	uint32_t mType;
	uint32_t mUsage;

	uint32_t mNumElements;
	uint32_t mElementSize;

	uint32_t mMapDevice;
	uint32_t mMapFlags;

	cudaGraphicsResource* mInteropCUDA;
};


#endif
