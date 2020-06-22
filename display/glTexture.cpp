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

#include "glUtility.h"
#include "glTexture.h"

#include "cudaMappedMemory.h"


// from glBuffer.cpp
cudaGraphicsRegisterFlags cudaGraphicsRegisterFlagsFromGL( uint32_t flags );


//-----------------------------------------------------------------------------------
inline const char* glTextureFormatToStr( uint32_t format )
{
	#define GL_FORMAT_STR(x) case x: return #x

	switch(format)
	{
		GL_FORMAT_STR(GL_LUMINANCE8);
		GL_FORMAT_STR(GL_LUMINANCE16);
		GL_FORMAT_STR(GL_LUMINANCE32UI_EXT);
		GL_FORMAT_STR(GL_LUMINANCE8I_EXT);
		GL_FORMAT_STR(GL_LUMINANCE16I_EXT);
		GL_FORMAT_STR(GL_LUMINANCE32I_EXT);
		GL_FORMAT_STR(GL_LUMINANCE16F_ARB);
		GL_FORMAT_STR(GL_LUMINANCE32F_ARB);

		GL_FORMAT_STR(GL_LUMINANCE8_ALPHA8);
		GL_FORMAT_STR(GL_LUMINANCE16_ALPHA16);
		GL_FORMAT_STR(GL_LUMINANCE_ALPHA32UI_EXT);
		GL_FORMAT_STR(GL_LUMINANCE_ALPHA8I_EXT);
		GL_FORMAT_STR(GL_LUMINANCE_ALPHA16I_EXT);
		GL_FORMAT_STR(GL_LUMINANCE_ALPHA32I_EXT);
		GL_FORMAT_STR(GL_LUMINANCE_ALPHA16F_ARB);
		GL_FORMAT_STR(GL_LUMINANCE_ALPHA32F_ARB);

		GL_FORMAT_STR(GL_RGB8);
		GL_FORMAT_STR(GL_RGB16);
		GL_FORMAT_STR(GL_RGB32UI);
		GL_FORMAT_STR(GL_RGB8I);
		GL_FORMAT_STR(GL_RGB16I);
		GL_FORMAT_STR(GL_RGB32I);
		GL_FORMAT_STR(GL_RGB16F_ARB);

		GL_FORMAT_STR(GL_RGBA8);
		GL_FORMAT_STR(GL_RGBA16);
		GL_FORMAT_STR(GL_RGBA32UI);
		GL_FORMAT_STR(GL_RGBA8I);
		GL_FORMAT_STR(GL_RGBA16I);
		GL_FORMAT_STR(GL_RGBA32I);
		GL_FORMAT_STR(GL_RGBA16F_ARB);

		case GL_RGB32F_ARB:   return "GL_RGB32F";
		case GL_RGBA32F_ARB:  return "GL_RGBA32F";
	}

	return "unknown";
}

inline uint32_t glTextureLayout( uint32_t format )
{
	switch(format)
	{
		case GL_LUMINANCE8:
		case GL_LUMINANCE16:			
		case GL_LUMINANCE32UI_EXT:
		case GL_LUMINANCE8I_EXT:
		case GL_LUMINANCE16I_EXT:
		case GL_LUMINANCE32I_EXT:
		case GL_LUMINANCE16F_ARB:
		case GL_LUMINANCE32F_ARB:		return GL_LUMINANCE;

		case GL_LUMINANCE8_ALPHA8:		
		case GL_LUMINANCE16_ALPHA16:
		case GL_LUMINANCE_ALPHA32UI_EXT:
		case GL_LUMINANCE_ALPHA8I_EXT:
		case GL_LUMINANCE_ALPHA16I_EXT:
		case GL_LUMINANCE_ALPHA32I_EXT:
		case GL_LUMINANCE_ALPHA16F_ARB:
		case GL_LUMINANCE_ALPHA32F_ARB: return GL_LUMINANCE_ALPHA;
		
		case GL_RGB8:					
		case GL_RGB16:
		case GL_RGB32UI:
		case GL_RGB8I:
		case GL_RGB16I:
		case GL_RGB32I:
		case GL_RGB16F_ARB:
		case GL_RGB32F_ARB:				return GL_RGB;

		case GL_RGBA8:
		case GL_RGBA16:
		case GL_RGBA32UI:
		case GL_RGBA8I:
		case GL_RGBA16I:
		case GL_RGBA32I:
		//case GL_RGBA_FLOAT32:
		case GL_RGBA16F_ARB:
		case GL_RGBA32F_ARB:			return GL_RGBA;
	}

	return 0;
}


inline uint32_t glTextureLayoutChannels( uint32_t format )
{
	const uint layout = glTextureLayout(format);

	switch(layout)
	{
		case GL_LUMINANCE:			return 1;
		case GL_LUMINANCE_ALPHA:	return 2;
		case GL_RGB:				return 3;
		case GL_RGBA:				return 4;
	}

	return 0;
}


inline uint32_t glTextureType( uint32_t format )
{
	switch(format)
	{
		case GL_LUMINANCE8:
		case GL_LUMINANCE8_ALPHA8:
		case GL_RGB8:
		case GL_RGBA8:					return GL_UNSIGNED_BYTE;

		case GL_LUMINANCE16:
		case GL_LUMINANCE16_ALPHA16:
		case GL_RGB16:
		case GL_RGBA16:					return GL_UNSIGNED_SHORT;

		case GL_LUMINANCE32UI_EXT:
		case GL_LUMINANCE_ALPHA32UI_EXT:
		case GL_RGB32UI:
		case GL_RGBA32UI:				return GL_UNSIGNED_INT;

		case GL_LUMINANCE8I_EXT:
		case GL_LUMINANCE_ALPHA8I_EXT:
		case GL_RGB8I:
		case GL_RGBA8I:					return GL_BYTE;

		case GL_LUMINANCE16I_EXT:
		case GL_LUMINANCE_ALPHA16I_EXT:
		case GL_RGB16I:
		case GL_RGBA16I:				return GL_SHORT;

		case GL_LUMINANCE32I_EXT:
		case GL_LUMINANCE_ALPHA32I_EXT:
		case GL_RGB32I:
		case GL_RGBA32I:				return GL_INT;


		case GL_LUMINANCE16F_ARB:
		case GL_LUMINANCE_ALPHA16F_ARB:
		case GL_RGB16F_ARB:
		case GL_RGBA16F_ARB:			return GL_FLOAT;

		case GL_LUMINANCE32F_ARB:
		case GL_LUMINANCE_ALPHA32F_ARB:
		//case GL_RGBA_FLOAT32:
		case GL_RGB32F_ARB:
		case GL_RGBA32F_ARB:			return GL_FLOAT;
	}

	return 0;
}


inline uint glTextureTypeSize( uint32_t format )
{
	const uint type = glTextureType(format);

	switch(type)
	{
		case GL_UNSIGNED_BYTE:
		case GL_BYTE:					return 1;

		case GL_UNSIGNED_SHORT:
		case GL_SHORT:					return 2;

		case GL_UNSIGNED_INT:
		case GL_INT:
		case GL_FLOAT:					return 4;
	}

	return 0;
}
//-----------------------------------------------------------------------------------

// constructor
glTexture::glTexture()
{
	mID     = 0;
	mWidth  = 0;
	mHeight = 0;
	mFormat = 0;
	mSize   = 0;

	mPackDMA   = 0;
	mUnpackDMA = 0;

	mMapDevice = 0;
	mMapFlags  = 0;

	mInteropPack   = NULL;
	mInteropUnpack = NULL;
}


// destructor
glTexture::~glTexture()
{
	if( mInteropPack != NULL )
	{
		CUDA(cudaGraphicsUnregisterResource(mInteropPack));
		mInteropPack = NULL;
	}

	if( mInteropUnpack != NULL )
	{
		CUDA(cudaGraphicsUnregisterResource(mInteropUnpack));
		mInteropUnpack = NULL;
	}

	if( mPackDMA != 0 )
	{
		GL(glDeleteBuffers(1, &mPackDMA));
		mPackDMA = 0;
	}

	if( mUnpackDMA != 0 )
	{
		GL(glDeleteBuffers(1, &mUnpackDMA));
		mUnpackDMA = 0;
	}

	if( mID != 0 )
	{
		GL(glDeleteTextures(1, &mID));
		mID = 0;
	}
}
	

// Create
glTexture* glTexture::Create( uint32_t width, uint32_t height, uint32_t format, void* data )
{
	glTexture* tex = new glTexture();
	
	if( !tex->init(width, height, format, data) )
	{
		LogError(LOG_GL "failed to create %ux%u texture (%s)\n", width, height, glTextureFormatToStr(format));
		return NULL;
	}
	
	return tex;
}
		
		
// Alloc
bool glTexture::init( uint32_t width, uint32_t height, uint32_t format, void* data )
{
	const uint32_t size = width * height * glTextureLayoutChannels(format) * glTextureTypeSize(format);

	if( size == 0 )
		return NULL;
		
	// generate texture objects
	uint32_t id = 0;
	
	GL(glEnable(GL_TEXTURE_2D));
	GL(glGenTextures(1, &id));
	GL(glBindTexture(GL_TEXTURE_2D, id));
	
	// set default texture parameters
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));

	LogVerbose(LOG_GL "creating %ux%u texture (%s format, %u bytes)\n", width, height, glTextureFormatToStr(format), size);
	
	// allocate texture
	GL_VERIFY(glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, glTextureLayout(format), glTextureType(format), data));
	
	mID     = id;
	mWidth  = width;
	mHeight = height;
	mFormat = format;
	mSize   = size;

	GL(glBindTexture(GL_TEXTURE_2D, 0));
	GL(glDisable(GL_TEXTURE_2D));

	return true;
}


// allocDMA
uint32_t glTexture::allocDMA( uint32_t type )
{
	if( type == GL_PIXEL_PACK_BUFFER_ARB && mPackDMA != 0 )
		return mPackDMA;
	else if( type == GL_PIXEL_UNPACK_BUFFER_ARB && mUnpackDMA != 0 )
		return mUnpackDMA;
	
	// allocate PBO
	uint32_t dma = 0;
	
	GL_VERIFY(glGenBuffers(1, &dma));
	GL_VERIFY(glBindBufferARB(type, dma));
	GL_VERIFY(glBufferDataARB(type, mSize, NULL, GL_DYNAMIC_DRAW_ARB));
	GL_VERIFY(glBindBufferARB(type, 0));

	if( type == GL_PIXEL_PACK_BUFFER_ARB )
		mPackDMA = dma;
	else if( type == GL_PIXEL_UNPACK_BUFFER_ARB )
		mUnpackDMA = dma;

	return dma;
}
	

// allocInterop
cudaGraphicsResource* glTexture::allocInterop( uint32_t type, uint32_t flags )
{
	if( type == GL_PIXEL_PACK_BUFFER_ARB && mInteropPack != NULL )
		return mInteropPack;
	else if( type == GL_PIXEL_UNPACK_BUFFER_ARB && mInteropUnpack != NULL )
		return mInteropUnpack;

	cudaGraphicsResource* interop = NULL;

	if( CUDA_FAILED(cudaGraphicsGLRegisterBuffer(&interop, allocDMA(type), cudaGraphicsRegisterFlagsFromGL(flags))) )
		return NULL;

	if( type == GL_PIXEL_PACK_BUFFER_ARB )
		mInteropPack = interop;
	else if( type == GL_PIXEL_UNPACK_BUFFER_ARB )
		mInteropUnpack = interop;

	LogVerbose(LOG_CUDA "registered openGL texture for interop access (%ux%u, %s, %u bytes)\n", mWidth, mHeight, glTextureFormatToStr(mFormat), mSize);
	return interop;
}


// Bind
bool glTexture::Bind()
{
	if( !mID )
		return false;

	GL_VERIFY(glEnable(GL_TEXTURE_2D));
	GL_VERIFY(glActiveTextureARB(GL_TEXTURE0_ARB));
	GL_VERIFY(glBindTexture(GL_TEXTURE_2D, mID));

	return true;
}


// Unbind
void glTexture::Unbind()
{
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
}


// Render
void glTexture::Render( const float4& rect )
{
	if( !Bind() )
		return;

	glBegin(GL_QUADS);

		glColor4f(1.0f,1.0f,1.0f,1.0f);

		glTexCoord2f(0.0f, 0.0f); 
		glVertex2f(rect.x, rect.y);

		glTexCoord2f(1.0f, 0.0f); 
		glVertex2f(rect.z, rect.y);	

		glTexCoord2f(1.0f, 1.0f); 
		glVertex2f(rect.z, rect.w);

		glTexCoord2f(0.0f, 1.0f); 
		glVertex2f(rect.x, rect.w);

	glEnd();
	Unbind();
}


// Render
void glTexture::Render( float x, float y )
{
	Render(x, y, mWidth, mHeight);
}


// Render
void glTexture::Render( float x, float y, float width, float height )
{
	Render(make_float4(x, y, x + width, y + height));
}



// dmaTypeFromFlags
static inline uint32_t dmaTypeFromFlags( uint32_t flags )
{
	if( flags == GL_READ_ONLY )
		return GL_PIXEL_PACK_BUFFER_ARB;
	else
		return GL_PIXEL_UNPACK_BUFFER_ARB;	// GL_READ_WRITE?
}


// Map
void* glTexture::Map( uint32_t device, uint32_t flags )
{
	if( mMapDevice != 0 )
	{
		LogError(LOG_GL "error -- glTexture is already mapped (call Unmap() first)\n");
		return NULL;
	}

	if( device != GL_MAP_CPU && device != GL_MAP_CUDA )
	{
		LogError(LOG_GL "glTexture::Map() -- invalid device (must be GL_MAP_CPU or GL_MAP_CUDA)\n");
		return NULL;
	}

	if( !Bind() )
		return NULL;

	// make sure DMA buffer is allocated	
	void* dmaPtr = NULL;

	const uint32_t dmaType = dmaTypeFromFlags(flags);
	const uint32_t dmaBuffer = allocDMA(dmaType);

	if( !dmaBuffer )
		return NULL;

	// set pixel alignment flags (default is 4, but sometimes fails on rgb8 depending on resolution)
	GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
	//GL(glPixelStorei(GL_UNPACK_ROW_LENGTH, mWidth));
	//GL(glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, mHeight));

	// either map in CPU mode or with CUDA interop
	if( device == GL_MAP_CPU )
	{
		GL(glBindBuffer(dmaType, dmaBuffer));

		if( flags == GL_WRITE_DISCARD )
		{
			// invalidate the old buffer so we can map without stalling
			GL(glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, mSize, NULL, GL_STREAM_DRAW_ARB));	
			flags = GL_WRITE_ONLY; // GL expects GL_WRITE_ONLY
		}
		else if( flags == GL_READ_ONLY )
		{
			// read data from texture to PBO
			GL(glReadPixels(0, 0, mWidth, mHeight, glTextureLayout(mFormat), glTextureType(mFormat), NULL));
		}

		// lock the PBO buffer
		dmaPtr = glMapBuffer(dmaType, flags);

		if( !dmaPtr )
		{
			LogError(LOG_GL "glMapBuffer() failed\n");
			GL_CHECK("glMapBuffer()\n");
			return NULL;
		}
	}
	else if( device == GL_MAP_CUDA )
	{
		if( flags == GL_READ_ONLY )
		{
			GL(glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, mPackDMA));
			GL(glReadPixels(0, 0, mWidth, mHeight, glTextureLayout(mFormat), glTextureType(mFormat), NULL));
			GL(glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0));
		}

		// make sure CUDA resource is registered
		cudaGraphicsResource* interop = allocInterop(dmaType, flags);

		if( !interop )
			return NULL;

		if( mMapFlags != 0 && mMapFlags != flags )	// TODO two buffers, but one set of flags
			CUDA(cudaGraphicsResourceSetMapFlags(interop, cudaGraphicsRegisterFlagsFromGL(flags)));

		if( CUDA_FAILED(cudaGraphicsMapResources(1, &interop)) )
			return NULL;

		// map CUDA device pointer
		size_t mappedSize = 0;

		if( CUDA_FAILED(cudaGraphicsResourceGetMappedPointer(&dmaPtr, &mappedSize, interop)) )
		{
			CUDA(cudaGraphicsUnmapResources(1, &interop));
			return NULL;
		}
		
		if( mSize != mappedSize )
			LogError(LOG_GL "glTexture::Map() -- CUDA size mismatch %zu bytes  (expected=%u)\n", mappedSize, mSize);
	}

	mMapDevice = device;
	mMapFlags  = flags;

	return dmaPtr;
}


// Unmap
void glTexture::Unmap()
{
	if( mMapDevice != GL_MAP_CPU && mMapDevice != GL_MAP_CUDA )
		return;

	if( !Bind() )
		return;

	const uint32_t dmaType = dmaTypeFromFlags(mMapFlags);

	if( mMapDevice == GL_MAP_CPU )
	{
		GL(glUnmapBuffer(dmaType));

		if( mMapFlags != GL_READ_ONLY )
			GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, glTextureLayout(mFormat), glTextureType(mFormat), NULL));

		GL(glBindBuffer(dmaType, 0));
	}
	else if( mMapDevice == GL_MAP_CUDA )
	{
		cudaGraphicsResource* interop = allocInterop(dmaType, mMapFlags);

		if( !interop )
			return;

		CUDA(cudaGraphicsUnmapResources(1, &interop));

		if( mMapFlags != GL_READ_ONLY )
		{
			GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mUnpackDMA));
			GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, glTextureLayout(mFormat), glTextureType(mFormat), NULL));
			GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0));	
		}
	}

	mMapDevice = 0;
	Unbind();
}


// Copy
bool glTexture::Copy( void* ptr, uint32_t offset, uint32_t size, uint32_t flags )
{
	if( !ptr || size == 0 || size >= mSize || offset >= mSize || offset > (mSize - size) )
		return false;

	uint32_t mapFlags = GL_READ_ONLY;

	if( flags == GL_FROM_CPU || flags == GL_FROM_CUDA )
	{
		if( size == mSize )
			mapFlags = GL_WRITE_DISCARD;
		else
			mapFlags = GL_WRITE_ONLY;
	}
	
	if( flags == GL_FROM_CPU )
	{
		// TODO for faster CPU path, see http://hacksoflife.blogspot.com/2015/06/glmapbuffer-no-longer-cool.html
		void* dst = Map(GL_MAP_CPU, mapFlags);

		if( !dst )
			return false;

		memcpy((uint8_t*)dst + offset, ptr, size);
	}
	else if( flags == GL_FROM_CUDA )
	{
		void* dst = Map(GL_MAP_CUDA, mapFlags);

		if( !dst )
			return false;

		if( CUDA_FAILED(cudaMemcpy((uint8_t*)dst + offset, ptr, size, cudaMemcpyDeviceToDevice)) )
		{
			Unmap();
			return false;
		}
	}
	else if( flags == GL_TO_CPU )
	{
		void* src = Map(GL_MAP_CPU, mapFlags);

		if( !src )
			return false;
	
		memcpy(ptr, (uint8_t*)src + offset, size);
	}
	else if( flags == GL_TO_CUDA )
	{
		void* src = Map(GL_MAP_CUDA, mapFlags);

		if( !src )
			return false;
	
		if( CUDA_FAILED(cudaMemcpy(ptr, (uint8_t*)src + offset, size, cudaMemcpyDeviceToDevice)) )
		{
			Unmap();
			return false;
		}
	}

	Unmap();
	return true;
}

// Copy
bool glTexture::Copy( void* ptr, uint32_t size, uint32_t flags )
{
	return Copy(ptr, 0, mSize, flags);
}

// Copy
bool glTexture::Copy( void* ptr, uint32_t flags )
{
	return Copy(ptr, 0, mSize, flags);
}

#if 0
// MapCUDA
void* glTexture::MapCUDA()
{
	if( !mInteropCUDA )
	{
		if( CUDA_FAILED(cudaGraphicsGLRegisterBuffer(&mInteropCUDA, mDMA, cudaGraphicsRegisterFlagsWriteDiscard)) )
			return NULL;

		printf(LOG_CUDA "registered %u byte openGL texture for interop access (%ux%u)\n", mSize, mWidth, mHeight);
	}
	
	if( CUDA_FAILED(cudaGraphicsMapResources(1, &mInteropCUDA)) )
		return NULL;
	
	void*  devPtr     = NULL;
	size_t mappedSize = 0;

	if( CUDA_FAILED(cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedSize, mInteropCUDA)) )
	{
		CUDA(cudaGraphicsUnmapResources(1, &mInteropCUDA));
		return NULL;
	}
	
	if( mSize != mappedSize )
		printf(LOG_GL "glTexture::MapCUDA() -- size mismatch %zu bytes  (expected=%u)\n", mappedSize, mSize);
		
	return devPtr;
}


// Unmap
void glTexture::Unmap()
{
	if( !mInteropCUDA )
		return;
		
	CUDA(cudaGraphicsUnmapResources(1, &mInteropCUDA));
	
	GL(glEnable(GL_TEXTURE_2D));
	GL(glBindTexture(GL_TEXTURE_2D, mID));
	GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mDMA));
	GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, glTextureLayout(mFormat), glTextureType(mFormat), NULL));
	
	GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0));
	GL(glBindTexture(GL_TEXTURE_2D, 0));
	GL(glDisable(GL_TEXTURE_2D));
}


// Upload
bool glTexture::UploadCPU( void* data )
{
	// activate texture & pbo
	GL(glEnable(GL_TEXTURE_2D));
	GL(glActiveTextureARB(GL_TEXTURE0_ARB));
	GL(glBindTexture(GL_TEXTURE_2D, mID));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0));
	GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mDMA));

	//GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
	//GL(glPixelStorei(GL_UNPACK_ROW_LENGTH, img->GetWidth()));
	//GL(glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, img->GetHeight()));

	// hint to driver to double-buffer
	// glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, mImage->GetSize(), NULL, GL_STREAM_DRAW_ARB);	

	// map PBO
	GLubyte* ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	        
	if( !ptr )
	{
		GL_CHECK("glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB)");
		return NULL;
	}

	memcpy(ptr, data, mSize);

	GL(glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB)); 

	//GL(glEnable(GL_TEXTURE_2D));
	//GL(glBindTexture(GL_TEXTURE_2D, mID));
	//GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mDMA));
	GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, glTextureLayout(mFormat), glTextureType(mFormat), NULL));
	
	GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0));
	GL(glBindTexture(GL_TEXTURE_2D, 0));
	GL(glDisable(GL_TEXTURE_2D));

	/*if( !mInteropHost || !mInteropDevice )
	{
		if( !cudaAllocMapped(&mInteropHost, &mInteropDevice, mSize) )
			return false;
	}
	
	memcpy(mInteropHost, data, mSize);
	
	void* devGL = MapCUDA();
	
	if( !devGL )
		return false;
		
	CUDA(cudaMemcpy(devGL, mInteropDevice, mSize, cudaMemcpyDeviceToDevice));
	Unmap();*/

	return true;
}
#endif


