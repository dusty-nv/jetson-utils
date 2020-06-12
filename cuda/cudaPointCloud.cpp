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

#include "cudaPointCloud.h"
#include "cudaMappedMemory.h"

#include "glUtility.h"
#include "glBuffer.h"
#include "glCamera.h"

#include "mat33.h"
#include "logging.h"


// constructor
cudaPointCloud::cudaPointCloud()
{
	mPointsCPU   = NULL;
	mPointsGPU   = NULL;
	mBufferGL    = NULL;
	mCameraGL    = NULL;
	mDepthResize = NULL;

	mDepthSize = 0;
	mNumPoints = 0;
	mMaxPoints = 0;

	mHasRGB         = false;
	mHasNewPoints	 = false;
	mHasCalibration = false;
}


// destructor
cudaPointCloud::~cudaPointCloud()
{
	if( mDepthResize != NULL )
	{
		CUDA(cudaFree(mDepthResize));
		mDepthResize = NULL;
	}

	if( mCameraGL != NULL )
	{
		delete mCameraGL;
		mCameraGL = NULL;
	}

	Free();
}


// Free
void cudaPointCloud::Free()
{
	if( mBufferGL != NULL )
	{
		delete mBufferGL;
		mBufferGL = NULL;
	}

	if( mPointsCPU != NULL )
	{
		CUDA(cudaFreeHost(mPointsCPU));

		mPointsCPU = NULL;
		mPointsGPU = NULL;
		mNumPoints = 0;
		mMaxPoints = 0;
	}
}


// Clear
void cudaPointCloud::Clear()
{
	if( mPointsGPU != NULL )
		CUDA(cudaMemset(mPointsGPU, 0, GetMaxSize()));

	mNumPoints = 0;
}


// Create
cudaPointCloud* cudaPointCloud::Create()
{
	return new cudaPointCloud();
}


// Reserve
bool cudaPointCloud::Reserve( uint32_t maxPoints )
{
	if( maxPoints == 0 )
		return false;

	// check if enough memory is already allocated
	if( maxPoints <= mMaxPoints && mPointsCPU != NULL )
		return true;

	// release old memory
	Free();

	// determine size of new memory
	const size_t maxSize = maxPoints * sizeof(Vertex);

	// allocate new memory
	if( !cudaAllocMapped((void**)&mPointsCPU, (void**)&mPointsGPU, maxSize) )
	{
		LogError(LOG_CUDA "failed to allocate %zu bytes for point cloud\n", maxSize);
		return false;
	}
	
	mMaxPoints = maxPoints;
	return true;
}


// allocBufferGL
bool cudaPointCloud::allocBufferGL()
{
	if( mBufferGL != NULL && mBufferGL->GetSize() == GetMaxSize() )
		return true;

	if( mBufferGL != NULL )
	{
		delete mBufferGL;
		mBufferGL = NULL;
	}

	mBufferGL = glBuffer::Create(GL_VERTEX_BUFFER, GetMaxPoints(), sizeof(Vertex), NULL, GL_DYNAMIC_DRAW);

	if( !mBufferGL )
		return false;

	return true;
}


// allocDepthResize
bool cudaPointCloud::allocDepthResize( size_t size )
{
	if( size == 0 )
		return false;

	if( mDepthResize != NULL && mDepthSize == size )
		return true;

	if( mDepthResize != NULL )
	{
		CUDA(cudaFree(mDepthResize));
		mDepthResize = NULL;
	}

	if( CUDA_FAILED(cudaMalloc(&mDepthResize, size)) )
		return false;

	mDepthSize = size;
	return true;
}


// Render
bool cudaPointCloud::Render()
{
	if( mNumPoints == 0 )
		return false;

	// make sure GL buffer is ready
	if( !allocBufferGL() )
		return false;

	// make sure camera is ready
	if( !mCameraGL )
	{
		mCameraGL = glCamera::Create(glCamera::YawPitchRoll);

		if( !mCameraGL )
			return false;

		mCameraGL->SetEye(0.0f, 5.0f, 5.0f);
		mCameraGL->StoreDefaults();
	}

	// copy to OpenGL if needed
	if( mHasNewPoints )
	{
		void* ptr = mBufferGL->Map(GL_MAP_CUDA, GL_WRITE_DISCARD);

		if( !ptr )
			return false;

		CUDA(cudaMemcpy(ptr, mPointsGPU, GetSize(), cudaMemcpyDeviceToDevice));
		
		mBufferGL->Unmap();		
		mHasNewPoints = false;
	}

	// enable the camera and buffer
	mCameraGL->Activate();
	mBufferGL->Bind();

	GL(glEnableClientState(GL_VERTEX_ARRAY));
	GL(glVertexPointer(3, GL_FLOAT, sizeof(Vertex), 0));

	GL(glEnableClientState(GL_COLOR_ARRAY));
	GL(glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(Vertex), (void*)offsetof(Vertex, color)));

	// draw the points
	GL(glDrawArrays(GL_POINTS, 0, mNumPoints));

	// disable the buffer and camera
	GL(glDisableClientState(GL_COLOR_ARRAY));
	GL(glDisableClientState(GL_VERTEX_ARRAY));

	mBufferGL->Unbind();
	mCameraGL->Deactivate();
	
	return true;
}


// SetCalibration
void cudaPointCloud::SetCalibration( const float2& focalLength, const float2& principalPoint )
{
	mFocalLength 	 = focalLength;
	mPrincipalPoint = principalPoint;
	mHasCalibration = true;
}


// SetCalibration
void cudaPointCloud::SetCalibration( const float K[3][3] )
{
	SetCalibration(make_float2(K[0][0], K[1][1]), make_float2(K[0][2], K[1][2]));
}


// SetCalibration
bool cudaPointCloud::SetCalibration( const char* filename )
{
	if( !filename )
		return false;

	// open the camera calibration file
	FILE* file = fopen(filename, "r");

	if( !file )
	{
		LogError(LOG_CUDA "cudaPointCloud::Extract() -- failed to open calibration file %s\n", filename);
		return false;
	}
 
	// parse the 3x3 calibration matrix
	float K[3][3];

	for( int n=0; n < 3; n++ )
	{
		char str[512];

		if( !fgets(str, 512, file) )
		{
			LogError(LOG_CUDA "cudaPointCloud::Extract() -- failed to read line %i from calibration file %s\n", n+1, filename);
			return false;
		}

		const int len = strlen(str);

		if( len <= 0 )
		{
			LogError(LOG_CUDA "cudaPointCloud::Extract() -- invalid line %i from calibration file %s\n", n+1, filename);
			return false;
		}

		if( str[len-1] == '\n' )
			str[len-1] = 0;

		if( sscanf(str, "%f %f %f", &K[n][0], &K[n][1], &K[n][2]) != 3 )
		{
			LogError(LOG_CUDA "cudaPointCloud::Extract() -- failed to parse line %i from calibration file %s\n", n+1, filename);
			return false;
		}
	}

	// close the file
	fclose(file);

	// dump the matrix
	LogVerbose(LOG_CUDA "cudaPointCloud::Extract() -- loaded intrinsic camera calibration from %s\n", filename);
	mat33_print(K, "K");

	// save the calibration
	SetCalibration(K);
	return true;
}


// Save
bool cudaPointCloud::Save( const char* filename )
{
	if( !filename || mNumPoints == 0 || !mPointsCPU )
		return false;

	// open the PCD file
	FILE* file = fopen(filename, "w");

	if( !file )
	{
		LogError(LOG_CUDA "cudaPointCloud::Save() -- failed to create %s\n", filename);
		return false;
	}

	// write the PCD header
	fprintf(file, "# .PCD v0.7 - Point Cloud Data file format\n");
	fprintf(file, "VERSION 0.7\n");

	if( mHasRGB )
	{
		fprintf(file, "FIELDS x y z rgb\n");
		fprintf(file, "SIZE 4 4 4 4\n");
		fprintf(file, "TYPE F F F U\n");
	}
	else
	{
		fprintf(file, "FIELDS x y z\n");
		fprintf(file, "SIZE 4 4 4\n");
		fprintf(file, "TYPE F F F\n");
	}

	fprintf(file, "COUNT 1 1 1 1\n");
	fprintf(file, "WIDTH %u\n", mNumPoints);
	fprintf(file, "HEIGHT 1\n");
	fprintf(file, "VIEWPOINT 0 0 0 1 0 0 0\n");
	fprintf(file, "POINTS %u\n", mNumPoints);
	fprintf(file, "DATA ascii\n");

	// wait for the GPU to finish any processing
	CUDA(cudaDeviceSynchronize());

	// write out points to the PCD file
	for( size_t n=0; n < mNumPoints; n++ )
	{
		Vertex* point = GetData(n);

		// output XYZ coordinates
		fprintf(file, "%f %f %f", point->pos.x, point->pos.y, point->pos.z);

		// output RGB color
		if( mHasRGB )
		{
			// pack the color into 24 bits
			const uint32_t rgb_packed = (uint32_t(point->color.x) << 16 |
	      					         uint32_t(point->color.y) << 8 | 
							         uint32_t(point->color.z));
		
			fprintf(file, " %u", rgb_packed);
		}

		fprintf(file, "\n");
	}
		
	fclose(file);
	return true;
}




