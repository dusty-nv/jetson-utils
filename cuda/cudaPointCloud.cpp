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

#include "mat33.h"


// constructor
cudaPointCloud::cudaPointCloud()
{
	mPointsCPU = NULL;
	mPointsGPU = NULL;

	mNumPoints = 0;
	mMaxPoints = 0;

	mHasRGB    = false;
}


// destructor
cudaPointCloud::~cudaPointCloud()
{
	Free();
}


// Free
void cudaPointCloud::Free()
{
	if( mPointsCPU != NULL )
	{
		CUDA(cudaFreeHost(mPointsCPU));

		mPointsCPU = NULL;
		mPointsGPU = NULL;
		mNumPoints = 0;
		mMaxPoints = 0;
	}
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
	mMaxPoints = maxPoints;
	const size_t maxSize = GetMaxSize();

	// allocate new memory
	if( !cudaAllocMapped((void**)&mPointsCPU, (void**)&mPointsGPU, maxSize) )
	{
		printf(LOG_CUDA "failed to allocate %zu bytes for point cloud\n", maxSize);
		mMaxPoints = 0;
		return false;
	}
	
	return true;
}


// Extract
bool cudaPointCloud::Extract( float* depth, float4* rgba, uint32_t width, uint32_t height )
{
	const float f_w = (float)width;
	const float f_h = (float)height;

	return Extract(depth, rgba, width, height, 
				make_float2(f_h, f_h),
				make_float2(f_w * 0.5f, f_h * 0.5f));
}


// Extract
bool cudaPointCloud::Extract( float* depth, float4* rgba, uint32_t width, uint32_t height,
					 	const float intrinsicCalibration[3][3] )
{
	return Extract(depth, rgba, width, height,
				make_float2(intrinsicCalibration[0][0], intrinsicCalibration[1][1]),
				make_float2(intrinsicCalibration[0][2], intrinsicCalibration[1][2]));
}


// Extract
bool cudaPointCloud::Extract( float* depth, float4* rgba, uint32_t width, uint32_t height,
					 	const char* intrinsicCalibrationPath )
{
	if( !intrinsicCalibrationPath )
		return Extract(depth, rgba, width, height);

	// open the camera calibration file
	FILE* file = fopen(intrinsicCalibrationPath, "r");

	if( !file )
	{
		printf(LOG_CUDA "cudaPointCloud::Extract() -- failed to open calibration file %s\n", intrinsicCalibrationPath);
		return false;
	}
 
	// parse the 3x3 calibration matrix
	float K[3][3];

	for( int n=0; n < 3; n++ )
	{
		char str[512];

		if( !fgets(str, 512, file) )
		{
			printf(LOG_CUDA "cudaPointCloud::Extract() -- failed to read line %i from calibration file %s\n", n+1, intrinsicCalibrationPath);
			return false;
		}

		const int len = strlen(str);

		if( len <= 0 )
		{
			printf(LOG_CUDA "cudaPointCloud::Extract() -- invalid line %i from calibration file %s\n", n+1, intrinsicCalibrationPath);
			return false;
		}

		if( str[len-1] == '\n' )
			str[len-1] = 0;

		if( sscanf(str, "%f %f %f", &K[n][0], &K[n][1], &K[n][2]) != 3 )
		{
			printf(LOG_CUDA "cudaPointCloud::Extract() -- failed to parse line %i from calibration file %s\n", n+1, intrinsicCalibrationPath);
			return false;
		}
	}

	// close the file
	fclose(file);

	// dump the matrix
	printf(LOG_CUDA "cudaPointCloud::Extract() -- loaded intrinsic camera calibration from %s\n", intrinsicCalibrationPath);
	mat33_print(K, "K");

	// proceed with extracting the point cloud
	return Extract(depth, rgba, width, height, K);
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
		printf(LOG_CUDA "cudaPointCloud::Save() -- failed to create %s\n", filename);
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
		float3* point = GetData(n);

		// output XYZ coordinates
		const float3 xyz = point[0];
		fprintf(file, "%f %f %f", xyz.x, xyz.y, xyz.z);

		// output RGB color
		if( mHasRGB )
		{
			const float3 rgb = point[1];

			// pack the color into 24 bits
			const uint32_t rgb_packed = (uint32_t(rgb.x) << 16 |
	      					         uint32_t(rgb.y) << 8 | 
							         uint32_t(rgb.z));
		
			fprintf(file, " %u", rgb_packed);
		}

		fprintf(file, "\n");
	}
		
	fclose(file);
	return true;
}




