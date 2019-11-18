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

#ifndef __CUDA_POINT_CLOUD_H__
#define __CUDA_POINT_CLOUD_H__


#include "cudaUtility.h"


/**
 * CUDA-accelerated point cloud processing.
 * @ingroup cuda
 */
class cudaPointCloud
{
public:
	/**
	 * Destructor
	 */
	~cudaPointCloud();
	
	/**
	 * Extract point cloud from depth map and optional RGBA image.
	 */
	bool Extract( float* depth, float4* rgba, uint32_t width, uint32_t height );

	/**
	 * Extract point cloud from depth map and optional RGBA image.
	 */
	bool Extract( float* depth, float4* rgba, uint32_t width, uint32_t height,
			    const float2& focalLength, const float2& principalPoint );

	/**
	 * Extract point cloud from depth map and optional RGBA image.
	 */
	bool Extract( float* depth, float4* rgba, uint32_t width, uint32_t height,
			    const float intrinsicCalibration[3][3] );

	/**
	 * Extract point cloud from depth map and optional RGBA image.
	 */
	bool Extract( float* depth, float4* rgba, uint32_t width, uint32_t height,
			    const char* intrinsicCalibrationFilename );

	/**
	 * Retrieve the number of points being used.
	 */
	inline uint32_t GetNumPoints() const		{ return mNumPoints; }

	/**
	 * Retrieve the max number of points in memory.
	 */
	inline uint32_t GetMaxPoints() const		{ return mMaxPoints; }

	/**
	 * Retrieve memory pointer to point cloud data.
	 */
	inline float3* GetData() const			{ return mPointsCPU; }

	/**
	 * Retrieve memory pointer to a specific point.
	 */
	inline float3* GetData( size_t index ) const	{ return mPointsCPU + index * (mHasRGB + 1); }
 
	/**
	 * Retrieve the size in bytes currently being used.
	 */
	inline size_t GetSize() const				{ return mNumPoints * sizeof(float3) * (mHasRGB + 1); }

	/**
	 * Retrieve the maximum size in bytes of the point cloud.
	 */
	inline size_t GetMaxSize() const			{ return mMaxPoints * sizeof(float3) * (mHasRGB + 1); }

	/**
	 * Does the point cloud have RGB data?
	 */
	inline bool HasRGB() const				{ return mHasRGB; }

	/**
	 * Allocate and reserve memory for the max number of points.
	 */
	bool Reserve( uint32_t maxPoints );

	/**
	 * Free the memory being used to store the point cloud.
	 */
	void Free();

	/**
	 * Save point cloud to PCD file.
	 */
	bool Save( const char* filename );


protected:
	cudaPointCloud();

	float3* mPointsCPU;
	float3* mPointsGPU;

	uint32_t mNumPoints;
	uint32_t mMaxPoints;

	bool mHasRGB;
};

#endif

