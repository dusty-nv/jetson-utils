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


// forward declarations
class glBuffer;
class glCamera;


/**
 * CUDA-accelerated point cloud processing.
 * @ingroup pointCloud
 */
class cudaPointCloud
{
public:
	/**
	 * Point vertex
	 */
	struct Vertex
	{
		/**
		 * The XYZ position of the point.
		 */
		float3 pos;

		/**
		 * The RGB color of the point.
		 * @note will be white if RGB data not provided
		 */
		uchar3 color;

		/**
		 * The class ID of the point.
		 * @note will be 0 if classification data no provided
		 */
		uint8_t classID;

	} __attribute__((packed));

	/**
	 * Create
	 */
	static cudaPointCloud* Create();

	/**
	 * Destructor
	 */
	~cudaPointCloud();
	
	/**
	 * Allocate and reserve memory for the max number of points.
	 *
	 * @note Memory is reserved automatically by Extract(), but
	 *       if you know in advance the maximum number of points
	 *       it is good practice to call Reserve() ahead of time.
	 */
	bool Reserve( uint32_t maxPoints );

	/**
	 * Free the memory being used to store the point cloud.
	 */
	void Free();

	/**
	 * Clear the points, but keep the memory allocated.
	 */
	void Clear();

	/**
	 * Extract point cloud from depth map and optional RGBA image.
	 */
	bool Extract( float* depth, float4* rgba, uint32_t width, uint32_t height );

	/**
	 * Extract point cloud from depth map and optional RGBA image.
	 */
	bool Extract( float* depth, uint32_t depth_width, uint32_t depth_height,
			    float4* rgba, uint32_t color_width, uint32_t color_height );

	/**
	 * Retrieve the number of points being used.
	 */
	inline uint32_t GetNumPoints() const		{ return mNumPoints; }

	/**
	 * Retrieve the max number of points in memory.
	 */
	inline uint32_t GetMaxPoints() const		{ return mMaxPoints; }

	/**
	 * Retrieve the size in bytes currently being used.
	 */
	inline size_t GetSize() const				{ return mNumPoints * sizeof(Vertex); }

	/**
	 * Retrieve the maximum size in bytes of the point cloud.
	 */
	inline size_t GetMaxSize() const			{ return mMaxPoints * sizeof(Vertex); }

	/**
	 * Retrieve memory pointer to point cloud data.
	 */
	inline Vertex* GetData() const			{ return mPointsCPU; }

	/**
	 * Retrieve memory pointer to a specific point.
	 */
	inline Vertex* GetData( size_t index ) const	{ return mPointsCPU + index; }
 
	/**
	 * Does the point cloud have RGB data?
	 */
	inline bool HasRGB() const				{ return mHasRGB; }

	/**
	 * Render the point cloud with OpenGL
	 */
	bool Render();

	/**
	 * Save point cloud to PCD file.
	 */
	bool Save( const char* filename );

	/**
	 * Set the intrinsic camera calibration.
	 */
	bool SetCalibration( const char* filename );

	/**
	 * Set the intrinsic camera calibration.
	 */
	void SetCalibration( const float K[3][3] );

	/**
	 * Set the intrinsic camera calibration.
	 */
	void SetCalibration( const float2& focalLength, const float2& principalPoint );

protected:
	cudaPointCloud();

	bool allocBufferGL();
	bool allocDepthResize( size_t size );
	
	Vertex* mPointsCPU;
	Vertex* mPointsGPU;

	glBuffer* mBufferGL;
	glCamera* mCameraGL;

	uint32_t mNumPoints;
	uint32_t mMaxPoints;

	float2 mFocalLength;
	float2 mPrincipalPoint;

	float* mDepthResize;
	size_t mDepthSize;

	bool mHasRGB;
	bool mHasNewPoints;
	bool mHasCalibration;
};

#endif

