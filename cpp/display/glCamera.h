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
 
#ifndef __GL_CAMERA_H__
#define __GL_CAMERA_H__


#include <stdint.h>
#include <stdio.h>


/**
 * OpenGL perspective camera supporting Look-At, Yaw/Pitch/Roll, and Ortho modes.
 * @ingroup OpenGL
 */
class glCamera
{
public:
	/**
	 * Enum specifying the camera mode
	 */
	enum CameraMode
	{
		LookAt,		/**< LookAt (orbit) */
		YawPitchRoll,	/**< YawPitchRoll (first person */
		Ortho		/**< Ortho (2D) */
	};
	
	/**
	 * Create OpenGL camera object with the specified CameraMode
	 *
	 * @param registerEvents the ID of the glDisplay window
	 *        to register with to recieve input events (for
	 *        moving the camera around with keyboard/mouse),
	 *        or -1 to not register for input events.
	 */
	static glCamera* Create( CameraMode mode, int registerEvents=0 );

	/**
	 * Create OpenGL camera object
	 *
	 * @param registerEvents the ID of the glDisplay window
	 *        to register with to recieve input events (for
	 *        moving the camera around with keyboard/mouse),
	 *        or -1 to not register for input events.
	 */
	static glCamera* Create( int registerEvents=0 );

	/**
	 * Free the camera
	 */
	~glCamera();
	
	/**
	 * Activate GL_PROJECTION and GL_MODELVIEW matrices
	 */
	void Activate();

	/**
	 * Activate GL_PROJECTION and GL_MODELVIEW matrices
	 */
	void Activate( CameraMode mode );

	/**
	 * Restore previous GL_PROJECTION and GL_MODELVIEW matrices
	 */
	void Deactivate();

	/**
	 * Get the camera mode
	 */
	inline CameraMode GetCameraMode() const						{ return mMode; }

	/**
	 * Set the camera mode
	 */
	inline void SetCameraMode( CameraMode mode )					{ mMode = mode; }

	/**
	 * Set the field of view (FOV), in degrees
	 */
	inline void SetFOV( float fov )							{ mFoV = fov; }

	/**
 	 * Set the near/far z-clipping plane
	 */
	inline void SetClippingPlane( float near, float far )			{ mNear = near; mFar = far; }

	/**
	 * Set the distance to the near clipping plane
	 */
	inline void SetNear( float near )							{ mNear = near; }

	/**
	 * Set the distance to the far clipping plane
	 */
	inline void SetFar( float far )							{ mFar = far; }

	/**
	 * Set the eye position
	 */
	inline void SetEye( float x, float y, float z )				{ mEye[0] = x; mEye[1] = y; mEye[2] = z; }

	/**
	 * Set the look-at point
	 */
	inline void SetLookAt( float x, float y, float z )			{ mLookAt[0] = x; mLookAt[1] = y; mLookAt[2] = z; }

	/**
	 * Set the yaw/pitch/roll angles, in radians
	 */
	inline void SetRotation( float yaw, float pitch, float roll )	{ mRotation[0] = pitch; mRotation[1] = yaw; mRotation[2] = roll; }

	/**
	 * Set the yaw angle, in radians
	 */
	inline void SetYaw( float yaw )							{ mRotation[1] = yaw; }

	/**
	 * Set the pitch angle, in radians
	 */
	inline void SetPitch( float pitch )						{ mRotation[0] = pitch; }

	/**
 	 * Set the roll angle, in radians
	 */
	inline void SetRoll( float roll )							{ mRotation[2] = roll; }

	/**
	 * Set the movement speed (in world units)
	 */
	inline void SetMovementSpeed( float speed )					{ mMovementSpeed = speed; }

	/**
	 * Enable or disable movement from user input
	 */
	inline void SetMovementEnabled( bool enabled )				{ mMovementEnabled = enabled; }

	/**
	 * Store the current configuration as defaults
	 */
	void StoreDefaults();

	/**
	 * Reset camera orientation to defaults
	 */
	void Reset();

	/**
	 * Register to recieve input events (enable movement)
	 */
	void RegisterEvents( uint32_t display=0 );
	
private:
	glCamera( CameraMode mode );

	bool mouseInViewport() const;

	bool onEventLookAt( uint16_t msg, int a, int b );
	bool onEventYawPitchRoll( uint16_t msg, int a, int b );

	static bool onEvent( uint16_t msg, int a, int b, void* user );

	CameraMode mMode;

	float mNear;
	float mFar;
	float mFoV;

	float mEye[3];		// camera location
	float mRotation[3];	// pitch/yaw/roll euler angles
	float mLookAt[3];	// look-at point
	float mUp[3];		// up direction

	float mDefaultEye[3];
	float mDefaultRotation[3];
	float mDefaultLookAt[3];

	float mPrevModelView[16];
	float mPrevProjection[16];

	float mMovementSpeed;
	bool  mMovementEnabled;
	
	void* mDisplay;
	int   mViewport[4];
	bool  mMouseActive;
};

#endif

