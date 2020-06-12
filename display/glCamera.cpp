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

#include "glDisplay.h"
#include "glCamera.h"

#include "pi.h"

#include <X11/keysym.h>
#include <math.h>


// constructor
glCamera::glCamera( CameraMode mode )
{
	mMode = mode;
	mNear = 1.0f;
	mFar  = 16128.0f;
	mFoV  = 65.0f;

	mUp[0] = 0.0f; 
	mUp[1] = 1.0f; 
	mUp[2] = 0.0f;

	mDefaultEye[0] = 500.0f; 
	mDefaultEye[1] = 500.0f; 
	mDefaultEye[2] = 500.0f;

	mDefaultLookAt[0] = 0.0f; 
	mDefaultLookAt[1] = 0.0f;
	mDefaultLookAt[2] = 0.0f;

	mDefaultRotation[0] = 60.0f * DEG_TO_RAD;
	mDefaultRotation[1] = 0.0f;
	mDefaultRotation[2] = 0.0f;

	mMovementSpeed   = 1.0f;
	mMovementEnabled = false;

	mDisplay     = NULL;
	mMouseActive = false;

	memset(mViewport, 0, sizeof(mViewport));
	memset(mPrevModelView, 0, sizeof(mPrevModelView));
	memset(mPrevProjection, 0, sizeof(mPrevProjection));

	Reset();
}


// destructor
glCamera::~glCamera()
{
	glUnregisterEvents(NULL, this);
}


// Create
glCamera* glCamera::Create( CameraMode mode, int registerEvents )
{
	glCamera* cam = new glCamera(mode);

	if( !cam )
	{
		LogError(LOG_GL "failed to create camera\n");
		return NULL;
	}

	if( registerEvents >= 0 )
		cam->RegisterEvents(registerEvents);

	return cam;
}


// Create
glCamera* glCamera::Create( int registerEvents )
{
	return Create(Ortho, registerEvents);
}


// Activate
void glCamera::Activate( CameraMode mode )
{
	SetCameraMode(mode);
	Activate();	
}


// Activate
void glCamera::Activate()
{
	// save the previous matrices
	glGetFloatv(GL_MODELVIEW_MATRIX, mPrevModelView);
	glGetFloatv(GL_PROJECTION_MATRIX, mPrevProjection);

	// get the viewport bounds				
	glGetIntegerv(GL_VIEWPORT, mViewport);
	//printf(LOG_GL "glCamera -- viewport %i %i %i %i\n", viewport[0], viewport[1], viewport[2], viewport[3]);
	const float aspect = float(mViewport[2]) / float(mViewport[3]);

	// set perspective matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	if( mMode == YawPitchRoll || mMode == LookAt )
		gluPerspective(mFoV, aspect, mNear, mFar);
	else if( mMode == Ortho )
		glOrtho(mViewport[0], mViewport[2], mViewport[3], mViewport[1], 0.0f, 1.0f);

	// set modelview matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	if( mMode == LookAt )
	{
		gluLookAt(mEye[0], mEye[1], mEye[2], mLookAt[0], mLookAt[1], mLookAt[2], mUp[0], mUp[1], mUp[2]);
	}
	else if( mMode == YawPitchRoll )
	{
		glRotatef(mRotation[0] * RAD_TO_DEG, 1.0f, 0.0f, 0.0f);
		glRotatef(mRotation[1] * RAD_TO_DEG, 0.0f, 1.0f, 0.0f);
		//glRotatef(mRotation[2] * RAD_TO_DEG, 0.0f, 0.0f, 1.0f);
		glTranslatef(-mEye[0], -mEye[1], -mEye[2]);
	}
}

// Deactivate
void glCamera::Deactivate()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(mPrevModelView);
	
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(mPrevProjection);
}


// Reset
void glCamera::Reset()
{
	mEye[0] = mDefaultEye[0]; 
	mEye[1] = mDefaultEye[1]; 
	mEye[2] = mDefaultEye[2];

	mLookAt[0] = mDefaultLookAt[0]; 
	mLookAt[1] = mDefaultLookAt[1]; 
	mLookAt[2] = mDefaultLookAt[2]; 

	mRotation[0] = mDefaultRotation[0]; 
	mRotation[1] = mDefaultRotation[1]; 
	mRotation[2] = mDefaultRotation[2]; 
}


// StoreDefaults
void glCamera::StoreDefaults()
{
	mDefaultEye[0] = mEye[0]; 
	mDefaultEye[1] = mEye[1]; 
	mDefaultEye[2] = mEye[2];

	mDefaultLookAt[0] = mLookAt[0]; 
	mDefaultLookAt[1] = mLookAt[1]; 
	mDefaultLookAt[2] = mLookAt[2]; 

	mDefaultRotation[0] = mRotation[0]; 
	mDefaultRotation[1] = mRotation[1]; 
	mDefaultRotation[2] = mRotation[2]; 
}


// RegisterEvents
void glCamera::RegisterEvents( uint32_t display )
{
	SetMovementEnabled(true);
	glRegisterEvents(&onEvent, this, display);
	mDisplay = glGetDisplay(display);
}


// onEvent
bool glCamera::onEvent( uint16_t msg, int a, int b, void* user )
{
	if( !user )
		return false;

	glCamera* cam = (glCamera*)user;

	if( msg == KEY_STATE && a == XK_r && b == KEY_PRESSED )
		cam->Reset();

	if( !cam->mMovementEnabled )
		return false;

	if( msg == MOUSE_BUTTON )
	{
		if( a == MOUSE_LEFT )
		{
			if( b == MOUSE_PRESSED && cam->mouseInViewport() )
				cam->mMouseActive = true;
			else
				cam->mMouseActive = false;
		}
	}

	if( cam->mMode == glCamera::LookAt )
		return cam->onEventLookAt(msg, a, b);
	else if( cam->mMode == glCamera::YawPitchRoll )
		return cam->onEventYawPitchRoll(msg, a, b);

	return false;
}


// onEventLookAt
bool glCamera::onEventLookAt( uint16_t msg, int a, int b )
{
	if( msg != KEY_STATE && msg != MOUSE_WHEEL && msg != MOUSE_DRAG )
		return false;

	float movement_speed = mMovementSpeed;
	float angle_speed = movement_speed * 0.075f;

	const float delta[] = { mEye[0] - mLookAt[0],
					    mEye[1] - mLookAt[1],
					    mEye[2] - mLookAt[2] };

	// https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
	float radius = sqrtf(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
	float phi    = atan2f(delta[2], delta[0]);
	float theta  = acosf(delta[1] / radius);

	if( msg == KEY_STATE && b == KEY_PRESSED )
	{
		const int key = a;

		if( key == XK_Up || key == XK_Down || key == XK_w || key == XK_s )
		{
			if( key == XK_Up || key == XK_w )	
				movement_speed *= -1.0;

			radius += movement_speed;
		}
		else if( key == XK_Left || key == XK_Right || key == XK_a || key == XK_d )
		{
			if( key == XK_Right || key == XK_d )	
				angle_speed *= -1.0;

			phi += angle_speed;
		}
		else if( key == XK_q || key == XK_z || key == XK_e )
		{
			if( key == XK_q )
				angle_speed *= -1.0;

			theta += angle_speed * 0.6f;
		}
	}
	else if( msg == MOUSE_DRAG )
	{
		if( mMouseActive )
		{
			phi += float(a) * 0.005f;
			theta += float(b) * 0.005f;
		}
	}
	else if( msg == MOUSE_WHEEL )
	{
		if( mMouseActive || mouseInViewport() )
		{
			radius += movement_speed * a;
		}
	}

	// convert from spherical to cartesian
	mEye[0] = radius * sinf(theta) * cosf(phi);
	mEye[1] = radius * cos(theta);
	mEye[2] = radius * sinf(theta) * sinf(phi);

	mEye[0] += mLookAt[0];
	mEye[1] += mLookAt[1];
	mEye[2] += mLookAt[2];

	//printf("radius %f  phi %f  theta %f\n", radius, phi, theta);
	//printf("eye %f %f %f\n", mEye[0], mEye[1], mEye[2]);

	return true;
}


// onEventYawPitchRoll
bool glCamera::onEventYawPitchRoll( uint16_t msg, int a, int b )
{
	float movement_speed = mMovementSpeed;

	if( msg == KEY_STATE && b == KEY_PRESSED )
	{
		const int key = a;

		if( key == XK_Up || key == XK_Down || key == XK_w || key == XK_s )
		{
			if( key == XK_Up || key == XK_w )	
				movement_speed *= -1.0;

			mEye[0] -= sinf(mRotation[1]) * movement_speed;
			mEye[1] += sinf(mRotation[0]) * movement_speed;
			mEye[2] += cosf(mRotation[1]) * movement_speed;
		}
		else if( key == XK_Left || key == XK_Right || key == XK_a || key == XK_d )
		{
			if( key == XK_Left || key == XK_a )	
				movement_speed *= -1.0;

			mEye[0] += cosf(mRotation[1]) * movement_speed;
			mEye[2] += sinf(mRotation[1]) * movement_speed;
		}
		else if( key == XK_q || key == XK_z || key == XK_e )
		{
			if( key == XK_z || key == XK_e )
				movement_speed *= -1.0;

			mEye[1] += movement_speed;
		}
	}
	else if( msg == MOUSE_DRAG )
	{
		if( mMouseActive )
		{
			mRotation[0] += float(b) * 0.0025f;
			mRotation[1] += float(a) * 0.0025f;
		}
	}
	else if( msg == MOUSE_WHEEL )
	{
		if( mMouseActive || mouseInViewport() )
		{
			movement_speed *= a;

			mEye[0] -= sinf(mRotation[1]) * movement_speed;
			mEye[1] += sinf(mRotation[0]) * movement_speed;
			mEye[2] += cosf(mRotation[1]) * movement_speed;
		}
	}

	return true;
}


// mouseInViewport
bool glCamera::mouseInViewport() const
{
	glDisplay* display = (glDisplay*)mDisplay;

	if( !display )
		return false;

	int x = 0;
	int y = 0;

	display->GetMousePosition(&x, &y);

	const int viewLeft   = mViewport[0];
	const int viewTop    = display->GetHeight() - mViewport[1] - mViewport[3];
	const int viewRight  = viewLeft + mViewport[2];
	const int viewBottom = viewTop + mViewport[3];

	if( x >= viewLeft && x <= viewRight && y >= viewTop && y <= viewBottom )
		return true;

	return false;
}


