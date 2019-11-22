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
		printf(LOG_GL "failed to create camera\n");
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
bool glCamera::onEvent( uint16_t msg, int a, int b )
{
	if( !mMovementEnabled )
		return false;

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
		else if( key == XK_r )
		{
			Reset();
		}
	}
	else if( msg == MOUSE_BUTTON )
	{
		if( a == MOUSE_LEFT )
		{
			if( b == MOUSE_PRESSED && mouseInViewport() )
				mMouseActive = true;
			else
				mMouseActive = false;
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


// onEvent
bool glCamera::onEvent( uint16_t msg, int a, int b, void* user )
{
	if( !user )
		return false;

	return ((glCamera*)user)->onEvent(msg, a, b);
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


