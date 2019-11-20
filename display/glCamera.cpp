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
	// get the viewport bounds
	GLint viewport[4];					
	glGetIntegerv(GL_VIEWPORT, viewport);
	printf(LOG_GL "glCamera -- viewport %i %i %i %i\n", viewport[0], viewport[1], viewport[2], viewport[3]);
	const float aspect = float(viewport[2]) / float(viewport[3]);	// TODO what if viewport origin is not (0,0)?

	// set perspective matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	if( mMode == YawPitchRoll || mMode == LookAt )
		gluPerspective(mFoV, aspect, mNear, mFar);
	else if( mMode == Ortho )
		glOrtho(viewport[0], viewport[2], viewport[3], viewport[1], 0.0f, 1.0f);

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
	glRegisterEvents(&onEvent, this);
}


// onEvent
bool glCamera::onEvent( uint16_t msg, int a, int b, void* user )
{
	if( !user )
		return false;

	glCamera* cam = (glCamera*)user;

	if( msg == KEY_RAW && b == KEY_PRESSED )
	{
		float motion_scale = 25.0f;

		if( a == XK_w || a == XK_s || a == XK_Up || a == XK_Down )
		{
			if( a == XK_w || a == XK_Up )	
				motion_scale *= -1.0;

			cam->mEye[0] -= sinf(cam->mRotation[1]) * motion_scale;
			cam->mEye[1] += sinf(cam->mRotation[0]) * motion_scale;
			cam->mEye[2] += cosf(cam->mRotation[1]) * motion_scale;
		}
		else if( a == XK_a || a == XK_d || a == XK_Left || a == XK_Right )
		{
			if( a == XK_a || a == XK_Left )	
				motion_scale *= -1.0;

			cam->mEye[0] += cosf(cam->mRotation[1]) * motion_scale;
			cam->mEye[2] += sinf(cam->mRotation[1]) * motion_scale;
		}
		else if( a == XK_q || a == XK_z || a == XK_e )
		{
			if( a == XK_z || a == XK_e )
				motion_scale *= -1.0;

			cam->mEye[1] += motion_scale;
		}
		else if( a == XK_r )
		{
			cam->Reset();
		}
	}
	
	return true;
}


