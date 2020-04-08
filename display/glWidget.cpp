/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 
#include "glWidget.h"
#include "glDisplay.h"


// constructor
glWidget::glWidget( Shape shape )
{
	initDefaults();
	mShape = shape;
}


// constructor
glWidget::glWidget( float x, float y, float width, float height, Shape shape )
{
	initDefaults();

	mX = x;
	mY = y;
	
	mShape  = shape;
	mWidth  = width;
	mHeight = height;
}


// initDefaults
void glWidget::initDefaults()
{
	mShape  = Rect;
	mX      = 0;
	mY      = 0;
	mWidth  = 0;
	mHeight = 0;

	mFillColor[0] = 1.0f; 
	mFillColor[1] = 1.0f; 
	mFillColor[2] = 1.0f;
	mFillColor[3] = 0.0f;

	mLineColor[0] = 1.0f;
	mLineColor[1] = 1.0f; 
	mLineColor[2] = 1.0f;
	mLineColor[3] = 1.0f;

	mLineWidth = 2.0f;
	mVisible   = true;
	mUserData  = NULL;
	mDisplay   = NULL;
}


// destructor
glWidget::~glWidget()
{
	if( mDisplay != NULL )
		mDisplay->RemoveWidget(this);
}


// Contains
bool glWidget::Contains( float x, float y ) const
{
	if( mShape == Rect )
	{
		if( x >= mX && y >= mY && x <= (mX + mWidth) && y <= (mY + mHeight) )
			return true;
	}
	
	// TODO other shape types
	return false;
}

	
// Render
void glWidget::Render()
{
	if( !mVisible )
		return;

	if( mShape == Rect )
	{
		if( mFillColor[3] > 0.0f )
		{
			glDrawRect(mX, mY, mWidth, mHeight,
					 mFillColor[0], mFillColor[1],
					 mFillColor[2], mFillColor[3]);
		}

		if( mLineColor[3] > 0.0f && mLineWidth > 0.0f )
		{
			glDrawOutline(mX, mY, mWidth, mHeight,
					    mLineColor[0], mLineColor[1],
					    mLineColor[2], mLineColor[3]);
		}
	}

	// TODO other shape types
}


// OnEvent
bool glWidget::OnEvent( uint16_t event, int a, int b, void* user )
{
	// TODO dispatch recursively, when child widgets are added
	return false;
}


// setDisplay
void glWidget::setDisplay( glDisplay* display )
{
	// TODO set recursively, when child widgets are added
	mDisplay = display;
}

	

