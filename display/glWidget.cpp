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

#include <X11/cursorfont.h>
#include <math.h>


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

	mSelectedFillColor[0] = 1.0f; 
	mSelectedFillColor[1] = 1.0f; 
	mSelectedFillColor[2] = 1.0f;
	mSelectedFillColor[3] = 0.0f;

	mSelectedLineColor[0] = 1.0f;
	mSelectedLineColor[1] = 1.0f; 
	mSelectedLineColor[2] = 1.0f;
	mSelectedLineColor[3] = 1.0f;

	mLineWidth  = 2.0f;
	mUserData   = NULL;
	mDisplay    = NULL;
	mDragState  = DragNone;
	mMoveable   = false;
	mResizeable = false;
	mVisible    = true;
}


// destructor
glWidget::~glWidget()
{
	if( mDisplay != NULL )
		mDisplay->RemoveWidget(this, false);
}


// GetIndex
int glWidget::GetIndex() const
{
	if( !mDisplay )
		return -1;

	return mDisplay->GetWidgetIndex(this);
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


// GlobalToLocal
void glWidget::GlobalToLocal( float x, float y, float* x_out, float* y_out ) const
{
	x -= mX;
	y -= mY;

	if( x_out != NULL )
		*x_out = x;

	if( y_out != NULL )
		*y_out = y;
}


// LocalToGlobal
void glWidget::LocalToGlobal( float x, float y, float* x_out, float* y_out ) const
{
	x += mX;
	y += mY;

	if( x_out != NULL )
		*x_out = x;

	if( y_out != NULL )
		*y_out = y;
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
					    mLineColor[2], mLineColor[3],
					    mLineWidth);
		}
	}

	// TODO other shape types
}


// OnEvent
bool glWidget::OnEvent( uint16_t event, int a, int b, void* user )
{
	if( event == MOUSE_BUTTON && a == MOUSE_LEFT )
	{
		if( b == MOUSE_PRESSED )
		{
			if( mMoveable || mResizeable )
			{
				mDragState = mMoveable ? DragMove : DragNone;

				if( mResizeable )
				{
					const int* mouseCoords = mDisplay->GetMousePosition();
					const DragState anchor = coordToBorder(mouseCoords[0], mouseCoords[1]);

					if( anchor != DragNone )
					{
						setCursor(anchor);
						mDragState = anchor;
					}						
				}
			}
		}
		else
		{
			mDragState = DragNone;
		}
	}
	else if( event == MOUSE_DRAG && mDragState != DragNone )
	{
		bool wasMoved = false;
		bool wasResized = false;

		if( mDragState == DragMove )
		{
			Move(a, b);
			wasMoved = true;
		}

		// Y resize
		if( mDragState == DragResizeN || mDragState == DragResizeNW || mDragState == DragResizeNE )
		{
			mHeight -= b;
			mY += b;

			wasMoved = true;
			wasResized = true;
		}
		else if( mDragState == DragResizeS || mDragState == DragResizeSW || mDragState == DragResizeSE )
		{
			mHeight += b;
			wasResized = true;
		}
		
		// X resize
		if( mDragState == DragResizeW || mDragState == DragResizeNW || mDragState == DragResizeSW )
		{
			mWidth -= a;
			mX += a;

			wasMoved = true;
			wasResized = true;
		}
		else if( mDragState == DragResizeE || mDragState == DragResizeNE || mDragState == DragResizeSE )
		{
			mWidth += a;
			wasResized = true;
		}

		if( mWidth < 1.0f )
			mWidth = 1.0f;

		if( mHeight < 1.0f )
			mHeight = 1.0f;

		if( wasMoved )
			dispatchEvent(WIDGET_MOVED, mX, mY);

		if( wasResized )
			dispatchEvent(WIDGET_RESIZED, mWidth, mHeight);
	}
	else if( event == MOUSE_MOVE )
	{
		if( mResizeable && mDragState == DragNone )
			setCursor(coordToBorder(a,b));
	}

	// TODO dispatch recursively, when child widgets are added
	return false;
}


// within_distance
inline static bool within_distance( float a, float b, float max_distance )
{
	return fabsf(a-b) <= max_distance;
}
		

// coordToBorder
glWidget::DragState glWidget::coordToBorder( float x, float y, float max_distance )
{
	const bool N = within_distance(y, mY, max_distance);
	const bool S = within_distance(y, mY + mHeight, max_distance);
	const bool W = within_distance(x, mX, max_distance);
	const bool E = within_distance(x, mX + mWidth, max_distance);

	if( N )
	{
		if( W )
			return DragResizeNW;
		else if( E )
			return DragResizeNE;
		else
			return DragResizeN;
	}
	else if( S )
	{
		if( W )
			return DragResizeSW;
		else if( E )
			return DragResizeSE;
		else
			return DragResizeS; 
	}
	else if( W )
	{
		return DragResizeW;
	}
	else if( E )
	{
		return DragResizeE;
	}

	return DragNone;	// interior point
}


// setCursor
void glWidget::setCursor( DragState anchor )
{
	if( !mDisplay )
		return;

	if( anchor == DragNone || anchor == DragMove )
	{
		if( mMoveable )
			mDisplay->SetCursor(XC_fleur);
		else
			mDisplay->ResetCursor();

		return;
	}

	uint32_t cursor = 0;

	if( anchor == DragResizeN )
		cursor = XC_top_side; 
	else if( anchor == DragResizeS )
		cursor = XC_bottom_side;
	else if( anchor == DragResizeW )
		cursor = XC_left_side;
	else if( anchor == DragResizeE )
		cursor = XC_right_side;
	else if( anchor == DragResizeNW )
		cursor = XC_top_left_corner;
	else if( anchor == DragResizeNE )
		cursor = XC_top_right_corner;
	else if( anchor == DragResizeSW )
		cursor = XC_ll_angle; //XC_bottom_left_corner;
	else if( anchor == DragResizeSE )
		cursor = XC_bottom_right_corner;

	//printf("set cursor: %u\n", cursor);
	mDisplay->SetCursor(cursor);
}

// setDisplay
void glWidget::setDisplay( glDisplay* display )
{
	// TODO set recursively, when child widgets are added
	mDisplay = display;
}


// AddEventHandler
void glWidget::AddEventHandler( glWidgetEventHandler callback, void* user )
{
	if( !callback )
		return;

	eventHandler handler;

	handler.callback = callback;
	handler.user = user;

	mEventHandlers.push_back(handler);
}


// RemoveEventHandler
void glWidget::RemoveEventHandler( glWidgetEventHandler callback, void* user )
{
	if( !callback && !user )
		return;

	for( int n=0; n < mEventHandlers.size(); n++ )
	{
		bool found = false;

		if( callback != NULL && user != NULL )
		{
			if( mEventHandlers[n].callback == callback && mEventHandlers[n].user == user )
				found = true;
		}
		else if( callback != NULL )
		{
			if( mEventHandlers[n].callback == callback )
				found = true;
		}
		else if( user != NULL )
		{
			if( mEventHandlers[n].user == user )
				found = true;
		}

		if( found )
		{
			mEventHandlers.erase(mEventHandlers.begin() + n);
			n--;	// keep searching for more matches
		}
	}
}


// dispatchEvent
void glWidget::dispatchEvent( uint16_t msg, int a, int b )
{
	const uint32_t numHandlers = mEventHandlers.size();

	for( uint32_t n=0; n < numHandlers; n++ )
		mEventHandlers[n].callback(this, msg, a, b, mEventHandlers[n].user);
}


