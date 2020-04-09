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
 
#ifndef __GL_WIDGET_H__
#define __GL_WIDGET_H__


#include <stdint.h>
#include <stdio.h>


// forward declarations
class glDisplay;


/**
 * OpenGL graphics widget for rendering moveable/resizable shapes.
 * @ingroup OpenGL
 */
class glWidget
{
public:
	/**
	 * Shape enum
	 */
	enum Shape
	{
		Rect,
		Line,
		Ellipse
	};

	/**
	 * Constructor
	 */
	glWidget( Shape shape=Rect );

	/**
	 * Constructor
	 */
	glWidget( float x, float y, float width, float height, Shape shape=Rect );

	/**
	 * Destructor
	 */
	virtual ~glWidget();

	/**
	 * Test if the point is inside the widget
	 */
	bool Contains( float x, float y ) const;

	/**
	 * Convert from global window coordinates to local widget offset
	 */
	void GlobalToLocal( float x, float y, float* x_out, float* y_out ) const;

	/**
	 * Convert from local widget offset to global window coordinates
	 */
	void LocalToGlobal( float x, float y, float* x_out, float* y_out ) const;

	/**
	 * Move the widget's position by the specified offset
	 */
	inline void Move( float x, float y )					{ mX += x; mY += y; }

	/**
	 * Get position of widget in global window coordinates
	 */
	inline void GetPosition( float* x, float* y ) const		{ if(x) *x=mX; if(y) *y=mY; }
	
	/**
	 * Set position of widget in global window coordinates
	 */
	inline void SetPosition( float x, float y )				{ mX = x; mY = y; }

	/**
	 * Get size
	 */
	inline void GetSize( float* width, float* height ) const	{ if(width) *width=mWidth; if(height) *height=mHeight; }
	
	/**
	 * Set size
	 */
	inline void SetSize( float width, float height )			{ mWidth = width; mHeight = height; }

	/**
	 * Get width
	 */
	inline float GetWidth() const							{ return mWidth; }

	/**
	 * Get height
	 */
	inline float GetHeight() const						{ return mHeight; }

	/**
	 * Set width
	 */
	inline void SetWidth( float width )					{ mWidth = width; }

	/**
	 * Set height
	 */
	inline void SetHeight( float height )					{ mHeight = height; }

	/**
	 * Get the shape
	 */
	inline Shape GetShape() const							{ return mShape; }

	/**
	 * Set the shape
	 */
	inline void SetShape( Shape shape )					{ mShape = shape; }

	/**
	 * Set fill alpha
	 */
	inline void SetFillAlpha( float a )					{ mFillColor[3] = a; }

	/**
	 * Set fill color
	 */
	inline void SetFillColor( float r, float g, float b, float a=1.0f )	{ mFillColor[0] = r; mFillColor[1] = g; mFillColor[2] = b; mFillColor[3] = a; }

	/**
	 * Set outline color
	 */
	inline void SetLineColor( float r, float g, float b, float a=1.0f )	{ mLineColor[0] = r; mLineColor[1] = g; mLineColor[2] = b; mLineColor[3] = a; }

	/**
	 * Set outline alpha
	 */
	inline void SetLineAlpha( float a )					{ mLineColor[3] = a; }

	/**
	 * Set outline width
	 */
	inline void SetLineWidth( float width ) 				{ mLineWidth = width; }

	/**
	 * Is the widget moveable/draggable by the user?
	 */
	inline bool IsMoveable() const						{ return mMoveable; }

	/**
	 * Toggle if the user can move/drag the widget
	 */
	inline void SetMoveable( bool moveable )				{ mMoveable = moveable; }

	/**
	 * Is the widget resizeable by the user?
	 */
	inline bool IsResizeable() const						{ return mResizeable; }

	/**
	 * Toggle if the user can resize the widget
	 */
	inline void SetResizeable( bool resizeable )				{ mResizeable = resizeable; }

	/**
	 * Is the widget visible
	 */
	inline bool IsVisible() const							{ return mVisible; }

	/**
	 * Show/hide the widget
	 */
	inline void SetVisible( bool visible )					{ mVisible = visible; }

	/**
	 * Retrieve user data
	 */
	inline void* GetUserData() const						{ return mUserData; }

	/**
	 * Set user-defined data
	 */
	inline void SetUserData( void* user )					{ mUserData = user; }

	/**
	 * Get the root window of the widget
	 */
	inline glDisplay* GetDisplay() const					{ return mDisplay; }

	/**
	 * Render (automatically called by parent)
	 */
	void Render();

	/**
	 * Event handler (automatically called by parent)
	 */
	bool OnEvent( uint16_t event, int a, int b, void* user );

	
protected:
	friend class glDisplay;

	enum DragState
	{
		DragNone,
		DragMove,
		DragResizeN,
		DragResizeNW,
		DragResizeNE,
		DragResizeS,
		DragResizeSW,
		DragResizeSE,
		DragResizeW,
		DragResizeE		
	};

	void initDefaults();
	void setCursor( DragState cursor );	
	void setDisplay( glDisplay* display );
	
	float mX;
	float mY;
	float mWidth;
	float mHeight;

	float mFillColor[4];
	float mLineColor[4];
	float mLineWidth;

	bool  mMoveable;
	bool  mResizeable;
	bool  mVisible;
	
	Shape mShape;
	void* mUserData;

	glDisplay* mDisplay;
	DragState  mDragState;

	DragState coordToBorder( float x, float y, float max_distance=10.0f );
};

#endif

