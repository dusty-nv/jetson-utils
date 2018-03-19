/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __GL_VIEWPORT_H__
#define __GL_VIEWPORT_H__


#include "glUtility.h"
#include "glTexture.h"

#include <time.h>


/**
 * OpenGL display window / video viewer
 */
class glDisplay
{
public:
	/**
	 * Create a new maximized openGL display window.
	 * @param r default background RGBA color, red component (0.0-1.0f)
	 * @param g default background RGBA color, green component (0.0-1.0f)
	 * @param b default background RGBA color, blue component (0.0-1.0f)
	 * @param a default background RGBA color, alpha component (0.0-1.0f)
	 */
	static glDisplay* Create( float r=0.05f, float g=0.05f, float b=0.05f, float a=1.0f );

	/**
	 * Create a new maximized openGL display window.
	 * @param title window title bar label string
	 * @param r default background RGBA color, red component (0.0-1.0f)
	 * @param g default background RGBA color, green component (0.0-1.0f)
	 * @param b default background RGBA color, blue component (0.0-1.0f)
	 * @param a default background RGBA color, alpha component (0.0-1.0f)
	 */
	static glDisplay* Create( const char* title, float r=0.05f, float g=0.05f, float b=0.05f, float a=1.0f );

	/**
	 * Destroy window
	 */
	~glDisplay();

	/**
 	 * Clear window and begin rendering a frame.
	 */
	void BeginRender();

	/**
	 * Finish rendering and refresh / flip the backbuffer.
	 */
	void EndRender();

	/**
	 * Process UI events.
	 */
	void UserEvents();
		
	/**
	 * UI event handler.
	 */
	void onEvent( uint msg, int a, int b );

	/**
	 * Set the window title string.
	 */
	void SetTitle( const char* str );

	/**
	 * Set the background color.
	 * @param r background RGBA color, red component (0.0-1.0f)
	 * @param g background RGBA color, green component (0.0-1.0f)
	 * @param b background RGBA color, blue component (0.0-1.0f)
	 * @param a background RGBA color, alpha component (0.0-1.0f)
	 */
	inline void SetBackgroundColor( float r, float g, float b, float a )	{ mBgColor[0] = r; mBgColor[1] = g; mBgColor[2] = b; mBgColor[3] = a; }

	/**
	 * Get the average frame time (in milliseconds).
	 */
	inline float GetFPS()	{ return 1000000000.0f / mAvgTime; }
		
protected:
	glDisplay();
		
	bool initWindow();
	bool initGL();

	static const int screenIdx = 0;
		
	Display*     mDisplayX;
	Screen*      mScreenX;
	XVisualInfo* mVisualX;
	Window       mWindowX;
	GLXContext   mContextGL;
		
	uint32_t mWidth;
	uint32_t mHeight;

	timespec mLastTime;
	float    mAvgTime;
	float    mBgColor[4];
};

#endif

