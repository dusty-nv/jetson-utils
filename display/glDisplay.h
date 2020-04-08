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
#include "glEvents.h"
#include "glWidget.h"

#include <time.h>
#include <vector>


/**
 * OpenGL display window / video viewer
 * @ingroup OpenGL
 */
class glDisplay
{
public:
	/**
	 * Create a new OpenGL display window with the specified options.
	 *
	 * @param title window title bar string, or NULL for a default title
	 * @param width desired width of the window, or -1 to be maximized
	 * @param height desired height of the window, or -1 to be maximized
	 * @param r default background RGBA color, red component (0.0-1.0f)
	 * @param g default background RGBA color, green component (0.0-1.0f)
	 * @param b default background RGBA color, blue component (0.0-1.0f)
	 * @param a default background RGBA color, alpha component (0.0-1.0f)
	 */
	static glDisplay* Create( const char* title=NULL, int width=-1, int height=-1,
						 float r=0.05f, float g=0.05f, float b=0.05f, float a=1.0f );

	/**
	 * Destroy window
	 */
	~glDisplay();

	/**
 	 * Clear window and begin rendering a frame.
	 * If processEvents is true, ProcessEvents() will automatically be called.
	 */
	void BeginRender( bool processEvents=true );

	/**
	 * Finish rendering and refresh / flip the backbuffer.
	 */
	void EndRender();

	/**
	 * Render an OpenGL texture
	 * @note for more texture rendering methods, @see glTexture
	 */
	void Render( glTexture* texture, float x=5.0f, float y=30.0f );

	/**
	 * Render a CUDA float4 image using OpenGL interop
	 * If normalize is true, the image's pixel values will be rescaled from the range of [0-255] to [0-1]
	 * If normalize is false, the image's pixel values are assumed to already be in the range of [0-1]
	 * Note that if normalization is selected to be performed, it will be done in-place on the image
	 */
	void Render( float* image, uint32_t width, uint32_t height, float x=0.0f, float y=30.0f, bool normalize=true );

	/**
	 * Begin the frame, render one CUDA float4 image using OpenGL interop, and end the frame.
	 * Note that this function is only useful if you are rendering a single texture per frame.
	 * If normalize is true, the image's pixel values will be rescaled from the range of [0-255] to [0-1]
	 * If normalize is false, the image's pixel values are assumed to already be in the range of [0-1]
	 * Note that if normalization is selected to be performed, it will be done in-place on the image
	 */
	void RenderOnce( float* image, uint32_t width, uint32_t height, float x=5.0f, float y=30.0f, bool normalize=true );

	/**
	 * Render a line in screen coordinates with the specified color
	 * @note the RGBA color values are expected to be in the range of [0-1]
	 */
	void RenderLine( float x1, float y1, float x2, float y2, float r, float g, float b, float a=1.0f, float thickness=2.0f );

	/**
	 * Render the outline of a rect in screen coordinates with the specified color
	 * @note the RGBA color values are expected to be in the range of [0-1]
	 */
	void RenderOutline( float x, float y, float width, float height, float r, float g, float b, float a=1.0f, float thickness=2.0f );

	/**
	 * Render a filled rect in screen coordinates with the specified color
	 * @note the RGBA color values are expected to be in the range of [0-1]
	 */
	void RenderRect( float x, float y, float width, float height, float r, float g, float b, float a=1.0f );

	/**
	 * Render a filled rect covering the current viewport with the specified color
	 * @note the RGBA color values are expected to be in the range of [0-1]
	 */
	void RenderRect( float r, float g, float b, float a=1.0f );

	/**
	 * Returns true if the window is open.
	 */
	inline bool IsOpen() const 		{ return !mWindowClosed; }

	/**
	 * Returns true if the window has been closed.
	 */
	inline bool IsClosed() const		{ return mWindowClosed; }

	/**
	 * Returns true if between BeginRender() and EndRender()
	 */
	inline bool IsRendering() const	{ return mRendering; }

	/**
	 * Get the average frame time (in milliseconds).
	 */
	inline float GetFPS() const		{ return 1000000000.0f / mAvgTime; }

	/**
	 * Get the width of the window (in pixels)
	 */
	inline uint32_t GetWidth() const	{ return mWidth; }

	/**
	 * Get the height of the window (in pixels)
	 */
	inline uint32_t GetHeight() const	{ return mHeight; }

	/**
	 * Get the ID of this display instance into glGetDisplay()
	 */
	inline uint32_t GetID() const		{ return mID; }

	/**
	 * Get the mouse position.
	 */
	inline const int* GetMousePosition() const			{ return mMousePos; }

	/**
	 * Get the mouse position.
	 */
	inline void GetMousePosition( int* x, int* y ) const	{ if(x) *x = mMousePos[0]; if(y) *y = mMousePos[1]; }

	/**
	 * Get the mouse button state.
	 *
	 * @param button the button number, starting with 1.
	 *               In X11, the left mouse button is 1.
	 *               Here are the mouse button numbers:
	 *
	 *	            - 1 MOUSE_LEFT       (left button)
	 *               - 2 MOUSE_MIDDLE     (middle button / scroll wheel button)
	 *               - 3 MOUSE_RIGHT      (right button)
	 *               - 4 MOUSE_WHEEL_UP   (scroll wheel up)
	 *               - 5 MOUSE_WHEEL_DOWN (scroll wheel down)
	 *
	 * @returns true if the button is pressed, otherwise false
	 */
	inline bool GetMouseButton( uint32_t button ) const	{ if(button > sizeof(mMouseButtons)) return false; return mMouseButtons[button]; }

	/**
	 * Get the state of a key (lowercase, without modifiers applied)
	 *
	 * Similar to glEvent::KEY_STATE, GetKey() queries the raw key state
	 * without being translated by modifier keys such as Shift, CapsLock, 
	 * NumLock, ect.  Alphanumeric keys will be left as lowercase, so
	 * query lowercase keys - uppercase keys will always return false.   
	 *       
	 * @param key the `XK_` key symbol (see `/usr/include/X11/keysymdef.h`)
	 *
	 *            Uppercase keys like XK_A or XK_plus will always return false.
	 *            Instead, query the lowercase keys such as XK_a, XK_1, ect.
	 *
	 *            Other keys like F1-F12, shift, tab, ctrl/alt, arrow keys, 
	 *            backspace, delete, escape, and enter can all be queried.
	 *
	 *            GetKey() caches the first 1024 key symbols. Other keys will
	 *            return false, but can be subscribed to through a glEventHander.
	 *
	 * @returns true if the key is pressed, otherwise false
	 */
	inline bool GetKey( uint32_t key ) const 			{ const uint32_t idx = key - KEY_OFFSET; if(idx > sizeof(mKeyStates)) return false; return mKeyStates[idx]; }

	/**
	 * Process UI event messages.  Any queued events will be dispatched to the
	 * event message handlers that were registered with RegisterEventHandler()
	 *
	 * ProcessEvents() usually gets called automatically by BeginFrame(), so it
	 * is not typically necessary to explicitly call it unless you passed `false`
	 * to BeginFrame() and wish to process events at another time of your choosing.
	 *
	 * @see glEventType
	 * @see glEventHandler 
	 */
	void ProcessEvents();

	/**
	 * Register an event message handler that will be called by ProcessEvents()
	 * @param callback function pointer to the event message handler callback
	 * @param user optional user-specified pointer that will be passed to all
	 *             invocations of this event handler (typically an object)
	 */
	void RegisterEventHandler( glEventHandler callback, void* user=NULL );

	/**
	 * Remove an event message handler from being called by ProcessEvents()
	 * RemoveEventHandler() will search for previously registered event
	 * handlers that have the same function pointer and/or user pointer,
	 * and remove them for being called again in the future.
	 */
	void RemoveEventHandler( glEventHandler callback, void* user=NULL );

	/**
	 * Add a widget to the window that recieves events and is rendered.
	 */
	void AddWidget( glWidget* widget );

	/**
	 * Remove a widget from the window (it will not be deleted)
	 */
	void RemoveWidget( glWidget* widget );

	/**
	 * Retrieve the number of widgets.
	 */
	inline uint32_t GetNumWidgets() const					{ return mWidgets.size(); }

	/**
	 * Retrieve a widget.
	 */
	inline glWidget* GetWidget( const uint32_t index ) const	{ return mWidgets[index]; }

	/**
	 * Enable debugging of events.
	 */
	void EnableDebug();

	/**
	 * Set the window title string.
	 */
	void SetTitle( const char* str );

	/**
	 * Set the window's size.
	 *
	 * @param width the desired width of the window, in pixels.
	 * @param height the desired height of the window, in pixels.
	 *
	 * @note modifying the window's size will reset the viewport
	 *       to cover the full area of the new size of the window.
	 */
	void SetSize( uint32_t width, uint32_t height );

	/**
	 * Maximize or un-maximize the window.
	 */
	void SetMaximized( bool maximized );

	/**
	 * Determine if the window is maximized or not.
	 */
	bool IsMaximized();

	/**
	 * Retrieve the window's background color.
	 */
	void GetBackgroundColor( float* r, float* g, float* b, float* a=NULL );

	/**
	 * Set the window's background color.
	 *
	 * @param r background RGBA color, red component (0.0-1.0f)
	 * @param g background RGBA color, green component (0.0-1.0f)
	 * @param b background RGBA color, blue component (0.0-1.0f)
	 * @param a background RGBA color, alpha component (0.0-1.0f)
	 */
	void SetBackgroundColor( float r, float g, float b, float a=1.0f );

	/**
	 * Set the active mouse cursor.
	 *
	 * @param cursor one of the cursor ID's from `X11/cursorfont.h`
	 * @see ResetCursor() to restore the cursor back to default
	 */
	void SetCursor( uint32_t cursor );

	/**
	 * Reset the mouse cursor back to it's default.
	 */
	void ResetCursor();

	/**
	 * Set the active viewport being rendered to.
	 *
	 * SetViewport() will update the GL_PROJECTION matrix
	 * with a new ortho matrix to reflect these changes.
	 *
	 * After done rendering to this viewport, you should
	 * reset it back to it's original with ResetViewport()
	 */
	void SetViewport( int left, int top, int right, int bottom );

	/**
	 * Reset to the full viewport (and change back GL_PROJECTION)
	 */
	void ResetViewport();

	/**
	 * Default title bar name
	 */
	static const char* DEFAULT_TITLE;

protected:
	glDisplay();
		
	bool initWindow( int width, int height );
	bool initGL();

	glTexture* allocTexture( uint32_t width, uint32_t height );	

	void activateViewport();
	void dispatchEvent( glEventType msg, int a, int b );

	static bool onEvent( uint16_t msg, int a, int b, void* user );

	struct eventHandler
	{
		glEventHandler callback;
		void* user;
	};

	static const int screenIdx = 0;
		
	Display*     mDisplayX;
	Screen*      mScreenX;
	XVisualInfo* mVisualX;
	Window       mWindowX;
	GLXContext   mContextGL;
	Cursor	   mCursors[256];
	int	        mActiveCursor;
	bool		   mRendering;
	bool		   mEnableDebug;
	bool		   mWindowClosed;
	Atom		   mWindowClosedMsg;

	uint32_t mID;
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mScreenWidth;
	uint32_t mScreenHeight;

	timespec mLastTime;
	float    mAvgTime;
	float    mBgColor[4];
	int      mViewport[4];

	int	    mMousePos[2];
	int	    mMouseDrag[2];
	bool	    mMouseButtons[16];
	bool     mKeyStates[1024];

	float*   mNormalizedCUDA;
	uint32_t mNormalizedWidth;
	uint32_t mNormalizedHeight;

	std::vector<glWidget*> mWidgets;
	std::vector<glTexture*> mTextures;
	std::vector<eventHandler> mEventHandlers;
};

/**
 * Retrieve a display window object
 * @ingroup OpenGL
 */
glDisplay* glGetDisplay( uint32_t display=0 );

/**
 * Return the number of created glDisplay windows
 * @ingroup OpenGL
 */
uint32_t glGetNumDisplays();

#endif

