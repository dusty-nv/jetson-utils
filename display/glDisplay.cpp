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
 
#include "glDisplay.h"
#include "cudaNormalize.h"
#include "timespec.h"

#include <X11/Xatom.h>
#include <X11/cursorfont.h>

#include <algorithm>
#include <cstdlib>


//--------------------------------------------------------------
std::vector<glDisplay*> gDisplays;

glDisplay* glGetDisplay( uint32_t display )
{
	if( display >= gDisplays.size() )
		return NULL;
	
	return gDisplays[display];
}

uint32_t glGetNumDisplays()
{
	return gDisplays.size();
}
//--------------------------------------------------------------

const char* glDisplay::DEFAULT_TITLE = "NVIDIA Jetson";

#define OriginalCursor XC_arrow

// Constructor
glDisplay::glDisplay()
{
	mWindowX       = 0;
	mScreenX       = NULL;
	mVisualX       = NULL;
	mContextGL     = NULL;
	mDisplayX      = NULL;
	mRendering     = false;
	mEnableDebug   = false;
	mWindowClosed  = false;
	mActiveCursor  = -1;
	mDefaultCursor = OriginalCursor;
	mDragMode      = DragDefault;

	mID		     = 0;
	mWidth         = 0;
	mHeight        = 0;
	mScreenWidth   = 0;
	mScreenHeight  = 0;
	mAvgTime       = 1.0f;

	mBgColor[0]    = 0.0f;
	mBgColor[1]    = 0.0f;
	mBgColor[2]    = 0.0f;
	mBgColor[3]    = 1.0f;

	mNormalizedCUDA   = NULL;
	mNormalizedWidth  = 0;
	mNormalizedHeight = 0;

	// initial input states
	mMousePos[0]  = 0;
	mMousePos[1]  = 0;

	mMouseDrag[0] = -1;
	mMouseDrag[1] = -1;

	mMouseDragOrigin[0] = -1;
	mMouseDragOrigin[1] = -1;

	memset(mMouseButtons, 0, sizeof(mMouseButtons));
	memset(mKeyStates, 0, sizeof(mKeyStates));
 
	// get the starting time for FPS counter
	clock_gettime(CLOCK_REALTIME, &mLastTime);
	
	// register default event handler
	AddEventHandler(&onEvent, this);
}


// Destructor
glDisplay::~glDisplay()
{
	// remove this instance from the global list
	const size_t numDisplays = gDisplays.size();

	for( size_t n=0; n < numDisplays; n++ )
	{
		if( gDisplays[n] == this )
		{
			gDisplays.erase(gDisplays.begin() + n);
			break;
		}
	}

	// release widgets from the window
	RemoveAllWidgets();

	// release textures used during rendering
	const size_t numTextures = mTextures.size();

	for( size_t n=0; n < numTextures; n++ )
	{
		if( mTextures[n] != NULL )
		{
			delete mTextures[n];
			mTextures[n] = NULL;
		}
	}

	mTextures.clear();

	// free CUDA memory used for normalization
	if( mNormalizedCUDA != NULL )
	{
		CUDA(cudaFree(mNormalizedCUDA));
		mNormalizedCUDA = NULL;
	}

	// destroy the OpenGL context
	glXDestroyContext(mDisplayX, mContextGL);
}


// Create
glDisplay* glDisplay::Create( const char* title, int width, int height, float r, float g, float b, float a )
{
	glDisplay* vp = new glDisplay();
	
	if( !vp )
		return NULL;
		
	if( !vp->initWindow(width, height) )
	{
		printf(LOG_GL "failed to create X11 Window.\n");
		delete vp;
		return NULL;
	}
	
	if( !vp->initGL() )
	{
		printf(LOG_GL "failed to initialize OpenGL.\n");
		delete vp;
		return NULL;
	}
	
	GLenum err = glewInit();
	
	if (GLEW_OK != err)
	{
		printf(LOG_GL "GLEW Error: %s\n", glewGetErrorString(err));
		delete vp;
		return NULL;
	}

	if( title != NULL )
		vp->SetTitle(title);

	vp->SetBackgroundColor(r, g, b, a);
	
	vp->mID = gDisplays.size();
	gDisplays.push_back(vp);

	printf(LOG_GL "glDisplay -- display device initialized\n");
	return vp;
}


// Create
/*glDisplay* glDisplay::Create( float r, float g, float b, float a )
{
	return Create(DEFAULT_TITLE, r, g, b, a);
}*/


// initWindow
bool glDisplay::initWindow( int width, int height )
{
	if( !mDisplayX )
		mDisplayX = XOpenDisplay(0);

	if( !mDisplayX )
	{
		printf(LOG_GL "failed to open X11 server connection.\n");
		return false;
	}

		
	if( !mDisplayX )
	{
		printf(LOG_GL "InitWindow() - no X11 server connection.\n" );
		return false;
	}

	// retrieve screen info
	const int screenIdx   = DefaultScreen(mDisplayX);
	const int screenWidth = DisplayWidth(mDisplayX, screenIdx);
	const int screenHeight = DisplayHeight(mDisplayX, screenIdx);
	
	if( width <= 0 )
		width = screenWidth;

	if( height <= 0 )
		height = screenHeight;

	printf(LOG_GL "glDisplay -- X screen %i resolution:  %ix%i\n", screenIdx, screenWidth, screenHeight);
	printf(LOG_GL "glDisplay -- X window resolution:    %ix%i\n", width, height);
	
	Screen* screen = XScreenOfDisplay(mDisplayX, screenIdx);

	if( !screen )
	{
		printf(LOG_GL "failed to retrieve default Screen instance\n");
		return false;
	}
	
	Window winRoot = XRootWindowOfScreen(screen);

	// get framebuffer format
	static int fbAttribs[] =
	{
			GLX_X_RENDERABLE, True,
			GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
			GLX_RENDER_TYPE, GLX_RGBA_BIT,
			GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
			GLX_RED_SIZE, 8,
			GLX_GREEN_SIZE, 8,
			GLX_BLUE_SIZE, 8,
			GLX_ALPHA_SIZE, 8,
			GLX_DEPTH_SIZE, 24,
			GLX_STENCIL_SIZE, 8,
			GLX_DOUBLEBUFFER, True,
			GLX_SAMPLE_BUFFERS, 0,
			GLX_SAMPLES, 0,
			None
	};

	int fbCount = 0;
	GLXFBConfig* fbConfig = glXChooseFBConfig(mDisplayX, screenIdx, fbAttribs, &fbCount);

	if( !fbConfig || fbCount == 0 )
		return false;

	// get a 'visual'
	XVisualInfo* visual = glXGetVisualFromFBConfig(mDisplayX, fbConfig[0]);

	if( !visual )
		return false;

	// populate windows attributes
	XSetWindowAttributes winAttr;
	winAttr.colormap = XCreateColormap(mDisplayX, winRoot, visual->visual, AllocNone);
	winAttr.background_pixmap = None;
	winAttr.border_pixel = 0;
	winAttr.event_mask = StructureNotifyMask|KeyPressMask|KeyReleaseMask|PointerMotionMask|ButtonPressMask|ButtonReleaseMask;

	
	// create window
	Window win = XCreateWindow(mDisplayX, winRoot, 0, 0, width, height, 0,
						  visual->depth, InputOutput, visual->visual, 
						  CWBorderPixel|CWColormap|CWEventMask, &winAttr);

	if( !win )
		return false;


	// setup WM_DELETE message
	mWindowClosedMsg = XInternAtom(mDisplayX, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(mDisplayX, win, &mWindowClosedMsg, 1);

	// set default window title
	XStoreName(mDisplayX, win, DEFAULT_TITLE);

	// show the window
	XMapWindow(mDisplayX, win);

	// store variables
	mWindowX = win;
	mScreenX = screen;
	mVisualX = visual;
	mWidth   = width;
	mHeight  = height;

	mScreenWidth  = screenWidth;
	mScreenHeight = screenHeight;

	mViewport[0] = 0; 
	mViewport[1] = 0; 
	mViewport[2] = mWidth; 
	mViewport[3] = mHeight;

	XFree(fbConfig);
	return true;
}


// SetTitle
void glDisplay::SetTitle( const char* str )
{
	XStoreName(mDisplayX, mWindowX, str);
}


// initGL
bool glDisplay::initGL()
{
	mContextGL = glXCreateContext(mDisplayX, mVisualX, 0, True);

	if( !mContextGL )
		return false;

	GL(glXMakeCurrent(mDisplayX, mWindowX, mContextGL));

	GL(glEnable(GL_LINE_SMOOTH));
	GL(glHint(GL_LINE_SMOOTH_HINT, GL_NICEST));

	return true;
}


// SetViewport
void glDisplay::SetViewport( int left, int top, int right, int bottom )
{
	const int width  = right - left;
	const int height = bottom - top;

	mViewport[0] = left;
	mViewport[1] = mHeight - bottom;
	mViewport[2] = width;
	mViewport[3] = height;

	if( mRendering )
		activateViewport();
}


// ResetViewport
void glDisplay::ResetViewport()
{
	mViewport[0] = 0;
	mViewport[1] = 0;
	mViewport[2] = mWidth;
	mViewport[3] = mHeight;

	if( mRendering )
		activateViewport();
}


// activateViewport
void glDisplay::activateViewport()
{
	GL(glViewport(mViewport[0], mViewport[1], mViewport[2], mViewport[3]));
	GL(glMatrixMode(GL_PROJECTION));
	GL(glLoadIdentity());
	GL(glOrtho(0.0f, mViewport[2], mViewport[3], 0.0f, 0.0f, 1.0f));
}


// MakeCurrent
void glDisplay::BeginRender( bool processEvents )
{
	if( processEvents )
		ProcessEvents();

	mRendering = true;

	GL(glXMakeCurrent(mDisplayX, mWindowX, mContextGL));

	GL(glClearColor(mBgColor[0], mBgColor[1], mBgColor[2], mBgColor[3]));
	GL(glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT));

	activateViewport();	
}


// EndRender
void glDisplay::EndRender()
{
	// render widgets
	const size_t numWidgets = mWidgets.size();

	for( size_t n=0; n < numWidgets; n++ )
		mWidgets[n]->Render();

	// dragging rect (ignore if origin inside an existing widget)
	if( (mDragMode == DragSelect || mDragMode == DragCreate) && IsDragging(mDragMode) )
	{
		int x, y;
		int width, height;

		if( GetDragRect(&x, &y, &width, &height) )
			RenderOutline(x, y, width, height, 1, 1, 1);
	}

	// present the backbuffer
	glXSwapBuffers(mDisplayX, mWindowX);

	// measure framerate
	timespec currTime;
	clock_gettime(CLOCK_REALTIME, &currTime);

	const timespec diffTime = timeDiff(mLastTime, currTime);
	const float ns = 1000000000 * diffTime.tv_sec + diffTime.tv_nsec;

	mAvgTime   = mAvgTime * 0.8f + ns * 0.2f;
	mLastTime  = currTime;
	mRendering = false;
}


// allocTexture
glTexture* glDisplay::allocTexture( uint32_t width, uint32_t height )
{
	if( width == 0 || height == 0 )
		return NULL;

	const size_t numTextures = mTextures.size();

	for( size_t n=0; n < numTextures; n++ )
	{
		glTexture* tex = mTextures[n];

		if( tex->GetWidth() == width && tex->GetHeight() == height )
			return tex;
	}

	glTexture* tex = glTexture::Create(width, height, GL_RGBA32F_ARB);

	if( !tex )
	{
		printf(LOG_GL "glDisplay.Render() failed to create OpenGL interop texture\n");
		return NULL;
	}

	mTextures.push_back(tex);
	return tex;
}


// Render
void glDisplay::Render( glTexture* texture, float x, float y )
{
	if( !texture )
		return;

	texture->Render(x,y);
}


// Render
void glDisplay::Render( float* img, uint32_t width, uint32_t height, float x, float y, bool normalize )
{
	if( !img || width == 0 || height == 0 )
		return;
	
	// obtain the OpenGL texture to use
	glTexture* interopTex = allocTexture(width, height);

	if( !interopTex )
		return;
	
	// normalize pixels from [0,255] -> [0,1]
	if( normalize )
	{
		if( !mNormalizedCUDA || mNormalizedWidth < width || mNormalizedHeight < height )
		{
			if( mNormalizedCUDA != NULL )
			{
				CUDA(cudaFree(mNormalizedCUDA));
				mNormalizedCUDA = NULL;
			}

			if( CUDA_FAILED(cudaMalloc(&mNormalizedCUDA, width * height * sizeof(float) * 4)) )
			{
				printf(LOG_GL "glDisplay.Render() failed to allocate CUDA memory for normalization\n");
				return;
			}

			mNormalizedWidth = width;
			mNormalizedHeight = height;
		}

		// rescale image pixel intensities for display
		CUDA(cudaNormalizeRGBA((float4*)img, make_float2(0.0f, 255.0f), 
						   (float4*)mNormalizedCUDA, make_float2(0.0f, 1.0f), 
 						   width, height));
	}

	// map from CUDA to openGL using GL interop
	void* tex_map = interopTex->Map(GL_MAP_CUDA, GL_WRITE_DISCARD); //interopTex->MapCUDA();

	if( tex_map != NULL )
	{
		CUDA(cudaMemcpy(tex_map, normalize ? mNormalizedCUDA : img, interopTex->GetSize(), cudaMemcpyDeviceToDevice));
		//CUDA(cudaDeviceSynchronize());
		interopTex->Unmap();
	}

	// draw the texture
	interopTex->Render(x,y);
}


// RenderOnce
void glDisplay::RenderOnce( float* img, uint32_t width, uint32_t height, float x, float y, bool normalize )
{
	BeginRender();
	Render(img, width, height, x, y, normalize);
	EndRender();
}


// RenderLine
void glDisplay::RenderLine( float x1, float y1, float x2, float y2, float r, float g, float b, float a, float thickness )
{
	glDrawLine(x1, y1, x2, y2, r, g, b, a);
}


// RenderOutline
void glDisplay::RenderOutline( float left, float top, float width, float height, float r, float g, float b, float a, float thickness )
{
	glDrawOutline(left, top, width, height, r, g, b, a);
}


// RenderRect
void glDisplay::RenderRect( float left, float top, float width, float height, float r, float g, float b, float a )
{
	glDrawRect(left, top, width, height, r, g, b, a);
}


// RenderRect
void glDisplay::RenderRect( float r, float g, float b, float a )
{
	RenderRect(0, 0, mViewport[2], mViewport[3], r, g, b, a);
}


// GetBackgroundColor
void glDisplay::GetBackgroundColor( float* r, float* g, float* b, float* a )
{
	if( r != NULL )
		*r = mBgColor[0];

	if( g != NULL )
		*g = mBgColor[1];

	if( b != NULL )
		*b = mBgColor[2];

	if( a != NULL )
		*a = mBgColor[3];
}


// SetBackgroundColor
void glDisplay::SetBackgroundColor( float r, float g, float b, float a )
{
	mBgColor[0] = r; 
	mBgColor[1] = g; 
	mBgColor[2] = b; 
	mBgColor[3] = a; 
}


// IsMaximied
bool glDisplay::IsMaximized()
{
	Atom _NET_WM_STATE = XInternAtom(mDisplayX, "_NET_WM_STATE", False);
	Atom _NET_WM_STATE_MAXIMIZED_VERT = XInternAtom(mDisplayX, "_NET_WM_STATE_MAXIMIZED_VERT", False);
	Atom _NET_WM_STATE_MAXIMIZED_HORZ = XInternAtom(mDisplayX, "_NET_WM_STATE_MAXIMIZED_HORZ", False);

	Atom actualType;
	int actualFormat;
	unsigned long numItems, bytesAfter;
	unsigned char* propertyValue = NULL;
	long maxLength = 1024;
	bool maximized = false;

	const int error = XGetWindowProperty(mDisplayX, mWindowX, _NET_WM_STATE,
                        				  0, maxLength, False, XA_ATOM, &actualType,
                        				  &actualFormat, &numItems, &bytesAfter,
                        				  &propertyValue);

	if( error != Success )
	{
		printf(LOG_GL "glDisplay -- failed to get window properties (error=%i)\n", error);
		return false;
	}

	Atom* atoms = (Atom*)propertyValue;

	for( unsigned long i = 0; i < numItems; i++ ) 
	{
		if( atoms[i] == _NET_WM_STATE_MAXIMIZED_VERT )
			maximized = true;
		else if( atoms[i] == _NET_WM_STATE_MAXIMIZED_HORZ)
			maximized = true;
 	}

	XFree(propertyValue);
	return maximized;
}


// SetMaximized
void glDisplay::SetMaximized( bool maximized )
{
	Atom _NET_WM_STATE = XInternAtom(mDisplayX, "_NET_WM_STATE", False);
	Atom _NET_WM_STATE_MAXIMIZED_VERT = XInternAtom(mDisplayX, "_NET_WM_STATE_MAXIMIZED_VERT", False);
	Atom _NET_WM_STATE_MAXIMIZED_HORZ = XInternAtom(mDisplayX, "_NET_WM_STATE_MAXIMIZED_HORZ", False);

	XEvent e;
	memset(&e, 0, sizeof(XEvent));

	e.xany.type            = ClientMessage;
	e.xclient.message_type = _NET_WM_STATE;
	e.xclient.format       = 32;
	e.xclient.window       = mWindowX;
	e.xclient.data.l[0]    = maximized ? 1 : 0;
	e.xclient.data.l[1]    = _NET_WM_STATE_MAXIMIZED_VERT;
	e.xclient.data.l[2]    = _NET_WM_STATE_MAXIMIZED_HORZ;
	e.xclient.data.l[3]    = 0;

	const int error = XSendEvent(mDisplayX, RootWindow(mDisplayX, 0), 0,
                   			    SubstructureNotifyMask | SubstructureRedirectMask, 
						    &e);

	if( error == 0 )
		printf(LOG_GL "glDisplay -- failed to %s window\n", maximized ? "maximize" : "un-maximize");
}
	

// SetSize
void glDisplay::SetSize( uint32_t width, uint32_t height )
{
	if( mWidth == width && mHeight == height )
		return;

	// un-maximized the window if new size not fullscreen
	if( width != mScreenWidth || height != mScreenHeight )
		SetMaximized(false);

	// resize the window to the new resolution
	const int error = XResizeWindow(mDisplayX, mWindowX, width, height);

	if( error != 1 )
	{
		printf(LOG_GL "glDisplay -- failed to set window size to %ux%u (error=%i)\n", width, height, error);
		return;
	}

	printf(LOG_GL "glDisplay -- set the window size to %ux%u\n", width, height); 

	mWidth = width;
	mHeight = height;

	ResetViewport();
}


// SetCursor
void glDisplay::SetCursor( uint32_t cursor )
{
	if( cursor >= XC_num_glyphs )
	{
		printf(LOG_GL "glDisplay -- invalid mouse cursor '%u'\n", cursor);
		return;
	}

	if( !mCursors[cursor] )
		mCursors[cursor] = XCreateFontCursor(mDisplayX, cursor);

	if( !mCursors[cursor] )
	{
		printf(LOG_GL "glDisplay -- failed to load mouse cursor '%u'\n", cursor);
		return;
	}

	const int error = XDefineCursor(mDisplayX, mWindowX, mCursors[cursor]);

	if( error != 1 )
		printf(LOG_GL "glDisplay -- failed to set mouse cursor '%u' (error=%i)\n", cursor, error);
}


// SetDefaultCursor
void glDisplay::SetDefaultCursor( uint32_t cursor, bool activate )
{
	if( cursor >= XC_num_glyphs )
	{
		printf(LOG_GL "glDisplay -- invalid mouse cursor '%u'\n", cursor);
		return;
	}

	mDefaultCursor = cursor;

	if( activate )
		ResetCursor();
}


// ResetCursor
void glDisplay::ResetCursor()
{
	SetCursor(mDefaultCursor);
}


// ResetDefaultCursor
void glDisplay::ResetDefaultCursor( bool activate )
{
	SetDefaultCursor(OriginalCursor, activate);
}


// GetDragRect
bool glDisplay::GetDragRect( int* x, int* y, int* width, int* height )
{
	if( mDragMode == DragDisabled || !IsDragging() )
		return false;

	if( x != NULL )
		*x = std::min(mMouseDragOrigin[0], mMouseDrag[0]);

	if( y != NULL )
		*y = std::min(mMouseDragOrigin[1], mMouseDrag[1]);

	if( width != NULL )
		*width = std::abs(mMouseDrag[0] - mMouseDragOrigin[0]);

	if( height != NULL )
		*height = std::abs(mMouseDrag[1] - mMouseDragOrigin[1]);

	return true;
}


// GetDragCoords
bool glDisplay::GetDragCoords( int* x1, int* y1, int* x2, int* y2 )
{
	if( mDragMode == DragDisabled || !IsDragging() )
		return false;

	if( x1 != NULL )
		*x1 = std::min(mMouseDragOrigin[0], mMouseDrag[0]);

	if( y1 != NULL )
		*y1 = std::min(mMouseDragOrigin[1], mMouseDrag[1]);

	if( x2 != NULL )
		*x2 = std::max(mMouseDragOrigin[0], mMouseDrag[0]);

	if( y2 != NULL )
		*y2 = std::max(mMouseDragOrigin[1], mMouseDrag[1]);

	return true;
}


// AddWidget
glWidget* glDisplay::AddWidget( glWidget* widget )
{
	if( !widget )
		return NULL;

	mWidgets.push_back(widget);
	widget->setDisplay(this);

	return widget;
}


// RemoveWidget
void glDisplay::RemoveWidget( glWidget* widget, bool deleteWidget )
{
	const int index = GetWidgetIndex(widget);

	if( index < 0 )
		return;

	RemoveWidget(index, deleteWidget);
}


// RemoveWidget
void glDisplay::RemoveWidget( uint32_t n, bool deleteWidget )
{
	mWidgets[n]->setDisplay(NULL);
	
	if( deleteWidget )
		delete mWidgets[n];

	mWidgets[n] = NULL;
	mWidgets.erase(mWidgets.begin()+n);
}


// RemoveAllWidgets
void glDisplay::RemoveAllWidgets( bool deleteWidgets )
{
	const size_t numWidgets = mWidgets.size();

	for( size_t n=0; n < numWidgets; n++ )
		RemoveWidget(n, deleteWidgets);

	mWidgets.clear();
}


// GetWidgetIndex
int glDisplay::GetWidgetIndex( const glWidget* widget ) const
{
	const size_t numWidgets = mWidgets.size();

	for( size_t n=0; n < numWidgets; n++ )
	{
		if( mWidgets[n] == widget )
			return n;
	}

	return -1;
}


// FindWidget
glWidget* glDisplay::FindWidget( int x, int y )
{
	const size_t numWidgets = mWidgets.size();

	for( size_t n=0; n < numWidgets; n++ )
	{
		if( mWidgets[n]->Contains(x,y) )
			return mWidgets[n];
	}

	return NULL;
}


// FindWidgets
std::vector<glWidget*> glDisplay::FindWidgets( int x, int y )
{
	std::vector<glWidget*> widgets;

	const size_t numWidgets = mWidgets.size();

	for( size_t n=0; n < numWidgets; n++ )
	{
		if( mWidgets[n]->Contains(x,y) )
			widgets.push_back(mWidgets[n]);
	}

	return widgets;
}


// EnableDebug
void glDisplay::EnableDebug()
{
	mEnableDebug = true;
}


// ProcessEvents()
void glDisplay::ProcessEvents()
{
	// reset input states
	/*mMouseEvent     = false;
	mMouseDownEvent = false;
	mMouseDblClick  = false;
	mMouseWheel     = 0;
	mKeyText		= 0;*/

	XEvent evt;

	while( XEventsQueued(mDisplayX, QueuedAlready) > 0 )
	{
		XNextEvent(mDisplayX, &evt);

		if( evt.type == KeyPress || evt.type == KeyRelease )
		{
			const int keyPressed = (evt.type == KeyPress) ? KEY_PRESSED : KEY_RELEASED;

			// ignores modifiers (raw)
			const KeySym keySymbolRaw = XLookupKeysym(&evt.xkey, 0);	

			if( keySymbolRaw != NoSymbol )
			{
				const uint32_t idx = (uint32_t)keySymbolRaw - KEY_OFFSET;

				if( idx < sizeof(mKeyStates) )
					mKeyStates[idx] = keyPressed;
			
				dispatchEvent(KEY_STATE, (int)keySymbolRaw, keyPressed);
			}

			// apply modifier translation
			char keyStr[32];
			KeySym keySymbol;
			const int strLen = XLookupString(&evt.xkey, keyStr, sizeof(keyStr), &keySymbol, NULL);

			if( keySymbol != NoSymbol )
			{
				dispatchEvent(KEY_MODIFIED, (int)keySymbol, keyPressed);

				if( evt.type == KeyPress && strLen == 1 )
					dispatchEvent(KEY_CHAR, (int)keyStr[0], 0);
			}
		}
		else if( evt.type == ButtonPress || evt.type == ButtonRelease )
		{
			const int buttonPressed = (evt.type == ButtonPress) ? MOUSE_PRESSED : MOUSE_RELEASED;

			if( evt.xbutton.button < sizeof(mMouseButtons) )
				mMouseButtons[evt.xbutton.button] = buttonPressed;			
		
			dispatchEvent(MOUSE_BUTTON, evt.xbutton.button, buttonPressed);

			if( buttonPressed )
			{
				// handle mouse wheel scrolling
				if( evt.xbutton.button == MOUSE_WHEEL_UP )
				{
					dispatchEvent(MOUSE_WHEEL, -1, 0);
				}
				else if( evt.xbutton.button == MOUSE_WHEEL_DOWN )
				{
					dispatchEvent(MOUSE_WHEEL, 1, 0);
				}
				else if( evt.xbutton.button == MOUSE_LEFT && mDragMode != DragDisabled )
				{
					// kick off dragging (except when in Creation mode and inside another widget)
					if( !(mDragMode == DragCreate && FindWidget(mMousePos[0], mMousePos[1]) != NULL) )
					{
						mMouseDragOrigin[0] = mMousePos[0];
						mMouseDragOrigin[1] = mMousePos[1];
					}
				}
			}
			else
			{
				// reset drag when left button released
				if( evt.xbutton.button == MOUSE_LEFT )
				{
					if( IsDragging(mDragMode) /*&& FindWidget(mMouseDragOrigin[0], mMouseDragOrigin[1]) == NULL*/ )
					{
						if( mDragMode == DragCreate )
						{
							int x, y;
							int width, height;

							GetDragRect(&x, &y, &width, &height);

							glWidget* widget = new glWidget(x, y, width, height);

							widget->SetMoveable(true);
							widget->SetResizeable(true);

							AddWidget(widget);
							dispatchEvent(WIDGET_CREATED, GetNumWidgets()-1, 0);
						}					
					}

					mMouseDrag[0] = -1;
					mMouseDrag[1] = -1;

					mMouseDragOrigin[0] = -1;
					mMouseDragOrigin[1] = -1;
				}
			}
		}
		else if( evt.type == MotionNotify )
		{
			// relative coordinates
			mMousePos[0] = evt.xmotion.x;
			mMousePos[1] = evt.xmotion.y;

			dispatchEvent(MOUSE_MOVE, evt.xmotion.x, evt.xmotion.y);

			// absolute coordinates
			XWindowAttributes attr;
			XGetWindowAttributes(mDisplayX, evt.xmotion.root, &attr);

			dispatchEvent(MOUSE_ABSOLUTE, evt.xmotion.x_root + attr.x, evt.xmotion.y_root + attr.y);

			// handle drag events
			if( mDragMode != DragDisabled && (evt.xmotion.state & Button1Mask) )
			{
				if( IsDragging() )
				{
					const int delta_x = evt.xmotion.x - mMouseDrag[0];
					const int delta_y = evt.xmotion.y - mMouseDrag[1];

					if( delta_x != 0 || delta_y != 0 )
						dispatchEvent(MOUSE_DRAG, delta_x, delta_y);
				}

				mMouseDrag[0] = evt.xmotion.x;
				mMouseDrag[1] = evt.xmotion.y;
			}

			// reset mouse cursor when outside of widgets
			if( FindWidget(mMousePos[0], mMousePos[1]) == NULL )
				ResetCursor();
		}
		else if( evt.type == ConfigureNotify )
		{
			if( evt.xconfigure.width != mWidth || evt.xconfigure.height != mHeight )
			{
				const int prevWidth = mWidth;
				const int prevHeight = mHeight;

				mWidth = evt.xconfigure.width;
				mHeight = evt.xconfigure.height;

				if( mViewport[2] == prevWidth && mViewport[3] == prevHeight )
					SetViewport(0, 0, evt.xconfigure.width, evt.xconfigure.height);

				dispatchEvent(WINDOW_RESIZED, mWidth, mHeight);
			}
		}
		else if( evt.type == ClientMessage )
		{
			if( evt.xclient.data.l[0] == mWindowClosedMsg )
				dispatchEvent(WINDOW_CLOSED, 0, 0);
		}
	}
}


// AddEventHandler
void glDisplay::AddEventHandler( glEventHandler callback, void* user )
{
	if( !callback )
		return;

	eventHandler handler;

	handler.callback = callback;
	handler.user = user;

	mEventHandlers.push_back(handler);
}


// RemoveEventHandler
void glDisplay::RemoveEventHandler( glEventHandler callback, void* user )
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
void glDisplay::dispatchEvent( uint16_t msg, int a, int b )
{
	const uint32_t numHandlers = mEventHandlers.size();

	for( uint32_t n=0; n < numHandlers; n++ )
		mEventHandlers[n].callback(msg, a, b, mEventHandlers[n].user);
}


// onEvent
bool glDisplay::onEvent( uint16_t msg, int a, int b, void* user )
{
	if( !user )
		return false;

	glDisplay* display = (glDisplay*)user;

	switch(msg)
	{
		case MOUSE_MOVE:
		{
			if( display->mEnableDebug )
				printf(LOG_GL "glDisplay -- event MOUSE_MOVE (%i, %i)\n", a, b);

			break;
		}
		case MOUSE_ABSOLUTE:
		{
			if( display->mEnableDebug )
				printf(LOG_GL "glDisplay -- event MOUSE_ABSOLUTE (%i, %i)\n", a, b);

			break;
		}
		case MOUSE_BUTTON:
		{
			if( display->mEnableDebug )
				printf(LOG_GL "glDisplay -- event MOUSE_BUTTON %i (%s)\n", a, b ? "pressed" : "released");
	
			break;
		}
		case MOUSE_DRAG:
		{
			if( display->mEnableDebug )
				printf(LOG_GL "glDisplay -- event MOUSE_DRAG (%i, %i)\n", a, b);

			break;
		}
		case MOUSE_WHEEL:
		{
			if( display->mEnableDebug )
				printf(LOG_GL "glDisplay -- event MOUSE_WHEEL %i\n", a);
	 
			break;
		}
		case KEY_STATE:
		{
			if( display->mEnableDebug )
				printf(LOG_GL "glDisplay -- event KEY_STATE %i %s (%s)\n", a, XKeysymToString(a), b ? "pressed" : "released");

			if( a == XK_Escape && b == KEY_PRESSED )
			{
				XEvent ev;
				memset(&ev, 0, sizeof(ev));

				ev.xclient.type = ClientMessage;
				ev.xclient.window = display->mWindowX;
				ev.xclient.message_type = XInternAtom(display->mDisplayX, "WM_PROTOCOLS", true);
				ev.xclient.format = 32;
				ev.xclient.data.l[0] = XInternAtom(display->mDisplayX, "WM_DELETE_WINDOW", false);
				ev.xclient.data.l[1] = CurrentTime;
				XSendEvent(display->mDisplayX, display->mWindowX, False, NoEventMask, &ev);
			}

			break;
		}
		case KEY_MODIFIED:
		{
			if( display->mEnableDebug )
				printf(LOG_GL "glDisplay -- event KEY_MODIFIED %i %s (%s)\n", a, XKeysymToString(a), b ? "pressed" : "released");

			break;
		}
		case KEY_CHAR:
		{
			if( display->mEnableDebug )
				printf(LOG_GL "glDisplay -- event KEY_CHAR %c (%i)\n", (char)a, a);

			break;
		}
		case WIDGET_CREATED:
		{
			if( display->mEnableDebug )
			{
				float x1, y1;
				float x2, y2;

				display->GetWidget(a)->GetCoords(&x1, &y1, &x2, &y2);
				printf(LOG_GL "glDisplay -- event WIDGET_CREATE (%i, %i) (%i, %i) (index=%i)\n", (int)x1, (int)y1, (int)x2, (int)y2, a);
			}

			break;
		}
		case WINDOW_RESIZED:
		{
			if( display->mEnableDebug )
				printf(LOG_GL "glDisplay -- event WINDOW_RESIZED (%i, %i)\n", a, b);

			return true;
		}
		case WINDOW_CLOSED:
		{
			printf(LOG_GL "glDisplay -- the window has been closed\n");
			display->mWindowClosed = true;
			return true;
		}
	}

	// dispatch event to applicable widgets
	const size_t numWidgets = display->mWidgets.size();

	for( size_t n=0; n < numWidgets; n++ )
	{
		glWidget* widget = display->mWidgets[n];

		if( !widget->IsVisible() )
			continue;

		if( widget->Contains(display->mMousePos[0], display->mMousePos[1]) || widget->mDragState != glWidget::DragNone )
			widget->OnEvent(msg, a, b, user);
	}

	return false;
}


