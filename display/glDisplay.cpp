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


// DEFAULT_TITLE
const char* glDisplay::DEFAULT_TITLE = "NVIDIA Jetson";


// Constructor
glDisplay::glDisplay()
{
	mWindowX    = 0;
	mScreenX    = NULL;
	mVisualX    = NULL;
	mContextGL  = NULL;
	mDisplayX   = NULL;

	mWidth      = 0;
	mHeight     = 0;
	mAvgTime    = 1.0f;

	mBgColor[0] = 0.0f;
	mBgColor[1] = 0.0f;
	mBgColor[2] = 0.0f;
	mBgColor[3] = 1.0f;

	mNormalizedCUDA   = NULL;
	mNormalizedWidth  = 0;
	mNormalizedHeight = 0;

	mWindowClosed = false;

	clock_gettime(CLOCK_REALTIME, &mLastTime);
}


// Destructor
glDisplay::~glDisplay()
{
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

	if( mNormalizedCUDA != NULL )
	{
		CUDA(cudaFree(mNormalizedCUDA));
		mNormalizedCUDA = NULL;
	}

	glXDestroyContext(mDisplayX, mContextGL);
}


// Create
glDisplay* glDisplay::Create( const char* title, float r, float g, float b, float a )
{
	glDisplay* vp = new glDisplay();
	
	if( !vp )
		return NULL;
		
	if( !vp->initWindow() )
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

	printf(LOG_GL "glDisplay -- display device initialized\n");
	return vp;
}


// Create
glDisplay* glDisplay::Create( float r, float g, float b, float a )
{
	return Create(DEFAULT_TITLE, r, g, b, a);
}


// initWindow
bool glDisplay::initWindow()
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
	
	printf(LOG_GL "glDisplay -- X screen %i resolution:  %ix%i\n", screenIdx, screenWidth, screenHeight);
	
	Screen* screen = XScreenOfDisplay(mDisplayX, screenIdx);

	if( !screen )
	{
		printf("failed to retrieve default Screen instance\n");
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
	Window win = XCreateWindow(mDisplayX, winRoot, 0, 0, screenWidth, screenHeight, 0,
							   visual->depth, InputOutput, visual->visual, CWBorderPixel|CWColormap|CWEventMask, &winAttr);

	if( !win )
		return false;


	// setup WM_DELETE message
	mWindowClosedMsg = XInternAtom(mDisplayX, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(mDisplayX, win, &mWindowClosedMsg, 1);

	// set default window title
	XStoreName(mDisplayX, win, DEFAULT_TITLE);

	// show the window
	XMapWindow(mDisplayX, win);

	// cleanup
	mWindowX = win;
	mScreenX = screen;
	mVisualX = visual;
	mWidth   = screenWidth;
	mHeight  = screenHeight;
	
	XFree(fbConfig);
	return true;
}


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

	return true;
}


// MakeCurrent
void glDisplay::BeginRender( bool userEvents )
{
	if( userEvents )
		UserEvents();

	GL(glXMakeCurrent(mDisplayX, mWindowX, mContextGL));

	GL(glClearColor(mBgColor[0], mBgColor[1], mBgColor[2], mBgColor[3]));
	GL(glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT));

	GL(glViewport(0, 0, mWidth, mHeight));
	GL(glMatrixMode(GL_PROJECTION));
	GL(glLoadIdentity());
	GL(glOrtho(0.0f, mWidth, mHeight, 0.0f, 0.0f, 1.0f));	
}


// EndRender
void glDisplay::EndRender()
{
	glXSwapBuffers(mDisplayX, mWindowX);

	// measure framerate
	timespec currTime;
	clock_gettime(CLOCK_REALTIME, &currTime);

	const timespec diffTime = timeDiff(mLastTime, currTime);
	const float ns = 1000000000 * diffTime.tv_sec + diffTime.tv_nsec;

	mAvgTime  = mAvgTime * 0.8f + ns * 0.2f;
	mLastTime = currTime;
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
	void* tex_map = interopTex->MapCUDA();

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



#define MOUSE_MOVE		0
#define MOUSE_BUTTON	1
#define MOUSE_WHEEL		2
#define MOUSE_DOUBLE	3
#define KEY_STATE		4
#define KEY_CHAR		5
#define WINDOW_CLOSED	6


// OnEvent
void glDisplay::onEvent( uint msg, int a, int b )
{
	switch(msg)
	{
		case MOUSE_MOVE:
		{
			//mMousePos.Set(a,b);
			break;
		}
		case MOUSE_BUTTON:
		{
			/*if( mMouseButton[a] != (bool)b )
			{
				mMouseButton[a] = b;

				if( b )
					mMouseDownEvent = true;

				// ignore right-mouse up events
				if( !(a == 1 && !b) )
					mMouseEvent = true;
			}*/

			break;
		}
		case MOUSE_DOUBLE:
		{
			/*mMouseDblClick = b;

			if( b )
			{
				mMouseEvent = true;
				mMouseDownEvent = true;
			}*/

			break;
		}
		case MOUSE_WHEEL:
		{
			//mMouseWheel = a;
			break;
		}
		case KEY_STATE:
		{
			//mKeys[a] = b;
			break;
		}
		case KEY_CHAR:
		{
			//mKeyText = a;
			break;
		}
		case WINDOW_CLOSED:
		{
			printf(LOG_GL "glDisplay -- the window has been closed\n");
			mWindowClosed = true;
			break;
		}
	}

	//if( msg == MOUSE_MOVE || msg == MOUSE_BUTTON || msg == MOUSE_DOUBLE || msg == MOUSE_WHEEL )
	//	mMouseEventLast = time();
}


// UserEvents()
void glDisplay::UserEvents()
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

		switch( evt.type )
		{
			case KeyPress:	      onEvent(KEY_STATE, evt.xkey.keycode, 1);		break;
			case KeyRelease:     onEvent(KEY_STATE, evt.xkey.keycode, 0);		break;
			case ButtonPress:	 onEvent(MOUSE_BUTTON, evt.xbutton.button, 1); 	break;
			case ButtonRelease:  onEvent(MOUSE_BUTTON, evt.xbutton.button, 0);	break;
			case MotionNotify:
			{
				XWindowAttributes attr;
				XGetWindowAttributes(mDisplayX, evt.xmotion.root, &attr);
				onEvent(MOUSE_MOVE, evt.xmotion.x_root + attr.x, evt.xmotion.y_root + attr.y);
				break;
			}
			case ClientMessage:
			{
				if( evt.xclient.data.l[0] == mWindowClosedMsg )
					onEvent(WINDOW_CLOSED, 0, 0);

				break;
			}
		}
	}
}

