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

#ifndef __OPENGL_UTILITY_H
#define __OPENGL_UTILITY_H


#include <GL/glew.h>
#include <GL/glx.h>

#include <stdio.h>
#include "logging.h"


/**
 * OpenGL logging prefix.
 * @ingroup OpenGL
 */
#define LOG_GL   			"[OpenGL] "

/**
 * OpenGL error-checking macro
 * @ingroup OpenGL
 */
#define GL(x)				{ x; glCheckError( #x, __FILE__, __LINE__ ); }

/**
 * Return false on OpenGL error.
 * @ingroup OpenGL
 */
#define GL_VERIFY(x)		{ x; if(glCheckError( #x, __FILE__, __LINE__ )) return false; }

/**
 * OpenGL NULL on OpenGL error.
 * @ingroup OpenGL
 */
#define GL_VERIFYN(x)		{ x; if(glCheckError( #x, __FILE__, __LINE__ )) return NULL; }

/**
 * Print a message on OpenGL error.
 * @ingroup OpenGL
 */
#define GL_CHECK(msg)		{ glCheckError(msg, __FILE__, __LINE__); }

/**
 * OpenGL error-checking messsage function.
 * @ingroup OpenGL
 */
inline bool glCheckError(const char* msg, const char* file, int line)
{
	GLenum err = glGetError();

	if( err == GL_NO_ERROR )
		return false;

	const char* e = NULL;

	switch(err)
	{
		  case GL_INVALID_ENUM:          e = "invalid enum";      break;
		  case GL_INVALID_VALUE:         e = "invalid value";     break;
		  case GL_INVALID_OPERATION:     e = "invalid operation"; break;
		  case GL_STACK_OVERFLOW:        e = "stack overflow";    break;
		  case GL_STACK_UNDERFLOW:       e = "stack underflow";   break;
		  case GL_OUT_OF_MEMORY:         e = "out of memory";     break;
		#ifdef GL_TABLE_TOO_LARGE_EXT
		  case GL_TABLE_TOO_LARGE_EXT:   e = "table too large";   break;
		#endif
		#ifdef GL_TEXTURE_TOO_LARGE_EXT
		  case GL_TEXTURE_TOO_LARGE_EXT: e = "texture too large"; break;
		#endif
		  default:						 e = "unknown error";
	}

	LogError(LOG_GL "Error %i - '%s'\n", (uint)err, e);
	LogError(LOG_GL "   %s::%i\n", file, line );
	LogError(LOG_GL "   %s\n", msg );
	
	return true;
}


/**
 * OpenGL error check + logging
 * @ingroup OpenGL
 */
inline bool glCheckError(const char* msg)
{
	GLenum err = glGetError();

	if( err == GL_NO_ERROR )
		return false;

	const char* e = NULL;

	switch(err)
	{
		  case GL_INVALID_ENUM:          e = "invalid enum";      break;
		  case GL_INVALID_VALUE:         e = "invalid value";     break;
		  case GL_INVALID_OPERATION:     e = "invalid operation"; break;
		  case GL_STACK_OVERFLOW:        e = "stack overflow";    break;
		  case GL_STACK_UNDERFLOW:       e = "stack underflow";   break;
		  case GL_OUT_OF_MEMORY:         e = "out of memory";     break;
		#ifdef GL_TABLE_TOO_LARGE_EXT
		  case GL_TABLE_TOO_LARGE_EXT:   e = "table too large";   break;
		#endif
		#ifdef GL_TEXTURE_TOO_LARGE_EXT
		  case GL_TEXTURE_TOO_LARGE_EXT: e = "texture too large"; break;
		#endif
		  default:						 e = "unknown error";
	}

	LogError(LOG_GL "%s    (error %i - %s)\n", msg, (uint)err, e);
	return true;
}


/**
 * Print the amount of free GPU memory.
 * @ingroup OpenGL
 */
inline void glPrintFreeMem()
{
	GLint total_mem_kb = 0;
	GLint cur_avail_mem_kb = 0;

	const GLenum GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX = 0x9048;
	const GLenum GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX = 0x9049;

	glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);
	glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX,&cur_avail_mem_kb);

	LogInfo(LOG_GL "GPU memory free    %i / %i kb\n", cur_avail_mem_kb, total_mem_kb);
}


/**
 * Render a line in screen coordinates with the specified color
 * @note the RGBA color values are expected to be in the range of [0-1]
 * @ingroup OpenGL
 */
inline void glDrawLine( float x1, float y1, float x2, float y2, float r, float g, float b, float a=1.0f, float thickness=2.0f )
{
	glLineWidth(thickness);
	glBegin(GL_LINES);

		glColor4f(r, g, b, a);
		
		glVertex2f(x1, y1);
		glVertex2f(x2, y2);

	glEnd();
}


/**
 * Render the outline of a rect in screen coordinates with the specified color
 * @note the RGBA color values are expected to be in the range of [0-1]
 * @ingroup OpenGL
 */
inline void glDrawOutline( float x, float y, float width, float height, float r, float g, float b, float a=1.0f, float thickness=2.0f )
{
	const float right = x + width;
	const float bottom = y + height;

	glLineWidth(thickness);
	glBegin(GL_LINE_LOOP);

		glColor4f(r, g, b, a);
		
		glVertex2f(x, y);
		glVertex2f(right, y);
		glVertex2f(right, bottom);
		glVertex2f(x, bottom);

	glEnd();
}


/**
 * Render a filled rect in screen coordinates with the specified color
 * @note the RGBA color values are expected to be in the range of [0-1]
 * @ingroup OpenGL
 */
inline void glDrawRect( float x, float y, float width, float height, float r, float g, float b, float a=1.0f )
{
	const float right = x + width;
	const float bottom = y + height;

	glBegin(GL_QUADS);

		glColor4f(r, g, b, a);

		glVertex2f(x, y);
		glVertex2f(right, y);	
		glVertex2f(right, bottom);
		glVertex2f(x, bottom);

	glEnd();
}


#endif

