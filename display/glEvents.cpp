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
#include "glEvents.h"


// glRegisterEvents
void glRegisterEvents( glEventHandler callback, void* user, uint32_t displayID )
{
	glDisplay* display = glGetDisplay(displayID);

	if( display != NULL )
		display->AddEventHandler(callback, user);
}


// glUnregisterEvents
void glUnregisterEvents( glEventHandler callback, void* user )
{
	const uint32_t numDisplays = glGetNumDisplays();

	for( uint32_t n=0; n < numDisplays; n++ )
	{
		glDisplay* display = glGetDisplay(n);

		if( display != NULL )
			display->RemoveEventHandler(callback, user);
	}
}



