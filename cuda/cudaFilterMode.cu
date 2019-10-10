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


#include "cudaFilterMode.h"
#include <strings.h>


// cudaFilterModeFromStr
cudaFilterMode cudaFilterModeFromStr( const char* str, cudaFilterMode default_value )
{
	if( !str )
		return default_value;

	if( strcasecmp(str, "linear") == 0 )
		return FILTER_LINEAR;
	else if( strcasecmp(str, "point") == 0 || strcasecmp(str, "nearest") == 0 )
		return FILTER_POINT;

	return default_value;
}


// cudaFilterModeToStr
const char* cudaFilterModeToStr( cudaFilterMode filter )
{
	if( filter == FILTER_LINEAR )
		return "linear";

	return "point";
}




