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
 
#ifndef __MATRIX_33_H_
#define __MATRIX_33_H_


#include <stdio.h>
#include <unistd.h>


/**
 * Copy src input matrix to dst output.
 */
template<typename T>
inline void mat33_copy( T dst[3][3], const T src[3][3] )
{
	memcpy(dst, src, sizeof(T) * 9);
}


/**
 * Initialize a 3x3 identity matrix.
 */
template<typename T>
inline void mat33_identity( double dst[3][3] )
{
	dst[0][0] = 1; dst[0][1] = 0; dst[0][2] = 0;
	dst[1][0] = 0; dst[1][1] = 1; dst[1][2] = 0;
	dst[2][0] = 0; dst[2][1] = 0; dst[2][2] = 1;
}


/**
 * Compute the inverse of a 3x3 matrix.
 */
template<typename T>
void mat33_inverse( const T m[3][3], T inv[3][3] )
{
	// determinant
	const T det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
			  - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
			  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

	// inverse
	inv[0][0] = + (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
	inv[0][1] = - (m[0][1] * m[2][2] - m[0][2] * m[2][1]);
	inv[0][2] = + (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
	inv[1][0] = - (m[1][0] * m[2][2] - m[1][2] * m[2][0]);
	inv[1][1] = + (m[0][0] * m[2][2] - m[0][2] * m[2][0]);
	inv[1][2] = - (m[0][0] * m[1][2] - m[0][2] * m[1][0]);
	inv[2][0] = + (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
	inv[2][1] = - (m[0][0] * m[2][1] - m[0][1] * m[2][0]);
	inv[2][2] = + (m[0][0] * m[1][1] - m[0][1] * m[1][0]);

	// scale by determinant
	for( uint32_t i=0; i < 3; i++ )
		for( uint32_t k=0; k < 3; k++ )
			inv[i][k] /= det;
}


/**
 * Multiply two 3x3 matrices, dst=a*b
 */
template<typename T>
inline void mat33_mul( T dst[3][3], const T a[3][3], const T b[3][3] )
{
	for( int i=0; i < 3; i++ )
	{
		for( int j=0; j < 3; j++ )
		{
			dst[i][j] = 0;

			for( int k=0; k < 3; k++ )
				dst[i][j] = dst[i][j] + a[i][k] * b[k][j];
		}
	}
}


/**
 * Print out a 3x3 matrix to stdout.
 */
template<typename T>
void mat33_print( const T m[3][3], const char* name=NULL )
{
	if( name != NULL )
		printf("%s = \n", name);

	printf(" [ ");

	for( uint32_t i=0; i < 3; i++ )
	{
		for( uint32_t k=0; k < 3; k++ )
			printf("%f ", m[i][k]);

		if( i < 2 )
			printf("\n   ");
		else
			printf("]\n");
	}
}


/**
 * Set a 3x3 matrix to all zero's.
 */
template<typename T>
inline void mat33_zero( double dst[3][3] )
{
	memset(dst, 0, sizeof(T) * 9);
}

#endif

