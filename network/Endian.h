/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __NETWORK_ENDIAN_H_
#define __NETWORK_ENDIAN_H_


#include <endian.h>


/*
 * endianess defines (big or little)
 */
// #define __LITTLE_ENDIAN
// #define __BIG_ENDIAN


/*
 * define byte-swap macros
 */
inline uint64_t bswap64( uint64_t value )		{ return __builtin_bswap64(value); }
inline uint32_t bswap32( uint32_t value )		{ return __builtin_bswap32(value); }
inline uint16_t bswap16( uint16_t value )		{ return ((value >> 8) | (value << 8)); }


/*
 * define network swapping macros, based on endianness
 */
#if (__BYTE_ORDER == __LITTLE_ENDIAN)

inline uint64_t netswap64( uint64_t value )		{ return bswap64(value); }
inline uint32_t netswap32( uint32_t value )		{ return bswap32(value); }
inline uint16_t netswap16( uint16_t value )		{ return bswap16(value); }

#elif (__BYTE_ORDER == __BIG_ENDIAN)

inline uint64_t netswap64( uint64_t value )		{ return value; }
inline uint32_t netswap32( uint32_t value )		{ return value; }
inline uint16_t netswap16( uint16_t value )		{ return value; }

#endif
#endif
