/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __NETWORK_IPV6_H_
#define __NETWORK_IPV6_H_

#include <string>


/**
 * Convert an IPv6 address string in "x:x:x:x:x:x:x:x" hexadecimal format to 128-bit binary representation.
 *
 * @param str the IPv6 string, in "x:x:x:x:x:x:x:x" hexadecimal format
 * @param ipAddress output pointer to converted 128-bit IPv6 address (16 bytes long), in network byte order.
 *                  this buffer can be a variety of representations (commonly uint8[16] or uint16[8]),
 *                  as long as it's at least 128 bits (16 bytes) long.
 *
 * @returns true, if str was a valid IPv6 address and the conversion was successful.
 *          false, if the conversion failed.
 *
 * @ingroup network
 */
bool IPv6AddressFromStr( const char* str, void* ipAddress );


/**
 * Return text string of IPv6 address in "x:x:x:x:x:x:x:x" hexadecimal format.
 * @param ipAddress pointer to 128-bit IPv6 address (16 bytes long), in network byte order.
 *                   this buffer can be a variety of representations (commonly uint8[16] or uint16[8]),
 *                   as long as it's at least 128 bits (16 bytes) long.
 * @ingroup network
 */
std::string IPv6AddressToStr( void* ipAddress );


/**
 * The size in bytes of an IPv6 address (16 bytes)
 * This is meant to compliment POSIX's INET6_ADDRSTRLEN (46)
 * @ingroup network
 */
#define INET6_ADDRLEN 16


#endif
