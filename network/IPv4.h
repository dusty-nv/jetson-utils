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
 
#ifndef __NETWORK_IPV4_H_
#define __NETWORK_IPV4_H_

#include <string>


/**
 * Convert an IPv4 address string in "xxx.xxx.xxx.xxx" format to binary representation.
 *
 * @param str the IPv4 string, in "xxx.xxx.xxx.xxx" format
 * @param ip_out output pointer to converted IPv4 address, in network byte order.
 *
 * @returns true, if str was a valid IPv4 address and the conversion was successful.
 *          false, if the conversion failed.
 *
 * @ingroup network
 */
bool IPv4Address( const char* str, uint32_t* ip_out );


/**
 * Return text string of IPv4 address in "xxx.xxx.xxx.xxx" format
 * @param ip_address IPv4 address, supplied in network byte order.
 * @ingroup network
 */
std::string IPv4AddressStr( uint32_t ip_address );


#endif
