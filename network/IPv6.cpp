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
 
#include "IPv6.h"
#include "Networking.h"
#include "logging.h"

#include <arpa/inet.h>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <errno.h>


// IPv6AddressFromStr
bool IPv6AddressFromStr( const char* str, void* ipAddress )
{
	if( !str || !ipAddress )
		return false;

	in6_addr addr;

	const int res = inet_pton(AF_INET6, str, &addr);

	if( res != 1 )
	{
		LogError(LOG_NETWORK "IPv6AddressFromStr() failed to convert '%s' to valid IPv6 address\n", str);
		return false;
	}
	
	memcpy(ipAddress, addr.s6_addr, INET6_ADDRLEN);
	return true;
}


// IPv6AddressToStr
std::string IPv6AddressToStr( void* ipAddress )
{
	if( !ipAddress )
		return "";
	
	char str[INET6_ADDRSTRLEN];
	memset(str, 0, INET6_ADDRSTRLEN);

	if( inet_ntop(AF_INET6, ipAddress, str, INET6_ADDRSTRLEN) == NULL )
	{
		uint16_t* i = (uint16_t*)ipAddress;
		LogError("IPv6AddressToStr() failed to convert %04hX:%04hX:%04hX:%04hX:%04hX:%04hX:%04hX:%04hX to string\n", i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]);
		return "";
	}
	
	return std::string(str);
}

