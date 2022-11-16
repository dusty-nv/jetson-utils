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
 
#include "IPv4.h"
#include "Networking.h"
#include "logging.h"

#include <arpa/inet.h>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <errno.h>


// IPv4AddressFromStr
bool IPv4AddressFromStr( const char* str, uint32_t* ipAddress )
{
	if( !str || !ipAddress )
		return false;

	in_addr addr;

	const int res = inet_pton(AF_INET, str, &addr);

	if( res != 1 )
	{
		LogError(LOG_NETWORK "IPv4AddressFromStr() failed to convert '%s' to valid IPv4 address\n", str);
		return false;
	}
	
	*ipAddress = addr.s_addr;
	return true;
}


// IPv4AddressStr
std::string IPv4AddressToStr( uint32_t ipAddress )
{
	char str[INET_ADDRSTRLEN];
	memset(str, 0, INET_ADDRSTRLEN);

	if( inet_ntop(AF_INET, &ipAddress, str, INET_ADDRSTRLEN) == NULL )
	{
		LogError(LOG_NETWORK "IPv4AddressToStr() failed to convert 0x%08X to string\n", ipAddress);
		return "";
	}
	
	return std::string(str);
}
