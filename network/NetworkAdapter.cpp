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
 
#include "NetworkAdapter.h"
#include "IPv4.h"

#include <arpa/inet.h>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <errno.h>

#include "logging.h"


// networkHostname
std::string networkHostname()
{
	char str[256];

	if( gethostname(str, sizeof(str)) != 0 )
		return "<error>";
	
	return str;
}


// networkAdapters
void networkAdapters( std::vector<networkAdapter_t>& interfaceList )
{
	struct ifaddrs* addrs;

	if( getifaddrs(&addrs) < 0 )
	{ 
		const int e = errno;
		const char* err = strerror(e);
		LogError("Network error %i : %s\n", e, err );
	}

	LogVerbose("Network Interfaces:\n");

	for( ifaddrs* n=addrs; n != NULL; n = n->ifa_next )
	{
		if( n->ifa_addr->sa_family != AF_INET /*AF_INET6*/ )
			continue;

		if( !addrs->ifa_name || strlen(addrs->ifa_name) == 0 )
			continue;

		networkAdapter_t entry;

		entry.name      = addrs->ifa_name;
		entry.ipAddress = IPv4AddressStr(((sockaddr_in*)n->ifa_addr)->sin_addr.s_addr);

		LogVerbose("  - %s %s\n", entry.name.c_str(), entry.ipAddress.c_str());

		interfaceList.push_back(entry);
	}
}
