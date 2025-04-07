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
 
#include "Networking.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <net/if.h>

#include <sys/types.h>
#include <cstring>
#include <unistd.h>
#include <errno.h>

#include "logging.h"


// getHostname
std::string getHostname()
{
	char str[512];

	if( gethostname(str, sizeof(str)) != 0 )
	{
		LogError(LOG_NETWORK "failed to retrieve system hostname\n");
		return "";
	}
	
	return str;
}


// getHostByName
std::string getHostByName( const char* name, uint32_t retries )
{
	uint8_t ipBuffer[INET6_ADDRLEN];
	
	const uint32_t size = getHostByName(name, ipBuffer, sizeof(ipBuffer), retries);
	
	if( size == INET4_ADDRLEN )
		return IPv4AddressToStr(*((uint32_t*)ipBuffer));
	else if( size == INET6_ADDRLEN )
		return IPv6AddressToStr(ipBuffer);
	
	return "";
}

	
// getHostByName
uint32_t getHostByName( const char* name, void* ipAddress, uint32_t size, uint32_t retries )
{
	if( !name )
		return 0;
	
	// hostent struct:  https://stackoverflow.com/a/57313156
	//   h_addrtype:  AF_INET = 2  AF_INET6 = 10
	//   h_length:    AF_INET = 4  AF_INET6 = 16
	struct hostent* ent = NULL;
	
	for( uint32_t r=0; r < retries; r++ )
	{
		ent = gethostbyname(name);
	
		if( ent != NULL )
			break;
		
		//sleep(1);
		LogVerbose(LOG_NETWORK "getHostByName() trying to resolve host '%s' (retry %u of %u)\n", name, r+1, retries);
	}
	
	if( !ent )
	{
		LogError(LOG_NETWORK "getHostByName() unable to resolve host '%s'\n", name);
		return 0;
	}
	
	if( !ent->h_addr_list || !ent->h_addr_list[0] )
	{
		LogError(LOG_NETWORK "getHostByName() returned empty address list for host '%s'\n", name);
		return 0;
	}
		
	if( ent->h_length != INET4_ADDRLEN && ent->h_length != INET6_ADDRLEN )
	{
		LogError(LOG_NETWORK "getHostByName() returned IP address with unknown length %i for host '%s'\n", ent->h_length, name);
		return 0;
	}
	
	if( ent->h_length > size )
	{
		LogError(LOG_NETWORK "getHostByName() returned IP address longer than the available buffer length (%i vs %u bytes) for host '%s'\n", ent->h_length, size, name);
		return 0;
	}
	
	// TODO there can be more than one address...
	memcpy(ipAddress, ent->h_addr_list[0], ent->h_length);
	return ent->h_length;
}


// getNetworkInterfaces
std::vector<NetworkInterface> getNetworkInterfaces()
{
	std::vector<NetworkInterface> list;
	struct ifaddrs* addrs = NULL;

	if( getifaddrs(&addrs) < 0 )	// https://man7.org/linux/man-pages/man3/getifaddrs.3.html
	{ 
		const int e = errno;
		const char* err = strerror(e);
		LogError(LOG_NETWORK "getNetworkInterfaces() -- error %i : %s\n", e, err);
		return list;
	}

	for( ifaddrs* n=addrs; n != NULL; n = n->ifa_next )
	{
		if( !n->ifa_name || strlen(n->ifa_name) == 0 )
			continue;

		if( n->ifa_addr->sa_family != AF_INET && n->ifa_addr->sa_family != AF_INET6 )
			continue;
		
		int i = findNetworkInterface(list, n->ifa_name);
		
		if( i < 0 )
		{
			list.push_back(NetworkInterface());
			i = list.size() - 1;
		}

		list[i].name = n->ifa_name;
		list[i].up = (n->ifa_flags & IFF_UP);	// https://man7.org/linux/man-pages/man7/netdevice.7.html
		
		if( n->ifa_addr->sa_family == AF_INET )
		{
			list[i].ipv4.address = IPv4AddressToStr(((sockaddr_in*)n->ifa_addr)->sin_addr.s_addr);
			list[i].ipv4.netmask = IPv4AddressToStr(((sockaddr_in*)n->ifa_netmask)->sin_addr.s_addr);	// https://stackoverflow.com/a/48375002
			
			if( n->ifa_flags & IFF_BROADCAST )
				list[i].ipv4.broadcast = IPv4AddressToStr(((sockaddr_in*)n->ifa_broadaddr)->sin_addr.s_addr);
		}
		else if( n->ifa_addr->sa_family == AF_INET6 )
		{
			list[i].ipv6.address = IPv6AddressToStr(((sockaddr_in6*)n->ifa_addr)->sin6_addr.s6_addr);
			list[i].ipv6.prefix  = IPv6AddressToStr(((sockaddr_in6*)n->ifa_netmask)->sin6_addr.s6_addr);
		}
	}
	
	return list;
}


// findNetworkInterface
int findNetworkInterface( const std::vector<NetworkInterface>& interfaces, const char* name )
{
	if( !name )
		return -1;
	
	const uint32_t numInterfaces = interfaces.size();
	
	for( uint32_t n=0; n < numInterfaces; n++ )
	{
		if( interfaces[n].name == name )
			return n;
	}
	
	return -1;
}


// printNetworkInterfaces
void printNetworkInterfaces( const std::vector<NetworkInterface>& i )
{
	const uint32_t numInterfaces = i.size();
	
	LogInfo("Network Interfaces:\n");
	
	for( uint32_t n=0; n < numInterfaces; n++ )
	{
		LogInfo("  > %s (%s)\n", i[n].name.c_str(), i[n].up ? "up" : "down" );
		
		if( i[n].ipv4.address.size() > 0 )
		{
			LogInfo("     * IPv4 %s  ", i[n].ipv4.address.c_str());
			
			if( i[n].ipv4.netmask.size() > 0 )
				LogInfo("netmask %s  ", i[n].ipv4.netmask.c_str());
			
			if( i[n].ipv4.broadcast.size() > 0 )
				LogInfo("broadcast %s  ", i[n].ipv4.broadcast.c_str());
			
			LogInfo("\n");
		}
		
		if( i[n].ipv6.address.size() > 0 )
		{
			LogInfo("     * IPv6 %s  ", i[n].ipv6.address.c_str());
			
			if( i[n].ipv6.prefix.size() > 0 )
				LogInfo("prefix %s  ", i[n].ipv6.prefix.c_str());

			LogInfo("\n");
		}
	}
}

