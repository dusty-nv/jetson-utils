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
 
#ifndef __NETWORKING_H_
#define __NETWORKING_H_

#include <string>
#include <vector>

#include "IPv4.h"
#include "IPv6.h"


/**
 * Retrieve the host system's network hostname.
 * @returns the system hostname, or an empty string on error.
 * @ingroup network
 */
std::string getHostname();


/**
 * Resolve the IP address of a given hostname or domain using DNS lookup, and return it as a string.
 * This uses the system function gethostbyname() and supports both IPv6/IPv6 addresses.
 *
 * @param name the hostname to lookup.
 * @param retries the number of times to retry the lookup should it fail.
 * @returns the IP address in string format, or an empty string if an error occurred.
 *
 * @ingroup network
 */
std::string getHostByName( const char* name, uint32_t retries=10 );


/**
 * Resolve the IP address of a given hostname or domain using DNS lookup.
 * This uses the system function gethostbyname() and supports both IPv6/IPv6 addresses.
 *
 * @param name the hostname or domain name to lookup.
 * @param ipAddress output pointer to buffer that the IPv4 or IPv6 address will be written to.
 *                  this buffer should be up to 16 bytes long for supporting IPv6 addresses,
 *                  or 4 bytes long for only supporting IPv4 addresses.
 * @param size the size of the ipAddress buffer (4 bytes for IPv6 or 16 bytes for IPv6)
 * @param retries the number of times to retry the lookup should it fail.
 *
 * @returns the size in bytes of the ipAddress that was written (4 bytes for IPv6 or 16 bytes for IPv6)
 *          0 is returned if an error occurred, or the provided ipAddress buffer was not large enough.
 *
 * @ingroup network
 */
uint32_t getHostByName( const char* name, void* ipAddress, uint32_t size, uint32_t retries=10 );


/**
 * Info about a particular network interface.
 * The IPv6 information will be filled out if it's enabled on the interface.
 * @ingroup network
 */
struct NetworkInterface
{
	std::string name;			/**< name of the network interface (e.g. eth0, wlan0, lo) */
	bool up;					/**< true if the interface is up, false if down */
	
	struct IPv4
	{
		std::string address;	/**< IPv4 address */
		std::string netmask;	/**< IPv4 netmask */
		std::string broadcast;	/**< IPv4 broadcast */
	} ipv4;
	
	struct IPv6
	{
		std::string address;	/**< IPv6 address */
		std::string prefix;		/**< IPv6 prefix */
	} ipv6;
};


/**
 * Retrieve info about the different IPv4/IPv6 network interfaces of the system.
 * Internally this uses the system function getifaddrs()
 * @returns a list of network interfaces, or an empty list if an error occurred.
 * @ingroup network
 */
std::vector<NetworkInterface> getNetworkInterfaces();

/**
 * Find the index of a network interface by name from the list of interfaces.
 * @returns the index of the interface, or -1 if it was not found.
 */
int findNetworkInterface( const std::vector<NetworkInterface>& interfaces, const char* name );

/**
 * Print out a list of network interfaces.
 * @ingroup network
 */
void printNetworkInterfaces( const std::vector<NetworkInterface>& interfaces );


/**
 * LOG_NETWORK logging string.
 * @ingroup network
 */
#define LOG_NETWORK "[network] "

#endif
