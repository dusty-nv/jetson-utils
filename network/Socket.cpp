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

#include "Socket.h"
#include "Endian.h"
#include "Networking.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <cstring>
#include <unistd.h>
#include <errno.h>

#include "logging.h"


// printErrno
static void printErrno()
{
	const int e = errno;
	const char* err = strerror(e);
	LogError(LOG_NETWORK "socket error %i : %s\n", e, err);
}


// constructor
Socket::Socket( SocketType type ) : mType(type)
{
	mSock       = -1;
	mLocalPort  = 0;
	mLocalIP 	= 0;
	mRemoteIP   = 0;
	mRemotePort = 0;	
	
	mListening        = false;
	mPktInfoEnabled   = false;
	mBroadcastEnabled = false;
}


// destructor
Socket::~Socket()
{
	if( mSock != -1 )
	{
		close(mSock);
		mSock = -1;
	}
}


// SetBufferSize
bool Socket::SetBufferSize( size_t size )
{
	if( size == 0 )
		return false;

	const int isz = size;

	if( setsockopt(mSock, SOL_SOCKET, SO_RCVBUF, &isz, sizeof(int)) != 0 )
	{
		LogError(LOG_NETWORK "Socket failed to set rx buffer size of %zu bytes.\n", size);
		printErrno();			
		return false;
	}

	if( setsockopt(mSock, SOL_SOCKET, SO_SNDBUF, &isz, sizeof(int)) != 0 )
	{
		LogError(LOG_NETWORK "Socket failed to set rx buffer size of %zu bytes.\n", size);
		printErrno();		
		return false;
	}

	LogVerbose(LOG_NETWORK "successfully set socket buffer size of %s:%u to %zu bytes\n", IPv4AddressToStr(mLocalIP).c_str(), (uint32_t)mLocalPort, size);
	return true;
}


bool Socket::EnableJumboBuffer()
{
	return SetBufferSize(167772160); //SetBufferSize(10485760);
}


// Create
Socket* Socket::Create( SocketType type )
{
	Socket* s = new Socket(type);

	if( !s )
		return NULL;

	if( !s->Init() )
	{
		delete s;
		return NULL;
	}

	return s;
}

	
// Init
bool Socket::Init()
{
	const int type = (mType == SOCKET_UDP) ? SOCK_DGRAM : SOCK_STREAM;
	
	if( (mSock = socket(AF_INET, type, 0)) < 0 )
	{
		printErrno();
		return false;
	}

	return true;
}


// Accept
bool Socket::Accept( uint64_t timeout )
{
	if( mType != SOCKET_TCP )
		return false;

	// set listen mode
	if( !mListening )
	{
		if( listen(mSock, 1) < 0 )
		{
			LogError(LOG_NETWORK "failed to listen() on socket.\n");
			return false;
		}

		mListening = true;
	}

	// select timeout
	if( timeout != 0 )
	{
		struct timeval tv;
		
		tv.tv_sec  = timeout / 1000000;
		tv.tv_usec = timeout - (tv.tv_sec * 1000000);

		fd_set fds;
		
		FD_ZERO(&fds);
		FD_SET(mSock, &fds);

		const int result = select(mSock + 1, &fds, NULL, NULL, &tv);

		if( result < 0 )
		{
			LogError(LOG_NETWORK "select() error occurred during Socket::Accept()   (code=%i)\n", result);
			printErrno();
			return false;
		}
		else if( result == 0 )
		{
			LogError(LOG_NETWORK "Socket::Accept() timeout occurred\n");
			return false;
		}
	}

	// accept connections
	struct sockaddr_in addr;
	memset(&addr, 0, sizeof(addr));
	socklen_t addrLen = sizeof(addr);

	const int fd = accept(mSock, (struct sockaddr*)&addr, &addrLen);

	if( fd < 0 )
	{
		LogError(LOG_NETWORK "Socket::Accept() failed  (code=%i)\n", fd);
		printErrno();
		return false;
	}

	mRemoteIP   = addr.sin_addr.s_addr;
	mRemotePort = ntohs(addr.sin_port);
	
	// swap out the old 'listening' port
	close(mSock);

	mSock      = fd;
	mListening = false;
	
	return true;
}	


// Bind
bool Socket::Bind( const char* ipStr, uint16_t port )
{
	if( !ipStr )
		return Bind(port);

	uint32_t ipAddress = 0;

	if( !IPv4AddressFromStr(ipStr, &ipAddress) )
		return false;

	return Bind(ipAddress, port);
}

	
// Bind
bool Socket::Bind( uint32_t ipAddress, uint16_t port )
{
	struct sockaddr_in addr;
	memset(&addr, 0, sizeof(addr));

	addr.sin_family 	 = AF_INET;
	addr.sin_addr.s_addr = ipAddress;
	addr.sin_port		 = htons(port);

	if( bind(mSock, (struct sockaddr*)&addr, sizeof(addr)) < 0 )
	{
		LogError(LOG_NETWORK "failed to bind socket to %s port %hu\n", IPv4AddressToStr(ipAddress).c_str(), port);
		printErrno();
		return false;
	}

	mLocalIP   = ipAddress;
	mLocalPort = port;

	return true;
}


// Bind
bool Socket::Bind( uint16_t port )
{
	return Bind( htonl(INADDR_ANY), port );
}


// Connect
bool Socket::Connect( uint32_t ipAddress, uint16_t port )
{
	if( mType != SOCKET_TCP )
		return false;

	struct sockaddr_in addr;
	memset(&addr, 0, sizeof(addr));

	addr.sin_family 	 = AF_INET;
	addr.sin_addr.s_addr = ipAddress;
	addr.sin_port		 = htons(port);

	if( connect(mSock, (struct sockaddr*)&addr, sizeof(addr)) < 0 )
	{
		LogError(LOG_NETWORK "socket failed to connect to %X port %hi.\n", ipAddress, port);
		printErrno();
		return false;
	}

	mRemoteIP = ipAddress;
	mRemotePort = port;

	return true;
}


// Connect
bool Socket::Connect( const char* ipStr, uint16_t port )
{
	if( !ipStr || mType != SOCKET_TCP )
		return false;

	uint32_t ipAddress = 0;

	if( !IPv4AddressFromStr(ipStr, &ipAddress) )
		return false;

	return Connect(ipAddress, port);
}


// Recieve
size_t Socket::Recieve( uint8_t* buffer, size_t size, uint32_t* srcIpAddress, uint16_t* srcPort )
{
	if( !buffer || size == 0 )
		return 0;	

	struct sockaddr_in srcAddr;
	socklen_t addrLen = sizeof(srcAddr);

	// recieve packet
	const int64_t res = recvfrom(mSock, (void*)buffer, size, 0, (struct sockaddr*)&srcAddr, &addrLen);

	if( res < 0 )
	{
		//LogVerbose(LOG_NETWORK "Socket::Recieve() timed out\n");
		//printErrno();
		return 0;
	}

	if( srcIpAddress != NULL )
		*srcIpAddress = srcAddr.sin_addr.s_addr;

	if( srcPort != NULL )
		*srcPort = ntohs(srcAddr.sin_port);

#if 0 // DEBUG
	LogDebug(LOG_NETWORK "recieved %04lli bytes from %s:%hu\n", res, IPv4AddressToStr(srcAddr.sin_addr.s_addr).c_str(), (uint16_t)ntohs(srcAddr.sin_port));
#endif

	return res;
}


// Recieve
size_t Socket::Recieve( uint8_t* buffer, size_t size, uint32_t* remoteIP, uint16_t* remotePort, uint32_t* localIP )
{
	if( !buffer || size == 0 )
		return 0;	

	
	// enable IP_PKTINFO if not already done so
	if( !mPktInfoEnabled )
	{
		int opt = 1;
		
		if( setsockopt(mSock, IPPROTO_IP, IP_PKTINFO, (const char*)&opt, sizeof(int)) != 0 )
		{
			LogError(LOG_NETWORK "Socket::Receive() failed to enabled extended PKTINFO\n");
			printErrno();
			return 0;
		}
		
		mPktInfoEnabled = true;
	}
	
	
	// setup msghdr to recieve addition address info
	union controlData {
		cmsghdr cmsg;
		uint8_t data[CMSG_SPACE(sizeof(struct in_pktinfo))];
	};

	iovec iov;
	msghdr msg;
	controlData cmsg;
	sockaddr_in remoteAddr;
	
	memset(&msg, 0, sizeof(msghdr));
	memset(&cmsg, 0, sizeof(cmsg));
	memset(&remoteAddr, 0, sizeof(sockaddr_in));
	
	iov.iov_base = buffer;
	iov.iov_len  = size;
	
	msg.msg_name    = &remoteAddr;
	msg.msg_namelen = sizeof(sockaddr_in);
	msg.msg_iov     = &iov;
	msg.msg_iovlen  = 1;
	msg.msg_control = &cmsg;
	msg.msg_controllen = sizeof(cmsg);
	
	
	// recieve message
	const ssize_t res = recvmsg(mSock, &msg, 0);
	
	if( res < 0 )
	{
		//printf(LOG_NETWORK "Socket::Recieve() timed out\n");
		//printErrno();
		return 0;
	}
	
	// output local address
	for( cmsghdr* c = CMSG_FIRSTHDR(&msg); c != NULL; c = CMSG_NXTHDR(&msg, c) )
	{
		if( c->cmsg_level == IPPROTO_IP && c->cmsg_type == IP_PKTINFO )
		{
			if( localIP != NULL )
				*localIP = ((in_pktinfo*)CMSG_DATA(c))->ipi_addr.s_addr;
				
			// TODO local port...not included in IP_PKTINFO?
		}
	}

	// output remote address
	if( remoteIP != NULL )
		*remoteIP = remoteAddr.sin_addr.s_addr;

	if( remotePort != NULL )
		*remotePort = ntohs(remoteAddr.sin_port);
		
	return res;
}


// SetRecieveTimeout
bool Socket::SetRecieveTimeout( uint64_t timeout )
{
	struct timeval tv;
		
	tv.tv_sec  = timeout / 1000000;
	tv.tv_usec = timeout - (tv.tv_sec * 1000000);
		
	if( setsockopt(mSock, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, sizeof(struct timeval)) != 0 )
	{
		LogError(LOG_NETWORK "Socket::SetRecieveTimeout() failed to set timeout of %zu microseconds.\n", timeout);
		printErrno();
		return false;
	}

	return true;
}


// Send
bool Socket::Send( void* buffer, size_t size, uint32_t remoteIP, uint16_t remotePort )
{
	if( !buffer || size == 0 )
		return false;
	

	// if sending broadcast, enable broadcasting if not already done so
	if( remoteIP == netswap32(IP_BROADCAST) && !mBroadcastEnabled )
	{
		int opt = 1;
		
		if( setsockopt(mSock, SOL_SOCKET, SO_BROADCAST, (const char*)&opt, sizeof(int)) != 0 )
		{
			LogError(LOG_NETWORK "Socket::Send() failed to enabled broadcasting...\n");
			printErrno();
			return false;
		}
		
		mBroadcastEnabled = true;
	}


	// send the message
	struct sockaddr_in addr;
	memset(&addr, 0, sizeof(addr));

	addr.sin_family 	 = AF_INET;
	addr.sin_addr.s_addr = remoteIP;
	addr.sin_port		 = htons(remotePort);
	
	const int64_t res = sendto(mSock, (void*)buffer, size, 0, (struct sockaddr*)&addr, sizeof(addr));
	
	if( res != size )
	{
		LogError(LOG_NETWORK "failed send() to %s port %hu  (%li of %zu bytes)\n", IPv4AddressToStr(remoteIP).c_str(), remotePort, res, size);
		printErrno();
		return false;
	}
	
	return true;
}


// PrintIP
void Socket::PrintIP() const
{
	LogInfo(LOG_NETWORK "socket %i   host %s:%hu   remote %s:%hu\n", mSock, IPv4AddressToStr(mLocalIP).c_str(), mLocalPort,
														  IPv4AddressToStr(mRemoteIP).c_str(), mRemotePort );
}


// GetMTU
size_t Socket::GetMTU()
{
	int mtu = 0;
	socklen_t mtuSize = sizeof(int);
	
	if( getsockopt(mSock, IPPROTO_IP, IP_MTU, &mtu, &mtuSize) < 0 )
	{
		LogError(LOG_NETWORK "Socket::GetMTU() -- getsockopt(SOL_IP, IP_MTU) failed.\n");
		printErrno();
		return 0;
	}
	
	return (size_t)mtu;
}

