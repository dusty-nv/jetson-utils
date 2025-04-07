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
 
#ifndef __NETWORK_SOCKET_H_
#define __NETWORK_SOCKET_H_

#include <stdint.h>
#include <cstddef>


/**
 * TCP/UDP enumeration.
 * @ingroup network
 */
enum SocketType
{
	/**
	 * flag indicating UDP datagram service (SOCK_DGRAM)
	 */
	SOCKET_UDP  = 0,

	/**
	 * flag indicating TCP virtual circuit service (SOCK_STREAM)
	 */
	SOCKET_TCP  = 1
};


#define IP_ANY			0x00000000
#define IP_BROADCAST	0xFFFFFFFF
#define IP_LOOPBACK     0x7F000001


/**
 * The Socket class provides TCP or UDP ethernet networking.   
 * To exchange data with a remote IP on the network using UDP, follow these steps:
 *
 * 	1.  Create a Socket instance with the static Create() function.
 *  2.  Bind() the Socket to a host IP address and port
 *  3.  Exchange data with the Send/Recv functions
 *
 * @ingroup network
 */
class Socket
{
public:
	/**
	 * Create
	 */
	static Socket* Create( SocketType type );

	/**
	 * Destructor
	 */
	~Socket();

	/**
	 * Accept incoming connections (TCP only).
	 * @param timeout The timeout (in microseconds) to wait for incoming connections before returning.
	 */
	bool Accept( uint64_t timeout=0 );	

	/**
	 * Bind the socket to a local host IP address and port.
	 * @param ipAddress IPv4 address in string format "xxx.xxx.xxx.xxx"
	 * @param port the port number (0-65536), in host byte order.
	 *             If the port specified is 0, the socket will be bound to any available port.
	 */
	bool Bind( const char* localIP, uint16_t port );

	/**
	 * Bind the socket to a local host IP address and port.
	 * @param hostIP IPv4 address, in network byte order.
	 *               If htonl(INADDR_ANY) is specified for the ipAddress, the socket will be bound to all available interfaces.
	 * @param port the port number (0-65536), in host byte order.
	 *             If the port specified is 0, the socket will be bound to any available port.
	 */
	bool Bind( uint32_t localIP, uint16_t port );

	/**
	 * Bind the socket to a local host port, on any interface (0.0.0.0 INADDR_ANY).
	 * @param port the port number (0-65536), in host byte order.
	 *             If the port specified is 0, the socket will be bound to any available port.
	 */
	bool Bind( uint16_t port=0 );

	/**
	 * Connect to a listening server (TCP only).
	 * @param remoteIP IP address of the remote host.
	 */
	bool Connect( const char* remoteIP, uint16_t port );

	/**
	 * Connect to a listening server (TCP only).
 	 * @param remoteIP IP address of the remote host.
	 */
	bool Connect( uint32_t remoteIP, uint16_t port );

	/**
	 * Wait for a packet to be recieved and dump it into the user-supplied buffer.
	 * Optionally return the IP address and port of the remote host which sent the packet.
	 * @param buffer user-allocated destination for the packet
	 * @param size size of the buffer (in bytes)
	 * @param remoteIpAddress optional output, the IPv4 address of where the packet originated (in network byte order).
	 * @param remotePort optional output, the port from where the packet originated (in host byte order).
	 * @returns the size (in bytes) of the packet sucessfully written to buffer.
	 */
	size_t Recieve( uint8_t* buffer, size_t size, uint32_t* remoteIP=NULL, uint16_t* remotePort=NULL );

	/**
	 * Recieve packet, with additional address info.
	 * In addition to returning the IP and port of the remote host which sent the packet, this function
	 * can also return the local IP and port which the packet was sent to. 
	 * @see Recieve()
	 */
	size_t Recieve( uint8_t* buffer, size_t size, uint32_t* remoteIP, uint16_t* remotePort, uint32_t* localIP );
	
	/**
	 * Send message to remote host.
	 */
	bool Send( void* buffer, size_t size, uint32_t remoteIP, uint16_t remotePort );
	
	/**
	 * Set Receive() timeout (in microseconds).
	 */
	bool SetRecieveTimeout( uint64_t timeout );
	
	/**
	 * Set the rx/tx buffer sizes
	 */
	bool SetBufferSize( size_t size );

	/**
	 * Enable jumbo buffer
	 */
	bool EnableJumboBuffer();

	/**
	 * Retrieve the socket's file descriptor.
	 */
	inline int GetFD() const											{ return mSock; }

	/**
	 * GetType
	 */
	inline SocketType GetType() const									{ return mType; }

	/**
	 * PrintIP
	 */
	void PrintIP() const;

	/**
	 * Retrieve the MTU (in bytes).
	 * Returns 0 on error.
	 */
	size_t GetMTU();
	
private:
	
	Socket( SocketType type );
	bool Init();

	int 	   mSock;
	SocketType mType;
	bool       mListening;
	bool	   mPktInfoEnabled;
	bool       mBroadcastEnabled;
	
	uint32_t   mLocalIP;
	uint16_t   mLocalPort;

	uint32_t   mRemoteIP;
	uint16_t   mRemotePort;	
};

#endif
