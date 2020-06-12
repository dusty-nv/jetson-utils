/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 
#include "URI.h"

#include <algorithm>
#include <iostream>
#include <stdio.h>

#include "filesystem.h"
#include "logging.h"


// toLower
static std::string toLower( const std::string& str )
{
	std::string dst;
	dst.resize(str.size());
	std::transform(str.begin(), str.end(), dst.begin(), ::tolower);
	return dst;
}


// constructor
URI::URI()
{
	port = -1;
}


// constructor
URI::URI( const char* uri )
{
	Parse(uri);
}


// Parse
bool URI::Parse( const char* uri )
{
	if( !uri )
		return false;

	string    = uri;
	protocol  = "";
	extension = "";
	path      = "";
     port      = -1;
	
	// look for protocol
	std::size_t pos = string.find("://");

	if( pos != std::string::npos )
	{
		protocol = string.substr(0, pos);
		path = string.substr(pos+3, std::string::npos);
	}
	else
	{
		// check for some formats without specified protocol
		pos = string.find("/dev/video");

		if( pos == 0 )
		{
			protocol = "v4l2";
		}
		else if( string.find(".") != std::string::npos || string.find("/") != std::string::npos || fileExists(string.c_str()) )
		{
			protocol = "file";
		}
		else if( sscanf(string.c_str(), "%i", &port) == 1 )
		{
			protocol = "csi";
		}
		else if( string == "display" )
		{
			protocol = "display";
		}
		else
		{
			LogError("URI -- invalid resource or file path:  %s\n", string.c_str());
			return false;
		}

		path = string;

		// reconstruct full URI string
		string = protocol + "://";

		if( protocol == "file" )
			string += absolutePath(path);	// URI paths should be absolute
		else
			string += path;
	}

	// protocol should be all lowercase for easier parsing
	protocol = toLower(protocol);

	// parse extra info (device ordinals, IP addresses, ect)
	if( protocol == "v4l2" )
	{
		if( sscanf(path.c_str(), "/dev/video%i", &port) != 1 )
		{
			LogError("URI -- failed to parse V4L2 device ID from %s\n", path.c_str());
			return false;
		}
	}
	else if( protocol == "csi" )
	{
		if( sscanf(path.c_str(), "%i", &port) != 1 )
		{
			LogError("URI -- failed to parse MIPI CSI device ID from %s\n", path.c_str());
			return false;
		}
	}
	else if( protocol == "display" )
	{
		if( sscanf(path.c_str(), "%i", &port) != 1 )
		{
			LogVerbose("URI -- using default display device 0\n");
			port = 0;
		}
	}
	else if( protocol == "file" )
	{
		extension = fileExtension(path);
	}
	else
	{		
		// search for ip/port format
		std::string port_str;
		pos = path.find(":");		

		if( pos != std::string::npos )	// "xxx.xxx.xxx.xxx:port"
		{
			port_str = path.substr(pos+1, std::string::npos);
			path = path.substr(0, pos);
		}
		else if( std::count(path.begin(), path.end(), '.') == 0 ) // "port"
		{
			port_str = path;
			path = "127.0.0.1";
		}

		// parse the port number
		if( port_str.size() != 0 )
		{
			if( sscanf(port_str.c_str(), "%i", &port) != 1 )
			{
				LogError("URI -- failed to parse IP port from %s\n", string.c_str());
				return false;
			}
		}

		// convert "@:port" format to localhost
		if( path == "@" )
			path = "127.0.0.1";
	}
		
	return true;
}


// Print
void URI::Print( const char* prefix ) const
{
	if( !prefix )
		prefix = "";

	LogInfo("%s-- URI: %s\n", prefix, string.c_str());
	LogInfo("%s   - protocol:  %s\n", prefix, protocol.c_str());
	LogInfo("%s   - path:      %s\n", prefix, path.c_str());
	LogInfo("%s   - extension: %s\n", prefix, extension.c_str());
	LogInfo("%s   - port:      %i\n", prefix, port);
}



