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
 
#include "filesystem.h"
#include "Process.h"

#include <sys/stat.h>
#include <QDir>
#include <algorithm>



// absolutePath
std::string absolutePath( const std::string& relative_path )
{
	const std::string proc = Process::ExecutableDirectory();
	return proc + relative_path;
}


// locateFile
std::string locateFile( const std::string& path )
{
	std::vector<std::string> locations;
	return locateFile(path, locations);
}


// locateFile
std::string locateFile( const std::string& path, std::vector<std::string>& locations )
{
	if( fileExists(path.c_str()) )
		return path;

	locations.push_back(Process::ExecutableDirectory());
	locations.push_back("/usr/local/bin/");
	locations.push_back("/usr/local/");
	locations.push_back("/opt/");

	const size_t numLocations = locations.size();

	for( size_t n=0; n < numLocations; n++ )
	{
		const std::string str = locations[n] + path;

		if( fileExists(str.c_str()) )
			return str;
	}

	return "";
}


// listDir
bool listDir( const char* path, std::vector<std::string>& output, bool includePath )
{
	if( !path )
		return false;

	// get the list of files in the directory
	QDir qDir(path);	
	QStringList list = qDir.entryList();

	if( list.size() == 0 )
	{
		printf("%s is empty or does not exist.\n", path);
		return false;
	}

	for( int i=0; i < list.size(); ++i )
	{
		if( list.at(i) == "." || list.at(i) == ".." )
			continue;

		if( includePath )
			output.push_back( qDir.filePath(list.at(i)).toLocal8Bit().constData() );
		else
			output.push_back( list.at(i).toLocal8Bit().constData() );
	}

	std::sort(output.begin(), output.end());

	if( output.size() == 0 )
	{
		printf("%s is empty or does not exist.\n", path);
		return false;
	}

	/*for( int i=0; i < output.size(); ++i )
	{
		printf("%06i  %s\n", i, output[i].c_str());
	}*/

	return true;
}


// fileExists
bool fileExists( const char* path, bool regularFilesOnly )
{
	if( !path )
		return false;

	struct stat fileStat;
	const int result = stat(path, &fileStat);

	if( result == -1 )
	{
		//printf("%s does not exist.\n", path);
		return false;
	}

	if( !regularFilesOnly )
		return true;

	if( S_ISREG(fileStat.st_mode) )
		return true;
	
	return false;
}


// fileSize
size_t fileSize( const char* path )
{
	if( !path )
		return 0;

	struct stat fileStat;

	const int result = stat(path, &fileStat);

	if( result == -1 )
	{
		printf("%s does not exist.\n", path);
		return 0;
	}

	//printf("%s  size %zu bytes\n", path, (size_t)fileStat.st_size);
	return fileStat.st_size;
}


// filePath
std::string filePath( const std::string& filename )
{
	const std::string::size_type slashIdx = filename.find_last_of("/");

	if( slashIdx == std::string::npos || slashIdx == 0 )
		return filename;

	return filename.substr(0, slashIdx + 1);
}


// fileExtension
std::string fileExtension( const std::string& path )
{
	std::string ext = path.substr(path.find_last_of(".") + 1);

	transform(ext.begin(), ext.end(), ext.begin(), tolower);

	return ext;
}


// fileRemoveExtension
std::string fileRemoveExtension( const std::string& filename )
{
	const std::string::size_type dotIdx   = filename.find_last_of(".");
	const std::string::size_type slashIdx = filename.find_last_of("/");

    if( dotIdx == std::string::npos )
		return filename;

	if( slashIdx != std::string::npos && dotIdx < slashIdx )
		return filename;

    return filename.substr(0, dotIdx);
}


// fileChangeExtension
std::string fileChangeExtension(const std::string& filename, const std::string& newExtension)  
{
	return fileRemoveExtension(filename).append(newExtension);
}


// processPath
std::string processPath()
{
	return Process::ExecutablePath();
}


// processDirectory
std::string processDirectory()
{
	return Process::ExecutableDirectory();
}


// workingDirectory
std::string workingDirectory()
{
	return Process::WorkingDirectory();
}


