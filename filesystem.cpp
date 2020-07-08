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
#include "alphanum.h"
#include "Process.h"

#include <sys/stat.h>
#include <algorithm>
#include <strings.h>
#include <glob.h>

#include "logging.h"


// absolutePath
std::string absolutePath( const std::string& relative_path )
{
	if( relative_path.size() != 0 )
	{
		const char first_char = relative_path[0];

		if( first_char == '/' || first_char == '\\' || first_char == '~' )
			return relative_path;
	}

	return pathJoin(workingDirectory(), relative_path);
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
	// check the given path first
	if( fileExists(path.c_str()) )
		return path;

	// add standard search locations
	locations.push_back(Process::ExecutableDirectory());

	locations.push_back("/usr/local/bin/");
	locations.push_back("/usr/local/");
	locations.push_back("/opt/");

	locations.push_back("images/");
	locations.push_back("/usr/local/bin/images/");

	// check each location until the file is found
	const size_t numLocations = locations.size();

	for( size_t n=0; n < numLocations; n++ )
	{
		const std::string str = pathJoin(locations[n], path);

		if( fileExists(str.c_str()) )
			return str;
	}

	return "";
}


// listDir
bool listDir( const std::string& path_in, std::vector<std::string>& output, uint32_t mask )
{
	std::string path = path_in;
 
	if( path.size() == 0 )
		return false;

	// add a wildcard under directories, otherwise just the dir will be returned
	const bool pathIsDir = fileIsType(path, FILE_DIR|FILE_LINK);

	if( pathIsDir )
		path = pathJoin(path, "*");

	// glob the files - https://www.man7.org/linux/man-pages/man3/glob.3.html
	glob_t globList;

	const int result = glob(path.c_str(), GLOB_PERIOD|GLOB_MARK|GLOB_BRACE|GLOB_TILDE_CHECK, NULL, &globList);

	if( result != 0 )
	{
		if( result == GLOB_NOSPACE )
		{
			LogError("listDir('%s') - ran out of memory\n", path.c_str());
		}		
		else if( result == GLOB_ABORTED )
		{
			LogError("listDir('%s') - aborted due to read error or permissions\n", path.c_str());
		}
		else if( result == GLOB_NOMATCH )
		{
			const char firstChar = path[0];

			// if nothing was found and a full path wasn't specified, try the exe path
			if( firstChar != '.' && firstChar != '/' && firstChar != '\\' && firstChar != '*' && firstChar != '?' && firstChar != '~' )
				return listDir(pathJoin(Process::ExecutableDirectory(), path), output, mask);
			else
				LogError("listDir('%s') - found no matches\n", path.c_str());
		}

		return false;
	}

	// populate the output vector, and filter by file type
	for( size_t n=0; n < globList.gl_pathc; n++ )
	{
		// if there's a type mask, check that it matches
		if( mask != 0 && !fileIsType(globList.gl_pathv[n], mask) )
			continue; 

		output.push_back(globList.gl_pathv[n]);
	}
		
	globfree(&globList);

	// sort list alphanumerically (glob actually already does this)
	std::sort(output.begin(), output.end(), doj::alphanum_less<std::string>());

	if( output.size() == 0 )
	{
		LogError("%s didn't match any files\n", path.c_str());
		return false;
	}

	return true;
}


// fileType
uint32_t fileType( const std::string& path )
{
	if( path.size() == 0 )
		return FILE_MISSING;

	struct stat fileStat;
	const int result = stat(path.c_str(), &fileStat);

	if( result == -1 )
	{
		//printf("%s does not exist.\n", path.c_str());
		return FILE_MISSING;
	}

	if( S_ISREG(fileStat.st_mode) )
		return FILE_REGULAR;
	else if( S_ISDIR(fileStat.st_mode) )
		return FILE_DIR;
	else if( S_ISLNK(fileStat.st_mode) )
		return FILE_LINK;
	else if( S_ISCHR(fileStat.st_mode) )
		return FILE_CHAR;
	else if( S_ISBLK(fileStat.st_mode) )
		return FILE_BLOCK;
	else if( S_ISFIFO(fileStat.st_mode) )
		return FILE_FIFO;
	else if( S_ISSOCK(fileStat.st_mode) )
		return FILE_SOCKET;
	
	return FILE_MISSING;
}


// fileIsType
bool fileIsType( const std::string& path, uint32_t mask )
{
	if( path.size() == 0 )
		return false;

	const uint32_t type = fileType(path);
	
	if( type == FILE_MISSING )
		return false;
	
	if( mask == 0 )
		return true;
	
	if( (type & mask) != type )
		return false;
	
	return true;
}


// fileExists
bool fileExists( const std::string& path, uint32_t mask )
{
	return fileIsType(path, mask);
}


// fileSize
size_t fileSize( const std::string& path )
{
	if( path.size() == 0 )
		return 0;

	struct stat fileStat;

	const int result = stat(path.c_str(), &fileStat);

	if( result == -1 )
	{
		LogError("%s does not exist.\n", path.c_str());
		return 0;
	}

	//printf("%s  size %zu bytes\n", path, (size_t)fileStat.st_size);
	return fileStat.st_size;
}


// pathDir
std::string pathDir( const std::string& filename )
{
	const std::string::size_type slashIdx = filename.find_last_of("/");

	if( slashIdx == std::string::npos || slashIdx == 0 )
		return filename;

	return filename.substr(0, slashIdx + 1);
}


// pathJoin
std::string pathJoin( const std::string& a, const std::string& b )
{
	if( a.size() == 0 )
		return b;

	if( b.size() == 0 )
		return a;

	// check if there is already a path separator at the end
	const char lastChar = a[a.size()-1];

	if( lastChar == '/' || lastChar == '\\' )
		return a + b;
	
	return a + "/" + b;
}


// fileExtension
std::string fileExtension( const std::string& path )
{
	const std::string::size_type dotIdx = path.find_last_of(".");

	if( dotIdx == std::string::npos )
		return "";

	std::string ext = path.substr(dotIdx + 1);
	transform(ext.begin(), ext.end(), ext.begin(), tolower);
	return ext;
}


// fileHasExtension
bool fileHasExtension( const std::string& path, const std::string& extension )
{
	std::vector<std::string> extensions;
	extensions.push_back(extension);
	return fileHasExtension(path, extensions);
}

	
// fileHasExtension
bool fileHasExtension( const std::string& path, const char** extensions )
{
	if( !extensions )
		return false;

	std::vector<std::string> extList;
	uint32_t extCount = 0;

	while(true)
	{
		if( !extensions[extCount] )
			break;

		extList.push_back(extensions[extCount]);
		extCount++;
	}

	return fileHasExtension(path, extList);
}


// fileHasExtension
bool fileHasExtension( const std::string& path, const std::vector<std::string>& extensions )
{
	const std::string pathExtension = fileExtension(path);
	const size_t numExtensions = extensions.size();

	if( pathExtension.size() == 0 )
		return false;

	if( numExtensions == 0 )
		return false;

	for( size_t n=0; n < numExtensions; n++ )
	{
		if( extensions[n].size() == 0 )
			continue;

		if( strcasecmp(pathExtension.c_str(), extensions[n].c_str()) == 0 )
			return true;
	}

	return false;
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


