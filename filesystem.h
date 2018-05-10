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
 
#ifndef __FILESYSTEM_UTIL_H__
#define __FILESYSTEM_UTIL_H__

#include <string>
#include <vector>


/**
 * Return a list of all the files in the specified directory
 * @param path the path of the directory
 * @param list the output list of all files in the directory
 * @param includePath if true, the list of filenames will be prefixed with the path
 *                    if false (default), the list of filenames will contain filenames/extensions only
 */
bool listDir( const char* path, std::vector<std::string>& list, bool includePath=false );


/**
 * Verify path and return true if the file exists.
 * @param regularFilesOnly If false (which is the default), then sysFileExists() includes filesystem entries 
 *                         like directories, device files, and sockets when verifying the path.
 *
 *                         If regularFilesOnly parameter is true, then sysFileExists() will verify the path 
 *                         is to a readable, "regular" file.  Other file types (for example directories)
 *                         will result in sysFileExists() returning false.
 */
bool fileExists( const char* path, bool regularFilesOnly=false );


/**
 * Return the size (in bytes) of the specified file
 * @param path the path of the file
 * @return if successful, the size of the file in bytes
 *         otherwise, 0 will be returned.
 */
size_t fileSize( const char* path );


/**
 * Extract the path out of the supplied string, removing the filename and extension
 * For example, filePath("~/user/somefile.xml") would return "~/user"
 */
std::string filePath( const std::string& filename );


/**
 * Extract the file extension from the path.
 * This function will return all contents of the path to the right of the right-most '.'
 * The extension will be returned in all lowercase characters.
 */
std::string fileExtension( const std::string& path );


/**
 * Return the input string with the file extension removed
 * For example, `strRemoveExtension("~/user/somefile.xml")`
 * would return `"~/user/somefile"`.
 */
std::string fileRemoveExtension( const std::string& filename );


/**
 * Return the input string with a changed file extension
 * For example, `strChangeExtension("~/user/somefile.xml", "zip")`
 * would return `"~/user/somefile.zip"`.
 */
std::string fileChangeExtension( const std::string& filename, const std::string& newExtension );


#endif
