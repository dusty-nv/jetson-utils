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
 * Given a relative path, resolve the absolute path using the working directory.
 *
 * For example, if the current working directory `/home/user/` and
 * `absolutePath("resources/example")` is called, then this function
 * would return the path `/home/user/resources/example`.
 * 
 * If the path is already an absolute path (i.e. it begins with `/` or `~/`)
 * then this function will be ignored and the path will be returned as-is.
 *
 * @ingroup filesystem
 */
std::string absolutePath( const std::string& relative_path );

/**
 * Locate a file from common system locations.
 * First, this function will check if the file exists at the path provided,
 * and if not it will check for the existance of the file in common system
 * locations such as "/opt", "/usr/local", and "/usr/local/bin".
 *
 * @return the confirmed path of the located file, or empty string if
 *         the file could not be found
 *
 * @ingroup filesystem
 */
std::string locateFile( const std::string& path );

/**
 * Locate a file from a set of locations provided by the user, in addition 
 * to common system locations such as "/opt" and "/usr/local".
 *
 * @return the confirmed path of the located file, or empty string if
 *         the file could not be found
 *
 * @ingroup filesystem
 */
std::string locateFile( const std::string& path, std::vector<std::string>& locations );

/**
 * Join two paths, and properly include a path separator (`/`) as needed.
 * For example, 'pathJoin("~/workspace", "somefile.xml")` would return `~/workspace/somefile.xml`.
 * @ingroup filesystem
 */
std::string pathJoin( const std::string& a, const std::string& b );

/**
 * Return the parent directory of the specified path, removing the filename and extension.
 * For example, `pathDir("~/workspace/somefile.xml")` would return `~/workspace/`
 * @ingroup filesystem
 */
std::string pathDir( const std::string& path );

/**
 * File types
 * @ingroup filesystem
 */
enum fileTypes
{
	FILE_MISSING = 0,
	FILE_REGULAR = (1 << 0),
	FILE_DIR     = (1 << 1),
	FILE_LINK    = (1 << 2),
	FILE_CHAR    = (1 << 3),
	FILE_BLOCK   = (1 << 4),
	FILE_FIFO    = (1 << 5),
	FILE_SOCKET  = (1 << 6)
};

/**
 * Return a sorted list of the files in the specified directory.  listDir() will glob files from
 * the specified path, and filter against wildcard characters including `*` and `?`.
 * For example, valid paths would include `~/workspace`, `~/workspace/*.jpg`, ect.
 * @see here for a description of wildcard matching:  https://www.man7.org/linux/man-pages/man7/glob.7.html
 * @param path the path of the directory (may include wildcard characters)
 * @param[out] list the alphanumerically sorted output list of the files in the directory
 * @param mask filter by file type (by default, any file including directories will be included).
 *             The mask should consist of fileTypes OR'd together (e.g. `FILE_REGULAR|FILE_DIR`).
 * @ingroup filesystem
 */
bool listDir( const std::string& path, std::vector<std::string>& list, uint32_t mask=0 );

/**
 * Verify path and return true if the file exists.
 * @param mask filter by file type (by default, any file including directories will be checked).
 *             The mask should consist of fileTypes OR'd together (e.g. `FILE_REGULAR|FILE_DIR`).
 * @ingroup filesystem
 */
bool fileExists( const std::string& path, uint32_t mask=0 );

/**
 * Return true if the file is one of the types in the fileTypes mask.
 * @param mask file types to check against (@see fileTypes)
 *             The mask should consist of fileTypes OR'd together (e.g. `FILE_REGULAR|FILE_DIR`).
 * @ingroup filesystem
 */
bool fileIsType( const std::string& path, uint32_t mask );

/**
 * Return the file type, or FILE_MISSING if it doesn't exist.
 * @see fileTypes
 * @ingroup filesystem
 */
uint32_t fileType( const std::string& path );

/**
 * Return the size (in bytes) of the specified file.
 *
 * @param path the path of the file
 * @return if successful, the size of the file in bytes
 *         otherwise, 0 will be returned.
 *
 * @ingroup filesystem
 */
size_t fileSize( const std::string& path );

/**
 * Extract the file extension from the path.
 * This function will return all contents of the path to the right of the right-most `'.'`
 * The extension will be returned in all lowercase characters.
 * @ingroup filesystem
 */
std::string fileExtension( const std::string& path );

/**
 * Return true if the file has the given extension, otherwise false.
 * For example, `fileHasExtension("~/workspace/image.jpg", "jpg")` would return true.
 * @ingroup filesystem
 */
bool fileHasExtension( const std::string& path, const std::string& extension );

/**
 * Return true if the file has one of the given extensions, otherwise false.
 * @ingroup filesystem
 */
bool fileHasExtension( const std::string& path, const std::vector<std::string>& extensions );

/**
 * Return true if the file has one of the given extensions, otherwise false.
 * For example, `fileHasExtension("image.jpg", {"jpg", "jpeg", NULL})` would return true.
 * @param extensions list of extensions, should end with `NULL` sentinel.
 * @ingroup filesystem
 */
bool fileHasExtension( const std::string& path, const char** extensions );

/**
 * Return the input string with the file extension removed
 * For example, `fileRemoveExtension("~/workspace/somefile.xml")`
 * would return `~/user/somefile`.
 * @ingroup filesystem
 */
std::string fileRemoveExtension( const std::string& filename );

/**
 * Return the input string with a changed file extension
 * For example, `fileChangeExtension("~/workspace/somefile.xml", "zip")`
 * would return `~/user/somefile.zip`.
 * @ingroup filesystem
 */
std::string fileChangeExtension( const std::string& filename, const std::string& newExtension );

/**
 * Return the absolute path that of the calling process executable,
 * include the process executable's filename.
 * @ingroup filesystem
 */
std::string processPath();

/**
 * Return the directory that the calling process resides in.
 * For example, if the process executable is located at `/usr/bin/exe`,
 * then `processDirectory()` would return the path `/usr/bin`.
 *
 * @note to retrieve the full path of the calling process, including
 *       the process executable's filename, @see processPath()
 *
 * @ingroup filesystem
 */
std::string processDirectory();

/**
 * Return the current working directory of the calling process.
 * @ingroup filesystem
 */
std::string workingDirectory();


#endif

