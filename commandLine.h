/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __COMMAND_LINE_H_
#define __COMMAND_LINE_H_


#include <stdlib.h>	
#include <stdint.h>


/**
 * Command line parser for extracting flags, values, and strings.
 * @ingroup commandLine
 */
class commandLine
{
public:
	/**
	 * Constructor, takes the command line from `main()`
	 */
	commandLine( const int argc, char** argv, const char* extraFlag=NULL );

	/**
	 * Constructor, takes the command line from `main()`
	 */
	commandLine( const int argc, char** argv, const char** extraArgs );

	/**
	 * Checks to see whether the specified flag was included on the 
	 * command line.   For example, if argv contained `--foo`, then 
	 * `GetFlag("foo")` would return `true`
	 *
	 * @param allowOtherDelimiters if true (default), the argName will be 
	 *          matched against occurances containing either `-` or `_`.  
	 *          For example, `--foo-bar` and `--foo_bar` would be the same.
	 *
	 * @returns `true`, if the flag with argName was found
	 *          `false`, if the flag with argName was not found
	 */
	bool GetFlag( const char* argName, bool allowOtherDelimiters=true ) const;
	
	/**
	 * Get float argument.  For example if argv contained `--foo=3.14159`, 
	 * then `GetInt("foo")` would return `3.14159f`
	 *
	 * @param allowOtherDelimiters if true (default), the argName will be 
	 *          matched against occurances containing either `-` or `_`.  
	 *          For example, `--foo-bar` and `--foo_bar` would be the same.
	 *
	 * @returns `defaultValue` if the argument couldn't be found. (`0.0` by default).
	 *          Otherwise, returns the value of the argument.
	 */
	float GetFloat( const char* argName, float defaultValue=0.0f, bool allowOtherDelimiters=true ) const;

	/**
	 * Get integer argument.  For example if argv contained `--foo=100`, 
	 * then `GetInt("foo")` would return `100`
	 *
	 * @param allowOtherDelimiters if true (default), the argName will be 
	 *          matched against occurances containing either `-` or `_`.  
	 *          For example, `--foo-bar` and `--foo_bar` would be the same.
	 *
	 * @returns `defaultValue` if the argument couldn't be found (`0` by default).
	 *          Otherwise, returns the value of the argument. 
	 */
	int GetInt( const char* argName, int defaultValue=0, bool allowOtherDelimiters=true ) const; 

	/**
	 * Get unsigned integer argument.  For example if argv contained `--foo=100`, 
	 * then `GetUnsignedInt("foo")` would return `100`
	 *
	 * @param allowOtherDelimiters if true (default), the argName will be 
	 *          matched against occurances containing either `-` or `_`.  
	 *          For example, `--foo-bar` and `--foo_bar` would be the same.
	 *
	 * @returns `defaultValue` if the argument couldn't be found, or if the value
	 *          was negative (`0` by default). Otherwise, returns the parsed value.
	 */
	uint32_t GetUnsignedInt( const char* argName, uint32_t defaultValue=0, bool allowOtherDelimiters=true ) const; 

	/**
	 * Get string argument.  For example if argv contained `--foo=bar`,
	 * then `GetString("foo")` would return `"bar"`
	 *
	 * @param allowOtherDelimiters if true (default), the argName will be 
	 *          matched against occurances containing either `-` or `_`.  
	 *          For example, `--foo-bar` and `--foo_bar` would be the same.
	 *
	 * @returns `defaultValue` if the argument couldn't be found (`NULL` by default).
	 *          Otherwise, returns a pointer to the argument value string
	 *          from the `argv` array.
	 */
	const char* GetString( const char* argName, const char* defaultValue=NULL, bool allowOtherDelimiters=true ) const;

	/**
	 * Get positional string argument.  Positional arguments aren't named, but rather
	 * referenced by their index in the list. For example if the command line contained
	 * `my-program --foo=bar /path/to/my_file.txt`, then `GetString(0)` would return
	 * `"/path/to/my_file.txt"
	 *
	 * @returns `defaultValue` if the argument couldn't be found (`NULL` by default).
	 *          Otherwise, returns a pointer to the argument value string
	 *          from the `argv` array.
	 */
	const char* GetPosition( unsigned int position, const char* defaultValue=NULL ) const;
	
	/**
	 * Get the number of positional arguments in the command line.
	 * Positional arguments are those that don't have a name.
	 */
	unsigned int GetPositionArgs() const;
	
	/**
	 * Add an argument to the command line.
	 */
	void AddArg( const char* arg );

	/**
	 * Add arguments to the command line.
	 */
	void AddArgs( const char** args );

	/**
	 * Add a flag to the command line.
	 */
	void AddFlag( const char* flag );

	/**
	 * Print out the command line for reference.
	 */
	void Print() const;

	/**
	 * The argument count that the object was created with from main()
	 */
	int argc;

	/**
	 * The argument strings that the object was created with from main()
	 */
	char** argv;
};


/**
 * Specify a positional argument index.
 * @ingroup commandLine
 */
#define ARG_POSITION(x) x


#endif

