/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __CSV_WRITER_H_
#define __CSV_WRITER_H_

#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include <string>
#include <vector>
#include <iostream>


/**
 * csvWriter
 * @ingroup csv
 */
class csvWriter
{
public:
	// constructor/destructor
	csvWriter( const char* filename, const char* delimiter=", " );
	~csvWriter();

	// open
	inline static csvWriter* Open( const char* filename, const char* delimiter=", " );

	// close/flush
	inline void Close();
	inline void Flush();

	// is open or closed
	inline bool IsOpen() const;
	inline bool IsClosed() const;

	// end the current line
	inline void EndLine();
	
	// write value
	template<typename T>
	inline csvWriter& Write( const T& value );

	// write values
	template<typename T, typename... Args>
	inline csvWriter& Write( const T& value, const Args&... args );

	// write values and end the line
	template<typename T, typename... Args>
	inline csvWriter& WriteLine( const T& value, const Args&... args );

	// stream insertion
	template<typename T>
	inline csvWriter& operator << ( const T& value );

	// stream manipulators
	inline csvWriter& operator << ( csvWriter& (*value)(csvWriter&) );

	// set default delimiter
	inline void SetDelimiter( const char* delimiters );

	// retrieve default delimiter
	inline const char* GetDelimiter() const;

	// retrieve the filename
	inline const char* GetFilename() const;

private:
	std::ofstream mFile;
	std::string   mFilename;
	std::string   mDelimiter;
	bool		    mNewLine;
};


/**
 * csv stream manipulators
 * @ingroup csv
 */
namespace csv
{
	inline static csvWriter& endl( csvWriter& file );
	inline static csvWriter& flush( csvWriter& file );
}


// internal functions
#include "csvWriter.hpp"

#endif

