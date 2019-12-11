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
 
#ifndef __CSV_READER_H_
#define __CSV_READER_H_

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include <string>
#include <vector>
#include <iostream>


/**
 * csvData
 * @ingroup csv
 */
class csvData
{
public:
	// constructors
	csvData( char* str ) 							{ string = str; }
	csvData( const char* str ) 						{ string = str; }
	csvData( std::string& str )						{ string = str; }

	// assignment
	template<typename T> csvData( T value )				{ set(value); }
	template<typename T> void set( T value )			{ string = std::to_string(value); }

	// cast to string
	inline operator std::string&() 					{ return string; }
	inline operator const std::string&() const			{ return string; }
	inline operator const char*() const				{ return string.c_str(); }
	
	// cast to number
	inline operator int() const						{ return toInt(); }
	inline operator float() const 					{ return toFloat(); }
	inline operator double() const					{ return toDouble(); }

	// convert to number (return true if valid)
	inline bool toInt( int* value ) const;
	inline bool toFloat( float* value ) const;
	inline bool toDouble( double* value ) const;

	// convert to number (valid->false on error)
	inline int toInt( bool* valid=NULL ) const;
	inline float toFloat( bool* valid=NULL ) const;
	inline double toDouble( bool* valid=NULL ) const;

	// stream insertion/extraction operators
	inline friend std::istream& operator >> (std::istream& in, csvData& obj); 
	inline friend std::ostream& operator << (std::ostream& out, const csvData& obj);
	
	// split string by delimiters into list of tokens
	inline static std::vector<csvData> Parse( const char* str, const char* delimiters=",;\t " );

	// fill list of tokens with string split by delimiters
	inline static bool Parse( std::vector<csvData>& data, const char* str, const char* delimiters=",;\t " );

	// data storage
	std::string string;
};


/**
 * csvReader
 * @ingroup csv
 */
class csvReader
{
public:
	// constructor/destructor
	csvReader( const char* filename, const char* delimiters=",;\t " );
	~csvReader();

	// open
	inline static csvReader* Open( const char* filename, const char* delimiters=",;\t " );

	// close
	inline void Close();

	// is open and not EOF
	inline bool IsOpen() const;
	inline bool IsClosed() const;

	// read line, return list of tokens
	inline std::vector<csvData> Read();
	inline std::vector<csvData> Read( const char* delimiters );

	// read line, fill list of tokens
	inline bool Read( std::vector<csvData>& data );
	inline bool Read( std::vector<csvData>& data, const char* delimiters );

	// set default delimiters
	inline void SetDelimiters( const char* delimiters );

	// retrieve default delimiters
	inline const char* GetDelimiters() const;

	// retrieve the filename
	inline const char* GetFilename() const;

	// maximum line length
	const size_t MaxLineLength=2048;

private:
	FILE* mFile;

	std::string mFilename;
	std::string mDelimiters;
};


// internal functions
#include "csvReader.hpp"

#endif

