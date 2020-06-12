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
 
#ifndef __CSV_WRITER_HPP_
#define __CSV_WRITER_HPP_


#include "csvWriter.h"
#include "logging.h"


// constructor
csvWriter::csvWriter( const char* filename, const char* delimiter )
{
	if( !filename || !delimiter )
		return;

	mFile.open(filename);

	if( !IsOpen() )
	{
		LogError("csvWriter -- failed to open file %s\n", filename);
		return;
	}

	mFilename  = filename;
	mDelimiter = delimiter;
	mNewLine   = true;
}


// destructor
csvWriter::~csvWriter()
{
	Close();
}

	
// open
inline csvWriter* csvWriter::Open( const char* filename, const char* delimiter )
{
	if( !filename || !delimiter )
		return NULL;

	csvWriter* csv = new csvWriter(filename, delimiter);

	if( !csv->IsOpen() )
	{
		delete csv;
		return NULL;
	}

	return csv;
}

// close
inline void csvWriter::Close()
{
	if( IsClosed() )
		return;

	Flush();
	mFile.close();
}

// flush
inline void csvWriter::Flush()
{
	mFile.flush();
}

// isOpen
inline bool csvWriter::IsOpen() const
{
	return mFile.is_open();
}

// isClosed
inline bool csvWriter::IsClosed() const
{
	return !IsOpen();
}

// EndLine
inline void csvWriter::EndLine()
{
	mFile << std::endl;
	mNewLine = true;
}

// Write
template<typename T>
inline csvWriter& csvWriter::Write( const T& value )
{
	if( !mNewLine )
		mFile << mDelimiter;
	else
		mNewLine = false;

	mFile << value;
	return *this;
}

// Write
template<typename T, typename... Args>
inline csvWriter& csvWriter::Write( const T& value, const Args&... args )
{
	Write(value);
	Write(args...);
	return *this;
}

// WriteLine
template<typename T, typename... Args>
inline csvWriter& csvWriter::WriteLine( const T& value, const Args&... args )
{
	Write(value);
	Write(args...);
	EndLine();
	return *this;
}

// stream insertion
template<typename T>
inline csvWriter& csvWriter::operator << ( const T& value )
{
	return Write(value);
}

// stream manipulator
inline csvWriter& csvWriter::operator << ( csvWriter& (*value)(csvWriter&) )
{
	return value(*this);
}

// SetDelimiter
inline void csvWriter::SetDelimiter( const char* delimiter )
{
	mDelimiter = delimiter;
}

// GetDelimiter
inline const char* csvWriter::GetDelimiter() const
{
	return mDelimiter.c_str();
}

// GetFilename
inline const char* csvWriter::GetFilename() const
{
	return mFilename.c_str();
}

//----------------------------------------------------------------
namespace csv
{

// endl
inline static csvWriter& endl( csvWriter& file )
{
    file.EndLine();
    return file;
}

// flush
inline static csvWriter& flush( csvWriter& file )
{
    file.Flush();
    return file;
}

}

#endif

