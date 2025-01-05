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
 
#ifndef __CSV_READER_HPP_
#define __CSV_READER_HPP_


#include "csvReader.h"
#include "logging.h"


// toInt()
inline bool csvData::toInt( int* value ) const
{
	char* e;
	errno = 0;

	const int x = strtol(string.c_str(), &e, 0);

	if( *e != '\0' || errno != 0 )
		return false;

	if( value != NULL )
		*value = x;

	return true;
}

// toFloat()
inline bool csvData::toFloat( float* value ) const
{
	char* e;
	errno = 0;

	const float x = strtof(string.c_str(), &e);

	if( *e != '\0' || errno != 0 )
		return false;

	if( value != NULL )
		*value = x;

	return true;
}

// toDouble()
inline bool csvData::toDouble( double* value ) const
{
	char* e;
	errno = 0;

	const float x = strtof(string.c_str(), &e);

	if( *e != '\0' || errno != 0 )
		return false;

	if( value != NULL )
		*value = x;

	return true;
}

// toInt()
inline int csvData::toInt( bool* valid ) const			
{ 
	int x=0; 
	const bool v=toInt(&x); 

	if( valid != NULL ) 
		*valid=v;

	return x; 
}

// toFloat()
inline float csvData::toFloat( bool* valid ) const		
{ 
	float x=0.0f; 
	const bool v=toFloat(&x); 

	if( valid != NULL ) 
		*valid=v; 

	return x; 
}

// toDouble()
inline double csvData::toDouble( bool* valid ) const		
{ 
	double x=0.0f; 
	const bool v=toDouble(&x); 

	if( valid != NULL ) 
		*valid=v; 

	return x; 
}

// operator >>
inline std::istream& operator >> (std::istream& in, csvData& obj)
{
	in >> obj.string;
	return in;
}

// operator <<
inline std::ostream& operator << (std::ostream& out, const csvData& obj)
{
	out << obj.string;
	return out;
}

// Parse()
inline std::vector<csvData> csvData::Parse( const char* str, const char* delimiters ) 
{
	std::vector<csvData> tokens;
	Parse(tokens, str, delimiters);
	return tokens;
}

// Parse
inline bool csvData::Parse( std::vector<csvData>& tokens, const char* str, const char* delimiters )
{
	if( !str || !delimiters )
		return false;

	tokens.clear();

	const size_t str_length = strlen(str);
	char* str_tokens = (char*)malloc(str_length + 1);

	if( !str_tokens )
		return false;

	strcpy(str_tokens, str);

	if( str_tokens[str_length] == '\n' )
		str_tokens[str_length] = '\0';

	if( str_tokens[str_length-1] == '\n' )
		str_tokens[str_length-1] = '\0';

	char* token = strtok(str_tokens, delimiters);

	while( token != NULL )
	{
		tokens.push_back(token);
		token = strtok(NULL, delimiters);
	}

	free(str_tokens);
	return tokens.size() > 0;
}

//-------------------------------------------------------------------------------------
// constructor
csvReader::csvReader( const char* filename, const char* delimiters ) : mFile(NULL)
{
	if( !filename || !delimiters )
		return;

	mFile = fopen(filename, "r");

	if( !mFile )
	{
		LogError("csvReader -- failed to open file %s\n", filename);
		perror("csvReader -- error");
		return;
	}

	mFilename = filename;
	mDelimiters = delimiters;
}

// destructor
csvReader::~csvReader()
{
	Close();
}

	
// open
inline csvReader* csvReader::Open( const char* filename, const char* delimiters )
{
	if( !filename || !delimiters )
		return NULL;

	csvReader* csv = new csvReader(filename, delimiters);

	if( !csv->IsOpen() )
	{
		delete csv;
		return NULL;
	}

	return csv;
}

// close
inline void csvReader::Close()
{
	if( IsClosed() )
		return;

	fclose(mFile);
	mFile = NULL;
}

// isOpen
inline bool csvReader::IsOpen() const
{
	return mFile != NULL;
}

// isClosed
inline bool csvReader::IsClosed() const
{
	return !IsOpen();
}

// readLine
inline std::vector<csvData> csvReader::Read()
{
	return Read(mDelimiters.c_str());
}

// readLine
inline std::vector<csvData> csvReader::Read( const char* delimiters )
{
	std::vector<csvData> tokens;
	Read(tokens, delimiters);
}

// readLine
inline bool csvReader::Read( std::vector<csvData>& data )
{
	return Read(data, mDelimiters.c_str());
}

// readLine
inline bool csvReader::Read( std::vector<csvData>& data, const char* delimiters )
{
	if( IsClosed() )
		return false;

	// read the next line
	char str[MaxLineLength];

	if( fgets(str, sizeof(str), mFile) == NULL )
	{
		if( ferror(mFile) )
		{
			LogError("csvReader -- error reading file %s\n", mFilename.c_str());
			perror("csvReader -- error");
		}

		Close();
		return false;
	}

	// check if EOF was reached
	if( feof(mFile) == EOF )
		Close();

	// disregard comments
	if( str[0] == '#' )
		return Read(data, delimiters);

	return csvData::Parse(data, str, delimiters);
}

// SetDelimiters
inline void csvReader::SetDelimiters( const char* delimiters )
{
	mDelimiters = delimiters;
}

// GetDelimiters
inline const char* csvReader::GetDelimiters() const
{
	return mDelimiters.c_str();
}

// GetFilename
inline const char* csvReader::GetFilename() const
{
	return mFilename.c_str();
}

#endif

