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

#include "commandLine.h"
#include "logging.h"

#include <string>
#include <string.h>
#include <strings.h>


#define ARGC_START 0


// strRemoveDelimiter
static inline int strRemoveDelimiter( char delimiter, const char* string )
{
    int string_start = 0;

    while( string[string_start] == delimiter )
        string_start++;

    if( string_start >= (int)strlen(string)-1 )
        return 0;

    return string_start;
}


// constructor
commandLine::commandLine( const int pArgc, char** pArgv, const char* extraFlag )
{
	argc = pArgc;
	argv = pArgv;

	AddFlag(extraFlag);

	Log::ParseCmdLine(*this);
}


// constructor
commandLine::commandLine( const int pArgc, char** pArgv, const char** extraArgs )
{
	argc = pArgc;
	argv = pArgv;

	AddArgs(extraArgs);

	Log::ParseCmdLine(*this);
}


// GetInt
int commandLine::GetInt( const char* string_ref, int default_value ) const
{
	if( argc < 1 )
		return 0;

	bool bFound = false;
    int value = -1;

	for( int i=ARGC_START; i < argc; i++ )
	{
		const int string_start = strRemoveDelimiter('-', argv[i]);
		
		if( string_start == 0 )
			continue;
		
		const char* string_argv = &argv[i][string_start];
		const int length = (int)strlen(string_ref);

		if (!strncasecmp(string_argv, string_ref, length))
		{
			if (length+1 <= (int)strlen(string_argv))
			{
				int auto_inc = (string_argv[length] == '=') ? 1 : 0;
				value = atoi(&string_argv[length + auto_inc]);
			}
			else
			{
				value = 0;
			}

			bFound = true;
			continue;
		}
	}
 

	if (bFound)
		return value;
 
	return default_value;
}


// GetUnsignedInt
uint32_t commandLine::GetUnsignedInt( const char* argName, uint32_t defaultValue ) const
{
	const int val = GetInt(argName, (int)defaultValue);

	if( val < 0 )
		return defaultValue;

	return val;
} 


// GetFloat
float commandLine::GetFloat( const char* string_ref, float default_value ) const
{
	if( argc < 1 )
		return 0;

	bool bFound = false;
	float value = -1;

	for (int i=ARGC_START; i < argc; i++)
	{
		const int string_start = strRemoveDelimiter('-', argv[i]);
		
		if( string_start == 0 )
			continue;
		
		const char* string_argv = &argv[i][string_start];
		const int length = (int)strlen(string_ref);

		if (!strncasecmp(string_argv, string_ref, length))
		{
			if (length+1 <= (int)strlen(string_argv))
			{
				int auto_inc = (string_argv[length] == '=') ? 1 : 0;
				value = (float)atof(&string_argv[length + auto_inc]);
			}
			else
			{
				value = 0.f;
			}

			bFound = true;
			continue;
		}
	}

	if( bFound )
		return value;

	return default_value;
}


// GetFlag
bool commandLine::GetFlag( const char* string_ref ) const
{
	if( argc < 1 )
		return false;

	for (int i=ARGC_START; i < argc; i++)
	{
		const int string_start = strRemoveDelimiter('-', argv[i]);
		
		if( string_start == 0 )
			continue;
		
		const char* string_argv = &argv[i][string_start];
		const char* equal_pos = strchr(string_argv, '=');
		
		const int argv_length = (int)(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);
		const int length = (int)strlen(string_ref);

		if( length == argv_length && !strncasecmp(string_argv, string_ref, length) )
			return true;
	}
    
	return false;
}


// GetString
const char* commandLine::GetString( const char* string_ref, const char* default_value ) const
{
	if( argc < 1 )
		return 0;

	for (int i=ARGC_START; i < argc; i++)
	{
		const int string_start  = strRemoveDelimiter('-', argv[i]);
		
		if( string_start == 0 )
			continue;
		
		char* string_argv = (char*)&argv[i][string_start];
		const int length = (int)strlen(string_ref);

		if (!strncasecmp(string_argv, string_ref, length))
			return (string_argv + length + 1);
			//*string_retval = &string_argv[length+1];
	}

	return default_value;
}


// GetPosition
const char* commandLine::GetPosition( unsigned int position, const char* default_value ) const
{
	if( argc < 1 )
		return 0;

	unsigned int position_count = 0;
	
	for (int i=1/*ARGC_START*/; i < argc; i++)
	{
		const int string_start = strRemoveDelimiter('-', argv[i]);
		
		if( string_start != 0 )
			continue;
		
		if( position == position_count )
			return argv[i];
		
		position_count++;
	}

	return default_value;
}


// GetPositionArgs
unsigned int commandLine::GetPositionArgs() const
{
	unsigned int position_count = 0;
	
	for (int i=1/*ARGC_START*/; i < argc; i++)
	{
		const int string_start = strRemoveDelimiter('-', argv[i]);
		
		if( string_start != 0 )
			continue;
		
		position_count++;
	}

	return position_count;
}


// AddArg
void commandLine::AddArg( const char* arg )
{
	if( !arg )
		return;

	const size_t arg_length = strlen(arg);

	if( arg_length == 0 )
		return;

	const int new_argc = argc + 1;
	char** new_argv = (char**)malloc(sizeof(char*) * new_argc);

	if( !new_argv )
		return;

	for( int n=0; n < argc; n++ )
		new_argv[n] = argv[n];

	new_argv[argc] = (char*)malloc(arg_length + 1);

	if( !new_argv[argc] )
		return;

	strcpy(new_argv[argc], arg);

	argc = new_argc;
	argv = new_argv;
}


// AddArgs
void commandLine::AddArgs( const char** args )
{
	if( !args )
		return;

	int arg_count = 0;

	while(true)
	{
		if( !args[arg_count] )
			return;
		
		AddArg(args[arg_count]);
		arg_count++;
	}
}


// AddFlag
void commandLine::AddFlag( const char* flag )
{
	if( !flag || strlen(flag) == 0 )
		return;

	if( GetFlag(flag) )
		return;

	const std::string arg = std::string("--") + flag;
	AddArg(arg.c_str());
}


// Print
void commandLine::Print() const
{
	for( int n=0; n < argc; n++ )
		printf("%s ", argv[n]);

	printf("\n");
}




