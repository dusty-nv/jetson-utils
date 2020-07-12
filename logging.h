/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __LOGGING_UTILS_H_
#define __LOGGING_UTILS_H_

#include "commandLine.h"

#include <stdio.h>
#include <string>


/**
 * Standard command-line options able to be passed to videoOutput::Create()
 * @ingroup log
 */
#define LOG_USAGE_STRING  "logging arguments: \n" 									\
		  "  --log-file=FILE        output destination file (default is stdout)\n"			\
		  "  --log-level=LEVEL      message output threshold, one of the following:\n"		\
		  "                             * silent\n"									\
		  "                             * error\n"									\
		  "                             * warning\n"									\
		  "                             * success\n"									\
 		  "                             * info\n"									\
		  "                             * verbose (default)\n"							\
		  "                             * debug\n"									\
		  "  --verbose              enable verbose logging (same as --log-level=verbose)\n"  \
		  "  --debug                enable debug logging   (same as --log-level=debug)\n\n"


/**
 * Message logging with a variable level of output and destinations.
 * @ingroup log
 */
class Log
{
public:
	/**
	 * Defines the logging level of a message, and the threshold
	 * used by the logger to either drop or output messages.
	 */
	enum Level
	{
		SILENT=0,		 /**< No messages are output. */
		ERROR,		 /**< Major errors that may impact application execution. */
		WARNING,		 /**< Warning conditions where the application may be able to proceed in some capacity. */
		SUCCESS,		 /**< Successful events (e.g. the loading or creation of a resource) */
		INFO,		 /**< Informational messages that are more important than VERBOSE messages */
		VERBOSE,		 /**< Verbose details about program execution */
		DEBUG,		 /**< Low-level debugging (disabled by default) */
		DEFAULT=VERBOSE /**< The default level is `VERBOSE` */
	};

	/**
	 * Get the current logging level.
	 */
	static inline Level GetLevel()			{ return mLevel; }

	/**
	 * Set the current logging level.
	 */
	static inline void SetLevel( Level level )	{ mLevel = level; }

	/**
	 * Get the current log output.
	 */
	static inline FILE* GetFile()				{ return mFile; }

	/**
	 * Get the filename of the log output.
	 * This may also return `"stdout"` or `"stderror"`.
	 */
	static inline const char* GetFilename()		{ return mFilename.c_str(); }

	/**
	 * Set the logging output.
	 * This can be a built-in file, like `stdout` or `stderr`,
	 * or a file that has been opened by the user.
	 */
	static void SetFile( FILE* file );

	/**
	 * Set the logging output.
	 * Can be `"stdout"`, `"stderr"`, `"log.txt"`, ect.
	 */
	static void SetFile( const char* filename );

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 			{ return LOG_USAGE_STRING; }
	
	/**
	 * Parse command line options (see Usage() above)
	 */
	static void ParseCmdLine( const int argc, char** argv );

	/**
	 * Parse command line options (see Usage() above)
	 */
	static void ParseCmdLine( const commandLine& cmdLine );

	/**
	 * Convert a logging level to string.
	 */
	static const char* LevelToStr( Level level );

	/**
	 * Parse a logging level from a string. 
	 */
	static Level LevelFromStr( const char* str );

protected:
	static Level 	    mLevel;
	static FILE* 	    mFile;
	static std::string mFilename;
};


/**
 * Log a printf-style message with the provided level.
 * @ingroup log
 * @internal
 */
#define LogMessage(level, format, args...) if( level <= Log::GetLevel() ) fprintf(Log::GetFile(), format, ## args)

/**
 * Log a printf-style error message (Log::ERROR)
 * @ingroup log
 */
#define LogError(format, args...)		LogMessage(Log::ERROR, LOG_COLOR_RED LOG_LEVEL_PREFIX_ERROR format LOG_COLOR_RESET, ## args)

/**
 * Log a printf-style warning message (Log::WARNING)
 * @ingroup log
 */
#define LogWarning(format, args...)	LogMessage(Log::WARNING, LOG_COLOR_YELLOW LOG_LEVEL_PREFIX_WARNING format LOG_COLOR_RESET, ## args)

/**
 * Log a printf-style success message (Log::SUCCESS)
 * @ingroup log
 */
#define LogSuccess(format, args...)	LogMessage(Log::SUCCESS, LOG_COLOR_GREEN LOG_LEVEL_PREFIX_SUCCESS format LOG_COLOR_RESET, ## args)

/**
 * Log a printf-style info message (Log::INFO)
 * @ingroup log
 */
#define LogInfo(format, args...)		LogMessage(Log::INFO, LOG_LEVEL_PREFIX_INFO format, ## args)

/**
 * Log a printf-style verbose message (Log::VERBOSE)
 * @ingroup log
 */
#define LogVerbose(format, args...)	LogMessage(Log::VERBOSE, LOG_LEVEL_PREFIX_VERBOSE format, ## args)

/**
 * Log a printf-style debug message (Log::DEBUG)
 * @ingroup log
 */
#define LogDebug(format, args...)		LogMessage(Log::DEBUG, LOG_LEVEL_PREFIX_DEBUG format, ## args)


///////////////////////////////////////////////////////////////////
/// @name Logging Internals
/// @internal
/// @ingroup log
///////////////////////////////////////////////////////////////////

///@{

#ifdef LOG_DISABLE_COLORS
	#define LOG_COLOR_RESET      ""
	#define LOG_COLOR_RED        ""
	#define LOG_COLOR_GREEN      ""
	#define LOG_COLOR_YELLOW     ""
	#define LOG_COLOR_BLUE       ""
	#define LOG_COLOR_MAGENTA    ""
	#define LOG_COLOR_CYAN       ""
	#define LOG_COLOR_LIGHT_GRAY ""
	#define LOG_COLOR_DARK_GRAY  ""
#else
	// https://misc.flogisoft.com/bash/tip_colors_and_formatting
	#define LOG_COLOR_RESET      "\033[0m"
	#define LOG_COLOR_RED        "\033[0;31m"
	#define LOG_COLOR_GREEN      "\033[0;32m"
	#define LOG_COLOR_YELLOW     "\033[0;33m"
	#define LOG_COLOR_BLUE       "\033[0;34m"
	#define LOG_COLOR_MAGENTA    "\033[0;35m"
	#define LOG_COLOR_CYAN       "\033[0;36m"
	#define LOG_COLOR_LIGHT_GRAY "\033[0;37m"
	#define LOG_COLOR_DARK_GRAY  "\033[0;90m"
#endif

#ifdef LOG_ENABLE_LEVEL_PREFIX
	#define LOG_LEVEL_PREFIX_ERROR	"[E]"
	#define LOG_LEVEL_PREFIX_WARNING	"[W]"
	#define LOG_LEVEL_PREFIX_SUCCESS	"[S]"
	#define LOG_LEVEL_PREFIX_INFO		"[I]"
	#define LOG_LEVEL_PREFIX_VERBOSE	"[V]"
	#define LOG_LEVEL_PREFIX_DEBUG	"[D]"
#else
	#define LOG_LEVEL_PREFIX_ERROR	""
	#define LOG_LEVEL_PREFIX_WARNING	""
	#define LOG_LEVEL_PREFIX_SUCCESS	""
	#define LOG_LEVEL_PREFIX_INFO		""
	#define LOG_LEVEL_PREFIX_VERBOSE	""
	#define LOG_LEVEL_PREFIX_DEBUG	""
#endif

///@}

#endif

