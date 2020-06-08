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

#ifndef __CUDA_FONT_H__
#define __CUDA_FONT_H__


#include "cudaUtility.h"
#include "imageFormat.h"

#include <string>
#include <vector>


/**
 * Determine an appropriate font size given a particular dimension to use
 * (typically an image's width). Then the font won't be radically unsized.
 * 
 * @param dimension The dimension's size to fit against (i.e. image width)
 * @returns a font size between 10 and 32 pixels tall. 
 * @ingroup cudaFont
 */
float adaptFontSize( uint32_t dimension );


/**
 * TTF font rasterization and image overlay rendering using CUDA.
 * @ingroup cudaFont
 */
class cudaFont
{
public:
	/**
	 * Create new CUDA font overlay object using baked fonts.
	 * @param size The desired height of the font, in pixels.
	 */
	static cudaFont* Create( float size=32.0f );

	/**
	 * Create new CUDA font overlay object using baked fonts.
	 * @param font The name of the TTF font to use.
	 * @param size The desired height of the font, in pixels.
	 */
	static cudaFont* Create( const char* font, float size );
	
	/**
	 * Create new CUDA font overlay object using baked fonts.
	 * @param font A list of font names that are acceptable to use.
	 *             If the first font isn't found on the system,
	 *             then the next font from the list will be tried.
	 * @param size The desired height of the font, in pixels.
	 */
	static cudaFont* Create( const std::vector<std::string>& fonts, float size );

	/**
	 * Destructor
	 */
	~cudaFont();
	
	/**
	 * Render text overlay onto image
	 */
	bool OverlayText( void* image, imageFormat format,
				   uint32_t width, uint32_t height, 
			        const char* str, int x, int y, 
				   const float4& color=make_float4(0, 0, 0, 255),
				   const float4& background=make_float4(0, 0, 0, 0),
				   int backgroundPadding=5 );

	/**
	 * Render text overlay onto image
	 */
	bool OverlayText( void* image, imageFormat format, 
				   uint32_t width, uint32_t height, 
			        const std::vector< std::pair< std::string, int2 > >& text,
			        const float4& color=make_float4(0, 0, 0, 255),
				   const float4& background=make_float4(0, 0, 0, 0),
				   int backgroundPadding=5 );

	/**
	 * Render text overlay onto image
	 */
	template<typename T> bool OverlayText( T* image, uint32_t width, uint32_t height, 
			        				    const char* str, int x, int y, 
				   				    const float4& color=make_float4(0, 0, 0, 255),
				   				    const float4& background=make_float4(0, 0, 0, 0),
				   				    int backgroundPadding=5 )		
	{ 
		return OverlayText(image, imageFormatFromType<T>(), width, height, str, x, y, color, background, backgroundPadding); 
	}
			
	/**
	 * Render text overlay onto image
	 */
	template<typename T> bool OverlayText( T* image, uint32_t width, uint32_t height, 
			        				    const std::vector< std::pair< std::string, int2 > >& text, 
				   				    const float4& color=make_float4(0, 0, 0, 255),
				   				    const float4& background=make_float4(0, 0, 0, 0),
				   				    int backgroundPadding=5 )		
	{ 
		return OverlayText(image, imageFormatFromType<T>(), width, height, text, color, background, backgroundPadding); 
	}

	/**
	 * Return the bounding rectangle of the given text string.
	 */
	int4 TextExtents( const char* str, int x=0, int y=0 );


protected:
	cudaFont();
	bool init( const char* font, float size );

	uint8_t* mFontMapCPU;
	uint8_t* mFontMapGPU;
	
	int mFontMapWidth;
	int mFontMapHeight;
	
	void* mCommandCPU;
	void* mCommandGPU;
	int   mCmdIndex;

	float4* mRectsCPU;
	float4* mRectsGPU;
	int     mRectIndex;

	static const uint32_t MaxCommands = 1024;
	static const uint32_t FirstGlyph  = 32;
	static const uint32_t LastGlyph   = 255;
	static const uint32_t NumGlyphs   = LastGlyph - FirstGlyph;

	struct GlyphInfo
	{
		uint16_t x;
		uint16_t y;
		uint16_t width;
		uint16_t height;

		float xAdvance;
		float xOffset;
		float yOffset;
	} mGlyphInfo[NumGlyphs];
};

#endif
