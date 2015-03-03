/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Container/String/hkUtf8.h>
#include <wchar.h>

#ifndef WCHAR_MAX
	#error wchar.h should provide WCHAR_MAX
#endif

namespace
{
	template<unsigned N>
	struct BinFromHex
	{
		enum { value = (N%8) + (BinFromHex<N/8>::value << 1) };
	};

	template<>
	struct BinFromHex<0>
	{
		enum { value = 0 };
	};

	#define NIBBLE( hi, lo ) BinFromHex<0##hi##lo>::value
}

// used to replace an incoming character whose value is unknown or unrepresentable in Unicode
static const hkUtf8::CodePoint UNICODE_REPLACEMENT_CHARACTER = 0xfffd;

inline bool s_isUtf8ContinuationByte( int c )
{
	// continuation bytes look like  binary(10......)
	return (c & NIBBLE(1100,0000)) == NIBBLE(1000,0000);
}

inline bool s_isUtf8LeadByte( int c )
{
	// lead bytes look like  binary(11......)
	return (c & NIBBLE(1100,0000)) == NIBBLE(1100,0000);
}

inline bool s_isPlainAscii( int c )
{
	// ascii bytes look like  binary(0.......)
	return (c & NIBBLE(1000,0000)) == 0x0;
}


int hkUtf8::strLen(const char* s)
{
	int count = 0;
	for( int i = 0; s[i]!=HK_NULL; ++i )
	{
		if( s_isUtf8ContinuationByte(s[i]) == false )
		{
			count += 1; // i.e. ascii byte or lead byte
		}
	}
	return count;
}

int hkUtf8::utf8FromWide( char buf[4], wchar_t cp )
{
	if (cp <= 0x007f)
	{
		buf[0] = char(cp);
		return 1;
	}
	else if( cp <= 0x07ff )
	{
		buf[0] = char(0xc0 | (cp>>6));
		buf[1] = char(0x80 | (cp & 0x3f));
		return 2;
	}
#if WCHAR_MAX <= 0xffff // two byte wchar_t
	else
	{
		buf[0] = char(0xe0 | (cp>>12));
		buf[1] = char(0x80 | ((cp>>6) & 0x3f));
		buf[2] = char(0x80 | (cp & 0x3f));
		return 3;
	}
#else // four byte wchar_t
	else if( cp <= 0xffff )
	{
		buf[0] = char(0xe0 | (cp>>12));
		buf[1] = char(0x80 | ((cp>>6) & 0x3f));
		buf[2] = char(0x80 | (cp & 0x3f));
		return 3;
	}
	else if( cp <= 0x001fffff )
	{
		buf[0] = char(0xf0 | (cp>>18));
		buf[1] = char(0x80 | ((cp>>12) & 0x3f));
		buf[2] = char(0x80 | ((cp>>6) & 0x3f));
		buf[3] = char(0x80 | (cp & 0x3f));
		return 4;
	}
	else
	{
		HK_ASSERT(0x79ab8123,0);
		return 0;
	}
 #endif
}

int hkUtf8::utf8FromWide( char* dst, int dstSize, const wchar_t* src, int srcCount)
{
	if( dstSize )
	{
		dst[0] = 0; // ensure we are null terminated even if no more chars are written
	}

	int dstIdx = 0;
	// loop until we hit null or srcCount. unsigned trick handles srcCount==-1
	for( unsigned srcIdx = 0; src[srcIdx] != 0 && srcIdx < unsigned(srcCount); ++srcIdx )
	{
		char buf[4];
		int nc = utf8FromWide(buf, src[srcIdx]);
		// if we have enough room for the codepoint
		if( dstIdx + nc < dstSize)
		{
			for( int i = 0; i < nc; ++i )
			{
				dst[dstIdx+i] = buf[i];
			}
			dst[dstIdx+nc] = 0; // null termination invariant
		}
		// bump the count in any case for the return value
		dstIdx += nc;
	}
	return dstIdx + 1; // plus null
}

int hkUtf8::wideFromUtf8( wchar_t* dst, int dstCap, const char* src, int srcCount)
{
	int dstIdx = 0;
	for( Iterator it(src); it.advance(); )
	{
		CodePoint cp = it.current();
		HK_ASSERT(0x210b019b, cp <= wchar_t(-1) );
		if( dstIdx < dstCap )
		{
			dst[dstIdx] = wchar_t(cp);
		}
		dstIdx += 1;
	}
	// ensure null termination
	if( dstIdx < dstCap )
	{
		dst[dstIdx] = 0;
	}
	else if( dstCap != 0 )
	{
		dst[dstCap-1] = 0;
	}
	
	return dstIdx + 1; // plus null
}

namespace hkUtf8
{
	template<unsigned N>
	static bool decodeLead(hkUint8 b, CodePoint& cp )
	{
		HK_COMPILE_TIME_ASSERT(N>=2);
		enum {
			mask = hkUint8(int(0x80000000) >> (24+N)),
			bits = hkUint8(int(0x80000000) >> (23+N))
		};
			
		if( (b & mask) == bits )
		{
			cp = b & ~mask;
			return true;
		}
		return false;
	}
}

bool hkUtf8::Iterator::advance(int* lenOut)
{
	if( m_utf8 == HK_NULL || m_utf8[0] == 0 )
	{
		m_current = CodePoint(-1);
		return false;
	}
	// default to "bad char", set the values if no decoding error
	CodePoint cp = UNICODE_REPLACEMENT_CHARACTER;
	int len = 1;
	if( s_isPlainAscii(m_utf8[0]) )
    {
		cp = m_utf8[0];
	}
	else if( s_isUtf8LeadByte(m_utf8[0]) )
	{
		// find out how many more bytes we expect and extract the initial bits
		if( decodeLead<2>(m_utf8[0], cp) )
		{
			len = 2;
		}
		else if( decodeLead<3>(m_utf8[0], cp) )
		{
			len = 3;
		}
		else if( decodeLead<4>(m_utf8[0], cp) )
		{
			len = 4;
		}
		else if( decodeLead<5>(m_utf8[0], cp) )
		{
			len = 5;
		}
		else if( decodeLead<6>(m_utf8[0], cp) )
		{
			len = 6;
		}

		// now extract the rest of the continuation
		for( int i = 1; i < len; ++i )
		{
			if( s_isUtf8ContinuationByte(m_utf8[i]) )
			{
			    cp <<= 6;
			    cp |= m_utf8[i] & 0x3f;
			}
			else // eh? it went bad, bail out and return a bad char.
			{
				cp = UNICODE_REPLACEMENT_CHARACTER;
				len = i;
				break;
			}
		}
	}
	
	if( lenOut )
	{
		*lenOut = len;
	}
	m_current = cp;
	m_utf8 += len;
	return true;
}


hkUtf8::Utf8FromWide::Utf8FromWide(const wchar_t* s)
{
	if( s )
	{
		// calculate required size
		int nbytes = 0;
		{
			char buf[4];
			for( int i = 0; s[i] != 0; ++i )
			{
				nbytes += utf8FromWide(buf, s[i]);
			}
		}
		// fill values
		m_utf8.setSize( nbytes + 1 );
		char* p = m_utf8.begin();
		for( int i = 0; s[i] != 0; ++i )
		{
			p += utf8FromWide(p, s[i]);
		}
		*p = 0;
	}
}

hkUtf8::WideFromUtf8::WideFromUtf8(const char* s)
{
	if( s )
	{
		m_wide.reserve( hkUtf8::strLen(s) + 1 );
		for( Iterator it(s); it.advance(); )
		{
			CodePoint cp = it.current();
			HK_ASSERT(0x210b019b, cp <= wchar_t(-1) );
			m_wide.pushBack( wchar_t(cp) ); // if encoding error, we may not have reserved enough space
		}
		// null terminate
		m_wide.pushBack(0);
		m_wide.popBack();
	}
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
