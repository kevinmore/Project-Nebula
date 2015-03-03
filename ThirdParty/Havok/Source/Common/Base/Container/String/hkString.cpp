/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/Base/hkBase.h>
#include <Common/Base/Fwd/hkcctype.h>
#include <Common/Base/Fwd/hkcstdarg.h>
#include <Common/Base/Fwd/hkcstdio.h>
#include <Common/Base/Fwd/hkcstdlib.h>
#include <Common/Base/Fwd/hkcstring.h>

using namespace std;

// very small strings will likely cause reallocations
//static const int MINIMUM_STRING_CAPACITY = 64 - 1;

// Some platforms don't have vsnprintf, so we use vsprintf instead.
// vsprintf is not safe from overflows, so we include asserts.

// Win32 renames some standard string manipulation functions.
#if defined(HK_PLATFORM_WIN32) || defined(HK_PLATFORM_XBOX) || defined(HK_PLATFORM_XBOX360)
#	define VSNPRINTF(BUF,LEN,FMT,ARGS) ::_vsnprintf(BUF,LEN,FMT,ARGS)
#	define STRTOLL(STR,ENDPTR,BASE) ::_strtoi64(STR,ENDPTR,BASE)
#	define STRTOULL(STR,ENDPTR,BASE) ::_strtoui64(STR,ENDPTR,BASE)
#else
#	define VSNPRINTF(BUF,LEN,FMT,ARGS) ::vsnprintf(BUF,LEN,FMT,ARGS)
#	define STRTOLL(STR,ENDPTR,BASE) ::strtoll(STR,ENDPTR,BASE)
#	define STRTOULL(STR,ENDPTR,BASE) ::strtoull(STR,ENDPTR,BASE)
#endif

// all the static methods go first so that they may be inlined
// in the nonstatic ones further on.

char HK_CALL hkString::toUpper( char c )
{
	return (c>='a'&&c<='z')
			? (char)(c - ('a' - 'A'))
			: c;
}
char HK_CALL hkString::toLower( char c )
{
	return (c>='A'&&c<='Z')
			? (char)(c + ('a' - 'A'))
			: c;
}


int HK_CALL hkString::vsnprintf( char* buf, int len, const char* fmt, hk_va_list hkargs)
{
#if !defined(HK_PLATFORM_ANDROID) && !defined(HK_COMPILER_ARMCC) && !defined(HK_PLATFORM_TIZEN)
	
	va_list args;	
#if !defined(HK_PLATFORM_PS3_SPU) && \
  ( !defined(HK_ARCH_PPC) || defined(HK_PLATFORM_XBOX) || defined(HK_PLATFORM_XBOX360) || defined(HK_PLATFORM_MACPPC) || defined(HK_PLATFORM_MAC386) )  \
  && !(defined(HK_ARCH_X64) && defined(HK_PLATFORM_LINUX)) \
  && !defined(HK_PLATFORM_NACL) \
  && !defined(HK_PLATFORM_LRB_XENSIM) && !defined(HK_PLATFORM_LRB) \
  && !defined(HK_PLATFORM_PS4)
	args = reinterpret_cast<va_list>(hkargs);
#else
	// to get arround void* --> array[1] reverse conversion we have to bypass the compiler 
	// in mwerks anyway
	hkString::memCpy(args,hkargs,sizeof(va_list));
#endif
	int ret = VSNPRINTF(buf, len, fmt, args);
#else
	int ret = VSNPRINTF(buf, len, fmt, hkargs); // CTR and Android just use hk_va_list == va_list
#endif 

	return ret;
}

int HK_CALL hkString::snprintf( char* buf, int len, const char* fmt, ...)
{
	va_list vlist;
	va_start(vlist, fmt);
	int ret = VSNPRINTF(buf, len, fmt, vlist);
	va_end(vlist);
	return ret;
}

int HK_CALL hkString::sprintf( char* buf, const char* fmt, ...)
{
	va_list vlist;
	va_start(vlist, fmt);
	int ret = vsprintf(buf, fmt, vlist);
	va_end(vlist);
	return ret;
}

int HK_CALL hkString::strCmp( const char* a, const char* b )
{
	HK_ASSERT(0x2540a587, a != HK_NULL);
	HK_ASSERT(0x571b5b6a, b != HK_NULL);
	return strcmp(a,b); 
}

int HK_CALL hkString::strNcmp( const char* a, const char* b, int n )
{
	HK_ASSERT(0x37ef1ca6, a != HK_NULL);
	HK_ASSERT(0x68ef40e3, b != HK_NULL);
	return strncmp(a,b,(unsigned)n);
}

int HK_CALL hkString::strCasecmp( const char* a, const char* b )
{
	HK_ASSERT(0x5cf1cfe9, a != HK_NULL);
	HK_ASSERT(0x49243d55, b != HK_NULL);
	int i = 0;
	while ( a[i] != 0 || b[i] != 0 )
	{
		if( toLower(a[i]) < toLower(b[i]) )
		{
			return -1;
		}
		else if( toLower(a[i]) > toLower(b[i]) )
		{
			return 1;
		}
		++i;
	}
	
	return 0;
}

int	HK_CALL hkString::strNcasecmp(const char* a, const char* b, int n)
{
	HK_ASSERT(0x7a31572d, a != HK_NULL);
	HK_ASSERT(0x54725cb9, b != HK_NULL);
	HK_ASSERT(0x7403b0c0, n >= 0);

	int i = 0;
	while ( (a[i] != 0 || b[i] != 0) && i < n )
	{
		if( toLower(a[i]) < toLower(b[i]) )
		{
			return -1;
		}
		else if( toLower(a[i]) > toLower(b[i]) )
		{
			return 1;
		}
		++i;
	}
	
	return 0;
}

void HK_CALL hkString::strCpy( char* dst, const char* src )
{
	HK_ASSERT(0x49fa77f1, src != HK_NULL);
	HK_ASSERT(0x7d3f002c, dst != HK_NULL);
	strcpy(dst, src);
}

void HK_CALL hkString::strNcpy(char *dst, const char *src, int n)
{
	HK_ASSERT(0x6e527462, src != HK_NULL || n == 0);
	HK_ASSERT(0x67d37b1f, dst != HK_NULL || n == 0);
	if( n )
	{
		strncpy(dst, src, (unsigned)n);
	}
}

int HK_CALL hkString::strLen( const char* src )
{
	HK_ASSERT(0x18be8938, src != HK_NULL);
	return (int) strlen(src);
}

void HK_CALL hkString::strCat(char* dst, const char* src)
{
	HK_ASSERT(0x2de33d38, src != HK_NULL);
	HK_ASSERT(0x1ef1a18e, dst != HK_NULL);
	strcat(dst, src);
}

void HK_CALL hkString::strNcat(char* dst, const char* src, int n)
{
	HK_ASSERT(0x3f082f11, src != HK_NULL || n == 0);
	HK_ASSERT(0x106ee28e, dst != HK_NULL || n == 0);
	if( n )
	{
		strncat(dst, src, (unsigned)n);
	}
}

int HK_CALL hkString::atoi( const char* in, int base)
{
	return strtoul(in, HK_NULL, base);
}

hkInt64 HK_CALL hkString::atoll( const char* in, int base)
{
	return STRTOLL(in, HK_NULL, base);
}

hkUint64 HK_CALL hkString::atoull( const char* in, int base)
{
	return STRTOULL(in, HK_NULL, base);
}

hkReal HK_CALL hkString::atof( const char* in )
{
	return hkReal( strtod( in, HK_NULL) );
}

const char* HK_CALL hkString::strStr(const char* haystack, const char* needle)
{
	return strstr(haystack, needle);
}

const char* HK_CALL hkString::strChr(const char* haystack, int needle)
{
	return strchr(haystack, needle);
}

const char* HK_CALL hkString::strRchr(const char* haystack, int needle)
{
	return strrchr(haystack, needle);
}

#ifndef HK_PLATFORM_SPU
char* HK_CALL hkString::strDup(const char* src, hkMemoryAllocator& alloc)
{
	HK_ASSERT(0x13e1d159, src != HK_NULL);
	char* r = (char*)hkMemoryRouter::easyAlloc(alloc, strLen(src)+1 );
	hkString::strCpy(r, src);
	return r;
}

void HK_CALL hkString::strFree(char* s, hkMemoryAllocator& alloc)
{
	hkMemoryRouter::easyFree(alloc, s);
}
void HK_CALL hkString::strFree(char* s)
{
	hkString::strFree(s, hkMemoryRouter::getInstance().heap());
}

char* HK_CALL hkString::strNdup(const char* src, int maxlen, hkMemoryAllocator& alloc)
{
	HK_ASSERT(0x2e27506f, src != HK_NULL);
	int len = strLen(src);
	if( len > maxlen )
	{
		len = maxlen;
	}
	char* r = (char*)hkMemoryRouter::easyAlloc(alloc, len+1);
	hkString::strNcpy(r, src, len);
	r[len] = 0;
	return r;
}

char* HK_CALL hkString::strDup(const char* src)
{
	return strDup(src, hkMemoryRouter::getInstance().heap() );
}

char* HK_CALL hkString::strNdup(const char* src, int maxlen)
{
	return strNdup(src, maxlen, hkMemoryRouter::getInstance().heap() );
}

#endif

char* HK_CALL hkString::strLwr(char* s)
{
	HK_ASSERT(0x3779fe34, s != HK_NULL);
	int i=0;
	while(s[i])
	{
		s[i] = (char)(toLower(s[i]));
		i++;
	}
	return s;
}

char* HK_CALL hkString::strUpr(char* s)
{
	HK_ASSERT(0x128f807f, s != HK_NULL);
	int i=0;
	while(s[i])
	{
		s[i] = (char)(toUpper(s[i]));
		i++;
	}
	return s;
}

#if defined(HK_PLATFORM_WIIU)


#if 0 && (CAFE_OS_SDK_VERSION >= 20900)
#define CHECK_OSBLOCKMOVE(dst, src, nbytes, flush) OSBlockMove(dst, src, nbytes, flush)
#else
#define CHECK_OSBLOCKMOVE(dst, src, nbytes, flush) if(dst != src) { OSBlockMove(dst, src, nbytes, flush); } else
#endif
void HK_CALL hkString::memCpy( void* dst, const void* src, int n)
{
	CHECK_OSBLOCKMOVE(dst,src,(size_t)n, false);
}

void HK_CALL hkString::memMove(void* dst, const void* src, int n)
{
	CHECK_OSBLOCKMOVE(dst,src,(size_t)n, false);
}

void HK_CALL hkString::memSet(void* dst, const int c, int n)
{
	OSBlockSet(dst,c,(size_t)n);
}
#else
void HK_CALL hkString::memCpy( void* dst, const void* src, int n)
{
	memcpy(dst,src,(unsigned)n);
}

void HK_CALL hkString::memMove(void* dst, const void* src, int n)
{
	memmove(dst,src,(unsigned)n);
}

void HK_CALL hkString::memSet(void* dst, const int c, int n)
{
	memset(dst,c,(unsigned)n);
}
#endif

int HK_CALL hkString::memCmp( const void *buf1, const void *buf2, int n)
{
	return memcmp(buf1,buf2,(unsigned)n);
}

hkBool HK_CALL hkString::findAllOccurrences(const char* haystack, const char* needle, hkArray<int>& indices, hkString::ReplaceType rtype)
{
	size_t needleLen = strlen( needle );
	// find the first occurrence
	const char* p = strStr(haystack, needle);
	hkBool found = false;
	while( p )
	{
		found = true;
		indices.pushBack( static_cast<int>( p - haystack ) );
		// go to the next one if requested
		p = (rtype == REPLACE_ONE) ? HK_NULL : strStr( p + needleLen, needle );
	}
	return found;
}

hkBool HK_CALL hkString::beginsWith(const char* a, const char* b)
{
	for(int i = 0; b[i] != 0; ++i)
	{
		if( a[i] != b[i] )
		{
			return false;
		}
	}
	return true;
}

hkBool HK_CALL hkString::beginsWithCase(const char* a, const char* b)
{
	for(int i = 0; b[i] != 0; ++i)
	{
		if( tolower(a[i]) != tolower(b[i]) )
		{
			return false;
		}
	}
	return true;
}

hkBool HK_CALL hkString::endsWith(const char* a, const char* b)
{
	int alen = hkString::strLen(a);
	int blen = hkString::strLen(b);
	if( alen < blen )
	{
		return false;
	}
	int offset = alen - blen;

	for(int i = 0; i < blen; ++i)
	{
		if( a[i+offset] != b[i] )
		{
			return false;
		}
	}
	return true;
}

hkBool HK_CALL hkString::endsWithCase(const char* a, const char* b)
{
	int alen = hkString::strLen(a);
	int blen = hkString::strLen(b);
	if( alen < blen )
	{
		return false;
	}
	int offset = alen - blen;

	for(int i = 0; i < blen; ++i)
	{
		if( tolower(a[i+offset]) != tolower(b[i]) )
		{
			return false;
		}
	}
	return true;
}

int HK_CALL hkString::lastIndexOf(const char* str, char c)
{
	if( const char* p = strRchr(str, c) )
	{
		return int(hkUlong(p - str));
	}
	return -1;
}

int HK_CALL hkString::indexOf(const char* str, char c, int startIndex, int endIndex)
{
	HK_ASSERT(0x44e02638, startIndex <= endIndex);
	for( int i = 0; i < startIndex; ++i )
	{
		if( str[i] == 0 )
		{
			return -1;
		}
	}
	for( int i = startIndex; i < endIndex && str[i]; ++i )
	{
		if( str[i] == c )
		{
			return i;
		}
	}
	return -1;
}

void HK_CALL hkString::memClear128(void* dst, int numBytes)
{
	HK_ASSERT2( 0xf0dcf45e, (hkUlong(dst) & 0x7f) == 0, "Your input array must be 128 byte aligned");
	HK_ASSERT2( 0xf0dcf45e, (numBytes & 0x7f) == 0, "Your size must be 128 byte aligned");
#if defined(HK_PLATFORM_XBOX360)
	for (int i =0; i < numBytes; i+= 128 )
	{
		__dcbz128(0, dst);
		dst = hkAddByteOffset(dst, 128 );
	}
#elif defined(HK_PLATFORM_WIIU)
	DCZeroRange(dst, numBytes);
#else
	hkString::memClear16( dst, numBytes>>4 );
#endif
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
