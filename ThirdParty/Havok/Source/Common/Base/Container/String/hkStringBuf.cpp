/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Fwd/hkcstdarg.h>

static inline char tolower(char i)
{
	return i >= 'A' && i <= 'Z'
		? i + ('a'-'A')
		: i;
}

static inline char toupper(char i)
{
	return i >= 'a' && i <= 'z'
		? i - ('a'-'A')
		: i;
}

#ifdef HK_DEBUG
static hkBool32 aliasSafe(const char* p, const char* a, int na)
{
	hkUlong up = hkUlong(p);
	hkUlong start = hkUlong(a);
	hkUlong end = hkUlong(a+na);
	return up < start || up >= end;
}
#endif

hkStringBuf::hkStringBuf(const char* s)
{
	if( s != HK_NULL )
	{
		int len = hkString::strLen(s);
		setLength( len );
		hkString::memCpy(m_string.begin(), s, len );
	}
	else
	{
		setLength(0);
	}
}

hkStringBuf::hkStringBuf(const hkStringPtr& s)
{
	operator=(s.cString());
}

hkStringBuf::hkStringBuf(const char* s0, const char* s1, const char* s2, const char* s3, const char* s4, const char* s5)
{
	m_string.pushBackUnchecked(0);
	appendJoin(s0,s1,s2,s3,s4,s5);
}

hkStringBuf::hkStringBuf(const char* b, int len)
{
	setLength(len);
	hkString::memCpy( m_string.begin(), b, len );
}

hkStringBuf::hkStringBuf(const hkStringBuf& s)
{
	m_string = s.m_string;
}

hkStringBuf& hkStringBuf::operator=(const hkStringBuf& s)
{
	m_string = s.m_string;
	return *this;
}

hkStringBuf& hkStringBuf::operator=(const char* s)
{
	if( s )
	{
		int sl = hkString::strLen(s);
		setLength(sl);
		hkString::memCpy(m_string.begin(), s, sl);
	}
	else
	{
		setLength(0);
	}
	return *this;
}

int hkStringBuf::indexOf(char c, int startIndex, int endIndex) const
{
	endIndex = hkMath::min2( endIndex, getLength() );
	for( int i = startIndex; i < endIndex; ++i )
	{
		if( m_string[i] == c )
		{
			return i;
		}
	}
	return -1;
}

int hkStringBuf::indexOf( const char* s, int startIndex, int endIndex ) const
{
	HK_ASSERT(0x1fbc29bf, endIndex == HK_INT32_MAX);
	if( const char* found = hkString::strStr(m_string.begin()+startIndex, s) )
	{
		return int(hkUlong( found - m_string.begin() ));
	}
	return -1;
}

int hkStringBuf::indexOfCase( const char* needle ) const
{
	// naive n^2 implementation
	const char* haystack = m_string.begin();
	for( int hi = 0; haystack[hi]; ++hi )
	{
		int ni;
		for( ni = 0; needle[ni]; ++ni )
		{
			if( tolower(haystack[hi+ni]) != tolower(needle[ni]) )
			{
				break;
			}
		}
		if( needle[ni] == 0 )
		{
			return hi;
		}
	}
// 	hkStringBuf haystack = *this; haystack.lowerCase();
// 	hkStringBuf needle = s; needle.lowerCase();
// 	if( const char* found = hkString::strStr( haystack.cString(), needle.cString() ) )
// 	{
// 		return int(hkUlong(found - haystack.cString()));
// 	}
	return -1;
}

int hkStringBuf::lastIndexOf (char c, int start, int end) const
{
	if( end > getLength() )
	{
		end = getLength();
	}
	for(int i = end - 1; i >= start ; --i)
	{
		if( m_string[i] == c )
		{
			return i;
		}
	}
	return -1;
}
int hkStringBuf::lastIndexOf(const char* needle, int startIndex, int endIndex) const
{
	
	HK_ASSERT(0x1fbc29bf, endIndex == HK_INT32_MAX);
	int ret = -1;
	const char* haystack = m_string.begin();
	while( const char* found = hkString::strStr(haystack, needle) )
	{
		ret = int(hkUlong( found - m_string.begin() ));
		haystack = found+1;
	}
	return ret;
}

int hkStringBuf::compareTo(const char* other) const
{
	return hkString::strCmp( cString(), other );
}

int hkStringBuf::compareToIgnoreCase(const char* other) const
{
	return hkString::strCasecmp( cString(), other );
}

hkBool hkStringBuf::operator< (const char* other) const
{
	return hkString::strCmp( cString(), other ) < 0;
}

hkBool32 hkStringBuf::operator==(const char* other) const
{
	return compareTo(other) == 0;
}

hkBool32 hkStringBuf::startsWith(const char* s) const
{
	int i;
	for( i = 0; m_string[i] && s[i]; ++i )
	{
		if( m_string[i] != s[i] )
		{
			return false;
		}
	}
	return s[i] == HK_NULL;
}

hkBool32 hkStringBuf::startsWithCase(const char* s) const
{
	for( int i = 0; m_string[i] && s[i]; ++i )
	{
		if( tolower(m_string[i]) != tolower(s[i]) )
		{
			return false;
		}
	}
	return true;
}

hkBool32 hkStringBuf::endsWith(const char* s) const
{
	int sl = hkString::strLen(s);
	if( sl > getLength() )
	{
		return false;
	}
	int start = getLength() - sl;

	for( int i = 0; i < sl; ++i )
	{
		if( m_string[start+i] != s[i] )
		{
			return false;
		}
	}
	return true;
}

hkBool32 hkStringBuf::endsWithCase(const char* s) const
{
	int sl = hkString::strLen(s);
	if( sl > getLength() )
	{
		return false;
	}
	int start = getLength() - sl;

	for( int i = 0; i < sl; ++i )
	{
		if( tolower(m_string[start+i]) != tolower(s[i]) )
		{
			return false;
		}
	}
	return true;
}

int hkStringBuf::split( int sep, hkArray<const char*>::Temp& bits )
{
	int cur = 0;
	bits.pushBack( m_string.begin() );
	while(1)
	{
		int c = indexOf(char(sep), cur);
		if( c >= 0 )
		{
			m_string[c] = '\0';
			cur = c + 1;
			bits.pushBack( m_string.begin()+cur );
		}
		else
		{
			break;
		}
	}
	return bits.getSize();
}

void hkStringBuf::clear()
{
	setLength(0);
}

void hkStringBuf::printf(const char* fmt, ...)
{
	va_list args; 
	va_start(args, fmt);
	while(1)
	{
		int size = m_string.getCapacity();
		int nchars = hkString::vsnprintf(m_string.begin(), size, fmt, args);

		if( nchars >= 0 && nchars < size ) 
		{
			// usual case, it worked. update length		
			setLength( nchars );
			break;
		}
		else if( nchars < 0 )
		{
			// there was not enough room, double capacity
			setLength( size*2 > 255 ? size*2 : 255 ); 
		}
		else
		{
			// there was not enough room and we were told how much
			// was needed (not including \0)
			setLength( nchars );
		}
	}
	va_end(args);
}

void hkStringBuf::appendPrintf(const char* fmt, ...)
{
	hkStringBuf s;
	va_list args; 
	va_start(args, fmt);
	while(1)
	{
		int size = s.m_string.getCapacity();
		int nchars = hkString::vsnprintf(s.m_string.begin(), size, fmt, args);

		if( nchars >= 0 && nchars < size ) 
		{
			// usual case, it worked. update length		
			s.setLength( nchars );
			break;
		}
		else if( nchars < 0 )
		{
			// there was not enough room, double capacity
			s.setLength( size*2 > 255 ? size*2 : 255 ); 
		}
		else
		{
			// there was not enough room and we were told how much
			// was needed (not including \0)
			s.setLength( nchars );
		}
	}
	va_end(args);
	append(s);
}

hkStringBuf& hkStringBuf::operator+= (const char* other)
{
	if ( other != HK_NULL )
	{
		int otherLen = hkString::strLen(other);
		m_string.insertAt( m_string.getSize()-1, other, otherLen);
	}
	return *this;
}

hkStringBuf& hkStringBuf::appendJoin(const char* s0, const char* s1, const char* s2, const char* s3, const char* s4, const char* s5)
{
	int len[6] = {};
	const char* ptr[6+1] = { s0, s1, s2, s3, s4, s5, HK_NULL }; // last one is sentinel
	int origLen = getLength();
	int totalLen = origLen;
	for( int i = 0; ptr[i] != HK_NULL; ++i )
	{
		len[i] = hkString::strLen(ptr[i]);
		totalLen += len[i];
	}
	setLength( totalLen );
	int dst = origLen;
	for( int i = 0; ptr[i] != HK_NULL; ++i )
	{
		hkString::memCpy( m_string.begin()+dst, ptr[i], len[i] );
		dst += len[i];
	}
	return *this;
}

hkStringBuf& hkStringBuf::setJoin(const char* s0, const char* s1, const char* s2, const char* s3, const char* s4, const char* s5)
{
	clear();
	appendJoin(s0,s1,s2,s3,s4,s5);
	return *this;
}

void hkStringBuf::chompStart(int n)
{
	n = hkMath::min2(n, getLength());
	if( n > 0 )
	{
		m_string.removeAtAndCopy(0, n);
	}
}

void hkStringBuf::chompEnd(int n)
{
	if( n > 0 )
	{
		setLength( hkMath::max2(0, getLength()-n) );
	}
}

void hkStringBuf::slice(int startOffset, int length)
{
	HK_ASSERT(0x79e7a47c, startOffset+length <= getLength());
	HK_ASSERT(0x79e7a47d, length >= 0);
	if( startOffset != 0 )
	{
		hkMemUtil::memMove( m_string.begin(), m_string.begin()+startOffset, length );
	}
	setLength(length);
}

void hkStringBuf::set(const char* s, int len)
{
	HK_ASSERT(0x75a34527, aliasSafe(s,m_string.begin(), m_string.getSize()));
	if( len < 0 )
	{
		len = hkString::strLen(s);
	}
	setLength(len);
	hkMemUtil::memCpy( m_string.begin(), s, len );
}

void hkStringBuf::append(const char* s, int len)
{
	if ( s != HK_NULL )
	{
		HK_ASSERT(0x75a34527, aliasSafe(s,m_string.begin(), m_string.getSize()));
		if( len < 0 )
		{
			len = hkString::strLen(s);
		}

		int oldLen = getLength();
		setLength( oldLen + len);
		hkMemUtil::memCpy( m_string.begin()+oldLen, s, len );
	}
}

void hkStringBuf::prepend(const char* s, int len)
{
	insert(0, s, len);
}

void hkStringBuf::insert(int pos, const char* s, int len)
{
	if ( s != HK_NULL )
	{
		if(len<0)
		{
			len = hkString::strLen(s);
		}
		m_string.insertAt(pos, s, len);
	}
}

void hkStringBuf::pathBasename()
{
	int lastSlash = hkMath::max2( lastIndexOf('\\'), lastIndexOf('/') );
	if( lastSlash >= 0 )
	{
		chompStart(lastSlash+1);
	}
}

void hkStringBuf::pathDirname()
{
	int lastSlash = hkMath::max2( lastIndexOf('\\'), lastIndexOf('/') );
	if( lastSlash >= 0 )
	{
		slice(0, lastSlash);
	}
	else
	{
		clear(); 
	}
}

void hkStringBuf::pathNormalize()
{	
	// Path with only '/'
	hkStringBuf path(*this);
	path.replace( '\\', '/', REPLACE_ALL );
	const char* startOfResult = path.startsWith("//") ? "//"
		: path.startsWith("/") ? "/"
		: "";
	hkArray<const char*>::Temp oldBits;
	path.split('/', oldBits);

	hkArray<const char*>::Temp newBits;

	int numNamedParts = 0; 
	for( int i = 0; i < oldBits.getSize(); ++i )
	{
		if( hkString::strCmp("..", oldBits[i]) == 0 )
		{
			if( numNamedParts )
			{
				newBits.popBack();
				numNamedParts -= 1;
			}
			else
			{
				newBits.pushBack(oldBits[i]);
			}
		}
		else if( hkString::strCmp(".", oldBits[i]) == 0 )
		{
			// ignore "." components
		}
		else
		{
			numNamedParts += 1;
			newBits.pushBack(oldBits[i]);
		}
	}

	*this = startOfResult;
	for( int i = 0; i < newBits.getSize(); ++i )
	{
		this->pathAppend(newBits[i]);
	}
}

void hkStringBuf::pathExtension()
{
	int lastDot = lastIndexOf('.');
	if( lastDot != -1 )
	{
		chompStart(lastDot);
	}
	else
	{
		clear();
	}
}

hkStringBuf& hkStringBuf::pathAppend(const char* p0, const char* p1, const char* p2)
{
	const char* parts[] = { p0, p1, p2, HK_NULL }; // last one is a sentinel
	m_string.popBack();
	// Don't add a leading / if we're empty
	bool needSlash = m_string.getSize() && m_string.back() != '/';
	for( int i = 0; parts[i] != HK_NULL; ++i )
	{
		const char* p = parts[i];
		if( p[0] ) // skip empty strings
		{
			while( p[0] == '/' )
			{
				p += 1;
			}
			
			if( int len = hkString::strLen(p) )
			{
				while( len && p[len-1] == '/' )
				{
					len -= 1;
				}
				if( len && needSlash )
				{
					m_string.pushBack('/');
				}
				m_string.append(p, len);
			}
			needSlash = true;
		}
	}
	m_string.pushBack(0);
	return *this;
}

hkBool32 hkStringBuf::replace(char from, char to, ReplaceType rt)
{
	hkBool32 replaced = false;
	for( int i = 0; i < getLength(); ++i )
	{
		if( m_string[i] == from )
		{
			m_string[i] = to;
			replaced = true;
			if( rt == REPLACE_ONE )
			{
				break;
			}
		}
	}
	return replaced;
}

hkBool32 hkStringBuf::replace(const char* from, const char* to, ReplaceType rtype)
{
	int fromLen = hkString::strLen(from);
	int toLen = hkString::strLen(to);
	hkBool32 replaced = false;
	
	if( toLen > fromLen )
	{
		hkStringBuf tmpStr = *this;
		clear();
		int tmpIdx = 0;
		for( ; ; )
		{
			int n = tmpStr.indexOf(from, tmpIdx);
			if( n >= 0 )
			{
				replaced = true;
				append( tmpStr.cString()+tmpIdx, n-tmpIdx );
				append( to, toLen );
				tmpIdx = n+fromLen;
				if( rtype == REPLACE_ONE )
				{
					break;
				}
			}
			else
			{
				break;
			}
		}

		append( tmpStr.cString()+tmpIdx, tmpStr.getLength()-tmpIdx );
	}
	else // replacement same or shorter, don't allocate temp buffer
	{
		char* self = m_string.begin();
		int readFrom=0, searchFrom=0, writeTo=0, tgtIndex;
		while(((tgtIndex = indexOf(from, searchFrom)) != -1))
		{
			while(readFrom < tgtIndex)
			{
				self[writeTo++] = self [readFrom++];
			}
			for(int a=0; a<toLen; a++)
			{
				self [writeTo++] = to [a];
			}
			readFrom += fromLen;
			searchFrom = tgtIndex + fromLen;
			if( rtype == REPLACE_ONE)
				break;
		}
		while(readFrom < getLength())
		{
			self [writeTo++] = self [readFrom++];
		}
		self [writeTo] = '\0';
		setLength(writeTo);
	}
	return replaced;
}

void hkStringBuf::lowerCase()
{
	for( int i = 0; i < getLength(); ++i )
	{
		m_string[i] = tolower(m_string[i]);
	}
}

void hkStringBuf::upperCase()
{
	for( int i = 0; i < getLength(); ++i )
	{
		m_string[i] = toupper(m_string[i]);
	}
}

hkArray<char>::Temp& hkStringBuf::getArray()
{
	return m_string;
}


#if 0
struct BoyerMooreHorspool
{
	BoyerMooreHorspool( const char* needle, int needleLen )
		: m_needle(needle), m_needleLen(needleLen >= 0 ? needleLen : hkString::strLen(needle) )
	{
		for( int i = 0; i < HK_COUNT_OF(m_badCharShift); ++i )
		{
			m_badCharShift[i] = needleLen;
		}
		hkString::memSet(m_badCharShift, , sizeof(m_badCharShift));
		for( int i = 0; i < m_needleLen; ++i )
		{
			m_badCharShift[ m_needle[i] ]
		}
	}
	int m_badCharShift[256];
	const char* m_needle;
	int m_needleLen;
};
#endif

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
