/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Base/System/Io/Reader/Memory/hkMemoryStreamReader.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>
#include <Common/Base/Container/String/hkStringBuf.h>

#include <Common/Base/Fwd/hkcstdarg.h>
#include <Common/Base/Fwd/hkcstdlib.h>

using namespace std;


// isOk should return true directly after a successful read
// isOk should return false directly after an unsuccessful read

const int TEMP_STORAGE_SIZE = 256;

hkIstream::hkIstream(hkStreamReader* sb)
	:	m_streamReader(sb)
{
}


hkIstream::hkIstream(const char* fname)
{
	m_streamReader = hkFileSystem::getInstance().openReader(fname);
}

hkIstream::hkIstream(const void* mem, int memSize)
{
	m_streamReader.setAndDontIncrementRefCount( new hkMemoryStreamReader(mem, memSize, hkMemoryStreamReader::MEMORY_INPLACE) );
}

hkIstream::hkIstream(const hkMemoryTrack* track )
{
	m_streamReader.setAndDontIncrementRefCount( new hkMemoryTrackStreamReader(track, hkMemoryTrackStreamReader::MEMORY_INPLACE) );
}

hkIstream::~hkIstream()
{
}

hkBool hkIstream::isOk() const
{
	if (m_streamReader)
	{
		return m_streamReader->isOk();
	}
	return false;
}


#define IS_DECIMALPOINT(c) ( ((c) == '.') || ((c) == ',') )
#define IS_INTEGER(c)      ( ((c) >= '0') && ((c) <= '9') )
#define IS_HEX(c)          (  ( ((c) >= 'a') && ((c) <= 'f') ) || ( ((c) >= 'A') && ((c) <= 'F') )  )
#define IS_SIGN(c)         ( ((c) == '+') || ((c) == '-') )
#define IS_SPACE(c)		   ( ((c) == ' ') || ((c) == '\t') )
#define IS_EXP(c)		   ( ((c) == 'E') || ((c) == 'e') )
#define IS_EOL(c)		   ( ( (c) == '\r') || ((c) == '\n') )

// eat white space from reader
static void HK_CALL eatWhiteSpace( hkStreamReader* reader )
{
	char buf[64];
	while(1)
	{
		int n = reader->peek(buf, sizeof(buf));
		if( n == 0 )
		{
			return; //eof
		}
		for( int i = 0; i < n; ++i )
		{
			if( IS_SPACE(buf[i]) == false && IS_EOL(buf[i]) == false ) // real char
			{
				reader->skip(i);
				return;
			}
		}
		reader->skip(n);
	}
}

static hkUint64 HK_CALL readInteger64( hkStreamReader* reader, hkBool& negOut )
{
	eatWhiteSpace(reader);

	negOut = false;
	hkUint64 u64 = 0;

	char storage[TEMP_STORAGE_SIZE];
	int startIndex = 0;

	int nbytes = reader->peek( storage, TEMP_STORAGE_SIZE-1 );

	if( nbytes )
	{
		if( storage[0] == '+')
		{
			startIndex = 1;
		}
		else if( storage[0] == '-')
		{
			negOut = true;
			startIndex = 1;
		}

		unsigned base = 0;
		// base16? need at least 3 chars 0x<digit|hex>.
		if( startIndex+3 < nbytes
			&& storage[startIndex] == '0'
			&& ( storage[startIndex+1] == 'x' || storage[startIndex+1] == 'X')
			&& ( IS_INTEGER(storage[startIndex+2]) || IS_HEX(storage[startIndex+2]) ) )
		{
			base = 16;
			startIndex += 2;
		}
		// base8? need at least two chars 0<digit>
		else if( startIndex+2 < nbytes
			&& storage[startIndex] == '0'
			&& IS_INTEGER( storage[startIndex+1] ) )
		{
			base = 8;
			startIndex += 1;
		}
		// no prefix, use base10
		else
		{
			base = 10;
		}

		int i;
		for( i = startIndex; i < nbytes; ++i )
		{
			unsigned next = unsigned(-1);
			if( IS_INTEGER(storage[i]) )
			{
				next = static_cast<unsigned>(storage[i] - '0');
			}
			else if( storage[i] > 'A' && storage[i] <= 'F' )
			{
				next = static_cast<unsigned>(storage[i] + 10 - 'A');
			}
			else if( storage[i] > 'a' && storage[i] <= 'f' )
			{
				next = static_cast<unsigned>(storage[i] + 10 - 'a');
			}

			if( next < base )
			{
				u64 *= base;
				u64 += next;
			}
			else
			{
				break;
			}
		}
		reader->skip(i);

		return u64;
	}
	else
	{
		reader->skip(1); // peek failed, set eof.
	}

	return 0;
}


static float HK_CALL readFloat(hkStreamReader* reader)
{
	eatWhiteSpace(reader);
	
	char storage[TEMP_STORAGE_SIZE];
	int validChars = 0;

	int nbytes = reader->peek( storage, TEMP_STORAGE_SIZE-1 );
	if( (nbytes != 0) && (IS_INTEGER(storage[0]) || IS_SIGN(storage[0])
						|| IS_DECIMALPOINT(storage[0]) ) ) // optional +-.
	{
		++validChars;

		while( validChars < nbytes )
		{
			char c = storage[validChars];
			if( IS_INTEGER(c) || IS_SIGN(c) || IS_EXP(c) || IS_DECIMALPOINT(c) )
			{
				if (IS_DECIMALPOINT(c))
				{
					// HVK-1548
					storage[validChars] = '.';
				}
				++validChars; // most likely
				continue;
			}
			break; // else not a valid float character
		}
		reader->skip(validChars);
	}
	else
	{
		reader->skip(1);
	}

	storage[validChars] = '\0';

	if( validChars > 0 )
	{
		return (float)strtod( storage, HK_NULL );
	}
	return -1;
}

hkIstream& hkIstream::operator>> (hkBool& b)
{
	eatWhiteSpace(m_streamReader);
	char buf[6];
	int nread = m_streamReader->peek(buf, 6);
	// "false":5 "true":4
	// Be careful about "true<EOF>" - if we read too far, isOk
	// will return false even though we have a good read.
	if( nread >= 4 && (hkString::strNcmp(buf, "true", 4) == 0) )
	{
		if( nread == 4 || (IS_SPACE(buf[4]) || (IS_EOL(buf[4]) ) ) )
		{
			b = true;
			m_streamReader->skip( 4 );
			return *this;
		}
	}
	if( nread >= 5 && (hkString::strNcmp(buf, "false", 4) == 0) )
	{
		if( nread == 5 || (IS_SPACE(buf[5]) || (IS_EOL(buf[5]) ) ) )
		{
			b = false;
			m_streamReader->skip( 5 );
			return *this;
		}
	}
	if( nread == 0 )
	{
		m_streamReader->skip(1); // force eof
	}
	b = false; 
	return *this;
}

hkIstream& hkIstream::operator>> (char& c)
{
	m_streamReader->read(&c, 1);
	return *this;
}

#if defined(HK_COMPILER_MSVC)
#	pragma warning(push)
#	pragma warning(disable:4146)
#endif
template <typename T>
inline void readInteger( hkStreamReader* reader, T& tOut )
{
	hkBool neg;
	hkUint64 u64 = readInteger64(reader, neg);
	T t = static_cast<T>(u64);
	tOut = neg ? static_cast<T>(-t) : t;
}
#if defined(HK_COMPILER_MSVC)
#	pragma warning(pop)
#endif

hkIstream& hkIstream::operator>> (short& s)
{
	readInteger(m_streamReader, s);
	return *this;
}

hkIstream& hkIstream::operator>> (unsigned short& s)
{
	readInteger(m_streamReader, s);
	return *this;

}

hkIstream& hkIstream::operator>> (int& i)
{
	readInteger(m_streamReader, i);
	return *this;
}

hkIstream& hkIstream::operator>> (unsigned int& i)
{
	readInteger(m_streamReader, i);
	return *this;
}

hkIstream& hkIstream::operator>> (float & f)
{
	f = readFloat(m_streamReader);
	return (*this);
}

hkIstream& hkIstream::operator>> (double& d)
{
	// XXX down size as Havok doesn't use doubles usually,
	// but change this if doubles required
	d = float( readFloat(m_streamReader) );
	return *this;
}

hkIstream& hkIstream::operator>> (hkInt64& i)
{
	readInteger(m_streamReader, i);
	return *this;
}

hkIstream& hkIstream::operator>> (hkUint64& i)
{
	readInteger(m_streamReader, i);
	return *this;
}

hkIstream& hkIstream::operator>> (hkStringBuf& str)
{
	str.clear();
	hkArray<char>::Temp& strBuf = str.getArray();
	eatWhiteSpace( m_streamReader );
	while( 1 )
	{
		char buf[64];
		if( int n = m_streamReader->peek(buf, sizeof(buf)) )
		{
			// look for a " " or "\n" terminator
			for( int i = 0; i < n; ++i )
			{
				char c = buf[i];
				if( IS_SPACE(c) || IS_EOL(c) )
				{
					m_streamReader->skip(i);
					strBuf.append(buf,i);
					strBuf.pushBack(0);
					return *this;
				}
			}
			// no terminator, go back for more
			m_streamReader->skip(n);
			strBuf.append(buf, n);
		}
		else // eof, return what we've got
		{
			if( strBuf.getSize() == 0 )
			{
				m_streamReader->skip(1);
			}
			strBuf.pushBack(0);
			return *this;
		}
	}
}

hkIstream& hkIstream::operator>> (hkStringPtr& str)
{
	hkStringBuf sb;
	(*this) >> sb;
	str = sb;
	return *this;
}

int hkIstream::getline(char* str, int maxlen, char delim)
{
	HK_ASSERT(0x50a0d3e2, maxlen > 0); //XXX batch me
	eatWhiteSpace( m_streamReader );

	int n = m_streamReader->peek(str, maxlen);
	// look for the terminator
	for( int i = 0; i < n; ++i )
	{
		if( str[i] == delim )
		{
			str[i] = '\0';
			m_streamReader->skip(i+1);
			return i;
		}
	}
	// maybe got something before eof
	if( n > 0 && n < maxlen )
	{
		m_streamReader->skip(n);
		str[n] = '\0';
		return n;
	}

	// we got nothing
	if( n == 0 && maxlen > 0 )
	{
		m_streamReader->skip(1); //force eof
	}
	// if we reach maxlen without seeing the delimiter, we return -1
	m_streamReader->skip(n);
	if( maxlen ) str[0] = 0;
	return -1;
}

int hkIstream::read( void* buf, int nbytes )
{
	return m_streamReader->read(buf, nbytes);
}

void hkIstream::setStreamReader(hkStreamReader* newReader)
{
	m_streamReader = newReader;
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
