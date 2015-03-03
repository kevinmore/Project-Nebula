/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Container/String/Deprecated/hkStringOld.h>
#include <Common/Base/Fwd/hkcstdarg.h>

using namespace std;

// very small strings will likely cause reallocations
static const int MINIMUM_STRING_CAPACITY = 64 - 1;
//
// nonstatic member functions
//
#if !defined(HK_PLATFORM_PS3_SPU) 
void HK_CALL hkStringOld::printf(const char* fmt, ...)
{
	if( getCapacity() < MINIMUM_STRING_CAPACITY )
	{
		setCapacity( 255 );
	}	

	va_list args; 
	va_start(args, fmt);
	while(1)
	{
		int size = m_string.getCapacity();
		int nchars = hkString::vsnprintf(m_string.begin(), size, fmt, args);

		if( nchars >= 0 && nchars < size ) 
		{
			// usual case, it worked. update length		
			m_string.setSizeUnchecked( nchars+1 ); // reducing size
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
	m_string.optimizeCapacity( 0, true ); // shrink the array
}
#endif

int hkStringOld::indexOf(char c, int start, int end) const
{
	for(int i = start; i < getLength() && i < end; ++i)
	{
		if( m_string[i] == c )
		{
			return i;
		}
	}
	return -1;
}

int hkStringOld::lastIndexOf(char c, int start, int end) const
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

hkStringOld hkStringOld::operator+ (const hkStringOld& other) const 
{
	int myLength = getLength();
	int otherLength = other.getLength();
	int totalLength= myLength + otherLength;
    // allocate a new chunk of memory to hold the result of concatenation
	char* p = hkAllocateChunk<char>( totalLength + 1, HK_MEMORY_CLASS_STRING ); 
	// copy the two strings
	hkString::memCpy( p, m_string.begin(), myLength );
	hkString::memCpy( p+myLength, other.m_string.begin(), otherLength+1 ); // copy null too

	return hkStringOld(p, totalLength + 1, totalLength + 1); // the string is not copied, the array uses the buffer
}

hkStringOld hkStringOld::operator+ (const char* other) const
{
	int myLength = getLength();
	int otherLength = hkString::strLen(other);
	int totalLength= myLength + otherLength;
	// allocate a new chunk of memory to hold the result of concatenation
	char* p = hkAllocateChunk<char>( totalLength + 1, HK_MEMORY_CLASS_STRING );
	// copy the two strings
	hkString::memCpy( p, m_string.begin(), myLength );
	hkString::memCpy( p+myLength, other, otherLength+1 ); // copy null too
	
	return hkStringOld( p, totalLength + 1, totalLength + 1 ); // the string is not copied, the array uses the buffer
}

hkStringOld& hkStringOld::operator+= (const hkStringOld& other)
{
	int myLength = getLength();
	int otherLength = other.getLength();
	// increase size and concatenate the strings
	setLength( myLength + otherLength );
	hkString::memCpy( m_string.begin()+myLength, other.cString(), otherLength+1 ); // copy null too
	
	return *this;
}

hkStringOld& hkStringOld::operator+= (const char* other)
{
	int myLength = getLength();
	int otherLength = hkString::strLen(other);
	// increase size and concatenate the strings
	setLength( myLength + otherLength );
	hkString::memCpy( m_string.begin()+myLength, other, otherLength+1 ); // copy null too
	
	return *this;
}

hkStringOld hkStringOld::asUpperCase() const
{
	char* p = hkAllocateChunk<char>( m_string.getSize(), HK_MEMORY_CLASS_STRING );
	// don't use strupr because that doesn't deal with embedded nulls
	for(int i = 0; i < getLength(); ++i)
	{
		p[i] = hkString::toUpper(m_string[i]);
	}
	p[ getLength() ] = 0; // null terminate

	return hkStringOld( p, m_string.getSize(), m_string.getSize() ); // memory is not copied, use the new buffer
}

void hkStringOld::makeUpperCase()
{
	for(int i = 0; i < getLength(); ++i)
	{
		m_string[i] = hkString::toUpper(m_string[i]);
	}
}


hkStringOld hkStringOld::asLowerCase() const
{
	char* p = hkAllocateChunk<char>( m_string.getSize(), HK_MEMORY_CLASS_STRING );
	// don't use strlwr because that doesn't deal with embedded nulls
	for(int i = 0; i < getLength(); ++i)
	{
		p[i] = hkString::toLower(m_string[i]);
	}
	p[ getLength() ] = 0; // null terminate
	
	return hkStringOld( p, m_string.getSize(), m_string.getSize() ); // memory is not copied, use the new buffer
}

void hkStringOld::makeLowerCase()
{
	for(int i = 0; i < getLength(); ++i)
	{
		m_string[i] = hkString::toLower(m_string[i]);
	}
}

hkBool hkStringOld::beginsWith (const char* other) const
{
	for(int i=0; other[i] != 0; ++i)
	{
		if( i >= getLength() || m_string[i] != other[i] )
		{
			return false;
		}
	}
	return true;
}

hkBool hkStringOld::endsWith (const hkStringOld& other) const   // use the char* varsion? to avoid strLen?
{
	if( getLength() < other.getLength() )
	{
		return false;
	}
	int offset = getLength() - other.getLength();

	for(int i=0; i < other.getLength(); ++i)
	{
		if( m_string[i+offset] != other[i] )
		{
			return false;
		}
	}
	return true;
}

hkBool hkStringOld::endsWith (const char* other) const
{
	int offset = getLength() - hkString::strLen(other);
	if(offset < 0)
	{
		return false;
	}
	for(int i=0; other[i] != 0; ++i)
	{
		if( m_string[i+offset] != other[i] )
		{
			return false;
		}
	}
	return true;
}

hkStringOld hkStringOld::replace(char from, char to, hkString::ReplaceType rtype ) const
{
	char* p = hkAllocateChunk<char>( m_string.getSize(), HK_MEMORY_CLASS_STRING );
	hkString::memCpy( p, m_string.begin(), m_string.getSize() );
	for(int i = 0; i < getLength(); ++i)
	{
		if(p[i] == from)
		{
			p[i] = to;
			if(rtype == REPLACE_ONE)
			{
				break;
			}
		}
	}
	p[ getLength() ] = 0; // null terminate

	return hkStringOld( p, m_string.getSize(), m_string.getSize() ); // memory is not copied, the new buffer is used
}

hkBool hkStringOld::replaceInplace( char from, char to, hkString::ReplaceType rtype ) 
{
	hkBool replaced = false;
	for(int i = 0; i < getLength(); ++i)
	{
		if(m_string[i] == from)
		{
			m_string[i] = to;
			replaced = true;
			if(rtype == REPLACE_ONE)
			{
				break;
			}
		}
	}
	return replaced;
}

void HK_CALL hkStringOld::copyAndReplace( char* dest, const char* orig, int origLength, const hkStringOld& from, const hkStringOld& to, const hkArray<int>& indices)
{
	int currentPosOld = 0;
	int currentPosNew = 0;
	int fromLen = from.getLength();
	int toLen = to.getLength();

	for(int i =0; i<indices.getSize(); ++i)
	{
		// copy original piece
		int sizeToCopy = (i==0)? indices[i] : ( indices[i] - indices[i-1] - fromLen );
		hkString::memCpy( dest+currentPosNew, orig+currentPosOld, sizeToCopy );
		currentPosNew += sizeToCopy;
		currentPosOld += sizeToCopy+fromLen;
		// copy replacement string "to"
		hkString::memCpy( dest+currentPosNew, to.cString(), toLen );
		currentPosNew += toLen;
	}
	// copy final piece
	hkString::memCpy( dest+currentPosNew, orig+currentPosOld, origLength - indices[indices.getSize()-1] - fromLen  );
}

hkStringOld hkStringOld::replace(const hkStringOld& from, const hkStringOld& to, hkString::ReplaceType rtype) const
{
	// indices of occurrences of "from"
	hkArray<int> indices;
	findAllOccurrences( cString(), from.cString(), indices, rtype );
	// allocate memory
	int totalSize = m_string.getSize() + (to.getLength() - from.getLength())*indices.getSize();
	char* dest = hkAllocateChunk<char>( totalSize, HK_MEMORY_CLASS_STRING );
	
	if( !indices.isEmpty() )
	{
		// replace occurrences of "from" with "to"
		copyAndReplace( dest, cString(), getLength(), from, to, indices );
	}
	else
	{
		// no occurrences found, just copy the string as it is
		hkString::memCpy( dest, cString(), getLength() );
	}
	dest[ totalSize-1 ] = 0; // null terminate
	// make a string from the allocated memory and return it
	return hkStringOld( dest, totalSize, totalSize );
}

hkBool hkStringOld::replaceInplace(const hkStringOld& from, const hkStringOld& to, hkString::ReplaceType rtype) 
{
	// keep the indices of the occurrences of "from" in this string
	hkInplaceArray<int,12> indices; 
	//hkArray<int> indices; 

	// if we found any occurrences
	if( findAllOccurrences( this->cString(), from.cString(), indices, rtype ) ) 
	{
		int totalSize = m_string.getSize() + (to.getLength() - from.getLength())*indices.getSize();
		// destination of the copying, maybe a new memory chunk or the current string

		char* destination;
		// if the new length is bigger than the old one we have to reallocate to
		// avoid overlapping, even if the capacity is bigger
		if( totalSize > m_string.getSize() )
		{
			//allocate a piece of memory
			destination = hkAllocateChunk<char>( totalSize, HK_MEMORY_CLASS_STRING );
		}
		else
		{
			//no need to allocate, use current buffer
			destination = m_string.begin();
		}
		
		// do the replacement
		copyAndReplace( destination, m_string.begin(), getLength(), from, to, indices );

		destination[ totalSize-1 ] = 0; // null terminate
		

		//if we allocated a new chunk we must free the old one
		if( destination != m_string.begin() ) 
		{
			m_string.clearAndDeallocate();
			//update data, size and capacity
			m_string.setDataAutoFree( destination, totalSize, totalSize );
		}
		else 
		{
			//otherwise just update the size
			m_string.setSize( totalSize );
		}
		return true;
	}
	else
	{
		return false;
	}
}

void hkStringOld::split(int ic, hkArray<hkStringOld>& bits ) const
{
	char c = (char)ic;

	int cur = 0;
	int end = indexOf(c, cur);
	while( end != -1 )
	{
		bits.expandOne() = substr(cur, end-cur);
		cur = end + 1;
		end = indexOf(c, cur);
	}
	if( m_string[cur] != 0 )
	{
		bits.expandOne() = substr(cur);
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
