/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/String/hkStringPtr.h>
#include <Common/Base/Fwd/hkcstdarg.h>

static const char*const s_emptyString = HK_NULL;

// These are copy & pasted from hkString so that hkBaseExt.cpp works easier.
static char* strDup(const char* src)
{
	HK_ASSERT(0x13e1d159, src != HK_NULL);
	char* r = hkAllocate<char>( hkString::strLen(src)+1, HK_MEMORY_CLASS_STRING );
	hkString::strCpy(r, src);
	return r;
}

static char* strNdup(const char* src, int maxlen)
{
	HK_ASSERT(0x2e27506f, src != HK_NULL);
	int len = hkString::strLen(src);
	if( len > maxlen )
	{
		len = maxlen;
	}
	char* r = hkAllocate<char>(len+1, HK_MEMORY_CLASS_STRING);
	hkString::strNcpy(r, src, len);
	r[len] = 0;
	return r;
}

static void assign( const char** ptr, const char* src, int len=-1 )
{
	if( hkClearBits(*ptr, hkStringPtr::OWNED_FLAG) == src )
	{
		return;
	}

	if( (hkUlong(*ptr) & hkStringPtr::OWNED_FLAG) == hkStringPtr::OWNED_FLAG )
	{
		hkDeallocate<char>( const_cast<char*>(*ptr) - 1);
	}
	if( src )
	{
		char* p = ( len == -1 )
			? strDup(src)
			: strNdup(src, len);
		HK_ASSERT(0x3d02cfc8, (hkUlong(p) & 1) == 0 );
		*ptr = p + 1; // set the owned flag
	}
	else
	{
		*ptr = s_emptyString;
	}
}

hkStringPtr::hkStringPtr()
	: m_stringAndFlag(s_emptyString)
{
}

hkStringPtr::hkStringPtr(const char* string)
	: m_stringAndFlag(s_emptyString)
{
	assign(&m_stringAndFlag, string);
}

hkStringPtr::hkStringPtr(const char* s, int len)
	: m_stringAndFlag(s_emptyString)
{
	assign(&m_stringAndFlag, s, len);
}

hkStringPtr::hkStringPtr(const hkStringPtr& strRef)
	: m_stringAndFlag(s_emptyString)
{
	assign(&m_stringAndFlag, strRef);
}

hkStringPtr::hkStringPtr(hkFinishLoadedObjectFlag f)
{
}

hkStringPtr::~hkStringPtr()
{
	assign(&m_stringAndFlag, HK_NULL);
}

hkStringPtr& hkStringPtr::operator=(const char* string)
{
	assign(&m_stringAndFlag, string);
	return *this;
}

hkStringPtr& hkStringPtr::operator=(const hkStringPtr& string)
{
	assign(&m_stringAndFlag, string);
	return *this;
}

int hkStringPtr::getLength() const
{
	const char* s = cString();
	return s ? hkString::strLen(s) : 0;
}

void hkStringPtr::printf(const char* fmt, ...)
{
	hkInplaceArray<char, 255> buf;
	va_list args; 
	va_start(args, fmt);
	while(1)
	{
		int size = buf.getCapacity();
		int nchars = hkString::vsnprintf(buf.begin(), size, fmt, args);

		if( nchars >= 0 && nchars < size ) 
		{
			// usual case, it worked.
			operator=(buf.begin());
			break;
		}
		else if( nchars < 0 )
		{
			// there was not enough room, double capacity
			buf.reserve( buf.getCapacity() * 2 );
		}
		else
		{
			// there was not enough room and we were told how much
			// was needed (not including \0)
			buf.reserve( nchars + 1 );
		}
	}
	va_end(args);
	*this = buf.begin();
}

void hkStringPtr::set(const char* s, int len)
{
	HK_ASSERT2(0x51d73058, cString() != s ,"Don't use set to resize string ptr");
	assign(&m_stringAndFlag, s, len);
}

void hkStringPtr::setPointerAligned(const char* s) 
{
	// make sure that the pointer is at least 2-aligned
	HK_ASSERT2(0x3d02cfc8, (hkUlong(s) & 1) == 0, "The pointer must refer to a 2-aligned storage");
	// set the pointer
	if(cString() != s) // do not reassign the same pointer
	{
		// deallocate the previous string (if owned) and set the internal pointer to NULL
		assign(&m_stringAndFlag, HK_NULL);
		// copy the given pointer (force not owned)
		m_stringAndFlag = reinterpret_cast<const char*>(hkUlong(s));
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
