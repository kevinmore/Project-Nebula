/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Util/hkLoadUtil.h>
#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>

hkLoadUtil::hkLoadUtil(const char* fname)
	: m_fileName(fname)
{
}

hkLoadUtil::hkLoadUtil( hkStreamReader* sr )
	: m_fileName(HK_NULL), m_reader(sr)
{
}

hkLoadUtil::hkLoadUtil( hkIstream& is )
	: m_fileName(HK_NULL), m_reader(is.getStreamReader())
{
}

hkLoadUtil::~hkLoadUtil()
{
	//out of line for refptr
}

// outputs

bool hkLoadUtil::toArray( hkArrayBase<char>& out, hkMemoryAllocator& mem )
{
	hkRefPtr<hkStreamReader> sr;
	hkInt64 oldOffset = -1;
	hkInt64 size = -1;
	if( m_fileName )
	{
		hkFileSystem::Entry entry;
		switch (hkFileSystem::getInstance().stat(m_fileName, entry))
		{
		case hkFileSystem::RESULT_OK:
			{
				sr = entry.openReader();
				if( sr == HK_NULL )
				{
					return false;
				}
				size = (int)entry.getSize();
				out._reserve( mem, out.getSize() + int(size) );
				break;
			}
		case hkFileSystem::RESULT_NOT_IMPLEMENTED:
			{
				// Stat is not currently implemented by the server filesystem
				sr = hkFileSystem::getInstance().openReader(m_fileName);
				if( sr == HK_NULL )
				{
					return false;
				}
				break;
			}
		case hkFileSystem::RESULT_ERROR:
			return false;
		}
	}
	else if( m_reader && m_reader->isOk() )
	{
		sr = m_reader;
		if( hkSeekableStreamReader* s = sr->isSeekTellSupported() )
		{
			oldOffset = s->tell();
		}
	}
	else
	{
		return false; // no source available
	}

	bool ret;
	while( true )
	{
		const int N = 4096;
		char buf[4096];
		int n = sr->read( buf, N );
		if( n > 0 )
		{
			hkMemUtil::memCpy( out._expandBy(mem, n), buf, n );
		}
		else if( n == 0 )
		{
			ret = true;
			break;
		}
		else // n<0
		{
			ret = false;
			break;
		}
	}

	if( oldOffset != -1 )
	{
		sr->isSeekTellSupported()->seek( int(oldOffset), hkSeekableStreamReader::STREAM_SET);
	}
	return ret;
}


bool hkLoadUtil::toArray( hkArray<char>::Temp& out )
{
	return toArray( out, hkMemoryRouter::getInstance().temp() );
}


bool hkLoadUtil::toArray( hkArray<char>& out )
{
	return toArray( out, hkMemoryRouter::getInstance().heap() );
}

void* hkLoadUtil::toAllocation( int* sizeOut, hkMemoryAllocator& mem )
{
	hkArray<char>::Temp temp;
	if( toArray(temp) )
	{
		*sizeOut = temp.getSize();
		void* p = mem.blockAlloc( temp.getSize() );
		hkMemUtil::memCpy(p, temp.begin(), temp.getSize() );
		return p;
	}
	return HK_NULL;
}

bool hkLoadUtil::toString( hkStringBuf& buf )
{
	// Pop back null at the end of the string
	HK_ASSERT(0x528933f, buf.getArray().back() == 0);
	buf.getArray().popBack();
	bool ret = toArray(buf.getArray());
	buf.getArray().pushBack(0);
	return ret;
}

bool hkLoadUtil::toString( hkStringPtr& buf )
{
	hkArray<char>::Temp temp;
	if( toArray(temp) )
	{
		buf.set( temp.begin(), temp.getSize() );
		return true;
	}
	return false;
}

int hkLoadUtil::toString( char* buf, int bufSize )
{
	hkArray<char>::Temp temp;
	if( toArray(temp) )
	{
		int len = hkMath::min2( temp.getSize(), bufSize-1 );
		hkMemUtil::memCpy(buf, temp.begin(), len );
		buf[len] = 0;
		return len;
	}
	return 0;
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
