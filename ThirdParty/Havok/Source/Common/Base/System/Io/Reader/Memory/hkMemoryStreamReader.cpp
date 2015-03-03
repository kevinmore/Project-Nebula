/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Reader/Memory/hkMemoryStreamReader.h>

hkMemoryStreamReader::hkMemoryStreamReader(const void* mem, int memSize, MemoryType mt)
	:	m_bufCurrent(0),
		m_bufSize(memSize),
		m_memType(mt),
		m_hitEof(false)
{
	if( m_memType == MEMORY_COPY )
	{
		m_buf = hkAllocate<char>( memSize, HK_MEMORY_CLASS_STREAM );
		hkString::memCpy( m_buf, mem, memSize );
	}
	else
	{
		m_buf = const_cast<char*>(static_cast<const char*>(mem));
	}
}

hkMemoryStreamReader::~hkMemoryStreamReader()
{
	if( m_memType == MEMORY_COPY || m_memType == MEMORY_TAKE )
	{
		hkDeallocate<char>(m_buf);
	}
}

int hkMemoryStreamReader::read(void* buf, int nbytes)
{
    int nread = hkMath::min2(m_bufSize - m_bufCurrent, nbytes);
    hkString::memCpy(buf, m_buf+m_bufCurrent, nread );
    m_bufCurrent += nread;
	if( nread == 0 && nbytes != 0 )
	{
		m_hitEof = true;
	}
    return nread;
}

int hkMemoryStreamReader::skip(int nbytes)
{
	int nread = hkMath::min2(m_bufSize - m_bufCurrent, nbytes);
	m_bufCurrent += nread;
	if( nread == 0 && nbytes != 0 )
	{
		m_hitEof = true;
	}
	return nread;
}

hkBool hkMemoryStreamReader::isOk() const
{
    return m_hitEof == false;
}

int hkMemoryStreamReader::peek(void* buf, int nbytes)
{
	int nread = hkMath::min2(m_bufSize - m_bufCurrent, nbytes);
	hkMemUtil::memCpy( buf, m_buf+m_bufCurrent, nread);
	return nread;
}

hkResult hkMemoryStreamReader::seek(int relOffset, SeekWhence whence)
{
	int pos = -1;
	switch(whence)
	{
		case STREAM_SET:
			pos = relOffset;
			break;
		case STREAM_CUR:
			pos = m_bufCurrent + relOffset;
			break;
		case STREAM_END:
			pos = m_bufSize - relOffset;
			break;
	}
	hkResult ok = HK_SUCCESS;
	if(	pos < 0 )
	{
		pos = 0;
		ok = HK_FAILURE;
	}
	else if( pos > m_bufSize )
	{
		pos = m_bufSize;
		ok = HK_FAILURE;
	}
	m_bufCurrent = pos;
	m_hitEof = false;
	return ok;
}

int hkMemoryStreamReader::tell() const
{
	return m_bufCurrent;
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
