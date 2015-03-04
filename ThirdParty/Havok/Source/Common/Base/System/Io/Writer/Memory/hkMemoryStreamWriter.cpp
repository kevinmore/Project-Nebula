/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Writer/Memory/hkMemoryStreamWriter.h>

int hkMemoryStreamWriter::write(const void* b, int nb)
{
	HK_ASSERT(0x50acbb51, nb >= 0);
	int n = hkMath::min2(nb, m_bufSize-m_cur);
	hkMemUtil::memCpy( hkAddByteOffset(m_buf,m_cur), b, n);
	m_cur += n;
	return n;
}

hkBool hkMemoryStreamWriter::isOk() const
{
	return m_cur < m_bufSize;
}

hkBool hkMemoryStreamWriter::seekTellSupported() const
{
	return true; 
}

int hkMemoryStreamWriter::tell() const
{
	return m_cur;
}

hkResult hkMemoryStreamWriter::seek(int offset, hkStreamWriter::SeekWhence whence)
{
	int absOffset = m_cur;
	switch( whence )
	{
		case STREAM_SET:
			absOffset = offset;
			break;
		case STREAM_CUR:
			absOffset = m_cur + offset;
			break;
		case STREAM_END:
			absOffset = m_bufSize - offset;
			break;
		default:
			HK_ASSERT2(0x55f1b803, 0, "Bad 'whence' passed to seek()");
			break;
	}
	if( (absOffset >= 0)  && (absOffset <= m_bufSize) )
	{
		m_cur = absOffset;
		return HK_SUCCESS;
	}
	return HK_FAILURE;
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
