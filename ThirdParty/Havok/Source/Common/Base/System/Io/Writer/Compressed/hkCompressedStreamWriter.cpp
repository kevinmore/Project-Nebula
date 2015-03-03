/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Writer/Compressed/hkCompressedStreamWriter.h>
#include <Common/Base/Algorithm/Compression/hkCompression.h>

hkCompressedStreamWriter::hkCompressedStreamWriter(hkStreamWriter* s, int bufSize)
	:	m_stream(s), m_bufSize(bufSize),
		m_uncompbufpos(0)
{
	HK_ASSERT(0x4325ab34, m_stream != HK_NULL);
	m_stream->addReference();
	m_uncompbufsize = bufSize - hkCompression::BLOCK_HEADER_SIZE;
	m_uncompbuf = hkMemHeapBlockAlloc<hkUchar>(m_uncompbufsize);
}

hkCompressedStreamWriter::~hkCompressedStreamWriter()
{
	m_stream->removeReference();
	hkMemHeapBlockFree(m_uncompbuf, m_uncompbufsize);
}

int hkCompressedStreamWriter::write(const void* buffer, int nbytes)
{
	hkUchar* cbuf = (hkUchar*)buffer;
	int total_copied = 0;
	while (nbytes > 0)
	{
		int copy = hkMath::min2(m_uncompbufsize - m_uncompbufpos, nbytes);
		hkString::memCpy(m_uncompbuf + m_uncompbufpos, cbuf, copy);

		m_uncompbufpos += copy;
		nbytes -= copy;
		cbuf += copy;
		total_copied += copy;

		if (m_uncompbufpos == m_uncompbufsize)
		{
			writeBlock(); 
			if (!isOk())
			{
				return 0; 
			}
		}
	}
	return total_copied;
}

void hkCompressedStreamWriter::writeBlock()
{
	const void* in = m_uncompbuf;
	void* out_start = hkMemTempBlockAlloc<char>(m_bufSize);
	void* out_end = out_start;
	hkCompression::Result res = hkCompression::compress(in, m_uncompbufpos, out_end, m_bufSize, m_bufSize);
	HK_ASSERT(0x4324992a, res == hkCompression::COMP_NEEDINPUT);
	m_stream->write(out_start, hkGetByteOffsetInt(out_start, out_end) );
	hkMemTempBlockFree(out_start, m_bufSize);
	m_uncompbufpos = 0;
}

void hkCompressedStreamWriter::flush()
{
	writeBlock();
	m_stream->flush();
}

hkBool hkCompressedStreamWriter::isOk() const
{
	return m_stream->isOk();
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
