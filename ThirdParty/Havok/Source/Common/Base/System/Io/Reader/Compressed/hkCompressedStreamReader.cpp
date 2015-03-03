/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Reader/Compressed/hkCompressedStreamReader.h>
#include <Common/Base/Algorithm/Compression/hkCompression.h>

enum { DEFAULT_BUFSIZE = 1<<16 };

hkCompressedStreamReader::hkCompressedStreamReader(hkStreamReader *s)
	: m_stream(s)
	, m_ok(true)
	, m_uncomppos(0)
	, m_uncomplen(0)
	, m_compbufpos(0)
	, m_compbuflen(0)
{
	HK_ASSERT( 0x3a82bd80, m_stream != HK_NULL );
	m_stream->addReference();
	m_compbufsize = DEFAULT_BUFSIZE;
	m_compbuf = hkMemHeapBufAlloc<hkUchar>(m_compbufsize);
	m_uncompbufsize = DEFAULT_BUFSIZE;
	m_uncompbuf = hkMemHeapBufAlloc<hkUchar>(m_uncompbufsize);
}

hkCompressedStreamReader::~hkCompressedStreamReader()
{
	m_stream->removeReference();
	hkMemHeapBufFree(m_compbuf, m_compbufsize);
	hkMemHeapBufFree(m_uncompbuf, m_uncompbufsize);
}


hkResult hkCompressedStreamReader::loadNewBlock()
{
	// Move any already-read compressed data to the start of the buffer
	if (m_compbufpos != 0)
	{
		hkString::memMove(m_compbuf, m_compbuf + m_compbufpos, m_compbuflen);
		m_compbufpos = 0;
	}

	// We need to get the header of the next block
	if (m_compbuflen < hkCompression::BLOCK_HEADER_SIZE)
	{
		int r = m_stream->read(m_compbuf + m_compbuflen, m_compbufsize - m_compbuflen);
		if (r == 0 && m_compbuflen == 0)
		{
			// EOF
			return HK_FAILURE;
		}
		m_compbuflen += r;
		if (m_compbuflen < hkCompression::BLOCK_HEADER_SIZE)
		{
			// partial read of a block header, corrupted/truncated stream
			m_ok = false;
			return HK_FAILURE;
		}
	}

	// Now, we need to get a whole block (at least)
	hk_size_t uncompressed_sz_u, compressed_sz_u;
	int uncompressed_sz, compressed_sz;
	if (hkCompression::getDecompressionBufferSize(m_compbuf, compressed_sz_u, uncompressed_sz_u) == HK_FAILURE)
	{
		// Corrupted header
		m_ok = false;
		return HK_FAILURE;
	}
	uncompressed_sz = (int)uncompressed_sz_u;
	compressed_sz = (int)compressed_sz_u;
	if (compressed_sz > m_compbufsize)
	{
		// need to reallocate input buffer to hold the next block
		int oldsize = m_compbufsize;
		m_compbufsize = compressed_sz;
		m_compbuf = hkMemHeapBufRealloc(m_compbuf, oldsize, m_compbufsize);
	}
	if (m_compbuflen < compressed_sz)
	{
		// need to read some more data to have a whole block
		int r = m_stream->read(m_compbuf + m_compbuflen, m_compbufsize - m_compbuflen);
		m_compbuflen += r;
		if (m_compbuflen < compressed_sz || r == 0)
		{
			// partial read of a block, corrupted/truncated stream
			m_ok = false;
			return HK_FAILURE;
		}
	}
	if (uncompressed_sz > m_uncompbufsize)
	{
		// need to reallocate output buffer to hold the next block
		int oldsize = m_uncompbufsize;
		m_uncompbufsize = uncompressed_sz;
		m_uncompbuf = hkMemHeapBufRealloc(m_uncompbuf, oldsize, m_uncompbufsize);
	}

	// By this stage our input and output buffers are big enough for a block, and at least
	// one entire block has been loaded into the input buffer

	HK_ASSERT(0x4325aabc, m_compbufpos == 0);
	const void* in = m_compbuf;
	void* out = m_uncompbuf;
	hkCompression::Result res = hkCompression::decompress(in, compressed_sz, out, m_uncompbufsize);
	if (res == hkCompression::COMP_ERROR)
	{
		// corrupted data
		m_ok = false;
		return HK_FAILURE;
	}
	HK_ASSERT(0x43259999, res == hkCompression::COMP_NEEDINPUT && out > m_uncompbuf);
	
	m_uncomplen = hkGetByteOffsetInt(m_uncompbuf, out);
	m_uncomppos = 0;
	m_compbufpos += hkGetByteOffsetInt(m_compbuf, in);
	m_compbuflen -= hkGetByteOffsetInt(m_compbuf, in);
	return HK_SUCCESS;
}

int hkCompressedStreamReader::read(void* buf, int nbytes)
{
	if (!m_ok)
	{
		return 0;
	}

	hkUchar* cbuf = (hkUchar*)buf;

	int total_copied = 0;
	while (nbytes > 0)
	{
		if (m_uncomplen == 0)
		{
			hkResult r = loadNewBlock();
			if (r == HK_FAILURE)
			{
				if (m_ok)
				{
					// EOF
					return total_copied;
				}
				else
				{
					return 0;
				}
			}
		}

		HK_ASSERT(0x5d82bc21, m_uncomplen > 0);
		int copy = hkMath::min2(nbytes, m_uncomplen);
		if (cbuf != HK_NULL)
		{
			hkString::memCpy(cbuf, m_uncompbuf + m_uncomppos, copy);
			cbuf += copy;
		}
		nbytes -= copy;
		m_uncomppos += copy;
		m_uncomplen -= copy;
		total_copied += copy;
	}
	return total_copied;
}

int hkCompressedStreamReader::skip(int nbytes)
{
	//> The current implementation of skip decompresses and ignores nbytes
	//> If performance of skip matters for large skips, this could be changed
	//> to jump over whole blocks.
	return read(HK_NULL, nbytes);
}

hkBool hkCompressedStreamReader::isOk() const
{
	return m_ok && m_stream->isOk();
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
