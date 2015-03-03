/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Reader/Buffered/hkBufferedStreamReader.h>

#ifdef HK_PLATFORM_WIIU
#	include <cafe.h>
	static const int READ_BUFFER_ALIGNMENT = PPC_IO_BUFFER_ALIGN;
#else
	static const int READ_BUFFER_ALIGNMENT = 64;
#endif
static const int READ_BUFFER_BLOCK_SIZE = 512;

#define IS_POWER_OF_2(A) (((A)&((A)-1))==0)

hkBufferedStreamReader::Buffer::Buffer(int cap)
	:	begin( hkAlignedAllocate<char>( READ_BUFFER_ALIGNMENT, cap, HK_MEMORY_CLASS_STREAM ) ),
		current(0),
		size(0),
		capacity(cap)
{
	HK_ASSERT3( 0x3a82bd7f, cap % READ_BUFFER_BLOCK_SIZE == 0, "block size needs to be a multiple of " << READ_BUFFER_BLOCK_SIZE );
}

hkBufferedStreamReader::Buffer::~Buffer()
{
    hkAlignedDeallocate<char>(begin);
}

hkBufferedStreamReader::hkBufferedStreamReader(hkStreamReader* s, int bufSize)
	:	m_stream(s),
		m_seekStream(s->isSeekTellSupported()),
		m_buf(bufSize)

{
	HK_ASSERT( 0x3a82bd80, m_stream != HK_NULL );
	m_stream->addReference();
}


hkBufferedStreamReader::~hkBufferedStreamReader()
{
	m_stream->removeReference();
}

static int refillBuffer(hkStreamReader* sr, void* buf, int nbytes)
{
	if( sr->isOk()==false )
	{
		return 0;
	}
	int numTodo = nbytes;
	int numDone = 0; // invariant: numTodo + numDone == nbytes. keep them in sync, compiler will remove dead stores
	while( numTodo )
	{
		HK_ASSERT(0x5ed27176, numDone + numTodo == nbytes);
		int n = sr->read( hkAddByteOffset(buf, numDone), numTodo );
		if( n > 0 )
		{
			numTodo -= n;
			numDone += n;
		}
		else
		{
			return numDone;
		}
	}

	return numDone;
}

int hkBufferedStreamReader::read(void* buf, int nbytes)
{
	int numTodo = nbytes;
	int numDone = 0;  // invariant: numTodo + numDone == nbytes. keep them in sync, compiler will remove dead stores
	int numInBuffer = m_buf.size - m_buf.current;

	while( numTodo > numInBuffer ) // while bytes left bigger than buffer
	{
		HK_ASSERT(0x65ccb444, numDone + numTodo == nbytes);

		hkString::memCpy( hkAddByteOffset(buf, numDone), m_buf.begin+m_buf.current, numInBuffer );
		numTodo -= numInBuffer;
		numDone += numInBuffer;

		// buffer is now empty
		m_buf.current = 0;
		m_buf.size = 0;

		if( int n = refillBuffer(m_stream, m_buf.begin, m_buf.capacity) )
		{
			m_buf.size = n;
			numInBuffer = n;
		}
		else // reached eof early out
		{
			return numDone;
		}
	}

	// bytes are satisfied by buffer, numTodo <= numInbuffer
	hkString::memCpy( hkAddByteOffset(buf, numDone), m_buf.begin+m_buf.current, numTodo );
	m_buf.current += numTodo;
	numDone += numTodo;
	//numTodo = 0;

	return numDone;
}

int hkBufferedStreamReader::skip(int nbytes)
{
	int numTodo = nbytes;
	int numDone = 0; // invariant: numTodo + numDone == nbytes. keep them in sync, compiler will remove dead stores
	int numInBuffer = m_buf.size - m_buf.current;

	if( numTodo > numInBuffer )
	{
		// we need to skip past our current buffer
		numTodo -= numInBuffer;
		numDone += numInBuffer;
		m_buf.current = 0;
		m_buf.size = 0;

		// now skip as many whole blocks as we need
		int blockSkips = numTodo / READ_BUFFER_BLOCK_SIZE;
		int ns = m_stream->skip( blockSkips );
		if( ns < blockSkips )
		{
			numTodo -= ns;
			numDone += ns;
			return numDone;
		}

		// refill again after the skip
		numTodo -= blockSkips;
		numDone += blockSkips;
		m_buf.size = refillBuffer(m_stream, m_buf.begin, m_buf.capacity );
		numInBuffer = m_buf.size;
	}

	// remaining skip is satisfied by the buffer
	int n = hkMath::min2( numTodo, numInBuffer );
	m_buf.current += n;
	numTodo -= n;
	numDone += n;
	return numDone;
}

hkBool hkBufferedStreamReader::isOk() const
{
	// we have some buffered or we can get some
	return m_buf.current != m_buf.size || m_stream->isOk();
}

int hkBufferedStreamReader::peek(void* buf, int nbytes)
{
	if( nbytes > m_buf.capacity-READ_BUFFER_BLOCK_SIZE )
	{
		return -1; // too big
	}
	if( m_buf.current + nbytes > m_buf.size )
	{
		// if there is some unconsumed stuff still in the buffer,
		// note that this can be greater than READ_BUFFER_BLOCK_SIZE
		int numBytesToPreserve = m_buf.size - m_buf.current;

		// move preserved to beginning of buffer, allowing for alignment
		int newCurrent = READ_BUFFER_BLOCK_SIZE - (numBytesToPreserve % READ_BUFFER_BLOCK_SIZE);
		if((newCurrent != m_buf.current) && numBytesToPreserve)
		{
			hkMemUtil::memMove(	m_buf.begin + newCurrent,
				m_buf.begin + m_buf.current,
				numBytesToPreserve ); 
		}
		m_buf.current = newCurrent;
		m_buf.size = newCurrent + numBytesToPreserve; // buf.size should be called buf.end

		// now read some more into the space between size and capacity
		int n = refillBuffer( m_stream, hkAddByteOffset(m_buf.begin, m_buf.size), m_buf.capacity-m_buf.size );
		m_buf.size += n;
	}

	int n = hkMath::min2( nbytes, m_buf.size - m_buf.current );
 	hkMemUtil::memCpy(buf, hkAddByteOffset(m_buf.begin, m_buf.current), n);

	return n;
}


hkSeekableStreamReader* hkBufferedStreamReader::isSeekTellSupported()
{
	return m_seekStream ? this : HK_NULL;
}

hkResult hkBufferedStreamReader::seek( int offset, SeekWhence whence)
{
	
	m_buf.current = 0;
	m_buf.size = 0; // and clears the buffer
	return m_seekStream->seek(offset, whence);
}

int hkBufferedStreamReader::tell() const
{
	int childPos = m_seekStream->tell();
	if( childPos >= 0 )
	{
		int offset = m_buf.size - m_buf.current;
		return childPos - offset;
	}
	return -1;
}

HK_COMPILE_TIME_ASSERT( IS_POWER_OF_2(READ_BUFFER_ALIGNMENT) );
HK_COMPILE_TIME_ASSERT( IS_POWER_OF_2(READ_BUFFER_BLOCK_SIZE) );

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
