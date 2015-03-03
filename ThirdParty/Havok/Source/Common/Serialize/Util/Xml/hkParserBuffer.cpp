/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
// this
#include <Common/Serialize/Util/Xml/hkParserBuffer.h>

hkParserBuffer::hkParserBuffer(hkStreamReader* reader)
{
	m_reader = reader;
	reader->addReference();

	// Set up the buffer as empty
	m_buffer.setSize(1);
	m_buffer[0] = 0;
	m_buffer.setSizeUnchecked(0);

	// Set the buffer to the start
	m_pos = m_buffer.begin();

	// 
	m_lexemeStart = 0;
	m_bufferStart = 0;

	// Reset the position
	m_row = 0;
	m_col = 0;
}

hkParserBuffer::~hkParserBuffer()
{
	m_reader->removeReference();
}

int hkParserBuffer::read(int size)
{
	HK_ASSERT(0x2342a3a4, size > 0);

	// Make the minimum read size 256
	if (size < MIN_READ_SIZE)
	{
		size = MIN_READ_SIZE;
	}

	// See if should shift back
	char* oldStart = m_buffer.begin();
	const int oldSize = m_buffer.getSize();

	// Add the needed space. Add 1, cos we need enough space to always zero terminate
	char* dst = m_buffer.expandBy(size + 1);
	// Try reading
	int numRead = m_reader->read(dst, size);

	// Zero terminate it
	m_buffer[oldSize + numRead] = 0;
	// Set the new buffer size
	m_buffer.setSizeUnchecked(oldSize + numRead);
	
	// See if the pointers need fixing
	char* start = m_buffer.begin();
	if (start != oldStart)
	{
		// Fix the pos
		m_pos = (m_pos - oldStart) + start;
	}

	return m_buffer.getSize() - oldSize; 
}

void hkParserBuffer::lexemeCommit()
{
	// Reset the start position
	const char cr = 0xd;
	const char lf = 0xa;

	const char* pos = m_buffer.begin() + m_lexemeStart;
	const char* end = m_pos;
	for (; pos < end; pos++)
	{
		const char c = *pos;

		if (c == cr)
		{
			if (pos + 1 < end && c == lf)
			{
				// cr/lf
				pos ++;
			}
			// cr
			m_row++;
			m_col = 0;
		}
		else if (c == lf)
		{
			m_row++;
			m_col = 0;
		}
	}
	// Move start
	m_lexemeStart = int(m_pos - m_buffer.begin());
}

void hkParserBuffer::bufferCommit()
{
	HK_ASSERT(0x323a2a54, m_lexemeStart >= m_bufferStart);

	m_bufferStart = m_lexemeStart;

	// Look if its worth copying back...
	if (m_bufferStart > 1024)
	{
		// Okay... Lets shift everything back
		hkString::memMove(m_buffer.begin(), m_buffer.begin() + m_bufferStart, m_buffer.getSize() - m_bufferStart);

		const int len = m_buffer.getSize() - m_bufferStart;
		// Calc new size
		m_buffer.setSizeUnchecked(len);

		// Make sure space for end mark
		HK_ASSERT(0x3423432a, len < m_buffer.getCapacity());
		// Mark the end
		m_buffer.begin()[len] = 0;

		// Shift the lexeme position
		m_pos -= m_bufferStart;

		// Reset the start
		m_lexemeStart -= m_bufferStart;
		m_bufferStart = 0;
	}
}

void hkParserBuffer::setLexemePosition(int pos)
{
	HK_ASSERT(0x4234aaab, pos >= 0 && pos < m_buffer.getSize() - m_lexemeStart);
	m_pos = m_buffer.begin() + m_lexemeStart + pos;
}

hkBool hkParserBuffer::match(const char* text, int len)
{
	int postSize = requirePostSize(len);
	if (postSize < len)
	{
		return false;
	}
	// Do the comparison
	return hkString::strNcmp(getCurrentPosition(), text, len) == 0;
}

hkBool hkParserBuffer::match(const char* text)
{
	return match(text, hkString::strLen(text));
}

hkBool hkParserBuffer::matchAndConsume(const char* text)
{
	int len = hkString::strLen(text);
	if (match(text, len))
	{
		advance(len);
		return true;
	}
	return false;
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
