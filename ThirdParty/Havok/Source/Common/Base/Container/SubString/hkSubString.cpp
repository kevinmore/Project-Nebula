/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

// this
#include <Common/Base/Container/SubString/hkSubString.h>

hkBool hkSubString::operator==(const hkSubString& rhs) const
{
	if (m_start == rhs.m_start)
	{
		return m_end == rhs.m_end;
	}
	// Else we need to compare

	int len = int(m_end - m_start);
	if (len != int(rhs.m_end - rhs.m_start))
	{
		return false;
	}
	const char*const c1 = m_start;
	const char*const c2 = rhs.m_start;

	for (int i = 0; i < len; i++)
	{
		if (c1[i] != c2[i])
		{
			return false;
		}
	}
	return true;
}

hkBool hkSubString::operator==(const char* rhs) const
{
	const char* cur = m_start;
	for (; cur < m_end && *rhs != 0; cur++, rhs++)
	{
		if (*cur != *rhs)
		{
			return false;
		}
	}
	// Check if at end
	return cur == m_end && rhs[0] == 0;
}

int hkSubString::getInt() const
{
	const int maxSize = 20;
	char buffer[maxSize + 1];
	// It's too big
	if (int(m_end - m_start) > maxSize)
	{
		return 0;
	}

	hkString::strNcpy(buffer, m_start, int(m_end - m_start));
	buffer[m_end - m_start] = 0;

	return hkString::atoi(buffer, 10);
}

hkOstream& operator<<(hkOstream& stream, const hkSubString& text)
{
	stream.write( text.m_start, text.length());
	return stream;
}

unsigned int hkSubString::calcHash() const
{
	const char* p = m_start;
	const char* end = m_end;

	hkUlong h = 0;
	for (; p < end; p++)
	{
		h = 31 * h + *p;
	}
	return (unsigned int)(h);
}

void hkSubString::operator=(const char* rhs)
{
	m_start = rhs;
	m_end = m_start + hkString::strLen(rhs);
}

int hkSubString::compareTo(const hkSubString& rhs) const
{
	const int lenA = length();
	const int lenB = rhs.length();
	int len = hkMath::min2(lenA, lenB);

	const char* a = m_start;
	const char* b = rhs.m_start;
	const char* end = a + len;

	while ( a < end && *a == *b)
	{
		a++;
		b++;
	}

	if (a >= end)
	{
		// Got to the end
		if (lenA == lenB)
		{
			return 0;
		}
		return (lenA < lenB) ? -1 : 1;
	}
	else
	{
		return (*a < *b) ? -1 : 1;
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
