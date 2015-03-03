/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>

// this
#include <Common/GeometryUtilities/Mesh/Utils/FindUniquePositionsUtil/hkFindUniquePositionsUtil.h>

void hkFindUniquePositionsUtil::reset(int sizeEstimate)
{
	m_positions.clear(); m_positions.reserve(sizeEstimate);
	m_hashMap.clear(); m_hashMap.reserve(sizeEstimate);
	m_entries.clear(); m_entries.reserve(sizeEstimate);
}


static HK_FORCE_INLINE hkBool32 hkFindUniquePositionsUtil_equals(const hkVector4& a, const hkVector4& b)
{
    //HK_ASSERT(0x8d7292c1, sizeof(a(0)) == sizeof(hkUint32));
    //const hkUint32* ia = (const hkUint32*)&a(0);
    //const hkUint32* ib = (const hkUint32*)&b(0);

    //return ((ia[0] ^ ib[0]) | (ia[1] ^ ib[1]) | (ia[2] ^ ib[2])) == 0;

	// Using SIMD is probably better as you want -0 == 0 which doesn't work with the binary version. The integer version
	// may be faster.

	return a.allExactlyEqual<3>(b);
}

int hkFindUniquePositionsUtil::addPosition(const hkVector4& pos)
{
    const hkUint32 hash = hashVector(pos);

    const hkPointerMap<hkUint32, int>::Iterator iter =  m_hashMap.findKey(hash);
	const int positionIndex = m_positions.getSize();

    if (!m_hashMap.isValid(iter))
    {
        // Not found so just add it
        m_positions.pushBack(pos);

        const int entryIndex = m_entries.getSize();
        Entry& entry = m_entries.expandOne();
        entry.m_positionIndex = positionIndex;
        entry.m_nextEntryIndex = -1;

		m_hashMap.insert(hash, entryIndex);
    }
    else
    {
        const int index = m_hashMap.getValue(iter);
		{
			Entry* curEntry = &m_entries[index];
			while (true)
			{
				if (hkFindUniquePositionsUtil_equals(m_positions[curEntry->m_positionIndex], pos))
				{
					// Found
					return curEntry->m_positionIndex;
				}

				if (curEntry->m_nextEntryIndex < 0)
				{
					// We are at the end
					break;
				}
				curEntry = &m_entries[curEntry->m_nextEntryIndex];
			}
		}

        // Not found, create a new entry
        m_positions.pushBack(pos);

        const int entryIndex = m_entries.getSize();
        Entry& entry = m_entries.expandOne();

		// Link in to the start (must be done here after the list has been expanded)
		Entry& start = m_entries[index];

        entry.m_positionIndex = positionIndex;
        entry.m_nextEntryIndex = start.m_nextEntryIndex;

		start.m_nextEntryIndex = entryIndex;
    }

	return positionIndex;
}

void hkFindUniquePositionsUtil::addPositions(const hkVector4* pos, int numPos)
{
	const hkVector4* end = pos + numPos;
	for (; pos != end; pos++)
	{
		addPosition(*pos);
	}
}

int hkFindUniquePositionsUtil::findPosition(const hkVector4& pos) const
{
    const hkUint32 hash = hashVector(pos);

    const hkPointerMap<hkUint32, int>::Iterator iter =  m_hashMap.findKey(hash);
    if (m_hashMap.isValid(iter))
    {
        const int index = m_hashMap.getValue(iter);
        const Entry* curEntry = &m_entries[index];

        while (true)
        {
            if (hkFindUniquePositionsUtil_equals(m_positions[curEntry->m_positionIndex], pos))
            {
				// Found
                return curEntry->m_positionIndex;
            }

            if (curEntry->m_nextEntryIndex < 0)
            {
                // We are at the end
                break;
            }
            curEntry = &m_entries[curEntry->m_nextEntryIndex];
        }
    }

	// Not found
	return -1;
}

int HK_CALL hkFindUniquePositionsUtil::getSizeInBytesFor( int N )
{
	int pointerMapSize = hkPointerMap<hkUint32, int>::getSizeInBytesFor(N);
	return ( N * sizeof(hkVector4) ) 
		+ HK_NEXT_MULTIPLE_OF(16, N * sizeof(Entry))
		+ HK_NEXT_MULTIPLE_OF(16, pointerMapSize );
}

void hkFindUniquePositionsUtil::setBuffer( void* _buffer, int numPoints )
{
	HK_CHECK_ALIGN16(_buffer);

	char* buffer = (char*) _buffer;
	hkVector4* pointData = reinterpret_cast<hkVector4*> (buffer);

	int entryOffset =  numPoints * sizeof(hkVector4) ;
	Entry* entryData = reinterpret_cast<Entry*> (buffer+entryOffset);

	int pointerMapOffset = entryOffset + HK_NEXT_MULTIPLE_OF(16, numPoints * sizeof(Entry));
	void* pointerMapData = (void*) (buffer+pointerMapOffset);

	m_positions.setDataUserFree(pointData, 0, numPoints);
	m_entries.setDataUserFree(entryData, 0, numPoints);

	// inplace new - can't initialize pointer map
	new (&m_hashMap) hkPointerMap<hkUint32, int>( pointerMapData, hkPointerMap<hkUint32, int>::getSizeInBytesFor(numPoints) );
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
