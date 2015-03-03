/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

#include <Common/Base/Memory/Allocator/Solver/hkSolverAllocator.h>

hkSolverAllocator::hkSolverAllocator()
	: m_bufferStart(HK_NULL)
	, m_bufferEnd(HK_NULL)
	, m_currentEnd(HK_NULL)
	, m_numAllocatedBlocks(0)
	, m_peakUse(0)
	, m_freeElems(&m_elemsBuf[0], 0, HK_COUNT_OF(m_elemsBuf))
{
}

hkSolverAllocator::~hkSolverAllocator()
{
}

void hkSolverAllocator::setBuffer( void* vbuffer, int bufferSize )
{
	hkCriticalSectionLock lock(&m_criticalSection);

	m_freeElems.setDataUserFree( &m_elemsBuf[0], 0, HK_COUNT_OF(m_elemsBuf));

	HK_ASSERT2( 0xf0321255, m_freeElems.isEmpty(), "hkSolverAllocator still has memory allocated" );
	HK_ASSERT2( 0xf0321254, 0 == int (hkUlong(vbuffer) & (HK_REAL_ALIGNMENT-1)), "Your input buffer must be aligned for SIMD" );
	if(!vbuffer)
	{
		HK_ASSERT2( 0x36275c25, bufferSize == 0, "hkSolverAllocator requires 0 size for a NULL buffer");
		bufferSize = 0;
	}

	char* buffer = static_cast<char*>(vbuffer);
	m_bufferStart = buffer;
	m_currentEnd = buffer;
	m_bufferEnd = buffer + bufferSize;
	m_numAllocatedBlocks = 0;
}

void* hkSolverAllocator::blockAlloc( int numBytes )
{
	if (numBytes)
	{
		numBytes = HK_NEXT_MULTIPLE_OF(128, numBytes);
		int size = numBytes;

		void* result = allocate(size, true);
		HK_ASSERT(0x65587bdb, size == numBytes);
		return result;
	}
	else
	{
		return HK_NULL;
	}
}

void hkSolverAllocator::blockFree( void* p, int numBytes )
{
	if (p && numBytes)
	{
		bufFree(p, HK_NEXT_MULTIPLE_OF(128, numBytes));
	}
}

void* hkSolverAllocator::bufAlloc(int& reqNumInOut)
{
	return allocate(reqNumInOut, false);
}

void* hkSolverAllocator::allocate( int& reqNumInOut, bool useExactSize )
{
	HK_ASSERT2( 0xf034dff6, (reqNumInOut&(HK_REAL_ALIGNMENT-1)) == 0, "Your size must be aligned for SIMD");

	hkCriticalSectionLock lock(&m_criticalSection);
	int size = reqNumInOut;

	// search smallest existing elements which still fits our size
	int bestSize = 0x7ffffff;
	int bestIndex = -1;

	if ( m_numAllocatedBlocks+1 >= 2*m_freeElems.getCapacity() - m_freeElems.getSize() )
	{
		// not enough space on the free list
		// above formula explained:
		// we get the maximum freelist usage if the buffer is split into
		// n blocks and every other block is in the free list.
		// So as a result freeList.getSize() + numAllocated <= freeList.getCapacity()*2
		return HK_NULL;
	}

	for (int i =  m_freeElems.getSize()-1; i>=0; i-- )
	{
		int elemSize = m_freeElems[i].m_size;
		if ( elemSize < size )
		{
			continue;
		}
		if ( elemSize >= bestSize )
		{
			continue;
		}
		bestSize = elemSize;
		bestIndex = i;
	}

	if ( bestIndex < 0 )
	{
		// no free element found
		HK_ASSERT2( 0xf032de54, m_bufferEnd - m_bufferStart >= size, "You requested a solver buffer allocation, which is bigger than the solver buffer size, please run Havok with Memory Limiting enabled or increase your solver buffer size (see hkSolverAllocator)" );
		int remainingFree = int(m_bufferEnd - m_currentEnd);
		if ( remainingFree >= size )
		{
			char* s = m_currentEnd;
			m_currentEnd += size;
			reqNumInOut = size;
			m_numAllocatedBlocks++;
			{
				hk_size_t inUse = hk_size_t(m_currentEnd - m_bufferStart);
				if (inUse > m_peakUse) m_peakUse = inUse;
			}
			return s;
		}
		return HK_NULL;
	}

	// split the block remaining bit is bigger than 128 and 30%
	Element& e = m_freeElems[bestIndex];
	HK_ASSERT ( 0xf0325dfe, e.m_size >= size );
	char* s = e.m_start;

	if ((useExactSize && bestSize != size) || 
		((bestSize * 2 > size * 3) && size > 1024))
	{
		reqNumInOut = size;
		e.m_size -= size;
		e.m_start += size;
	}
	else
	{
		reqNumInOut = e.m_size;
		m_freeElems.removeAtAndCopy(bestIndex);
	}
	m_numAllocatedBlocks++;
	{
		hk_size_t inUse = hk_size_t(m_currentEnd - m_bufferStart);
		if (inUse > m_peakUse) m_peakUse = inUse;
	}
	return s;
}


void hkSolverAllocator::bufFree(void* data, int size)
{
	hkCriticalSectionLock lock(&m_criticalSection);
	m_numAllocatedBlocks--;

	char* start = (char*) data;
	char* end = start + size;
	HK_ASSERT( 0xf0dfed34, end <= m_bufferEnd && start >= m_bufferStart && size > 0  );

	// check if we are at the end of the allocated space
	if ( end == m_currentEnd )
	{
		m_currentEnd -= size;
		while ( m_freeElems.getSize() && m_freeElems.back().getEnd() == start )
		{
			m_currentEnd = m_freeElems.back().m_start;
			m_freeElems.popBack();
		}
		return;
	}

	if (!m_freeElems.getSize())
	{
		Element& e = *m_freeElems.expandByUnchecked(1);
		e.m_start = start;
		e.m_size = size;
		return;
	}

	// find the insertion point
	int insertPoint;
	for (insertPoint = m_freeElems.getSize()-1; insertPoint>=0; insertPoint-- )
	{
		Element& e = m_freeElems[insertPoint];
		if ( e.m_start < start )
		{
			break;
		}
	}

	// now m_freeElems[insertPoint].m_start < start
	if ( insertPoint >= 0 )
	{
		Element& e = m_freeElems[insertPoint];
		// merge with left object
		if ( e.getEnd() == start )
		{
			e.m_size += size;
			// check merge with right
			if ( insertPoint < m_freeElems.getSize()-1 && end == m_freeElems[insertPoint+1].m_start )
			{
				e.m_size += m_freeElems[insertPoint+1].m_size;
				m_freeElems.removeAtAndCopy(insertPoint+1);
			}
			return;
		}
	}
	if ( insertPoint < m_freeElems.getSize()-1  )
	{
		Element& e = m_freeElems[insertPoint+1];
		if ( e.m_start == end )
		{
			e.m_size += size;
			e.m_start = start;
			return;
		}
	}

	HK_ASSERT(0x3b273670, m_freeElems.getSize() < m_freeElems.getCapacity());
	// add a new element
	{
		Element e;
		e.m_start = start;
		e.m_size = size;
		m_freeElems.expandByUnchecked(1);
		for(int i = m_freeElems.getSize() - 1; i > (insertPoint + 1); i--)
		{
			m_freeElems[i] = m_freeElems[i-1];
		}
		m_freeElems[insertPoint + 1] = e;
	}

	return;
}

bool hkSolverAllocator::canAllocSingleBlock( int numBytes )
{
	return numBytes <= int(m_bufferEnd - m_bufferStart);
}

void hkSolverAllocator::getMemoryStatistics( hkMemoryAllocator::MemoryStatistics& u ) const
{
	u.m_allocated = hkGetByteOffset( m_bufferStart, m_bufferEnd );
	u.m_available = hk_size_t(m_bufferEnd - m_currentEnd);
	u.m_inUse = hk_size_t(m_currentEnd - m_bufferStart);
	u.m_peakInUse = m_peakUse;
}

void hkSolverAllocator::resetPeakMemoryStatistics()
{
	m_peakUse = hk_size_t(m_currentEnd - m_bufferStart);
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
