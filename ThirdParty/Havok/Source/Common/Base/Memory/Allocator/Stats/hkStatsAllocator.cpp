/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Stats/hkStatsAllocator.h>

hkStatsAllocator::hkStatsAllocator(hkMemoryAllocator* a)
	: m_criticalSection(1000)
	, m_alloc(a)
{
	m_stats.m_allocated = 0;
	m_stats.m_inUse = 0;
	m_stats.m_peakInUse = 0;
	//m_stats.m_available = 0;
	//m_stats.m_totalAvailable = 0;
	//m_stats.m_largestBlock = 0;
}

void hkStatsAllocator::init(hkMemoryAllocator* a)
{
	HK_ASSERT(0x1c66d9f5, m_alloc == HK_NULL );
	m_alloc = a;
}

void* hkStatsAllocator::blockAlloc( int numBytes )
{
	hkCriticalSectionLock lock( &m_criticalSection );
	m_stats.m_allocated += numBytes;
	m_stats.m_inUse = m_stats.m_allocated;
	m_stats.m_peakInUse = hkMath::max2( m_stats.m_peakInUse, m_stats.m_inUse );

	return m_alloc->blockAlloc(numBytes);
}

void hkStatsAllocator::blockFree( void* p, int numBytes )
{
	hkCriticalSectionLock lock( &m_criticalSection );
	m_stats.m_allocated -= numBytes;
	m_alloc->blockFree(p, numBytes);
}

void hkStatsAllocator::getMemoryStatistics( MemoryStatistics& u ) const
{
	hkCriticalSectionLock lock( &m_criticalSection );
	u = m_stats;
}

void hkStatsAllocator::resetPeakMemoryStatistics()
{
	m_stats.m_peakInUse = m_stats.m_inUse;
}

int hkStatsAllocator::getAllocatedSize( const void* obj, int numBytes ) const
{
	return numBytes;
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
