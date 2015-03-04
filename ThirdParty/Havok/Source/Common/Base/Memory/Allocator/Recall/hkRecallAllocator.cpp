/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Recall/hkRecallAllocator.h>

hkRecallAllocator::hkRecallAllocator(hkMemoryAllocator* a)
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

void hkRecallAllocator::init(hkMemoryAllocator* a)
{
	HK_ASSERT(0x1c66d9f5, m_alloc == HK_NULL );
	m_alloc = a;
}

void* hkRecallAllocator::blockAlloc( int numBytes )
{
	hkCriticalSectionLock lock( &m_criticalSection );

	HK_COMPILE_TIME_ASSERT( sizeof(Header) <= 16 );
	int request = HK_NEXT_MULTIPLE_OF(16, numBytes) + 16;
	void* block = m_alloc->blockAlloc( request );

	Header* a = reinterpret_cast<Header*>(block);
	a->m_payloadSize = numBytes;
	a->m_requestedSize = request;
	a->m_next = m_head;
	m_head = a;

	m_stats.m_allocated += request;
	m_stats.m_inUse += numBytes;
	m_stats.m_peakInUse = hkMath::max2( m_stats.m_peakInUse, m_stats.m_inUse );

	return a->getPayload();
}

void hkRecallAllocator::blockFree( void* p, int numBytes )
{
	if(p == HK_NULL)
	{
		// Some Havok classes (hkMap) can deallocate an empty block.
		HK_ASSERT(0x35a765d8, numBytes == 0);
		return;
	}
	hkCriticalSectionLock lock( &m_criticalSection );

	int request = HK_NEXT_MULTIPLE_OF(16, numBytes) + 16;
	Header* toFree = reinterpret_cast<Header*>( hkAddByteOffset(p,-16) );
	HK_ASSERT(0x19e086e0, toFree->getPayload() == p );
	HK_ASSERT(0x2ec72e1c, toFree->m_payloadSize == numBytes );

	// Remove from singly linked list
	// Use a placeholder so we don't have to treat the head specially
	Header tmpHead;
	tmpHead.m_next = m_head;
	tmpHead.m_payloadSize = -1;
	tmpHead.m_requestedSize = -1;
	for( Header* cur = &tmpHead; cur->m_next != HK_NULL; cur=cur->m_next )
	{
		if( cur->m_next == toFree )
		{
			// unlink
			cur->m_next = toFree->m_next;
			// maybe the head was deleted
			m_head = tmpHead.m_next;
			// update stats
			m_stats.m_allocated -= request;
			m_stats.m_inUse -= numBytes;
			// free
			m_alloc->blockFree(toFree, toFree->m_requestedSize);
			// early out
			return;
		}
	}
	HK_ASSERT(0x7b1047b0,0); // not found?
}

void hkRecallAllocator::getMemoryStatistics( MemoryStatistics& u ) const
{
	hkCriticalSectionLock lock( &m_criticalSection );
	u = m_stats;
}

void hkRecallAllocator::resetPeakMemoryStatistics()
{
	m_stats.m_peakInUse = m_stats.m_inUse;
}

int hkRecallAllocator::getAllocatedSize(const void* obj, int numBytes) const
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
