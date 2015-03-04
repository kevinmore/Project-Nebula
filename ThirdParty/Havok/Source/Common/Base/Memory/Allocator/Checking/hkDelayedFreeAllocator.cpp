/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Checking/hkDelayedFreeAllocator.h>
#include <Common/Base/Memory/System/Util/hkMemorySnapshot.h>

hkDelayedFreeAllocator::hkDelayedFreeAllocator()
	: m_next(HK_NULL)
	, m_prng(0)
{
}

void hkDelayedFreeAllocator::init(hkMemoryAllocator* next, Limits* limits)
{
	Limits defaultLimits;
	m_next = next;
	m_kept._reserveExactly( *next, m_limits.m_numAllocsKept );
	m_limits = limits ? *limits : defaultLimits;
	m_prng.setSeed( m_limits.m_randomSeed );
	m_curBytesKept = 0;
}

void hkDelayedFreeAllocator::quit()
{
	releaseDelayedFrees();
	m_kept._clearAndDeallocate(*m_next);
	m_next = HK_NULL;
}

void hkDelayedFreeAllocator::releaseDelayedFrees()
{
	if (m_next)
	{
		for( int i = 0; i < m_kept.getSize(); ++i )
		{
			const Alloc& a = m_kept[i];
			m_next->blockFree(a.p, a.size);
		}
		m_kept.clear();
		m_curBytesKept = 0;
	}
}

void* hkDelayedFreeAllocator::blockAlloc( int numBytes )
{
	m_inUse += numBytes;
	return m_next->blockAlloc( numBytes );
}

static hkBool32 blockOk( const void* p, int size )
{
	HK_COMPILE_TIME_ASSERT(sizeof(int)==4);
	for( int i = 0; i < size/4; ++i )
	{
		if( static_cast<const int*>(p)[i] != 0xfeee )
		{
			return false;
		}
	}
	return true;
}

void hkDelayedFreeAllocator::blockFree( void* pfree, int numBytes )
{
	m_inUse -= numBytes;

#if 0 && defined(HK_DEBUG)
	int total = 0;
	for( int i = 0; i < m_kept.getSize(); ++i )
	{
		total += m_kept[i].size;
	}
	HK_ASSERT(0x1f377c7d, total == m_curBytesKept );
#endif

	if( numBytes > m_limits.m_blockTooBig )
	{
		m_next->blockFree( pfree, numBytes ); // keeping such a large block is probably a waste
		return;
	}
	
	
	while( numBytes + m_curBytesKept > m_limits.m_maxBytesKept || m_kept.getSize() >= m_limits.m_numAllocsKept )
	{
		int kill = m_prng.getRand32() % m_kept.getSize();
		Alloc a = m_kept[kill];
		m_kept.removeAt(kill);
		if( blockOk( a.p, a.size) == hkFalse32 )
		{
			// memory modified after being freed
			HK_BREAKPOINT(0);
		}
		m_curBytesKept -= a.size;
		m_next->blockFree(a.p, a.size);
	}


#if defined(HK_DELAYED_FREE_LIST_ALLOCATOR_ENABLE_SERIAL_NUMBER)
	static int serial = 0;
	Alloc a = {pfree, numBytes, ++serial};
	if ( serial == 0xc38 )	// check for your serial number of you bad allocation
	{
		serial = serial;	// and set a breakpoint
	}
#else
	Alloc a = { pfree, numBytes };
#endif

	hkString::memSet4( pfree, 0xfeee, numBytes/4 );
	m_curBytesKept += numBytes;
	m_kept.pushBackUnchecked( a);
}

hkBool32 hkDelayedFreeAllocator::isOk() const
{
	for( int i = 0; i < m_kept.getSize(); ++i )
	{
		if( blockOk( m_kept[i].p, m_kept[i].size ) == hkFalse32 )
		{
			return false;
		}
	}
	return true;
}

void hkDelayedFreeAllocator::getMemoryStatistics( hkMemoryAllocator::MemoryStatistics& u ) const
{
	u.m_allocated = m_inUse + m_curBytesKept + (m_kept.getCapacity() * hkSizeOf(Alloc));
	u.m_inUse = m_inUse;
	u.m_available = 0;
}

int hkDelayedFreeAllocator::getAllocatedSize( const void* obj, int numBytes ) const
{
	return m_next->getAllocatedSize( obj, numBytes );
}

int hkDelayedFreeAllocator::addToSnapshot(hkMemorySnapshot& snapshot, int parentId) const
{
	int thisId = snapshot.addProvider("hkDelayedFreeAllocator", parentId);
	if(m_kept.getCapacity() > 0)
	{
		snapshot.addOverhead( thisId, m_kept.begin(), m_kept.getCapacity()*sizeof(Alloc) );
		for( int i = 0; i < m_kept.getSize(); ++i )
		{
			// show the delayed frees as overhead
			snapshot.addOverhead( thisId, m_kept[i].p, m_kept[i].size );
		}
	}
	return thisId;
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
