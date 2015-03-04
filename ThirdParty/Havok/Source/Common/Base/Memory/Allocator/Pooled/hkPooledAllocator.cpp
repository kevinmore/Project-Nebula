/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Pooled/hkPooledAllocator.h>

hkPooledAllocator::hkPooledAllocator()
: m_statsAllocated(0), m_statsUsed(0), m_statsPeak(0), m_large(HK_NULL), m_block(HK_NULL), m_blockSize(0)
{
}

void hkPooledAllocator::init(hkMemoryAllocator* internal, hkMemoryAllocator* large, hkMemoryAllocator* block, int blockSize, void *initialBlock, int initialBlockSize)
{
	m_curPtr = initialBlock;
	m_curBytesLeft = initialBlockSize;

	m_internal = internal;
	m_large = large;
	m_block = block;

	if( blockSize == 0 ) // autodetect block size
	{
		void* p = m_block->bufAlloc(blockSize);
		m_block->bufFree(p, blockSize);
	}
	m_blockSize = blockSize;
	HK_ASSERT2(0x728f0c5b, blockSize > 1024, "blockSize is probably too small to be useful");
	HK_ASSERT2(0x728f0c5c, (blockSize&0xf)==0, "blockSize must be a multiple of 16");

	m_statsAllocated = 0;
	m_statsUsed = 0;
	m_statsPeak = 0;
}

void hkPooledAllocator::quit()
{
	for( int i = 0; i < m_blockAllocs.getSize(); ++i )
	{
		m_block->bufFree( m_blockAllocs[i], m_blockSize );
	}
	m_blockAllocs._clearAndDeallocate(*m_internal);
	for( hkMapBase<void*, int>::Iterator it = m_largeAllocs.getIterator();
		m_largeAllocs.isValid(it); it = m_largeAllocs.getNext(it) )
	{
		void* p = m_largeAllocs.getKey(it);
		int s = m_largeAllocs.getValue(it);
		m_large->blockFree(p, s);
	}
	m_largeAllocs.clearAndDeallocate(*m_internal);

	m_block = HK_NULL;
	m_large = HK_NULL;
	m_blockSize = 0;
}


void* hkPooledAllocator::blockAlloc( int numBytes )
{
	// Need to keep a minimum alignment, serialization can
	// request as little as 1 byte as a time
	numBytes = HK_NEXT_MULTIPLE_OF(MIN_ALIGNMENT, numBytes);
	if(numBytes < MAX_ALIGNMENT)
	{
		numBytes = hkNextPowerOf2(numBytes);
	}

	// extra offset due to alignment requirements
	// there are quite a lot of small allocations (<16 bytes) so aligning
	// everything to 16 bytes will waste a lot of space
	int offset = static_cast<int>((HK_NEXT_MULTIPLE_OF((numBytes < MAX_ALIGNMENT ? numBytes : MAX_ALIGNMENT), reinterpret_cast<hkUlong>(m_curPtr))) - reinterpret_cast<hkUlong>(m_curPtr));
	// update stats
	m_statsUsed += numBytes;
	m_statsPeak = hkMath::max2(m_statsPeak, m_statsUsed);

	if( m_curBytesLeft >= numBytes + offset ) 
	{
		// It fits in our current block, let's do that.
		m_curBytesLeft -= offset;
		m_curPtr = static_cast<char *>(m_curPtr) + offset;
		// Check alignment
		void* p = static_cast<char *>(m_curPtr);
		HK_ASSERT(0x49f3b3a7, (reinterpret_cast<hkUlong>(p) & ((numBytes < MAX_ALIGNMENT ? numBytes : MAX_ALIGNMENT)-1)) == 0);
		m_curBytesLeft -= numBytes;
		m_curPtr = hkAddByteOffset(m_curPtr, numBytes);

		return p;
	}
	else if( numBytes <= m_blockSize ) 
	{
		// It doesn't fit, make a new block
		// If the block size is too small, handling allocs which of nearly
		// the same size (say 25%+) can lead to lots of wasted space

		void* b = m_block->bufAlloc(m_blockSize);
		m_statsAllocated += m_blockSize;
		m_blockAllocs._pushBack(*m_internal, b);

		if( m_blockSize - numBytes > m_curBytesLeft ) // take larger leftover bit
		{
			m_curBytesLeft = m_blockSize - numBytes;
			m_curPtr = hkAddByteOffset(b, numBytes);
		}

		return b;
	}
	else // must be large
	{
		void* r = m_large->bufAlloc(numBytes);
		m_statsAllocated += numBytes;
		m_largeAllocs.insert(*m_internal, r, numBytes);

		return r;
	}
}


void hkPooledAllocator::blockFree( void* pfree, int numBytes )
{
	numBytes = HK_NEXT_MULTIPLE_OF(MIN_ALIGNMENT, numBytes);
	if(numBytes < MAX_ALIGNMENT)
	{
		numBytes = hkNextPowerOf2(numBytes);
	}
	m_statsUsed -= numBytes;

	// We don't bother tracking small allocations
	// This data is not persistent anyway and will go
	// away when the loading process is complete

	if(numBytes > m_blockSize)
	{
		// Large block
		m_statsAllocated -= numBytes;
		m_large->blockFree(pfree, numBytes);
		m_largeAllocs.remove(pfree);
	}
}

hkBool32 hkPooledAllocator::isOk() const
{
	return true;
}

void hkPooledAllocator::getMemoryStatistics( hkPooledAllocator::MemoryStatistics& u ) const
{
	u.m_allocated = m_statsAllocated;
	u.m_available = m_curBytesLeft;
	u.m_inUse = m_statsUsed;
	u.m_peakInUse = m_statsPeak;
	u.m_totalAvailable = hkMemoryAllocator::MemoryStatistics::INFINITE_SIZE;
	u.m_largestBlock = hkMemoryAllocator::MemoryStatistics::INFINITE_SIZE;
}

int hkPooledAllocator::getAllocatedSize(const void* obj, int numBytes) const
{
	if(numBytes < MIN_ALIGNMENT)
	{
		return hkNextPowerOf2(HK_NEXT_MULTIPLE_OF(MIN_ALIGNMENT, numBytes));
	}
	else if( numBytes <= m_blockSize )
	{
		return HK_NEXT_MULTIPLE_OF(MAX_ALIGNMENT, numBytes);
	}
	return m_large->getAllocatedSize( obj, numBytes );
}

bool hkPooledAllocator::purge()
{
	if ( m_statsUsed != 0 )
	{
		// Cannot purge, there still are active allocations
		return false;
	}

	for( int i = 0; i < m_blockAllocs.getSize(); ++i )
	{
		m_block->bufFree( m_blockAllocs[i], m_blockSize );
	}
	m_blockAllocs._clearAndDeallocate(*m_internal);
	m_curPtr = HK_NULL;
	m_curBytesLeft = 0;
	m_statsAllocated = 0;

	return true;
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
