/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/BlockStream/Allocator/Dynamic/hkDynamicBlockStreamAllocator.h>

hkDynamicBlockStreamAllocator::hkDynamicBlockStreamAllocator( int initialSize )
	: m_maxBytesUsed(0)
{
#ifdef HK_PLATFORM_PS3
	HK_ASSERT( 0x2313eaff, !"This allocator is not yet supported on PS3 (SPU)" );
	return;
#endif
	if( initialSize > 0 )
	{
		expand( initialSize );
	}
}

hkDynamicBlockStreamAllocator::~hkDynamicBlockStreamAllocator()
{
	for( int i = 0; i < m_blocks.getSize(); ++i )
	{
		hkAlignedDeallocate<Block>( m_blocks[i] );
	}
}

void hkDynamicBlockStreamAllocator::blockAllocBatch( Block** blocksOut, int nblocks )
{
	if( nblocks == 0 )
	{
		return;
	}

	m_criticalSection.enter();
	{
		// If there are not enough blocks in the free list, we need to allocate more blocks.
		if( m_freeList.getSize() < nblocks )
		{
			// By default we will double the capacity.
			// But we need to make sure we allocate enough blocks for the request to be fulfilled.
			int expandSize = hkMath::max2( getCapacity(), (nblocks - m_freeList.getSize()) * sizeof(Block) );
			expand( expandSize );
		}

		for( int i = 0; i < nblocks; ++i )
		{
			blocksOut[i] = m_freeList.back();
			HK_ASSERT( 0x0079fa42, blocksOut[i]->m_allocator == this );
			m_freeList.popBack();
		}

		m_maxBytesUsed = hkMath::max2( m_maxBytesUsed, getBytesUsed() );
	}
	m_criticalSection.leave();
}

void hkDynamicBlockStreamAllocator::blockFreeBatch( Block** blocks, int nblocks )
{
	if( nblocks == 0 )
	{
		return;
	}

	m_criticalSection.enter();
	{
		for( int i = 0 ; i < nblocks; ++i )
		{
			if( blocks[i] )
			{
				HK_ASSERT( 0x0079fa43, blocks[i]->m_allocator == this );
				m_freeList.pushBack( blocks[i] );
			}
		}
	}
	m_criticalSection.leave();
}

int hkDynamicBlockStreamAllocator::getBytesUsed() const
{
	return (m_blocks.getSize() - m_freeList.getSize()) * sizeof(Block);
}

int hkDynamicBlockStreamAllocator::getMaxBytesUsed() const
{
	return m_maxBytesUsed;
}

int hkDynamicBlockStreamAllocator::getCapacity() const
{
	return m_blocks.getSize() * sizeof(Block);
}

void hkDynamicBlockStreamAllocator::freeAllRemainingAllocations()
{
	HK_ASSERT2(0x0079fa43,m_freeList.getSize() == m_blocks.getSize(), "Trying to free blocks that are still used.");
	m_criticalSection.enter();
	{
		for( int i = 0; i < m_blocks.getSize(); ++i )
		{
			hkAlignedDeallocate( m_blocks[i] );
		}
		m_blocks.clear();
		m_freeList.clear();
	}
	m_criticalSection.leave();
}

void hkDynamicBlockStreamAllocator::getMemoryStatistics( hkMemoryAllocator::MemoryStatistics& statsOut ) const
{
	statsOut.m_allocated		= getCapacity();
	statsOut.m_inUse			= getBytesUsed();
	statsOut.m_peakInUse		= getMaxBytesUsed();
	statsOut.m_available		= statsOut.m_allocated - getBytesUsed();
	statsOut.m_totalAvailable	= statsOut.m_available;
	statsOut.m_largestBlock		= sizeof(Block);
}

void hkDynamicBlockStreamAllocator::releaseUnusedBlocks()
{
	m_criticalSection.enter();
	{
		// Sort both pointer lists for easier processing
		hkAlgorithm::quickSort( m_blocks.begin(), m_blocks.getSize() );
		hkAlgorithm::quickSort( m_freeList.begin(), m_freeList.getSize() );
		int storageIdx = 0;
		int freeListIdx = 0;
		hkArray<Block*> newStorage;

		while( storageIdx < m_blocks.getSize() )
		{
			if( freeListIdx < m_freeList.getSize() && m_blocks[storageIdx] == m_freeList[freeListIdx] )
			{
				// The current block is free, so we give it back
				hkAlignedDeallocate( m_blocks[storageIdx] );
				++freeListIdx;
			}
			else
			{
				newStorage.pushBack( m_blocks[storageIdx] );
			}
			++storageIdx;
		}

		HK_ASSERT( 0x298ffa37, newStorage.getSize() + m_freeList.getSize() == m_blocks.getSize() );
		m_blocks = newStorage;

		// All blocks in the free list have been freed
		m_freeList.clear();
	}
	m_criticalSection.leave();
}

void hkDynamicBlockStreamAllocator::expand( int numBytes )
{
	const int numBlocks = HK_NEXT_MULTIPLE_OF( sizeof(Block), numBytes ) / sizeof(Block);
	for( int i = 0; i < numBlocks; ++i )
	{
		Block* block = hkAlignedAllocate<Block>( Block::BLOCK_ALIGNMENT, 1, HK_MEMORY_CLASS_BASE );
		HK_ASSERT2( 0x93a64de4, block != HK_NULL, "Out of Memory" );
		block->m_allocator = this;
		m_blocks.pushBack( block );
		m_freeList.pushBack( block );
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
