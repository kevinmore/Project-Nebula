/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/BlockStream/Allocator/hkBlockStreamAllocator.h>
#if !defined(HK_PLATFORM_SPU)
#	include <Common/Base/Container/BlockStream/Allocator/Fixed/hkFixedBlockStreamAllocator.h>
#endif

extern HK_THREAD_LOCAL( int ) hkThreadNumber;

#if defined(HK_PLATFORM_SPU)
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#include <Common/Base/Spu/Dma/Buffer/hkDmaBuffer.h>
#endif

typedef hkBlockStreamAllocator::Block Block;

//
// Simple BSA
//

#if !defined(HK_PLATFORM_SPU)

hkFixedBlockStreamAllocator::hkFixedBlockStreamAllocator(int allocSize): m_criticalSection (4000)
{
	m_capacity = 0;
	m_dontDeallocate = false;
	m_enableAllocationTimerInfo = false;
	m_blocks = 0;
	if ( allocSize )
	{
		init( allocSize );
	}
}

void hkFixedBlockStreamAllocator::init( int byteSize )
{
	HK_ASSERT2( 0xf0345466, m_blocks==HK_NULL, "You cannot initialize a stream allocator twice" );
	m_capacity = byteSize/sizeof( Block );
	m_dontDeallocate = false;
	m_blocks = hkAlignedAllocate<Block>( Block::BLOCK_ALIGNMENT, m_capacity, HK_MEMORY_CLASS_COLLIDE );
	HK_ASSERT(0x6acbf233, (hkUlong(m_blocks) & (Block::BLOCK_ALIGNMENT-1) ) == 0);

	rebuildFreelist();
}

void hkFixedBlockStreamAllocator::init(void* buffer, int byteSize)
{
	HK_ASSERT2( 0xf0345465, m_blocks==HK_NULL, "You cannot initialize a stream allocator twice" );
	char* alignedBuffer = (char*)(HK_NEXT_MULTIPLE_OF(Block::BLOCK_ALIGNMENT, hkUlong(buffer)));
	int bytesUsedForAlignment = int(alignedBuffer - (char*)buffer);
	byteSize -= bytesUsedForAlignment;

	m_capacity = byteSize/sizeof( Block );
	m_blocks = (Block*)alignedBuffer;

	rebuildFreelist();

	m_dontDeallocate = true;
}

void hkFixedBlockStreamAllocator::rebuildFreelist()
{
	m_freeList.setSize( m_capacity );

	int d = 0;
	for(int i = m_capacity-1; i>=0; d++, i--)
	{
		Block* block = &m_blocks[ i ];
		//HK_ON_DEBUG( block->setHeaderToZero() );
		block->m_allocator = this;
		m_freeList[d] = block;
	}
	m_minFreeListSize = m_freeList.getSize();
}

hkFixedBlockStreamAllocator::~hkFixedBlockStreamAllocator()
{
	HK_ON_CPU( clear() );
	HK_ASSERT2( 0xf0345466, m_blocks==HK_NULL, "You must call clear before the destructor" );
}

void hkFixedBlockStreamAllocator::freeAllRemainingAllocations()
{
	if ( m_freeList.getSize() != m_capacity )
	{
		rebuildFreelist();
	}
	HK_ASSERT2(0xed4db99, m_freeList.getSize() == m_capacity, "Delete all the thread local cache?" );
}

void hkFixedBlockStreamAllocator::clear()
{
	HK_ASSERT2(0xed4db99, m_freeList.getSize() == m_capacity, "Delete all the thread local cache?" );
	if( !m_dontDeallocate )
	{
		hkAlignedDeallocate( m_blocks );
	}
	m_dontDeallocate = true;
	m_freeList.clear();
	m_capacity = 0;
	m_blocks = HK_NULL;
}

int hkFixedBlockStreamAllocator::getCapacity() const
{
	return m_capacity * sizeof(hkBlockStreamBase::Block);
}

int hkFixedBlockStreamAllocator::getBytesUsed() const
{
	int numAllocated = m_capacity - m_freeList.getSize();
	return numAllocated * sizeof (Block);
}

int hkFixedBlockStreamAllocator::getMaxBytesUsed() const
{
	int numAllocated = m_capacity - m_minFreeListSize;
	return numAllocated * sizeof (Block);
}

void hkFixedBlockStreamAllocator::getMemoryStatistics( hkMemoryAllocator::MemoryStatistics& statsOut ) const
{
	statsOut.m_allocated		= getCapacity();
	statsOut.m_inUse			= getBytesUsed();
	statsOut.m_peakInUse		= getMaxBytesUsed();
	statsOut.m_available		= statsOut.m_allocated - getBytesUsed();
	statsOut.m_totalAvailable	= statsOut.m_available;
	statsOut.m_largestBlock		= sizeof(Block);
}

#endif	// !HK_PLATFORM_SPU

HK_COMPILE_TIME_ASSERT((HK_OFFSET_OF( hkFixedBlockStreamAllocator, m_blocks ) > HK_OFFSET_OF( hkFixedBlockStreamAllocator, m_criticalSection )));

void hkFixedBlockStreamAllocator::blockAllocBatch( Block** blocksOut, int nblocks )
{
#if !defined(HK_PLATFORM_SPU)

	m_criticalSection.enter();

	//HK_ON_DEBUG( HK_MONITOR_ADD_VALUE("BlockStreamManagerCsAlloc", 1.f, HK_MONITOR_TYPE_INT) );

	for(int i=0; i<nblocks; i++)
	{
		if ( m_freeList.isEmpty() )
		{
			HK_ASSERT2( 0xf03df676, false, "Out of blockstream memory" );
			HK_BREAKPOINT( 0xf03df676 );
		}
		*blocksOut = m_freeList.back();
		m_freeList.popBack();
		blocksOut++;
	}
	if ( m_freeList.getSize() < m_minFreeListSize )
	{
		m_minFreeListSize = m_freeList.getSize();
	}
	if ( m_enableAllocationTimerInfo)
	{
		HK_MONITOR_ADD_VALUE( "MemSizeUsed", float(getBytesUsed()), HK_MONITOR_TYPE_INT );
	}

	m_criticalSection.leave();

#else

	hkCriticalSection::enter( &m_criticalSection );

	{
		// Transfer 'this' object from PPU skipping everything before m_storage to avoid wiping it on the SPU simulator
		// and overwriting it on the DMA back
		const int offset = HK_OFFSET_OF(hkFixedBlockStreamAllocator, m_blocks);
		const int transferSize = sizeof(hkFixedBlockStreamAllocator) - offset;
		void* allocatorPpu = ((char*)this) + offset;
		hkDmaBuffer<transferSize> buffer(HK_DMA_WAIT, allocatorPpu, hkSpuDmaManager::READ_WRITE);
		hkFixedBlockStreamAllocator* allocator = (hkFixedBlockStreamAllocator*)(((char*)buffer.getContents()) - offset);

		// Get nblocks pre-allocated blocks
		for(int i=0; i<nblocks; i++)
		{
			void** blockPntr = (void**)&allocator->m_freeList.back();
			*blocksOut = (Block*)hkSpuDmaUtils::getPntrFromMainMemory( blockPntr );
			allocator->m_freeList.popBack();
			blocksOut++;
		}

		// Update the minimum size of the free list
		allocator->m_minFreeListSize = hkMath::min2( allocator->m_minFreeListSize, allocator->m_freeList.getSize() );

		buffer.dmaPutAndWait(allocatorPpu, hkSpuDmaManager::WRITE_BACK);
	}

	hkCriticalSection::leave( &m_criticalSection );

#endif
}

void hkFixedBlockStreamAllocator::blockFreeBatch( Block** blocks, int nblocks )
{
	if( !nblocks )
	{
		return;
	}
#if !defined(HK_PLATFORM_SPU)
	m_criticalSection.enter();
	//HK_ON_DEBUG( HK_MONITOR_ADD_VALUE("BlockStreamManagerCsFree", 1.f, HK_MONITOR_TYPE_INT) );

	for(int i=0; i<nblocks; blocks++, i++)
	{
		Block* block = *blocks;
		if( block )
		{
			HK_ASSERT( 0xf04df1d4, block->m_allocator == this );
			m_freeList.pushBackUnchecked( block );
		}
	}
	HK_ASSERT( 0xf0457678, m_freeList.getSize() <= m_capacity );
	if ( m_enableAllocationTimerInfo)
	{
		HK_MONITOR_ADD_VALUE( "MemSizeUsed", float(getBytesUsed()), HK_MONITOR_TYPE_INT );
	}
	m_criticalSection.leave();
#else
	hkCriticalSection::enter( &m_criticalSection );

	{
		// Transfer 'this' object from PPU skipping everything before m_storage to avoid wiping it on the SPU simulator
		// and overwriting it on the DMA back
		const int offset = HK_OFFSET_OF(hkFixedBlockStreamAllocator, m_blocks);
		const int transferSize = sizeof(hkFixedBlockStreamAllocator) - offset;
		void* allocatorPpu = ((char*)this) + offset;
		hkDmaBuffer<transferSize> buffer(HK_DMA_WAIT, allocatorPpu, hkSpuDmaManager::READ_WRITE);
		hkFixedBlockStreamAllocator* allocator = (hkFixedBlockStreamAllocator*)(((char*)buffer.getContents()) - offset);

		for(int i=0; i<nblocks; blocks++, i++)
		{
			Block* block = *blocks;
			if( !block )
			{
				continue;
			}
			Block** blockPOnPPu = allocator->m_freeList.expandByUnchecked( 1 );
			hkSpuDmaUtils::setPntrInMainMemory( (void**)blockPOnPPu, block );
		}

		buffer.dmaPutAndWait(allocatorPpu, hkSpuDmaManager::WRITE_BACK);
	}

	hkCriticalSection::leave( &m_criticalSection );
#endif
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
