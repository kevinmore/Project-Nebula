/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>
extern HK_THREAD_LOCAL( int ) hkThreadNumber;

typedef hkBlockStreamAllocator::Block Block;

Block* hkThreadLocalBlockStreamAllocator::blockAlloc()
{
#if defined(HK_DEBUG) && !defined(HK_PLATFORM_HAS_SPU)
	HK_ASSERT2( 0xf04f4565, m_threadId == -1 || m_threadId == HK_THREAD_LOCAL_GET(hkThreadNumber),
		"Your thread local allocator belongs to a different thread" );
	checkConsistency();
#endif

	hkBlockStreamAllocator::Block* block;
	if( m_numFreeBlocks > 0 )
	{
		m_numFreeBlocks = m_numFreeBlocks - 1;
		block = m_freeBlocks[ m_numFreeBlocks ];
		HK_ON_CPU(HK_ASSERT(0xf05678ad, block->getNumElements() == 0));
	}
	else
	{
		m_blockStreamAllocator->blockAllocBatch( &m_freeBlocks[0], BATCH_ALLOC_SIZE );
		HK_ON_CPU(HK_ON_DEBUG(for(int i=0;i<BATCH_ALLOC_SIZE;i++) m_freeBlocks[i]->setNumElements(0); ));
		m_numFreeBlocks = BATCH_ALLOC_SIZE-1;
		block = m_freeBlocks[BATCH_ALLOC_SIZE-1];
	}
	return block;
}

void hkThreadLocalBlockStreamAllocator::blockFree(hkBlockStreamAllocator::Block* blockPpu)
{
#if !defined(HK_PLATFORM_SPU) && defined(HK_DEBUG)
	HK_ASSERT( 0xf04df1d4, blockPpu->m_allocator == (hkBlockStreamAllocator*)m_blockStreamAllocator);
	HK_ASSERT2( 0xf04f4565, m_threadId == -1 || m_threadId == HK_THREAD_LOCAL_GET(hkThreadNumber),
		"Your thread local allocator belongs to a different thread" );

	blockPpu->setNumElements(0);	// for check consistency
	checkConsistency();
#endif

	HK_ASSERT(0x3e515b17, m_numFreeBlocks < MAX_FREE_LIST_SIZE);
	m_freeBlocks[ m_numFreeBlocks ] = blockPpu;
	m_numFreeBlocks = m_numFreeBlocks+1;
	if ( m_numFreeBlocks < MAX_FREE_LIST_SIZE )
	{
		return;
	}

	const int newEndIndex = m_numFreeBlocks - BATCH_ALLOC_SIZE;
	m_blockStreamAllocator->blockFreeBatch( &m_freeBlocks[newEndIndex], BATCH_ALLOC_SIZE );
	m_numFreeBlocks = newEndIndex;
}

void hkThreadLocalBlockStreamAllocator::clear()
{
	if ( m_numFreeBlocks )
	{
		m_blockStreamAllocator->blockFreeBatch( &m_freeBlocks[0], m_numFreeBlocks );
		m_numFreeBlocks = 0;
	}
}

#if !defined(HK_PLATFORM_SPU)
void hkThreadLocalBlockStreamAllocator::checkConsistency()
{
	HK_ASSERT2( 0xf04f4565, m_threadId == -1 || m_threadId == HK_THREAD_LOCAL_GET(hkThreadNumber),
		"Your thread local allocator belongs to a different thread" );
	hkThreadLocalBlockStreamAllocator* lc = this;
	for(int i=0; i< lc->m_numFreeBlocks; i++)
	{
		HK_ASSERT(0xf0768790, lc->m_freeBlocks[i]->getNumElements() == 0);
		for(int j=i+1; j<lc->m_numFreeBlocks; j++)
		{
			HK_ASSERT(0xf0768791, lc->m_freeBlocks[i] != lc->m_freeBlocks[j]);
		}
	}
}
#endif

#ifdef HK_DEBUG
#include <stdio.h>
#define SHOW_FREELIST \
	printf("freelist %x\n",this); \
	for(int i=0; i<m_numFreeBlocks; i++) printf("%d (%p)\n", i, m_freeBlocks[i]); printf("\n");
#else
#define SHOW_FREELIST
#endif


void hkThreadLocalBlockStreamAllocator::blockFreeBatch( Block** blocks, int nblocks )
{
	HK_ON_DEBUG(for(int i=0;i<nblocks;i++) blocks[i]->setNumElements(0); );
	if ( nblocks >= BATCH_ALLOC_SIZE )
	{
		m_blockStreamAllocator->blockFreeBatch( blocks, nblocks );
		return;
	}

	if ( nblocks + m_numFreeBlocks >= MAX_FREE_LIST_SIZE)
	{
		// batch free existing elements
		int numBlocksToFree = hkMath::_min2<int>( BATCH_ALLOC_SIZE, m_numFreeBlocks);
		const int newEndIndex = m_numFreeBlocks - numBlocksToFree;
		m_blockStreamAllocator->blockFreeBatch( &m_freeBlocks[newEndIndex], numBlocksToFree );
		m_numFreeBlocks = newEndIndex;
	}

	// append elements to free list
	for (int i = 0; i < nblocks; i++)
	{
		HK_ASSERT( 0xf04df1d4, blocks[i]->m_allocator == (hkBlockStreamAllocator*)m_blockStreamAllocator);
		m_freeBlocks[ i + m_numFreeBlocks ] = blocks[i];
	}
	m_numFreeBlocks = m_numFreeBlocks + nblocks;

	//	SHOW_FREELIST;
	HK_ON_DEBUG( checkConsistency() );
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
