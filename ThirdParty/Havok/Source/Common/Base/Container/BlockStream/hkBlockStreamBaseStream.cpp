/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/BlockStream/hkBlockStreamBaseStream.h>
#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#if defined(HK_PLATFORM_SPU)
	#include <Common/Base/Container/ArraySpu/hkArraySpuUtil.h>
#endif

// Block management
#if !defined(HK_PLATFORM_SPU)
void hkBlockStreamBase::Stream::initBlockStream( Allocator* HK_RESTRICT tlAllocator, bool zeroNewBlocks )
{
	//HK_ASSERT2( 0xf0343234, tlAllocator->m_blockStreamManager == allocator, "Your thread local allocator does not match the allocator" );
	m_allocator = tlAllocator->m_blockStreamAllocator;
	m_numTotalElements = 0;
	HK_ON_DEBUG(m_isLocked = false);
	HK_ON_DEBUG(m_spuWronglySentConsumedBlockStreamBack = false);
	m_partiallyFreed = false;
	m_zeroNewBlocks = zeroNewBlocks;
	HK_ON_PLATFORM_HAS_SPU(m_blockStreamPpu = this);
	blockAlloc(tlAllocator);

	HK_ON_DEBUG(checkConsistency());
}

void hkBlockStreamBase::Stream::reset(Allocator* HK_RESTRICT tlAllocator)
{
	HK_ASSERT( 0xf034defd, tlAllocator->m_blockStreamAllocator == m_allocator );
	if ( !m_blocks.isEmpty() )
	{
		if ( isEmpty() && m_blocks.getSize()==1 && m_partiallyFreed == false )
		{
			return;
		}
		clear(tlAllocator);
	}
	blockAlloc(tlAllocator);

	HK_ON_DEBUG(checkConsistency());
}

void hkBlockStreamBase::Stream::clear(Allocator* HK_RESTRICT tlAllocator)
{
	HK_ASSERT( 0xf0345dff, !m_spuWronglySentConsumedBlockStreamBack && !m_isLocked );
	int size = m_blocks.getSize();

	if ( m_partiallyFreed )
	{
		// compress used blocks
		hkBlockStreamBase::Block** src = m_blocks.begin();
		hkBlockStreamBase::Block** dst = src;
		for (int i =0; i < size; i++)
		{
			if ( *src )
			{
				*(dst++) = *(src);
			}
			src++;
		}
		size = int(dst - m_blocks.begin());
		m_partiallyFreed = false;
	}

	if ( size )
	{
		tlAllocator->blockFreeBatch( m_blocks.begin(),  size );
	}
	m_blocks.clear();	// needs to be done always since m_partiallyFreed
	m_numTotalElements = 0;

	HK_ON_DEBUG(checkConsistency());
}

void hkBlockStreamBase::Stream::append( Allocator* HK_RESTRICT tlAllocator, hkBlockStreamBase::Stream* HK_RESTRICT inStream )
{
	HK_ASSERT( 0xf034defd, tlAllocator->m_blockStreamAllocator == m_allocator );

	HK_ASSERT( 0xf0345dfe, !m_spuWronglySentConsumedBlockStreamBack && !m_isLocked && !inStream->m_spuWronglySentConsumedBlockStreamBack && !inStream->m_isLocked && !m_partiallyFreed && !inStream->m_partiallyFreed);
	if( inStream->isEmpty() )
	{
		if ( m_blocks.getSize() == 0)	// just cleared so we need to create a dummy block
		{
			blockAlloc(tlAllocator);
		}
		inStream->clear( tlAllocator );
		HK_ON_DEBUG(checkConsistency());
		return;
	}

	if ( m_blocks.getSize() == 0)	// just cleared so we can append
	{
	}else if( isEmpty() )
	{
		tlAllocator->blockFree( m_blocks[0] );
		m_blocks.clear();
	}
	else
	{
		//	modify nextPtr
		hkBlockStreamBase::Block* prevLast = lastRw();
		HK_ASSERT(0xf0345456, !prevLast->m_next && prevLast->getNumElements());
		prevLast->m_next = inStream->beginRw();
	}
	int oldSize = m_blocks.getSize();
	m_blocks.insertAt( oldSize, inStream->m_blocks.begin(), inStream->m_blocks.getSize() );
	inStream->m_blocks.clear();

	for (int i = oldSize; i< m_blocks.getSize(); i++)
	{
		m_blocks[i]->m_blockIndexInStream = i;
		HK_ON_DEBUG( m_blocks[i]->m_blockStream = this);
	}
	m_numTotalElements += inStream->m_numTotalElements;
	inStream->m_numTotalElements = 0;

	HK_ON_DEBUG(checkConsistency());

	return;
}

void hkBlockStreamBase::Stream::fixupConsumedBlocks(Allocator* HK_RESTRICT tlAllocator)
{
	if ( !m_partiallyFreed )
	{
		HK_ASSERT( 0xf0dfe565, m_blocks.indexOf(HK_NULL) == -1 );
		return;
	}
	int size = m_blocks.getSize();
	// compress used blocks
	hkBlockStreamBase::Block** src = m_blocks.begin();
	hkBlockStreamBase::Block** dst = src;
	int d = 0;
	int numTotalElements = 0;
	hkBlockStreamBase::Block* lastBlock = HK_NULL;
	for (int i =0; i < size; i++)
	{
		if ( *src )
		{
			hkBlockStreamBase::Block* block = *(src);
			if ( lastBlock )
			{
				lastBlock->m_next = block;
			}
			lastBlock = block;
			*dst = block;
			int numElements = block->getNumElements();
			numTotalElements += numElements;
			block->m_blockIndexInStream = d;
			dst++;
			d++;
		}
		src++;
	}
	if (lastBlock)
	{
		lastBlock->m_next = HK_NULL;
	}
	size = int(dst - m_blocks.begin());
	m_numTotalElements = numTotalElements;
	m_blocks.setSizeUnchecked(size);
	if (!size)
	{
		blockAlloc(tlAllocator);
	}
	checkConsistency();
	m_partiallyFreed = false;
}

void hkBlockStreamBase::Stream::freeAllBlocksBeforeRange(Allocator* HK_RESTRICT tlAllocator, const hkBlockStreamBase::Range *range)
{
	checkBlockOwnership( range->m_startBlock );
	for (int i = 0; i<range->m_startBlock->m_blockIndexInStream; i++)
	{
		freeBlockWithIndex( tlAllocator, m_blocks[i], i );
	}
	fixupConsumedBlocks( tlAllocator );
}

void hkBlockStreamBase::Stream::moveExclusiveRangeToBack( Allocator* HK_RESTRICT tlAllocator, const hkBlockStreamBase::Range* rangeInOut )
{
	HK_ASSERT( 0xf034ab33, tlAllocator->m_blockStreamAllocator == m_allocator );
	HK_ASSERT2( 0xf034ab35, rangeInOut->m_startByteLocation==0, "An exclusive range must have m_startByteLocation 0" );

	hkUint32 elems = rangeInOut->getNumElements();
	if (!elems)
	{
		return;
	}

	// We currently support only moving a range from the same stream
	checkBlockOwnership(rangeInOut->m_startBlock);

	int numblocks = 1;
	hkBlockStreamBase::Block* block = rangeInOut->m_startBlock;
	for (; elems > block->getNumElements(); block = block->m_next)
	{
		elems -= block->getNumElements();
		numblocks++;
	}

	HK_ASSERT2( 0xf034ab36, elems == block->getNumElements(), "An exclusive range must fill the entire last block" );
	HK_ASSERT2( 0xf034ab37, block->getBytesUsed() == Block::BLOCK_DATA_SIZE, "An exclusive range must fill the entire last block" );

	int index = rangeInOut->m_startBlock->m_blockIndexInStream;
	m_blocks.reserve( m_blocks.getSize() + numblocks ); // The next line is only safe if it doesn't trigger a resize, so make sure it doesn't.
	m_blocks.insertAt( m_blocks.getSize(), &m_blocks[index], numblocks );
	m_blocks.removeAtAndCopy(index, numblocks);

	for (int i=index; i<m_blocks.getSize(); i++)
	{
		m_blocks[i]->m_blockIndexInStream = i;
	}

	// Patch up next pointers
	if (index>0) m_blocks[index-1]->m_next = m_blocks[index];
	int patchIdx = m_blocks.getSize()-numblocks-1;
	if (patchIdx>=0) m_blocks[patchIdx]->m_next = m_blocks[patchIdx+1];
	m_blocks[m_blocks.getSize()-1]->m_next = HK_NULL;
	HK_ON_DEBUG(checkConsistency());
}
#endif

// this is called by a writer, which assumes the inplace block array to be in spu memory
#if !defined(HK_PLATFORM_SPU)
hkBlockStreamBase::Block* hkBlockStreamBase::Stream::blockAlloc(Allocator* HK_RESTRICT tlAllocator)
#else
hkBlockStreamBase::Block* hkBlockStreamBase::Stream::blockInit(hkBlockStreamBase::Block* blockOnPpu, hkBlockStreamBase::Block* blockBufferOnSpu )
#endif
{
	// Get a block from the allocator
#if !defined(HK_PLATFORM_SPU)
	HK_ASSERT( 0xf034defd, tlAllocator->m_blockStreamAllocator == m_allocator );
	hkBlockStreamBase::Block* block = tlAllocator->blockAlloc();
#else
	hkBlockStreamBase::Block* block = blockBufferOnSpu;
#endif

	// zero if needed
	if ( m_zeroNewBlocks )
	{
		HK_ASSERT( 0xf0fdde34, (hkUlong(block) & 0x7f) == 0);
		hkString::memClear128( block, sizeof(hkBlockStreamBase::Block) );
	}
	else
	{
		block->setHeaderToZero();
	}

	int blockIndex = m_blocks.getSize();

	block->m_allocator = m_allocator;

	block->m_blockIndexInStream = blockIndex;

#if !defined(HK_PLATFORM_SPU)
	HK_ON_DEBUG( block->m_blockStream = this);
	m_blocks.pushBack( block );
	return block;
#else
	#if defined(HK_DEBUG_SPU) || defined(HK_DEBUG)
	block->m_blockStream = m_blockStreamPpu;
	#endif

	hkArraySpuUtil::reserveMore( (hkArray<char>&)m_blocks, m_blocks.m_storage, INPLACE_PBLOCK_STORAGE,  sizeof(void*), 1 );
	m_blocks.setSizeUnchecked( blockIndex+1);

	// if we are using the inplace part, simply update the spu array,
	// else update the ppu array
	if ( m_blocks.stillInplaceUsingMask() )
	{
		m_blocks.getStorage(blockIndex) = blockOnPpu;
	}
	else
	{
		hkSpuDmaUtils::setPntrInMainMemory( (void**)&m_blocks[blockIndex], blockOnPpu );
	}
	return blockOnPpu;
#endif
}


hkBlockStreamBase::Block* hkBlockStreamBase::Stream::popBack(Allocator* HK_RESTRICT tlAllocator)
{
	int numBlocks = m_blocks.getSize();
	hkBlockStreamBase::Block* lastBlockPpu;
	hkBlockStreamBase::Block* secondLastBlockPpu;

#if !defined(HK_PLATFORM_SPU)

	HK_ASSERT( 0xf034defd, tlAllocator->m_blockStreamAllocator == m_allocator );
	lastBlockPpu = m_blocks[ numBlocks-1 ];
	HK_ASSERT( 0xf034deff, lastBlockPpu->m_blockIndexInStream == m_blocks.getSize() - 1 );
	secondLastBlockPpu = m_blocks[ numBlocks-2 ];
	HK_ASSERT(0xf034fddf, secondLastBlockPpu->m_next == lastBlockPpu);
	tlAllocator->blockFree( lastBlockPpu );
	secondLastBlockPpu->m_next = HK_NULL;

#else

	if ( m_blocks.stillInplaceUsingMask() )
	{
		lastBlockPpu = m_blocks.getStorage(numBlocks - 1);
		secondLastBlockPpu  = m_blocks.getStorage(numBlocks - 2);
	}
	else
	{
		lastBlockPpu = (hkBlockStreamBase::Block*)hkSpuDmaUtils::getPntrFromMainMemory( (void**)&m_blocks[numBlocks - 1] );
		secondLastBlockPpu  = (hkBlockStreamBase::Block*)hkSpuDmaUtils::getPntrFromMainMemory( (void**)&m_blocks[numBlocks - 2] );
	}
	tlAllocator->blockFree( lastBlockPpu );
	union { hkBlockStreamBase::Block** b; void** v; } b2v; b2v.b = &(secondLastBlockPpu->m_next);
	hkSpuDmaUtils::setPntrInMainMemory( b2v.v, HK_NULL );

#endif

	m_blocks.popBack();
	HK_ON_DEBUG(checkConsistency());
	return secondLastBlockPpu;
}

void hkBlockStreamBase::Stream::freeBlockWithIndex(Allocator* HK_RESTRICT tlAllocator, hkBlockStreamBase::Block* HK_RESTRICT blockPpu, int index)
{
	hkBlockStreamBase::Block** blocks = m_blocks.begin();
#if !defined(HK_PLATFORM_SPU)
	HK_ASSERT( 0xf03df1d8, blockPpu->m_blockIndexInStream == index && m_blocks[index] == blockPpu );
	checkBlockOwnership(blockPpu);
	blocks[index] = HK_NULL;
	m_partiallyFreed = true;	// on spu this is set by the consumer::setToRange()
#else
	// this is called only by the consumer, so we always update the ppu data
	m_spuWronglySentConsumedBlockStreamBack = true; // set this on SPU, as this blockstream should never go back to spu
	hkSpuDmaUtils::setPntrInMainMemory( (void**)&blocks[index], HK_NULL );
#endif
	tlAllocator->blockFree( blockPpu );
}

// SPU version of inline functions.
#if defined(HK_PLATFORM_SPU)
const hkBlockStreamBase::Block* hkBlockStreamBase::Stream::begin() const
{
	hkBlockStreamBase::Block* startBlockPpu;
	{
		if ( m_blocks.stillInplaceUsingMask() )
		{
			startBlockPpu = m_blocks.getStorage(0);
		}
		else
		{
			startBlockPpu = (hkBlockStreamBase::Block*)hkSpuDmaUtils::getPntrFromMainMemory( (void**)&m_blocks[0] );
		}
	}
	return startBlockPpu;
}

const hkBlockStreamBase::Block* hkBlockStreamBase::Stream::last() const
{
	int lastIndex = m_blocks.getSize()-1;
	if ( m_blocks.stillInplaceUsingMask() )
	{
		return m_blocks.getStorage(lastIndex);
	}
	return (const hkBlockStreamBase::Block*)hkSpuDmaUtils::getPntrFromMainMemory( (void**)&m_blocks[lastIndex] );
}
#endif

// Consistency checks
void hkBlockStreamBase::Stream::checkConsistency() const
{
#if !defined(HK_PLATFORM_SPU) && defined(HK_DEBUG)
	int totalNumElems = 0;
	for (int bi = 0; bi < m_blocks.getSize(); bi++)
	{
		const hkBlockStreamBase::Block* block = m_blocks[bi];
		if ( bi < m_blocks.getSize() - 1 )
		{
			// Not last block
			HK_ASSERT( 0xf0ccfe34, block->m_next == m_blocks[bi+1] );
			HK_ASSERT2(0xad731113, 0 < block->getNumElements(), "Empty blocks found within stream. This will cause asserts "
				"with the iterators." );
		}
		else
		{
			HK_ASSERT( 0xf0ccfe34, block->m_next == HK_NULL );
		}
		totalNumElems += block->getNumElements();
		checkBlockOwnership(block);
		HK_ASSERT( 0xf0ccfe34, block->m_blockIndexInStream == bi );
		HK_ASSERT( 0xf0ccfe37, block->m_allocator == m_allocator );
	}
	HK_ASSERT( 0xf0ccfe37, totalNumElems == m_numTotalElements );
#endif
}

void hkBlockStreamBase::Stream::checkConsistencyOfRange( const hkBlockStreamBase::Range& range )
{
#if !defined(HK_PLATFORM_SPU) && defined(HK_DEBUG)
	if (range.isEmpty())
	{
		return;
	}
	// check if the range points to a block owned by this stream
	HK_ASSERT( 0xf034df34, m_blocks[range.m_startBlock->m_blockIndexInStream] == range.m_startBlock );

	HK_ASSERT( 0xf034df34, range.m_startBlockNumElements <= range.m_startBlock->getNumElements() );
#endif
}


void hkBlockStreamBase::Stream::checkConsistencyWithGrid(
	const hkBlockStreamBase::Range* rangesIn, int numRanges, int rangeStriding, bool allowForUnusedData ) const
{
#if !defined(HK_PLATFORM_SPU) && defined(HK_DEBUG)
	hkLocalBuffer<hkBlockStreamBase::Range> ranges(numRanges);
	{
		for (int i =0; i < numRanges; i++)
		{
			ranges[i] = *rangesIn;
			rangesIn = hkAddByteOffsetConst( rangesIn, rangeStriding );
		}
		hkSort( ranges.begin(), numRanges, Range::compareRange );
	}

	checkConsistencyWithSortedRanges(ranges.begin(), numRanges, rangeStriding, allowForUnusedData);
#endif
}


void hkBlockStreamBase::Stream::checkConsistencyWithSortedRanges(
	const hkBlockStreamBase::Range* sortedRanges, int numRanges, int rangeStriding, bool allowForUnusedData) const
{
#if !defined(HK_PLATFORM_SPU) && defined(HK_DEBUG)
	checkConsistency();

	const hkBlockStreamBase::Block* currentBlock = HK_NULL;
	hkUint32 numElementsInCurrentBlock = 0;
	int curentBlockLastByte = -1;
	int totalNumElemsInRanges = 0;

	for (int i =0; i < numRanges; i++)
	{
		const hkBlockStreamBase::Range& range = sortedRanges[i];
		if (range.isEmpty())
		{
			continue;
		}
		totalNumElemsInRanges += range.getNumElements();

		// check if the range points to a block owned by this stream
		HK_ASSERT( 0xf034df34, m_blocks[range.m_startBlock->m_blockIndexInStream] == range.m_startBlock );

		if ( range.m_startBlock != currentBlock )
		{
			if ( currentBlock )
			{
				HK_ASSERT(0xf034df34, numElementsInCurrentBlock <= currentBlock->getNumElements());
			}
			currentBlock = range.m_startBlock;
			curentBlockLastByte = range.m_startByteLocation;
			numElementsInCurrentBlock = range.m_startBlockNumElements;
		}
		else
		{
			HK_ASSERT(0xf034df34,  range.m_startByteLocation > curentBlockLastByte);
		}
	}
	HK_ASSERT(0xf034df34, totalNumElemsInRanges <= m_numTotalElements );
	if ( !allowForUnusedData)
	{
		HK_ASSERT(0xf034df34, totalNumElemsInRanges == m_numTotalElements );
	}
#endif
}


//
//#if !defined(HK_PLATFORM_SPU)
//void hkBlockStreamBase::Stream::append( Allocator* HK_RESTRICT tlAllocator, hkBlockStreamBase::Range* HK_RESTRICT ranges, int numRanges, int rangeStriding )
//{
//	HK_ASSERT( 0xf034defd, tlAllocator->m_blockStreamAllocator == m_allocator );
//
//	hkBlockStreamBase::Range* range = ranges;
//	for (int i =0; i < numRanges; range = hkAddByteOffset( range, rangeStriding ), i++)
//	{
//		if ( range->isEmpty() )
//		{
//			continue;
//		}
//
//		if ( m_blocks.getSize() == 0)	// just cleared so we can append
//		{
//
//		}else if( isEmpty() )
//		{
//#if !defined(HK_PLATFORM_SPU)
//			tlAllocator->blockFree( m_blocks[0] );
//#endif
//			m_blocks.clear();
//		}
//		else
//		{
//			// see if we can fit the combined data
//			//	modify nextPtr
//			hkBlockStreamBase::Block* lastBlock = last();
//			HK_ASSERT(0xf0345456, !lastBlock->m_next && lastBlock->getNumElements());
//
//			int bytesUsed = range->m_startBlock->getBytesUsed()- range->m_startByteLocation;
//			if ( lastBlock->getBytesUsed()+ bytesUsed <= hkBlockStreamBase::Block::BLOCK_DATA_SIZE )
//			{
//				void* src = hkAddByteOffset(range->m_startBlock->begin(), range->m_startByteLocation );
//				void *dst = hkAddByteOffset(lastBlock->begin(), lastBlock->getBytesUsed());
//				hkString::memCpy16NonEmpty( dst, src, bytesUsed );
//				lastBlock->setBytesUsed(hkBlockStreamBase::Block::CountType(lastBlock->getBytesUsed()+ hkBlockStreamBase::Block::CountType(bytesUsed)));
//				m_numTotalElements += range->getNumElements();
//			}
//			else
//			{
//				if ( range->m_startByteLocation > 0)
//				{
//					// create new block
//					blockAlloc( tlAllocator );
//
//				}
//				lastBlock->m_next = range->m_startBlock;
//			}
//		}
//		//m_blocks.insertAt( m_blocks.getSize(), range->m_startBlock, range->getNumBlocks() );
//		HK_ASSERT(0x680e156b,0);	// ranges are not freed yet, but the blocks are partially appended
//		//inStream->m_blocks.clear();
//	}
//
//	HK_ON_DEBUG(checkConsistency());
//}

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
