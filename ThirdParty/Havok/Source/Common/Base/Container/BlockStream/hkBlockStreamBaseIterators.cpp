/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/BlockStream/hkBlockStreamBaseIterators.h>
#include <Common/Base/Container/BlockStream/hkBlockStreamBaseRange.h>
#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>

//
// Reader
//

// Setters

void hkBlockStreamBase::Reader::setToStartOfStream( const hkBlockStreamBase::Stream* HK_RESTRICT stream )
{
#if !defined(HK_PLATFORM_SPU)
	const hkBlockStreamBase::Block* startBlock = stream->begin();
#else
	const hkBlockStreamBase::Block* startBlockPpu = stream->begin();

	// get first block
	const hkBlockStreamBase::Block* startBlock;
	{
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( &m_blockBuffer[0], startBlockPpu, sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::READ_COPY, m_dmaGroup );
		hkSpuDmaManager::performFinalChecks( startBlockPpu, &m_blockBuffer[0], sizeof(hkBlockStreamBase::Block));
		startBlock = &m_blockBuffer[0];
	}

	// prefetch next block
	{
		m_nextBuffer = -1;
		hkBlockStreamBase::Block* nextBlockPpu = startBlock->m_next;
		if ( nextBlockPpu )
		{
			hkSpuDmaManager::getFromMainMemory( &m_blockBuffer[1], nextBlockPpu, sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::READ_COPY, m_dmaGroup );
			m_nextBuffer = 1;
		}
	}
	m_currentBlockPpu = const_cast<hkBlockStreamBase::Block*>(startBlockPpu);
#endif

	HK_ON_CPU(HK_ASSERT( 0xf0dfde39, startBlock != HK_NULL ));

	m_currentBlock			= startBlock;
	m_currentByteLocation   = ((char*)startBlock->begin());

	int numDatasThisBlock			 = startBlock->getNumElements();
	m_numElementsToReadInThisBlock	 = numDatasThisBlock;
	m_numElementsToReadInOtherBlocks = HK_INT32_MAX;
	if ( numDatasThisBlock == 0)
	{
		m_currentByteLocation = HK_NULL;
	}
}

void hkBlockStreamBase::Reader::setToRange( const hkBlockStreamBase::Range* HK_RESTRICT range )
{
	if ( 0 == range->m_numElements )
	{
		setEmpty();
		return;
	}

#if !defined(HK_PLATFORM_SPU)
	m_currentBlock = range->m_startBlock;
#else

	// get first block
	{
		hkBlockStreamBase::Block* startBlockPpu = range->m_startBlock;
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( &m_blockBuffer[0], startBlockPpu, sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::READ_COPY, m_dmaGroup );
		hkSpuDmaManager::performFinalChecks( startBlockPpu, &m_blockBuffer[0], sizeof(hkBlockStreamBase::Block));
		m_currentBlock = &m_blockBuffer[0];
		m_currentBlockPpu = startBlockPpu;
	}

	// prefetch next block
	{
		m_nextBuffer = -1;
		hkBlockStreamBase::Block* nextBlockPpu = m_blockBuffer[0].m_next;
		if ( nextBlockPpu )
		{
			hkSpuDmaManager::getFromMainMemory( &m_blockBuffer[1], nextBlockPpu, sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::READ_COPY, m_dmaGroup );
			m_nextBuffer = 1;
		}
	}
#endif
	m_currentByteLocation   = ((char*)m_currentBlock->begin()) + range->m_startByteLocation;

	m_numElementsToReadInThisBlock	 = range->m_startBlockNumElements;
	HK_ASSERT( 0xf0343443, m_numElementsToReadInThisBlock <= range->m_numElements && m_numElementsToReadInThisBlock <= int(m_currentBlock->getNumElements()) );
	m_numElementsToReadInOtherBlocks = range->m_numElements - m_numElementsToReadInThisBlock;
	if ( m_numElementsToReadInThisBlock == 0)
	{
		m_currentByteLocation = HK_NULL;
	}
}

// SPU memory management functions

#if defined(HK_PLATFORM_SPU)
#	if defined(HK_PLATFORM_SIM)
void hkBlockStreamBase::Reader::initSpu( int dmaGroup, int numActiveBlocks, const char* what )
{
	HK_ASSERT( 0xf0345456, numActiveBlocks > 0);	// reader must be double buffered
	m_dmaGroup = dmaGroup;
	m_numBlocksInBuffer = numActiveBlocks+1;
	int allocSizeBlock = (numActiveBlocks+1) * HK_NEXT_MULTIPLE_OF(128, sizeof(hkBlockStreamBase::Block));
	m_blockBuffer = (hkBlockStreamBase::Block*)hkSpuStack::getInstance().allocateStack(allocSizeBlock, what);
	m_nextBuffer = -1;
}
#	else
void hkBlockStreamBase::Reader::initSpu( int dmaGroup, int numActiveBlocks )
{
	m_dmaGroup = dmaGroup;
	m_numBlocksInBuffer = numActiveBlocks+1;
	int allocSizeBlock = (numActiveBlocks+1) * HK_NEXT_MULTIPLE_OF(128, sizeof(hkBlockStreamBase::Block));
	m_blockBuffer = (hkBlockStreamBase::Block*)hkSpuStack::getInstance().allocateStack(allocSizeBlock, "ReadItBlock");
	m_nextBuffer = -1;
}
#	endif

void hkBlockStreamBase::Reader::exitSpu(  )
{
	int nextBuffer = m_nextBuffer;
	if ( nextBuffer >= 0) // we have to wait for the last buffer to arrive
	{
		// wait till next buffer arrives
		hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup );
		hkSpuDmaManager::performFinalChecks( m_currentBlock->m_next, &m_blockBuffer[nextBuffer], sizeof(hkBlockStreamBase::Block));
	}

	int allocSizeBlock = (m_numBlocksInBuffer) * HK_NEXT_MULTIPLE_OF(128, sizeof(hkBlockStreamBase::Block));
	hkSpuStack::getInstance().deallocateStack(m_blockBuffer.val(), allocSizeBlock);
}
#endif

// Internal functions

const void* hkBlockStreamBase::Reader::advanceToNewBlock( )
{
#if !defined(HK_PLATFORM_SPU)
	const Block* currentBlock = m_currentBlock->m_next;
	m_currentBlock = currentBlock;
	if( !currentBlock )
	{
		m_currentByteLocation = HK_NULL;
		return HK_NULL;
	}
#else
	int nextBuffer = m_nextBuffer;
	if ( nextBuffer < 0) // no more buffers
	{
		m_currentByteLocation = HK_NULL;
		return HK_NULL;
	}

	// wait till next buffer arrives and write buffer finishes
	hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup );

	hkBlockStreamBase::Block* currentBlockPpu = m_currentBlock->m_next;
	hkBlockStreamBase::Block* currentBlock    = &m_blockBuffer[ nextBuffer ];
	HK_SPU_DMA_PERFORM_FINAL_CHECKS( currentBlockPpu, &m_blockBuffer[nextBuffer], sizeof(hkBlockStreamBase::Block));

	// advance current block (spu) to next buffer
	m_currentBlock    = currentBlock;
	m_currentBlockPpu = currentBlockPpu;

	// advance and prefetch next buffer
	hkBlockStreamBase::Block* nextBlockPpu = currentBlock->m_next;
	if (nextBlockPpu && m_numElementsToReadInOtherBlocks)
	{
		nextBuffer = m_nextBuffer+1;
		if ( nextBuffer >= m_numBlocksInBuffer ){ nextBuffer = 0; }
		hkSpuDmaManager::getFromMainMemory( &m_blockBuffer[nextBuffer], nextBlockPpu, sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::READ_COPY, m_dmaGroup );
	}
	else
	{
		nextBuffer = -1;
	}
	m_nextBuffer = nextBuffer;
#endif

	int numElementsToReadInThisBlock = hkMath::min2((int)currentBlock->getNumElements(), (int)m_numElementsToReadInOtherBlocks );
	if( numElementsToReadInThisBlock<=0 )
	{
		m_currentByteLocation = HK_NULL;
		return HK_NULL;
	}
	m_numElementsToReadInThisBlock	 = numElementsToReadInThisBlock;
	m_numElementsToReadInOtherBlocks = m_numElementsToReadInOtherBlocks - m_numElementsToReadInThisBlock;
	m_currentByteLocation			 = (const char*)currentBlock->begin();

	hkMath::prefetch128(currentBlock->m_next);	// prefetch the next block header
	hkMath::prefetch128( hkAddByteOffsetConst(currentBlock,128) );

	HK_ASSERT( 0xf06576df, m_numElementsToReadInThisBlock > 0 );
	return currentBlock->begin();
}

//
// Writer
//

// Initializers
void hkBlockStreamBase::Writer::setToStartOfStream( hkBlockStreamBase::Stream::Allocator* allocator, hkBlockStreamBase::Stream* blockStream)
{
	HK_ASSERT( 0xf0345dfe, !blockStream->m_spuWronglySentConsumedBlockStreamBack && !blockStream->m_isLocked );
	HK_ASSERT2( 0x1990dde0, allocator->m_blockStreamAllocator == blockStream->m_allocator, "The writer and the stream being written to must use the same block allocator" );
	HK_ASSERT( 0xf0456567, blockStream->isEmpty() && blockStream->m_numTotalElements == 0 );
	HK_ON_DEBUG( blockStream->m_isLocked = true );
	hkBlockStreamBase::Block* firstBlockPpu	= blockStream->beginRw();
	m_tlAllocator			= allocator;
	m_blockStream			= blockStream;
	m_currentByteLocation	= 0;
	m_currentBlockNumElems  = 0;

#if !defined(HK_PLATFORM_SPU)
	HK_ON_DEBUG(m_finalized = false);
	hkMath::prefetch128( firstBlockPpu );
	m_currentBlock			= firstBlockPpu;
#else
	m_nextBuffer = m_blockBufferCapacity > 1 ? 1 : 0;
	hkString::memClear16( m_blocksPpu, sizeof(m_blocksPpu)>>4);
	m_blocksPpu[0] = firstBlockPpu;
	m_currentBlock = &m_blockBuffer[0];
	m_spuToPpuOffset = hkGetByteOffsetInt(&m_blockBuffer[0], firstBlockPpu );
#endif
}


void hkBlockStreamBase::Writer::setToEndOfStream( hkBlockStreamBase::Stream::Allocator* allocator, hkBlockStreamBase::Stream* blockStream )
{
	HK_ASSERT( 0xf0345dfe, !blockStream->m_spuWronglySentConsumedBlockStreamBack && !blockStream->m_isLocked );
	HK_ASSERT2( 0x1990dde0, allocator->m_blockStreamAllocator == blockStream->m_allocator, "The writer and the stream being written to must use the same block allocator" );
	HK_ON_DEBUG( blockStream->m_isLocked = true );

	hkBlockStreamBase::Block* lastBlock = blockStream->lastRw();
	m_tlAllocator = allocator;
	m_blockStream = blockStream;

#if !defined(HK_PLATFORM_SPU)

	HK_ON_DEBUG( m_finalized = false );

	m_currentByteLocation	= lastBlock->getBytesUsed();
	m_currentBlockNumElems  = lastBlock->getNumElements();

	
	blockStream->m_numTotalElements = blockStream->m_numTotalElements - m_currentBlockNumElems;
	HK_ASSERT( 0xf0dfde34, blockStream->m_numTotalElements >= 0 );
	m_currentBlock = lastBlock;

#else

	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( m_blockBuffer, lastBlock, sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::READ_COPY );
	HK_SPU_DMA_PERFORM_FINAL_CHECKS( lastBlock, m_blockBuffer, sizeof(hkBlockStreamBase::Block));
	m_currentByteLocation	= m_blockBuffer->getBytesUsed();
	m_currentBlockNumElems  = m_blockBuffer->getNumElements();
	blockStream->m_numTotalElements = blockStream->m_numTotalElements - m_blockBuffer->getNumElements();
	m_nextBuffer = m_blockBufferCapacity > 1 ? 1 : 0;
	hkString::memClear16( m_blocksPpu, sizeof(m_blocksPpu)>>4);
	m_blocksPpu[0] = lastBlock;
	m_currentBlock = &m_blockBuffer[0];
	m_spuToPpuOffset = hkGetByteOffsetInt(&m_blockBuffer[0], lastBlock );

#endif
}

void hkBlockStreamBase::Writer::batchReserveAndAdvance( int numReservations, int numBytesPerReservation )
{
	HK_ASSERT( 0xf03d3401, m_currentBlock != HK_NULL && numBytesPerReservation <= Block::BLOCK_DATA_SIZE );
	HK_ON_DEBUG( m_finalized  = false    );

	int reservationsLeft = numReservations;

	// Consume any reservations from the current block if possible.
	{
		const int maxReservationsCurrBlock = (Block::BLOCK_DATA_SIZE - m_currentByteLocation) / numBytesPerReservation;
		int consumedReservationsCurrBlock = hkMath::min2( reservationsLeft, maxReservationsCurrBlock );

		if ( consumedReservationsCurrBlock > 0 )
		{
			// Advance
			m_currentByteLocation  += consumedReservationsCurrBlock * numBytesPerReservation;
			m_currentBlockNumElems += consumedReservationsCurrBlock;

			reservationsLeft -= consumedReservationsCurrBlock;
		}
	}

	const int maxReservationsPerBlock = Block::BLOCK_DATA_SIZE / numBytesPerReservation;

	// Keep allocating and advancing by full blocks until we are out of reservations.
	while ( reservationsLeft > 0)
	{
		const int consumedReservationsCurrBlock = hkMath::min2( reservationsLeft, maxReservationsPerBlock );

		allocateAndAccessNewBlock( );

		// Advance
		m_currentByteLocation  += consumedReservationsCurrBlock * numBytesPerReservation;
		m_currentBlockNumElems += consumedReservationsCurrBlock;

		reservationsLeft -= consumedReservationsCurrBlock;
	}
}


void hkBlockStreamBase::Writer::finalize()
{
	HK_ASSERT( 0xf0445dfe, !m_blockStream->m_spuWronglySentConsumedBlockStreamBack && m_blockStream->m_isLocked );
	HK_ON_DEBUG( m_blockStream->m_isLocked = false );
	HK_ON_DEBUG( m_finalized               = true  );

	// Only remove the last block of the stream if it is empty and if it is not the only block in the stream (which
	// has been allocated when the block stream itself was created).
	if ( m_currentBlockNumElems == 0 && m_blockStream->m_blocks.getSize() > 1 )
	{
	#if defined(HK_PLATFORM_SPU)
		// Popping the current block from the block stream sets the next pointer of the previous block to HK_NULL in PPU.
		// If we have just two block buffers the previous one may still be being DMAed so we need to wait for it.
		if ( m_blockBufferCapacity == 2 && m_blocksPpu[m_nextBuffer] )
		{
			hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup );
		}
	#endif

		m_blockStream->popBack( m_tlAllocator );

	#if defined(HK_PLATFORM_SPU)
		// Update the 'next' pointer of the previous block if it is in the block buffer
		if ( m_blockBufferCapacity > 2 )
		{
			int prevBuffer = m_nextBuffer - 2;
			if ( prevBuffer < 0 )
			{
				prevBuffer += m_blockBufferCapacity;
			}
			hkBlockStreamBase::Block* blockSpu = &m_blockBuffer[prevBuffer];
			blockSpu->m_next = HK_NULL;
		}
	#endif
	}
	else
	{
		finalizeLastBlock( m_currentBlock, HK_NULL, m_currentBlockNumElems, m_currentByteLocation );
	}

#if !defined(HK_PLATFORM_SPU)

	HK_ON_DEBUG( m_blockStream->checkConsistency() );

#else

	// Send all open blocks back to PPU
	for (int i = 0; i < m_blockBufferCapacity; i++)
	{
		// Skip block buffer if it is not in use or if it is the next available one in which case the DMA would had
		// already been started from allocateAndAccessNewBlock()
		hkBlockStreamBase::Block* blockPpu = m_blocksPpu[i];
		if (m_blockBufferCapacity > 1 && (blockPpu == HK_NULL || m_nextBuffer == i))
		{
			continue;
		}

		// Start DMA
		hkBlockStreamBase::Block* blockSpu = &m_blockBuffer[i];
		hkSpuDmaManager::putToMainMemory        ( blockPpu, blockSpu, sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::WRITE_NEW, m_dmaGroup );
		HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( blockPpu, blockSpu, sizeof(hkBlockStreamBase::Block) );

		// Only wait for dma completion every 7 steps. This will also wait for any 'puts' done in
		// allocateAndAccessNewBlock().
		if ( (i & 0x7) == 3 )
		{
			hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup );
		}
	}
	hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup );

#endif

	m_currentBlock = HK_NULL;
}

// SPU memory management.
#if defined(HK_PLATFORM_SPU)

	#if defined(HK_PLATFORM_SIM)
		void hkBlockStreamBase::Writer::initSpu( int dmaGroup, int numActiveBlocks, const char* what )
		{
			HK_ASSERT( 0xf03cff12, numActiveBlocks < MAX_NUM_ACTIVE_BLOCKS);
			m_dmaGroup = dmaGroup;
			m_blockBufferCapacity = numActiveBlocks+1;
			int allocSizeBlock = (numActiveBlocks+1) * HK_NEXT_MULTIPLE_OF(128, sizeof(hkBlockStreamBase::Block));
			m_blockBuffer = (hkBlockStreamBase::Block*)hkSpuStack::getInstance().allocateStack(allocSizeBlock, what);
		}
	#else
		void hkBlockStreamBase::Writer::initSpu( int dmaGroup, int numActiveBlocks )
		{
			HK_ASSERT( 0xf03cff12, numActiveBlocks < MAX_NUM_ACTIVE_BLOCKS);
			m_dmaGroup = dmaGroup;
			m_blockBufferCapacity = numActiveBlocks+1;
			int allocSizeBlock = (numActiveBlocks+1) * HK_NEXT_MULTIPLE_OF(128, sizeof(hkBlockStreamBase::Block));
			m_blockBuffer = (hkBlockStreamBase::Block*)hkSpuStack::getInstance().allocateStack(allocSizeBlock, "WriterBlocks");
		}
	#endif
#endif


// Internal functions
void* hkBlockStreamBase::Writer::allocateAndAccessNewBlock()
{
#if !defined(HK_PLATFORM_SPU)

	hkBlockStreamBase::Block* nextBlock = m_blockStream->blockAlloc( m_tlAllocator );

	finalizeLastBlock( m_currentBlock, nextBlock, m_currentBlockNumElems, m_currentByteLocation );

	m_currentBlock         = nextBlock;
	m_currentByteLocation  = 0;
	m_currentBlockNumElems = 0;

	return nextBlock->begin();

#else

	// Allocate new block on PPU and finalize current one on SPU
	hkBlockStreamBase::Block* newBlockPpu = m_tlAllocator->blockAlloc();
	finalizeLastBlock( m_currentBlock, newBlockPpu, m_currentBlockNumElems, m_currentByteLocation );

	const int newBuffer = m_nextBuffer;
	if ( m_blockBufferCapacity == 1 )
	{
		// If we have just one block buffer we need to DMA it to PPU and wait
		hkBlockStreamBase::Block* currentBlockPpu = m_blocksPpu[0];
		hkSpuDmaManager::putToMainMemory(         currentBlockPpu, &m_blockBuffer[0], sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::WRITE_NEW, m_dmaGroup );
		HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( currentBlockPpu, &m_blockBuffer[0], sizeof(hkBlockStreamBase::Block) );
		hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup );
	}
	else
	{
		// If the new buffer was in use we need to wait for the put DMA to finish
		if ( m_blocksPpu[newBuffer] )
		{
			hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup );
		}

		// Calculate next buffer and initiate DMA to PPU if it is in use
		m_nextBuffer = (m_nextBuffer == m_blockBufferCapacity - 1) ? 0 : m_nextBuffer + 1;
		hkBlockStreamBase::Block* nextBlockPpu = m_blocksPpu[m_nextBuffer];
		if ( nextBlockPpu )
		{
			hkSpuDmaManager::putToMainMemory(         nextBlockPpu, &m_blockBuffer[m_nextBuffer], sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::WRITE_NEW, m_dmaGroup );
			HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( nextBlockPpu, &m_blockBuffer[m_nextBuffer], sizeof(hkBlockStreamBase::Block) );
		}
	}

	// Initialize new block
	hkBlockStreamBase::Block* newBlockSpu = &m_blockBuffer[newBuffer];
	m_blockStream->blockInit( newBlockPpu, newBlockSpu );
	m_blocksPpu[newBuffer] = newBlockPpu;
	m_currentBlock = newBlockSpu;
	m_currentByteLocation = 0;
	m_currentBlockNumElems = 0;
	m_spuToPpuOffset = hkGetByteOffsetInt( newBlockSpu, newBlockPpu );

	return m_currentBlock->begin();

#endif
}

//
// Consumer
//

void hkBlockStreamBase::Consumer::setToStartOfStream( hkBlockStreamBase::Stream::Allocator* HK_RESTRICT allocator, hkBlockStreamBase::Stream* HK_RESTRICT stream, hkBlockStreamBase::Stream* streamPpu )
{
	stream->m_partiallyFreed = 1;
#if defined(HK_PLATFORM_SPU)
	// set the partially freed flag, don't wait for dma as this is done in ReadIterator::setToRange
	if ( streamPpu )
	{
		hkSpuDmaManager::putToMainMemorySmall   ( &streamPpu->m_partiallyFreed, &stream->m_partiallyFreed, sizeof(hkBool), hkSpuDmaManager::WRITE_NEW, m_dmaGroup );
		HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( &streamPpu->m_partiallyFreed, &stream->m_partiallyFreed, sizeof(hkBool) );
	}
#endif
	// 	HK_ASSERT( 0xf0345dfe, !blockStream->m_isLocked );	// enable this, but it requires a finalize()
	// 	HK_ON_DEBUG( blockStream->m_isLocked = true );
	hkBlockStreamBase::Reader::setToStartOfStream(stream);
	m_blockStream = stream;
	m_allocator = allocator;
	m_numElementsToFreeInThisBlock	= m_numElementsToReadInThisBlock;
}

void hkBlockStreamBase::Consumer::setToRange( hkBlockStreamBase::Stream::Allocator* allocator, hkBlockStreamBase::Stream* HK_RESTRICT stream, hkBlockStreamBase::Stream* streamPpu, const hkBlockStreamBase::Range* HK_RESTRICT range )
{
	if ( 0 == range->m_numElements )
	{
		setEmpty();
		return;
	}
	stream->m_partiallyFreed = 1;
#if defined(HK_PLATFORM_SPU)
	// set the partially freed flag, don't wait for dma as this is done in ReadIterator::setToRange
	if ( stream )
	{
		hkSpuDmaManager::putToMainMemorySmall( &streamPpu->m_partiallyFreed, &stream->m_partiallyFreed, sizeof(hkBool), hkSpuDmaManager::WRITE_NEW_NON_DETERMINISTIC, m_dmaGroup );
		HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( &streamPpu->m_partiallyFreed, &stream->m_partiallyFreed, sizeof(hkBool) );
	}
#endif
	hkBlockStreamBase::Reader::setToRange(range);
	m_blockStream = stream;
	m_allocator						 = allocator;
	m_numElementsToFreeInThisBlock	 = m_numElementsToReadInThisBlock;

	if (HK_NULL == range->m_startBlock || 0 == range->m_numElements)
	{
		m_numElementsToFreeInThisBlock = 0;
	}
}

const void* hkBlockStreamBase::Consumer::freeAndAdvanceToNewBlock( )
{
	hkBlockStreamBase::Block* HK_RESTRICT toFree = getCurrentBlock();
#if !defined(HK_PLATFORM_SPU)
	hkBlockStreamBase::Block* HK_RESTRICT toFreePpu = getCurrentBlock();
#else
	hkBlockStreamBase::Block* HK_RESTRICT toFreePpu = m_currentBlockPpu;
#endif

	int numElementsToFreeInThisBlock = m_numElementsToFreeInThisBlock;

	const void* ret = hkBlockStreamBase::Reader::advanceToNewBlock();
	m_numElementsToFreeInThisBlock	= m_numElementsToReadInThisBlock;


	//
	//	Free consumed block
	//
	if ( int(toFree->getNumElements()) == numElementsToFreeInThisBlock )
	{
		m_blockStream->freeBlockWithIndex( m_allocator, toFreePpu,   toFree->m_blockIndexInStream );
	}
	else
	{
		toFreePpu->atomicDecreaseElementCount(numElementsToFreeInThisBlock);
		// Note: there is still a tiny chance that 2 threads will not free a block,
		// but we blocks will be free later anyway in the destructor of the block stream
	}

	return ret;
}


//
// Modifier
//

void* hkBlockStreamBase::Modifier::advanceToNewBlock()
{
#if !defined(HK_PLATFORM_SPU)
	m_currentBlock = m_currentBlock->m_next;
	if( !m_currentBlock )
	{
		m_currentByteLocation = HK_NULL;
		return HK_NULL;
	}
#else
	// wait till prefetch(=next) buffer arrives and write block is written back
	hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup );

	//
	//	Write back current block
	//
	{
		int startOffset = m_writeBackBlockStartOffset;
		const int endOffset = (hkUlong) m_currentByteLocation.val() - (hkUlong) m_currentBlock->begin();

		// Handle non-aligned start
		if ( startOffset & 0xF )
		{
			void* writeBackBlockPpu	= hkAddByteOffset( (void*) m_currentBlockPpu, HK_OFFSET_OF(hkBlockStreamBase::Block, m_data) + startOffset);
			void*  srcSpu			= hkAddByteOffset( (void*) m_currentBlock->m_data, startOffset);
			int nonAlignedTransferSize = hkMath::min2(16 - (startOffset & 0xF), endOffset - startOffset);
			hkSpuDmaManager::putToMainMemorySmallAnySize( writeBackBlockPpu, srcSpu, nonAlignedTransferSize, hkSpuDmaManager::WRITE_NEW, m_dmaGroup );
			HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( writeBackBlockPpu, srcSpu, nonAlignedTransferSize);
			startOffset += nonAlignedTransferSize;
		}

		// Handle aligned part
		const int alignedTransferSize = (endOffset - startOffset) & ~0xF;
		if ( alignedTransferSize > 0 )
		{
			void* writeBackBlockPpu	= hkAddByteOffset( (void*) m_currentBlockPpu, HK_OFFSET_OF(hkBlockStreamBase::Block, m_data) + startOffset);
			void*  srcSpu			= hkAddByteOffset( (void*) m_currentBlock->m_data, startOffset);
			hkSpuDmaManager::putToMainMemory( writeBackBlockPpu, srcSpu, alignedTransferSize, hkSpuDmaManager::WRITE_NEW, m_dmaGroup );
			HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( writeBackBlockPpu, srcSpu, alignedTransferSize);
			startOffset += alignedTransferSize;
		}

		// Handle remainder
		const int reaminingTransferSize = endOffset - startOffset;
		if ( reaminingTransferSize > 0 )
		{
			void* writeBackBlockPpu	= hkAddByteOffset( (void*) m_currentBlockPpu, HK_OFFSET_OF(hkBlockStreamBase::Block, m_data) + startOffset);
			void*  srcSpu			= hkAddByteOffset( (void*) m_currentBlock->m_data, startOffset);
			hkSpuDmaManager::putToMainMemoryAnySize( writeBackBlockPpu, srcSpu, reaminingTransferSize, hkSpuDmaManager::WRITE_NEW, m_dmaGroup );
			HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT( writeBackBlockPpu, srcSpu, reaminingTransferSize);
		}

		m_writeBackBlockStartOffset = 0;	// the next block needs to be written back fully
	}

	int nextBuffer = m_nextBuffer;
	if ( nextBuffer < 0) // no more buffers
	{
		// wait for write back buffer
		HK_ON_SPU(hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup ));

		m_currentByteLocation = HK_NULL;
		return HK_NULL;
	}

	hkBlockStreamBase::Block* nextBlockPpu = m_currentBlock->m_next;
	hkBlockStreamBase::Block* nextBlock    = &m_blockBuffer[ nextBuffer ];
	HK_SPU_DMA_PERFORM_FINAL_CHECKS( nextBlockPpu, &m_blockBuffer[nextBuffer], sizeof(hkBlockStreamBase::Block));


	// advance and prefetch next buffer
	hkBlockStreamBase::Block* prefetchBlockPpu = nextBlock->m_next;
	if (prefetchBlockPpu && m_numElementsToReadInOtherBlocks)
	{
		nextBuffer = m_nextBuffer+1;
		if ( nextBuffer >= m_numBlocksInBuffer ){ nextBuffer = 0; }
		hkSpuDmaManager::getFromMainMemory( &m_blockBuffer[nextBuffer], prefetchBlockPpu, sizeof(hkBlockStreamBase::Block), hkSpuDmaManager::READ_COPY, m_dmaGroup );
	}
	else
	{
		nextBuffer = -1;
	}

	// advance current block (spu) to next buffer
	m_currentBlock    = nextBlock;
	m_currentBlockPpu = nextBlockPpu;
	m_nextBuffer =		nextBuffer;

#endif

	int numElementsToReadInThisBlock = hkMath::min2((int)m_currentBlock->getNumElements(), (int)m_numElementsToReadInOtherBlocks);
	if( numElementsToReadInThisBlock<=0 )
	{
		// wait for write back buffer
		HK_ON_SPU(hkSpuDmaManager::waitForDmaCompletion( m_dmaGroup ));

		m_currentByteLocation = HK_NULL;
		return HK_NULL;
	}
	m_numElementsToReadInThisBlock	 = numElementsToReadInThisBlock;
	m_numElementsToReadInOtherBlocks = m_numElementsToReadInOtherBlocks - m_numElementsToReadInThisBlock;
	m_currentByteLocation			 = (const char*)m_currentBlock->begin();

	HK_ASSERT( 0xf06576df, m_numElementsToReadInThisBlock > 0 );
	return const_cast<void*>(m_currentBlock->begin());
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
