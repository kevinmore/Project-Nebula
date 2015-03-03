/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/BlockStream/hkBlockStreamBaseRange.h>

#if defined (HK_PLATFORM_SPU)
	HK_COMPILE_TIME_ASSERT( (sizeof(hkBlockStreamBase::Range)&0xf) == 0 );	// we are writing back ranges to PPU 
#endif 

void hkBlockStreamBase::Range::setEndPoint(const Writer* HK_RESTRICT it)
{
	// the elements in the last block minus the existing elements in the first
	HK_ASSERT( 0xf0345456, it->m_blockStream->m_partiallyFreed == false );

	int currentTotal = it->m_blockStream->m_numTotalElements + it->m_currentBlockNumElems;
	int numElements  = currentTotal + m_numElements /* m_numElements is actually negative and initialized in setStartPoint */;

	HK_ASSERT2( 0xad341112, numElements >= 0, "Range corrupted." );

	if ( numElements )
	{
#if !defined(HK_PLATFORM_SPU)
		hkBlockStreamBase::Block* startBlock = m_startBlock;
		if ( (startBlock != it->m_currentBlock) && startBlock->getNumElements() == m_startBlockNumElements )	
		{
			// first block is empty,
			// advance to next block
			startBlock = startBlock->m_next;
			m_startBlockNumElements = 0;
			m_startByteLocation = 0;
		}
		if (startBlock == it->m_currentBlock)
		{
			m_startBlockNumElements = hkBlockStreamBase::Block::CountType(it->m_currentBlockNumElems) - m_startBlockNumElements;
		}
		else
		{
			m_startBlockNumElements = hkBlockStreamBase::Block::CountType(startBlock->getNumElements() - m_startBlockNumElements);
		}
		HK_ASSERT( 0xf045456, numElements >= m_startBlockNumElements );
		m_startBlock = startBlock;
#else
		hkBlockStreamBase::Block* startBlockPpu = m_startBlock;
		HK_ALIGN16( char startBlockBuffer[hkBlockStreamBase::Block::BLOCK_HEADER_SIZE]);
		// search our block in the write iterator
		const hkBlockStreamBase::Block* startBlockHeader = it->getBlockHeaderOnSpu( startBlockPpu, startBlockBuffer );
		hkBlockStreamBase::Block* currentBlockPpu = it->spuToPpu(it->m_currentBlock.val());
		if ( (startBlockPpu != currentBlockPpu) && startBlockHeader->getNumElements() == m_startBlockNumElements )	
		{
			// first block is empty,
			// advance to next block
			startBlockPpu = startBlockHeader->m_next;
			startBlockHeader = it->getBlockHeaderOnSpu( startBlockPpu, startBlockBuffer );
			m_startBlockNumElements = 0;
			m_startByteLocation = 0;
		}
		if (startBlockPpu == currentBlockPpu )
		{
			m_startBlockNumElements = hkBlockStreamBase::Block::CountType(it->m_currentBlockNumElems) - m_startBlockNumElements;
		}
		else
		{
			m_startBlockNumElements = hkBlockStreamBase::Block::CountType(startBlockHeader->getNumElements() - m_startBlockNumElements);
		}
		HK_ASSERT( 0xf045456, numElements >= m_startBlockNumElements );
		m_startBlock = startBlockPpu;
#endif
	}
	m_numElements = numElements;
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
