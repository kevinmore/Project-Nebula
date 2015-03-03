/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerScanSnapshot.h>

namespace
{
	struct RouterSizeInfo
	{
		void* m_allocator;
		hkInt32 m_size;
		hkInt32 m_alignedOffset; // only used in aligned{Allocate,Deallocate}
		#if HK_POINTER_SIZE==4
			int pad16;
		#endif
	};
	HK_COMPILE_TIME_ASSERT(sizeof(RouterSizeInfo)==16);
}


hkTrackerScanSnapshot::hkTrackerScanSnapshot(const hkTrackerSnapshot* snapshot, hkTrackerLayoutCalculator* layoutCalc):
	m_blockFreeList(sizeof(Block), HK_ALIGN_OF(Block), 4096),
	m_layoutCalc(layoutCalc)
{
	// Eventually the above m_allocations will go away and we'll use this.
	m_rawSnapshot.swap( const_cast<hkMemorySnapshot&>( snapshot->getRawSnapshot() ) );

	const char* stats = snapshot->getMemorySystemStatistics();
	if(stats)
	{
		m_memSysStatistics.append(stats, hkString::strLen(stats)+1);
	}
}

hkTrackerScanSnapshot::hkTrackerScanSnapshot()
	: m_blockFreeList(sizeof(Block), HK_ALIGN_OF(Block), 4096)
{
}

hkTrackerScanSnapshot::~hkTrackerScanSnapshot()
{
	clear();
}

const char* hkTrackerScanSnapshot::getTraceText(hkUlong addr) const
{
	const int index = m_traceMap.getWithDefault(addr, -1);
	if (index >= 0)
	{
		return &m_traceText[index];
	}
	return HK_NULL;
}

void hkTrackerScanSnapshot::setTraceText(hkUlong addr, const char* text)
{
	int index = m_traceMap.getWithDefault(addr, -1);
	if (index >= 0)
	{
		HK_ASSERT2(0x424aa234, false, "Snapshot already has text assoicated with the address");
		return;
	}

	index = m_traceText.getSize();
	int len = hkString::strLen(text);

	hkString::strCpy(m_traceText.expandBy(len + 1), text);
	m_traceMap.insert(addr, index);
}

void hkTrackerScanSnapshot::clear()
{
	m_blockFreeList.freeAll();
	m_references.clearAndDeallocate();
	m_blocks.clearAndDeallocate();
	m_blockMap.clear();
	m_typeText.clear();
}

hkTrackerScanSnapshot::Block* hkTrackerScanSnapshot::addBlock(const Node* type, const void* start, hk_size_t size)
{
	// Add the block
	Block* block = new (m_blockFreeList.alloc()) Block(type, start, size);
	m_blocks.pushBack(block);	
	m_blockMap.insert(start, block);

	return block;
}

hkTrackerScanSnapshot::Block* hkTrackerScanSnapshot::findBlock(const void* ptr) const 
{
	// If it points to a block... its a pointer
	Block* block = m_blockMap.getWithDefault(ptr, HK_NULL);
	if (block)
	{
		return block;
	}

	// Hack for handling string pointers
	void* easyPtr = (void*)((hkUlong(ptr) - 16) & (hkUlong(0) - 2));
	block = m_blockMap.getWithDefault(easyPtr, HK_NULL);
	if (block)
	{
		return block;
	}

	return HK_NULL;
}

const char* hkTrackerScanSnapshot::getMemorySystemStatistics() const
{
	if(m_memSysStatistics.isEmpty())
	{
		return HK_NULL;
	}
	else
	{
		return &m_memSysStatistics[0];
	}
}

HK_FORCE_INLINE static hkBool HK_CALL _orderBlocks(const hkTrackerScanSnapshot::Block* a, const hkTrackerScanSnapshot::Block* b)
{
	if (a->m_start == b->m_start)
	{
		// If they are the same order from largest to smallest block
		// We want this so when working out total used we always hit the largest (ie containing all other blocks) first
		return a->m_size > b->m_size;
	}
	else
	{
		return (char*)a->m_start < (char*)b->m_start;
	}
}

void hkTrackerScanSnapshot::orderBlocks()
{
	hkSort(m_blocks.begin(), m_blocks.getSize(), _orderBlocks);
}

hk_size_t hkTrackerScanSnapshot::calcTotalUsed() const
{
	return calcOrderedTotalUsed(m_blocks.begin(), m_blocks.getSize());
}

HK_FORCE_INLINE static hkBool HK_CALL _orderUsedBlocks(const hkTrackerScanSnapshot::Block* a, const hkTrackerScanSnapshot::Block* b)
{
	if (a->m_start == b->m_start)
	{
		// If they are the same order from largest to smallest block
		// We want this so when working out total used we always hit the largest (ie containing all other blocks) first
		return a->m_size > b->m_size;
	}
	else
	{
		return (char*)a->m_start < (char*)b->m_start;
	}
}

/* static */void hkTrackerScanSnapshot::orderBlocks(const Block** blocksIn, int numBlocks) 
{
	hkSort(blocksIn, numBlocks, _orderBlocks);
}

/* static */hk_size_t hkTrackerScanSnapshot::calcTotalUsed(const Block*const* blocksIn, int numBlocks)
{
	hkArray<const Block*> blocks;
	blocks.setSize(numBlocks);

	hkString::memCpy(blocks.begin(), blocksIn, numBlocks * sizeof(const Block*));

	orderBlocks(blocks.begin(), blocks.getSize());
	
	return calcOrderedTotalUsed(blocks.begin(), numBlocks);
}

/* static */hk_size_t hkTrackerScanSnapshot::calcOrderedTotalUsed(const Block*const* blocks, int numBlocks)
{
	if (numBlocks == 0)
	{
		return 0;
	}

	// The first block must be added
	const Block* prevBlock = blocks[0];
	hk_size_t totalSize = prevBlock->m_size;

	// Add other blocks if they are not inside
	for (int i = 1; i < numBlocks; i++)
	{
		const Block* block = blocks[i];
		
#ifdef HK_DEBUG
		{
			const Block* lastBlock = blocks[i - 1];
			// Must be later 
			HK_ASSERT(0x32423a32, (char*)lastBlock->m_start <= (char*)block->m_start);
			// If same address the larger block should be first
			HK_ASSERT(0x23432432, lastBlock->m_start != block->m_start || lastBlock->m_size >= block->m_size);
		}
#endif

		
		char* prevStart = (char*)prevBlock->m_start;
		char* prevEnd = prevStart + prevBlock->m_size;
		char* ptr = (char*)block->m_start;

		if (ptr >= prevStart && ptr < prevEnd)
		{
			// This block is contained in the previous block
			continue;
		}
		
		totalSize += block->m_size;
		prevBlock = block;
	}

	return totalSize;
}

void hkTrackerScanSnapshot::findBlocksByType(const char* name, hkArray<Block*>& blocksOut) const
{
	blocksOut.clear();
	for (int i = 0; i < m_blocks.getSize(); i++)
	{
		Block* block = m_blocks[i];

		if (block->m_type)
		{
			if (block->m_type->isNamedType() && block->m_type->m_name == name)
			{
				blocksOut.pushBack(block);
			}
		}
	}
}

int hkTrackerScanSnapshot::findReferenceIndex(const Block* block, const Block* refdBlock) const
{
	if (block->m_numReferences > 0)
	{
		Block*const* refs = getBlockReferences(block);
		for (int i = 0; i < block->m_numReferences; i++)
		{
			if (refs[i] == refdBlock)
			{
				return i;
			}
		}
	}

	return -1;
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
