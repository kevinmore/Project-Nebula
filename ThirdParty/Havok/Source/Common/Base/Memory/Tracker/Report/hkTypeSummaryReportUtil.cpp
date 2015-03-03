/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkTypeSummaryReportUtil.h>

/* static */void HK_CALL hkTypeSummaryReportUtil::report(hkTrackerScanSnapshot* scanSnapshot, FollowFilter* filter, hkOstream& stream)
{
	DontFollowMap dontFollowMap;
	hkArray<const Block*> rootBlocks;
	ParentMap parentMap;

	hkScanReportUtil::calcTypeRootBlocks(scanSnapshot, dontFollowMap, filter, parentMap, rootBlocks);

	hkArray<const Block*> remainingBlocks;
	hkArray<const Block*> children;

	// Any blocks that remain (ie that are in the snapshot, but not in the dont follow map, must be remaining blocks that were not reached)
	{
		const hkArray<Block*>& allBlocks = scanSnapshot->getBlocks();
		for (int i = 0; i < allBlocks.getSize(); i++)
		{
			const Block* block = allBlocks[i];
			hkArray<const Block*>& blocks = dontFollowMap.hasKey(block) ? children : remainingBlocks;
			blocks.pushBack(block);
		}
	}

	// 
	hkArray<TypeSummary> summaries;
	calcTypeSummaries(children, parentMap, summaries);
	// Okay lets order the summaries
	orderTypeSummariesBySize(summaries.begin(), summaries.getSize());

	hkTrackerLayoutCalculator* layoutCalc = scanSnapshot->getLayoutCalculator();
	for (int i = 0; i < summaries.getSize(); i++)
	{
		const TypeSummary& summary = summaries[i];

		hk_size_t typeSize = layoutCalc->calcTypeSize(summary.m_type);

		stream << MemorySize(summary.m_totalSize) << " for ";
		hkTrackerTypeTreeNode::dumpType(summary.m_type, stream);	
		stream << " ( " << summary.m_numInstances << " object/s X " << MemorySize(typeSize) << ", " << summary.m_numBlocks << " allocs)\n";
	}

	{
		hk_size_t blocksTotalSize = hkScanReportUtil::calcTotalSize(children);
		//hk_size_t rootBlocksTotalSize = hkScanReportUtil::calcTotalSize(rootBlocks);

		// << " Block total used:" << MemorySize(rootBlocksTotalSize) 
		stream << "Block total: " << MemorySize(blocksTotalSize) << "\n";
	}

	if (remainingBlocks.getSize() > 0)
	{
		hkStackTracer tracer;

		hk_size_t remainingSize = hkScanReportUtil::calcTotalSize(remainingBlocks);
		stream << remainingBlocks.getSize() << " block/s were not reached. Total unreached: " << MemorySize(remainingSize) << "\n";

		for (int i = 0; i < remainingBlocks.getSize(); i++)
		{
			const Block* block = remainingBlocks[i];

			hkTrackerTypeTreeNode::dumpType(block->m_type, stream);
			stream << " ";
			stream << "Not owned: " << (void*)block->m_start <<" Size: " << MemorySize(block->m_size) << "\n";
			// Dump the call stack
			hkScanReportUtil::dumpAllocationCallStack(&tracer, scanSnapshot, (void*)(block->m_start), stream);
		}
	}
}


/* static */void HK_CALL hkTypeSummaryReportUtil::calcTypeSummaries(const hkArray<const Block*>& children, const ParentMap& parentMap, hkArray<TypeSummary>& summaries)
{
	summaries.clear();

	for (int i = 0; i < children.getSize(); i++)
	{
		const Block* block = children[i];

		int summaryIndex = -1;

		// The type to add it to 
		const RttiNode* type = HK_NULL;

		if (block->m_arraySize < 0 && block->m_type && block->m_type->isNamedType())
		{
			type = block->m_type;
		}
		else
		{
			// Find the block that owns it
			const Block* parentBlock = parentMap.getWithDefault(block, HK_NULL);
			while (parentBlock)
			{
				if (parentBlock->m_arraySize < 0 && parentBlock->m_type && parentBlock->m_type->isNamedType() && children.indexOf(parentBlock) >= 0)
				{
					// Add to the memory of this block
					type = parentBlock->m_type;
					break;
				}

				parentBlock = parentMap.getWithDefault(parentBlock, HK_NULL);
			}
		}

		// Look for the type in the type summary
		summaryIndex = findSummaryByType(summaries, type);
		if (summaryIndex < 0)
		{
			// Look for the type in the type summary
			summaryIndex = findSummaryByType(summaries, block->m_type);
			if (summaryIndex < 0)
			{
				summaryIndex = summaries.getSize();
				TypeSummary& summary = summaries.expandOne();
				summary.m_type = type;
				summary.m_numInstances = 0;
				summary.m_totalSize = 0;
				summary.m_numBlocks = 0;
				summary.m_totalInstanceSize = 0;
			}
		}

		// Total up the summary
		TypeSummary& summary = summaries[summaryIndex];
		summary.m_totalSize += block->m_size;

		if (type == block->m_type)
		{
			summary.m_numInstances ++;
			summary.m_totalInstanceSize += block->m_size;
		}
		summary.m_numBlocks++;
	}
}

/* static */void hkTypeSummaryReportUtil::dumpSummaries(const hkArray<TypeSummary>& summaries, hkOstream& stream)
{
	{
		TypeSummary summary;
		calcTotalSummary(summaries.begin(), summaries.getSize(), summary);
		stream << "TOTAL - Size: " << MemorySize(summary.m_totalSize) << " Num Types: " << summaries.getSize() << " Num Instances: " << summary.m_numInstances << " Num Blocks: " << summary.m_numBlocks << "\n";
	}

	for (int j = 0; j < summaries.getSize(); j++)
	{
		const TypeSummary& summary = summaries[j];

		stream << "  ";
		if (summary.m_type)
		{
			hkTrackerTypeTreeNode::dumpType(summary.m_type, stream);
			stream << " (" << summary.m_numInstances << ")";
		}
		else
		{
			stream << "(Blocks)";
		}
		stream << " Num Blocks: " << summary.m_numBlocks << " Total size: " << MemorySize(summary.m_totalSize) << "\n";
	}
}

/* static */void HK_CALL hkTypeSummaryReportUtil::calcTotalSummary(const TypeSummary* summaries, int numSummaries, TypeSummary& summaryOut)
{
	summaryOut.m_type = HK_NULL;
	summaryOut.m_numInstances = 0;
	summaryOut.m_numBlocks = 0;
	summaryOut.m_totalSize = 0;
	summaryOut.m_totalInstanceSize = 0;

	for (int i = 0; i < numSummaries; i++)
	{
		const TypeSummary& summary = summaries[i];

		summaryOut.m_numBlocks += summary.m_numBlocks;
		summaryOut.m_numInstances += summary.m_numInstances;
		summaryOut.m_totalSize += summary.m_totalSize;
		summaryOut.m_totalInstanceSize += summary.m_totalInstanceSize;
	}
}

static hkBool _orderSummaryBySize(const hkTypeSummaryReportUtil::TypeSummary& a, const hkTypeSummaryReportUtil::TypeSummary& b)
{
	return a.m_totalSize > b.m_totalSize; 
}

/* static */void HK_CALL hkTypeSummaryReportUtil::orderTypeSummariesBySize(TypeSummary* summaries, int numSummaries)
{
	hkSort(summaries, numSummaries, _orderSummaryBySize);
}

/* static */int HK_CALL hkTypeSummaryReportUtil::findSummaryByType(const hkArray<TypeSummary>& summaries, const RttiNode* type)
{
	for (int i = 0; i < summaries.getSize(); i++)
	{
		if (summaries[i].m_type == type)
		{
			return i;
		}
	}

	return -1;
}

static hkBool _orderSummaryByName(const hkTypeSummaryReportUtil::TypeSummary& a, const hkTypeSummaryReportUtil::TypeSummary& b)
{
	if (a.m_type == HK_NULL || b.m_type == HK_NULL)
	{
		return a.m_type < b.m_type;
	}
	return a.m_type->m_name.compareTo(b.m_type->m_name) < 0;
}

/* static */void HK_CALL hkTypeSummaryReportUtil::orderTypeSummariesByName(TypeSummary* summaries, int numSummaries)
{
	hkSort(summaries, numSummaries, _orderSummaryByName);
}

/* static */void HK_CALL hkTypeSummaryReportUtil::calcTypeSummaries(const hkArray<const Block*>& blocks, hkArray<TypeSummary>& summaries)
{
	summaries.clear();

	hkPointerMap<const RttiNode*, int> map;

	for (int i = 0; i < blocks.getSize(); i++)
	{
		const Block* block = blocks[i];

		const RttiNode* type = HK_NULL;
		if (block->m_arraySize < 0)
		{
			type = block->m_type;
		}

		int index = map.getWithDefault(type, -1);
		if (index < 0)
		{
			index = summaries.getSize();
			TypeSummary& summary = summaries.expandOne();

			summary.m_type = type;
			summary.m_numInstances = 0;
			summary.m_numBlocks = 0;
			summary.m_totalSize = 0;
			summary.m_totalInstanceSize = 0;

			map.insert(type, index);
		}

		TypeSummary& summary = summaries[index];

		HK_ASSERT(0x4234324, summary.m_type == type);
		summary.m_numInstances++;
		summary.m_numBlocks++;
		summary.m_totalSize += block->m_size;
		summary.m_totalInstanceSize += block->m_size;
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
