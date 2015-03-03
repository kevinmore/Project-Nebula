/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkTulipReportUtil.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Memory/Tracker/Report/hkCategoryReportUtil.h>
#include <Common/Base/Types/Color/hkColor.h>

static void HK_CALL _reportLabelProperties(hkTrackerScanSnapshot* scanSnapshot, const hkPointerMap<const hkTrackerScanSnapshot::Block*, int>& map, hkOstream& stream)
{
	typedef hkTulipReportUtil::Block Block;
	const hkArray<Block*>& blocks = scanSnapshot->getBlocks();

	for (int i = 0; i < blocks.getSize(); i++)
	{
		const Block* block = blocks[i];
		const int index = map.getWithDefault(block, -1);

		stream << "  (node " << index;
		
		
		stream << " \"";
		hkTrackerTypeTreeNode::dumpType(block->m_type, stream);	
		if (block->m_arraySize >= 0)
		{
			stream << "[" << block->m_arraySize << "] ";
		}
		stream << "@" << (void*)block->m_start << " Size: " << hkTulipReportUtil::MemorySize(block->m_size);
		stream <<"\")\n"; 
	}
}

static void HK_CALL _reportProperties(hkTrackerScanSnapshot* scanSnapshot, const hkPointerMap<const hkTrackerScanSnapshot::Block*, int>& map, hkOstream& stream)
{
	typedef hkTulipReportUtil::Block Block;
	typedef hkTulipReportUtil::MemorySize MemorySize;

	const hkArray<Block*>& blocks = scanSnapshot->getBlocks();

	// Do the address property
	{
		stream << "(property 0 string \"startAddress\"\n";
		stream << "  (default \"null\" \"null\")\n";

		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			const int index = map.getWithDefault(block, -1);
			stream << "  (node " << index << " \"" << (void*)block->m_start<<"\")\n"; 
		}

		stream << ")\n\n";
	}

	// Do the size property
	{
		stream << "(property 0 double \"size\"\n";
		stream << "  (default \"0\" \"0\")\n";

		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			const int index = map.getWithDefault(block, -1);
			stream << "  (node " << index << " \"" << MemorySize(block->m_size, MemorySize::FLAG_RAW) <<"\")\n"; 
		}

		stream << ")\n\n";
	}

	// viewSize

	{
		stream << "(property 0 size \"viewSize\"\n";
		stream << "  (default \"(2,2,2)\" \"(1,1,0)\")\n";

		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			const int index = map.getWithDefault(block, -1);

			// Work out the size - to display as a volume
			const hkReal size = hkMath::pow(hkReal(block->m_size), hkReal(1.0f / 3.0f) );
			stream << "  (node " << index << " \"(" << size << "," << size << "," << size << ")\")\n"; 
		}

		stream << ")\n\n";
	}

	
}

void HK_CALL hkTulipReportUtil::report(hkTrackerScanSnapshot* scanSnapshot, hkOstream& stream)
{
	// First I have to output the nodes -> in this case they are the blocks

	const hkArray<Block*>& blocks = scanSnapshot->getBlocks();

	hkPointerMap<const Block*, int> map;

	stream << "(tlp \"2.0\"\n";


	stream << "(nodes ";
	for (int i = 0; i < blocks.getSize(); i++)
	{
		const Block* block = blocks[i];
		stream << i << " ";
		map.insert(block, i);
	}
	stream << ")";

	{
		int edgeId = 0;

		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			const int fromIndex = map.getWithDefault(block, -1);

			Block*const* refs = scanSnapshot->getBlockReferences(block);
			const int numRefs = block->m_numReferences;

			for (int j = 0; j < numRefs; j++)
			{
				const Block* to = refs[j];
				if (to)
				{
					int toIndex = map.getWithDefault(to, -1);
					HK_ASSERT(0x23432aa, toIndex >= 0);

					stream << "(edge " << edgeId << " " << fromIndex << " " << toIndex << ")\n";

					edgeId++;
				}
			}
		}
	}

	_reportProperties(scanSnapshot, map, stream);
	
	// The view label
	{
		stream << "(property 0 string \"viewLabel\"\n";
		stream << "  (default \"unknown\" \"unknown\")\n";
		_reportLabelProperties(scanSnapshot, map, stream);
		stream << ")\n\n";
	}

	stream << ")\n";
}


void HK_CALL hkTulipReportUtil::reportCategorySummary(hkTrackerScanSnapshot* scanSnapshot, hkScanReportUtil::FollowFilter* filter, hkOstream& stream)
{
	hkScanReportUtil::ParentMap parentMap;
	hkScanReportUtil::TypeIndexMap typeIndexMap;
	hkArray<const Block*> rootBlocks;

	hkCategoryReportUtil::calcCategories(scanSnapshot, filter, parentMap, typeIndexMap, rootBlocks);

	// Work out the child map
	ChildMultiMap childMultiMap;
	hkScanReportUtil::calcChildMap(parentMap, childMultiMap);

	// 
	hkPointerMap<const Block*, int> categoryMap;

	int nodeIndex = 0;

	// 
	nodeIndex += 1 + hkCategoryReportUtil::CATEGORY_LAST;

	// Okay I'm now ready for reporting... 
	{
		hkArray<const Block*> blocks;

		for (int i = hkCategoryReportUtil::CATEGORY_WORLD; i < hkCategoryReportUtil::CATEGORY_LAST; i++)
		{
			blocks.clear();
			for (int j = 0; j < rootBlocks.getSize(); j++)
			{
				const Block* rootBlock = rootBlocks[j];

				if (hkScanReportUtil::getTypeIndex(rootBlock, typeIndexMap) == i)
				{
					blocks.pushBack(rootBlock);
					// Add the children
					hkScanReportUtil::appendChildren(rootBlock, childMultiMap, blocks);
				}
			}

			for (int j = 0; j < blocks.getSize(); j++)
			{
				categoryMap.insert(blocks[j], i);
			}
		}
	}

	{
		const hkArray<hkTrackerScanSnapshot::Block*>& blocks = scanSnapshot->getBlocks();
		hkArray<const Block*> remainingBlocks;
		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			if (!parentMap.hasKey(block))
			{
				categoryMap.insert(block, hkCategoryReportUtil::CATEGORY_REMAINING);
			}
		}
	}

	{
		const hkArray<Block*>& blocks = scanSnapshot->getBlocks();

		//const char* categoryName = getCategoryName(TypeCategory(i));

		hkPointerMap<const Block*, int> map;	
		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			map.insert(block, i + nodeIndex);
		}

		stream << "(tlp \"2.0\"\n";

		// 
		stream << "(nodes \n";
		
		stream << "0\n";
		for (int i = 0; i < hkCategoryReportUtil::CATEGORY_LAST; i++)
		{
			stream << i + 1 << " ";
		}
		stream << "\n";

		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			stream << map.getWithDefault(block, -1) << " ";
		}
		stream << ")";

		int edgeId = 0;

		{
			for (int i = 0; i < hkCategoryReportUtil::CATEGORY_LAST; i++)
			{
				stream << "(edge " << edgeId << " 0 " << i + 1 << ")\n";
				edgeId++;
			}
		}

		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];

			if (parentMap.getWithDefault(block, HK_NULL) == HK_NULL)
			{
				int category = categoryMap.getWithDefault(block, -1);
				HK_ASSERT(0x3423a432, category >= 0);

				int toIndex = map.getWithDefault(block, -1);

				stream << "(edge " << edgeId << " " << (category + 1) << " " << toIndex << ")\n";
				edgeId++;
			}
		}

		{

			for (int i = 0; i < blocks.getSize(); i++)
			{
				const Block* block = blocks[i];
				const int fromIndex = map.getWithDefault(block, -1);

				// Get the children
				ChildMultiMap::Iterator iter = childMultiMap.findKey(block);

				for (; childMultiMap.isValid(iter); iter = childMultiMap.getNext(iter, block))
				{
					const Block* to = childMultiMap.getValue(iter);
					int toIndex = map.getWithDefault(to, -1);
					HK_ASSERT(0x23432aa, toIndex >= 0);

					stream << "(edge " << edgeId << " " << fromIndex << " " << toIndex << ")\n";

					edgeId++;
				}
			}
		}

		_reportProperties(scanSnapshot, map, stream);

		{
			stream << "(property 0 string \"viewLabel\"\n";
			stream << "  (default \"unknown\" \"unknown\")\n";

			for (int i = 0; i < hkCategoryReportUtil::CATEGORY_LAST; i++)
			{
				stream << "  (node " << i + 1;
					
				stream << " \"";
				const char* categoryName = hkCategoryReportUtil::getCategoryName(hkCategoryReportUtil::Category(i));
				stream << categoryName;
				
				stream <<"\")\n"; 
			}

			_reportLabelProperties(scanSnapshot, map, stream);

			stream << ")\n\n";
		}

		hkPseudoRandomGenerator rand(0x4234);
		hkColor::Argb colors[hkCategoryReportUtil::CATEGORY_LAST];

		for (int i = 0; i < hkCategoryReportUtil::CATEGORY_LAST; i++)
		{
			colors[i] = hkColor::getRandomColor(rand);
		}

		stream << "(property  0 color \"viewColor\"\n";
		stream << "(default \"(0,0,0,255)\" \"(0,0,0,255)\")\n";

		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			const int index = map.getWithDefault(block, -1);

			int category = categoryMap.getWithDefault(block, -1);
			HK_ASSERT(0x3423a432, category >= 0);

			hkColor::Argb color = colors[category];

			int r = (color >> 16) & 0xff;
			int g = (color >> 8) & 0xff;
			int b = (color) & 0xff;

			stream << "  (node " << index << " \"(255," << r << "," << g << "," << b << ")\")\n";
		}
		stream << ")\n";

		stream << ")\n";
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
