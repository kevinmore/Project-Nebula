/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkProductReportUtil.h>

/* static */void HK_CALL hkProductReportUtil::calcCategories(hkTrackerScanSnapshot* scanSnapshot, FollowFilter* filter, ParentMap& parentMap, TypeIndexMap& typeIndexMap, hkArray<const Block*>& rootBlocks)
{
	rootBlocks.clear();
	parentMap.clear();

	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hk", true, PRODUCT_COMMON, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkp", true, PRODUCT_PHYSICS_2012, typeIndexMap);	
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hknp", true, PRODUCT_PHYSICS, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkd", true, PRODUCT_DESTRUCTION_2012, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hknd", true, PRODUCT_DESTRUCTION, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkg", true, PRODUCT_GRAPHICS, typeIndexMap);

	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hka", true, PRODUCT_ANIMATION, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkai", true, PRODUCT_AI, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkb", true, PRODUCT_BEHAVIOR, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkgp", true, PRODUCT_GEOMETRY_PROCESSING, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hcl", true, PRODUCT_CLOTH, typeIndexMap);

	/// Any remaining
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "", false, PRODUCT_OTHER, typeIndexMap);

	hkScanReportUtil::appendBlocksWithTypeIndex(scanSnapshot, typeIndexMap, rootBlocks);

	// Make them all owned
	DontFollowMap dontFollowMap;
	for (int i = 0; i < rootBlocks.getSize(); i++)
	{
		dontFollowMap.insert(rootBlocks[i], 1);
	}

	for (int i = 0; i < rootBlocks.getSize(); i++)
	{
		hkScanReportUtil::appendParentMap(scanSnapshot, rootBlocks[i], &dontFollowMap, filter, parentMap);
	}
}

/* static */ const char* hkProductReportUtil::getProductName(Product type)
{
	switch (type)
	{
		case PRODUCT_AI:					return "Ai";
		case PRODUCT_ANIMATION:				return "Animation";
		case PRODUCT_BEHAVIOR:				return "Behavior";
		case PRODUCT_CLOTH:					return "Cloth";
		case PRODUCT_COMMON:				return "Common";
		case PRODUCT_DESTRUCTION_2012:		return "Destruction 2012";
		case PRODUCT_DESTRUCTION:			return "Destruction";
		case PRODUCT_PHYSICS_2012:			return "Physics 2012";
		case PRODUCT_PHYSICS:				return "Physics";
		case PRODUCT_GEOMETRY_PROCESSING:	return "Geometry Processing";
		case PRODUCT_GRAPHICS:				return "Havok Graphics";
		case PRODUCT_OTHER:					return "Other";
		case PRODUCT_LAST:					break; 
	}
	return "Invalid";
}

static hkBool _orderSummaryBySize(const hkTypeSummaryReportUtil::TypeSummary& a, const hkTypeSummaryReportUtil::TypeSummary& b)
{
	return a.m_totalSize > b.m_totalSize; 
}

/* static */void HK_CALL hkProductReportUtil::reportSummary(hkTrackerScanSnapshot* scanSnapshot, FollowFilter* filter, hkOstream& stream)
{
	stream << 
		"hkProductReportUtil::reportSummary\n" 
		"----------------------------------\n\n"
		"Report puts blocks into categories based on product, and then reports\n"
		"in summary the amount of blocks and memory usage by type. \n"
		"\n"
		"PRODUCT - the product the memory belongs to \n"
		"  Allocated size - total memory allocated for the blocks in the product\n"
		"TOTAL - Totals for all of that product\n"
		"  Num types - the number of types found in that product\n"
		"  Num instances - number of instances of those types\n"
		"  Num blocks - number of blocks owned by the instances\n"
		"\n"
		"For a type - \n"
		"  The number in brackets is the number of instances of the type\n"
		"  Num blocks - number of blocks owned by instances of the type\n"
		"  Total size - total size in bytes of all blocks owned by that type\n" 
		"\n"
		"Blocks listed as '(Blocks)' have indeterminate type\n\n";

	typedef hkScanReportUtil::MemorySize MemorySize;

	ParentMap parentMap;
	TypeIndexMap typeIndexMap;
	hkArray<const Block*> rootBlocks;

	calcCategories(scanSnapshot, filter, parentMap, typeIndexMap, rootBlocks);

	// Work out the child map
	ChildMultiMap childMultiMap;
	hkScanReportUtil::calcChildMap(parentMap, childMultiMap);

	{
		const hkArray<hkTrackerScanSnapshot::Block*>& blocks = scanSnapshot->getBlocks();

		hk_size_t usedSize = hkTrackerScanSnapshot::calcTotalUsed(blocks.begin(), blocks.getSize());

		stream << "Total used: " << MemorySize(usedSize) << "\n\n";
	}

	// Okay I'm now ready for reporting... 

	{
		hkArray<hkTypeSummaryReportUtil::TypeSummary> summaries;
		hkArray<const Block*> blocks;

		for (int i = 0; i < PRODUCT_LAST; i++)
		{
			summaries.clear();
			blocks.clear();

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

			hkTypeSummaryReportUtil::calcTypeSummaries(blocks, parentMap, summaries);

			if (summaries.getSize() > 0)
			{
				// Order largest to smallest
				hkTypeSummaryReportUtil::orderTypeSummariesBySize(summaries.begin(), summaries.getSize());

				stream << "PRODUCT: " << getProductName(Product(i))<< "\n";

				hkTypeSummaryReportUtil::dumpSummaries(summaries, stream);
				stream << "\n";
			}
		}
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
