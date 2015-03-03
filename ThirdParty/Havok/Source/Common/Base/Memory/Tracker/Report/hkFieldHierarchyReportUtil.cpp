/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkFieldHierarchyReportUtil.h>

namespace // anonymous
{

struct HierarchyInfo
{
	int m_depth;								///< The depth
	const hkTrackerScanSnapshot::Block* m_block;		///< The block
	const char* m_name;							///< The name of the member that references this
};

} // anonymous


void HK_CALL hkFieldHierarchyReportUtil::report(hkTrackerScanSnapshot* scanSnapshot, const Block* rootBlock, FollowFilter* filter, hkOstream& stream)
{
	hkArray<HierarchyInfo> stack;
	ParentMap parentMap;

	// Output the block
	{
		HierarchyInfo& info = stack.expandOne();
		info.m_block = rootBlock;
		info.m_depth = 0;
		info.m_name = HK_NULL;

		// Mark as visited
		parentMap.insert(rootBlock, HK_NULL);
	}

	hkTrackerLayoutCalculator* layoutCalc = scanSnapshot->getLayoutCalculator();
	while (stack.getSize() > 0)
	{
		// Output the block
		const HierarchyInfo& info = stack.back();
		int depth = info.m_depth;
		const Block* block = info.m_block;
		const char* name = info.m_name;
		stack.popBack();

		const hkTrackerTypeLayout* layout = HK_NULL;
		if (block->m_type)
		{
			layout = layoutCalc->getLayout(block->m_type);
		}

		hkScanReportUtil::appendSpaces(stream, depth * 2);
		if (name)
		{
			stream << name << " = ";
		}

		if (block == HK_NULL)
		{
			stream << "NULL\n";
			continue;
		}
		else
		{
			hkScanReportUtil::appendBlockType(block, stream);
			stream << "@" << (void*)block->m_start << " ";
			stream << " Size:" << MemorySize(block->m_size) << "\n";
		}

		depth++;

		const int numRefs = block->m_numReferences;
		Block*const* refs = scanSnapshot->getBlockReferences(block);
		// Add the members which are not visited
		for (int j = numRefs - 1; j >= 0; j--)
		{
			Block* childBlock = refs[j];

			if (childBlock == HK_NULL || parentMap.hasKey(childBlock))
			{
				continue;
			}

			int memberIndex = -1;
			if (layout && layout->m_fullScan == false)
			{
				memberIndex = j % layout->m_members.getSize();
			}

			if (filter && !filter->shouldFollow(block, childBlock, layoutCalc, layout, memberIndex))
			{
				continue;
			}

			parentMap.insert(childBlock, block);

			// Add to the stack

			HierarchyInfo& childInfo = stack.expandOne();
			childInfo.m_block = childBlock;
			childInfo.m_depth = depth;
			childInfo.m_name = HK_NULL;

			if (memberIndex >= 0 && layout)
			{
				childInfo.m_name = layout->m_members[memberIndex].m_name;
			}
		}

		depth--;
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
