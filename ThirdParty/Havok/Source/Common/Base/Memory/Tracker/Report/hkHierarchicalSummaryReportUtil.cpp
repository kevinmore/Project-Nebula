/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkHierarchicalSummaryReportUtil.h>

namespace // anonymous
{

struct SimplifiedSummaryReport
{
	typedef hkScanReportUtil::ChildMultiMap ChildMultiMap;
	typedef hkScanReportUtil::ParentMap ParentMap;
	typedef hkScanReportUtil::Block Block;
	typedef hkScanReportUtil::MemorySize MemorySize;

	SimplifiedSummaryReport(hkOstream& stream, int maxDepth, hkTrackerLayoutCalculator* layoutCalc):
		m_layoutCalc(layoutCalc),
		m_maxDepth(maxDepth),
		m_stream(stream)
	{
	}

		// Report
	void report(hkTrackerScanSnapshot* scanSnapshot, const Block* rootBlock, hkScanReportUtil::FollowFilter* filter);
	void report(hkTrackerScanSnapshot* scanSnapshot, const Block* rootBlock, const ParentMap& parentMap, const ChildMultiMap& childMultiMap);
	void _report(const Block* block, int depth);

	const ChildMultiMap* m_childMultiMap;
	const ParentMap* m_parentMap;
	hkPointerMap<const Block*, hk_size_t> m_sizeMap;

	hkTrackerScanSnapshot* m_scanSnapshot;
	hkTrackerLayoutCalculator* m_layoutCalc;
	int m_maxDepth;
	hkOstream& m_stream;
};

void SimplifiedSummaryReport::report(hkTrackerScanSnapshot* scanSnapshot, const Block* rootBlock, hkScanReportUtil::FollowFilter* filter)
{
	ParentMap parentMap;
	hkScanReportUtil::calcParentMap(scanSnapshot, rootBlock, filter, parentMap);
	ChildMultiMap childMultiMap;
	hkScanReportUtil::calcChildMap(parentMap, childMultiMap);

	report(scanSnapshot, rootBlock, parentMap, childMultiMap);
}

void SimplifiedSummaryReport::report(hkTrackerScanSnapshot* scanSnapshot, const Block* rootBlock, const ParentMap& parentMap, const ChildMultiMap& childMultiMap)
{
	m_parentMap = &parentMap;
	m_childMultiMap = &childMultiMap;
	m_scanSnapshot = scanSnapshot;

	{
		ParentMap::Iterator iter = parentMap.getIterator();

		for (; parentMap.isValid(iter); iter = parentMap.getNext(iter))
		{
			const Block* block = parentMap.getKey(iter);
			const Block* cur = block;

			while (cur)
			{
				// Total it up
				m_sizeMap.insert(cur, m_sizeMap.getWithDefault(cur, 0) + block->m_size);
				// 
				cur = parentMap.getWithDefault(cur, HK_NULL);
			}
		}
	}

	_report(rootBlock, 0);
}

void SimplifiedSummaryReport::_report(const Block* block, int depth)
{
	hkScanReportUtil::appendSpaces(m_stream, depth * 2);

	// Output this blocks info

	if (block->m_type)
	{
		hkTrackerTypeTreeNode::dumpType(block->m_type, m_stream);
	}
	else
	{
		{
			const Block* parentBlock = m_parentMap->getWithDefault(block, HK_NULL);
			if (parentBlock)
			{
				const char* name = hkScanReportUtil::calcMemberName(m_scanSnapshot, parentBlock, block);
				if (name)
				{
					m_stream << "." << name << " ";
				}
			}
		}

		m_stream << "#";
	}
	if (block->m_arraySize >= 0)
	{
		m_stream << "[" << block->m_arraySize << "]";
	}
	m_stream << " Size:" << MemorySize(block->m_size) << " Total Size: " << MemorySize(m_sizeMap.getWithDefault(block, 0)) << "\n";

	if (depth >= m_maxDepth)
	{
		// Find all of the child
		hkArray<const Block*> children;
		hkScanReportUtil::findChildren(block, *m_childMultiMap, children);

		hkArray<hkTypeSummaryReportUtil::TypeSummary> summaries;
		hkTypeSummaryReportUtil::calcTypeSummaries(children, *m_parentMap, summaries);
		hkTypeSummaryReportUtil::orderTypeSummariesBySize(summaries.begin(), summaries.getSize());

		for (int i = 0; i < summaries.getSize(); i++)
		{
			const hkTypeSummaryReportUtil::TypeSummary& summary = summaries[i];

			hkScanReportUtil::appendSpaces(m_stream, depth * 2 + 2);
			if (summary.m_type)
			{
				hkTrackerTypeTreeNode::dumpType(summary.m_type, m_stream);
				m_stream << " (" << summary.m_numInstances << ")";
			}
			else
			{
				m_stream << "(Blocks)";
			}
			m_stream << " Num Blocks: " << summary.m_numBlocks << " Total size: " << MemorySize(summary.m_totalSize) << "\n";
		}
	}
	else
	{
		ChildMultiMap::Iterator iter = m_childMultiMap->findKey(block);
		for (; m_childMultiMap->isValid(iter); iter = m_childMultiMap->getNext(iter, block))
		{
			const Block* child = m_childMultiMap->getValue(iter);
			_report(child, depth + 1);
		}
	}
}

} // anonymous


/* static */ void HK_CALL hkHierarchicalSummaryReportUtil::report(hkTrackerScanSnapshot* scanSnapshot, int maxDepth, const Block* rootBlock, FollowFilter* filter, hkOstream& stream)
{
	SimplifiedSummaryReport report(stream, maxDepth, scanSnapshot->getLayoutCalculator());
	report.report(scanSnapshot, rootBlock, filter);
}

/* static */ void HK_CALL hkHierarchicalSummaryReportUtil::report(hkTrackerScanSnapshot* scanSnapshot, int maxDepth, const hkArray<const Block*>& rootBlocks, const ParentMap& parentMap, const ChildMultiMap& childMultiMap, hkOstream& stream)
{
	// Get the roots
	SimplifiedSummaryReport report(stream, maxDepth, scanSnapshot->getLayoutCalculator());
	for (int i = 0; i < rootBlocks.getSize(); i++)
	{
		report.report(scanSnapshot, rootBlocks[i], parentMap, childMultiMap);
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
