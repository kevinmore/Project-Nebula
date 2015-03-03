/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkStackTraceReportUtil.h>
#include <Common/Base/Memory/Tracker/Report/hkTypeSummaryReportUtil.h>

namespace // anonymous
{

	struct Entry
	{
		hkUlong m_addr;
		const hkMemorySnapshot::Allocation* m_alloc;
	};
}


HK_FORCE_INLINE static hkBool _orderEntries(const Entry& a, const Entry& b)
{
	if (a.m_addr == b.m_addr)
	{
		// Order by size 
		return a.m_alloc->m_size > b.m_alloc->m_size;
	}
	return a.m_addr < b.m_addr;
}

HK_FORCE_INLINE static hkBool _orderSummaries(const hkStackTraceReportUtil::Summary& a, const hkStackTraceReportUtil::Summary& b)
{
	return a.m_totalSize > b.m_totalSize; 
}

static void HK_CALL _dumpStackTrace(const char* text, void* context)
{
	hkOstream& stream = *(hkOstream*)context;
	stream << text;
}

/* static */void HK_CALL hkStackTraceReportUtil::report(const hkTrackerScanSnapshot* scanSnapshot, FollowFilter* filter, hkOstream& stream)
{

	stream << 
		"hkStackTraceReportUtil::report\n"
		"------------------------------\n"
		"This report reports in summary all of the memory that was allocated\n"
		"under a specific stack trace address. To clarify - when memory is allocated\n"
		"a stack trace is recordeded. In this report, the memory will be added to every\n"
		"address specified in each callstack. This will lead to all memory being under 'main'\n"
		"and less under functions.\n"
		"Basically the report shows the hot spots of allocation\n\n"
		"Total Alloc Size - Total allocations made under the method\n"
		"Num allocs - Total number of allocations made under the method\n"
		"TOTAL - Totals for all of that category\n"
		"  Num types - the number of types found in that category\n"
		"  Num instances - number of instances of those types\n"
		"  Num blocks - number of blocks owned by the instances\n"
		"\n"
		"For a type - \n"
		"  The number in brackets is the number of instances of the type\n"
		"  Num blocks - number of blocks owned by instances of that type\n"
		"  Total size - total size in bytes of all blocks owned by that type\n" 
		"\n"
		"Blocks listed as '(Blocks)' have indeterminate type\n\n";

	// Get all the allocations 
	hkStackTracer tracer;

	hkArray<Entry> entries;
	hkArray<hkUlong> callStack;

	{
		hkPointerMap<hkUlong, int> uniqueAddrMap;

		const hkStackTracer::CallTree& callTree = scanSnapshot->getRawSnapshot().getCallTree();
		const hkArrayBase<hkMemorySnapshot::Allocation>& allocs = scanSnapshot->getRawSnapshot().getAllocations();
		for (int i = 0; i < allocs.getSize(); i++)
		{
			const hkMemorySnapshot::Allocation& alloc = allocs[i];
			if (alloc.m_traceId >= 0)
			{
				// Get the trace
				int stackSize = callTree.getCallStackSize(alloc.m_traceId);

				callStack.setSize(stackSize);
				callTree.getCallStack(alloc.m_traceId, callStack.begin(), stackSize);

				// Find the unique addresses in the call stack
				uniqueAddrMap.clear();
				for (int j = 0; j < stackSize; j++)
				{
					uniqueAddrMap.insert(callStack[j], 1);
				}

				Entry* dstEntries = entries.expandBy(uniqueAddrMap.getSize());

				// Add an entry for each one
				hkPointerMap<hkUlong, int>::Iterator iter = uniqueAddrMap.getIterator();
				for (; uniqueAddrMap.isValid(iter); iter = uniqueAddrMap.getNext(iter))
				{
					const hkUlong addr = uniqueAddrMap.getKey(iter);

					Entry& entry = *dstEntries++;
					entry.m_addr = addr;
					entry.m_alloc = &alloc;
				}
			}
		}
	}

	// Order
	hkSort(entries.begin(), entries.getSize(), _orderEntries);

	// Maps from each addr to the entries about that types start in entries
	hkPointerMap<hkUlong, int> startMap;
	{
		hkUlong cur = 0;
		for (int i = 0; i < entries.getSize(); i++)
		{
			hkUlong addr = entries[i].m_addr;
			if (addr != cur)
			{
				startMap.insert(addr, i);
				cur = addr;
			}
		}
	}

	// Get the summaries in order of largest to smallest
	hkArray<Summary> summaries;
	calcSummaries(scanSnapshot, summaries);

	{
		DontFollowMap dontFollowMap;

		{
			const hkArray<Block*>& allBlocks = scanSnapshot->getBlocks();
			for (int i = 0; i < allBlocks.getSize(); i++)
			{
				dontFollowMap.insert(allBlocks[i], 1);
			}
		}

		hkTrackerLayoutCalculator* layoutCalc = scanSnapshot->getLayoutCalculator();

		hkArray<const Block*> blocks;
		hkArray<const Block*> rootBlocks;
		ParentMap parentMap;
		hkArray<hkTypeSummaryReportUtil::TypeSummary> typeSummaries;

		for (int i = 0; i < summaries.getSize(); i++)
		{
			const Summary& summary = summaries[i];
			const hkUlong addr = summary.m_addr;

			int start = startMap.getWithDefault(addr, -1);
			HK_ASSERT(0x3242a423, start >= 0);

			// 
			tracer.dumpStackTrace( &summary.m_addr, 1, _dumpStackTrace, &stream);
			//stream << "\n";
			stream << "Total Alloc Size: " << MemorySize(summary.m_totalSize) << " Num allocs: " << int(summary.m_totalNumAllocs) << "\n";

			stream << "Blocks summary (generally smaller than allocations)\n";

			// Dump information about the types etc.

			blocks.clear();
			rootBlocks.clear();
			parentMap.clear();

			for (int j = start; j < entries.getSize() && entries[j].m_addr == addr; j++)
			{
				const Entry& entry = entries[j];
				Block* block = scanSnapshot->findBlock(entry.m_alloc->m_start);
				// 
				if (block)
				{
					blocks.pushBack(block);
					// Remove from the map
					dontFollowMap.remove(block);
				}
			}

			// Work out the root blocks
			for (int j = 0; j < blocks.getSize(); j++)
			{
				const Block* block = blocks[j];
				if (block->m_type == HK_NULL)
				{
					continue;
				}

				const hkTrackerTypeLayout* typeLayout = layoutCalc->getLayout(block->m_type);
				if (typeLayout && typeLayout->m_isVirtual)
				{
					// Add it to the blockers
					dontFollowMap.insert(block, 1);

					// Add the block... they will be the roots
					rootBlocks.pushBack(block);
				}
			}

			// Go through root blocks, working out the parent map
			hkScanReportUtil::appendParentAndDontFollowMap(scanSnapshot, rootBlocks, dontFollowMap, filter, parentMap);

			// 
			hkTypeSummaryReportUtil::calcTypeSummaries(blocks, parentMap, typeSummaries);
			hkTypeSummaryReportUtil::orderTypeSummariesBySize(typeSummaries.begin(), typeSummaries.getSize());

			// Dump the summaries
			hkTypeSummaryReportUtil::dumpSummaries(typeSummaries, stream);

			// Add a gap
			stream << "\n\n";

			// Re-add the blocks
			for (int j = 0; j < blocks.getSize(); j++)
			{
				dontFollowMap.insert(blocks[j], 1);
			}
		}
	}
}

/* static */void HK_CALL hkStackTraceReportUtil::calcSummaries(const hkTrackerScanSnapshot* scanSnapshot, hkArray<Summary>& summaries)
{
	const hkStackTracer::CallTree& callTree = scanSnapshot->getRawSnapshot().getCallTree();
	const hkArrayBase<hkMemorySnapshot::Allocation>& allocs = scanSnapshot->getRawSnapshot().getAllocations();

	hkPointerMap<hkUlong, int> addrMap;
	hkArray<hkUlong> callStack;

	hkPointerMap<hkUlong, int> uniqueAddrMap;

	for (int i = 0; i < allocs.getSize(); i++)
	{
		const hkMemorySnapshot::Allocation& alloc = allocs[i];
		if (alloc.m_traceId >= 0)
		{
			// Get the trace
			int stackSize = callTree.getCallStackSize(alloc.m_traceId);

			callStack.setSize(stackSize);
			callTree.getCallStack(alloc.m_traceId, callStack.begin(), stackSize);

			// Find the unique addresses in the call stack
			uniqueAddrMap.clear();
			for (int j = 0; j < stackSize; j++)
			{
				uniqueAddrMap.insert(callStack[j], 1);
			}

			// Add an entry for each one
			hkPointerMap<hkUlong, int>::Iterator iter = uniqueAddrMap.getIterator();
			for (; uniqueAddrMap.isValid(iter); iter = uniqueAddrMap.getNext(iter))
			{
				const hkUlong addr = uniqueAddrMap.getKey(iter);

				int summaryIndex = addrMap.getWithDefault(addr, -1);
				if (summaryIndex < 0)
				{
					summaryIndex = summaries.getSize();
					Summary& summary = summaries.expandOne();

					summary.m_addr = addr;
					summary.m_totalSize = 0;
					summary.m_totalNumAllocs = 0;

					addrMap.insert(addr, summaryIndex);
				}

				Summary& summary = summaries[summaryIndex];
				summary.m_totalSize += alloc.m_size;
				summary.m_totalNumAllocs++;
			}
		}
	}

	// Order
	hkSort(summaries.begin(), summaries.getSize(), _orderSummaries);
}


/* static */void HK_CALL hkStackTraceReportUtil::reportSummary(const hkTrackerScanSnapshot* scanSnapshot, hkOstream& stream)
{
	stream << 
		"hkStackTraceReportUtil::reportSummary\n"
		"-------------------------------------\n"
		"This report totals all of the allocations allocated under a specific stack trace\n"
		"address. As such all memory will be under 'main' and less under functions.\n"
		"Total Size - Total size of allocations allocated under method\n"
		"Num allocs - Total number of allocations allocated under the method\n\n";
	
	// Get all the allocations 
	hkStackTracer tracer;

	hkArray<Summary> summaries;
	calcSummaries(scanSnapshot, summaries);

	// 
	for (int i = 0; i < summaries.getSize(); i++)
	{
		const Summary& summary = summaries[i];
		tracer.dumpStackTrace( &summary.m_addr, 1, _dumpStackTrace, &stream);
		//stream << "\n";
		stream << "Total Size: " << MemorySize(summary.m_totalSize) << " Num allocs: " << int(summary.m_totalNumAllocs) << "\n\n";
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
