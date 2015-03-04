/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkCategoryReportUtil.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Memory/Tracker/Util/hkTextReportLayoutUtil.h>

void HK_CALL hkCategoryReportUtil::calcCategories(hkTrackerScanSnapshot* scanSnapshot, FollowFilter* filter, ParentMap& parentMap, TypeIndexMap& typeIndexMap, hkArray<const Block*>& rootBlocks)
{
	rootBlocks.clear();
	parentMap.clear();
	
	hkTrackerTypeTreeCache* typeCache = scanSnapshot->getLayoutCalculator()->getTypeCache();
	hkMemoryTracker& memoryTracker = hkMemoryTracker::getInstance();

	// Define a special category for 'shapes' that is not used by other categories.
	enum
	{
		CATEGORY_SHAPE = CATEGORY_LAST
	};

	{
		const hkScanReportUtil::NameTypePair pairs [] = 
		{
			{"hkpWorld", CATEGORY_WORLD} ,
			//{"hkdWorld", CATEGORY_DESTRUCTION_WORLD},
			{"hkpContactMgr", CATEGORY_CONTACT_MGR},
			{"hkpRigidBody", CATEGORY_RIGID_BODY},
			{"hkpPhantom", CATEGORY_PHANTOM},
			{"hkpConstraintInstance", CATEGORY_CONSTRAINT}, 
			{"hkPackfileData", CATEGORY_PACK_FILE_DATA},
			{"hkpAction", CATEGORY_ACTION},
			// All collision
			{"hkpCollisionDispatcher", CATEGORY_COLLISION},
			{"hkpCollisionAgent", CATEGORY_COLLISION},
			{"hkpAgent1nSector", CATEGORY_COLLISION},
			{"hkpAgentNnSector", CATEGORY_COLLISION},
			{"hkpTypedBroadPhaseDispatcher", CATEGORY_COLLISION},
			{"hkpBroadPhaseBorder", CATEGORY_COLLISION},
			{"hkp3AxisSweep", CATEGORY_COLLISION},
			
		};
		hkScanReportUtil::addTypes(scanSnapshot, pairs, HK_COUNT_OF(pairs), typeIndexMap);
	}
	
	// Put in categories type by name 
	//hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkp", true, CATEGORY_PHYSICS, typeIndexMap);

	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkd", true, CATEGORY_DESTRUCTION, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkg", true, CATEGORY_GRAPHICS, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hka", true, CATEGORY_ANIMATION, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkai", true, CATEGORY_AI, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkb", true, CATEGORY_BEHAVIOR, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hcl", true, CATEGORY_CLOTH, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hkgp", true, CATEGORY_GEOMETRY_PROCESSING, typeIndexMap);
	hkScanReportUtil::setTypeIndexByNamePrefix(scanSnapshot, "hknp", true, CATEGORY_PHYSICS, typeIndexMap);

	// Add all the derived types
	hkScanReportUtil::findAllDerivedTypes(memoryTracker, typeCache, typeIndexMap);

	// Find all the types, that are roots
	const int typeIndices[] = 
	{ 
		CATEGORY_CONTACT_MGR, 
		//CATEGORY_DESTRUCTION_WORLD, 
		CATEGORY_WORLD, 
		CATEGORY_RIGID_BODY, 
		CATEGORY_PHANTOM, 
		CATEGORY_CONSTRAINT, 
		CATEGORY_PACK_FILE_DATA, 
		CATEGORY_ACTION,
		CATEGORY_COLLISION,

		CATEGORY_GRAPHICS, 
		CATEGORY_DESTRUCTION,
		CATEGORY_AI,
		CATEGORY_ANIMATION,
		CATEGORY_BEHAVIOR,
		CATEGORY_CLOTH,
		CATEGORY_GEOMETRY_PROCESSING,
		CATEGORY_PHYSICS,
	};

	hkScanReportUtil::appendBlocksWithTypeIndices(scanSnapshot, typeIndexMap, typeIndices, HK_COUNT_OF(typeIndices), rootBlocks);

	// Make them all owned
	DontFollowMap dontFollowMap;
	for (int i = 0; i < rootBlocks.getSize(); i++)
	{
		dontFollowMap.insert(rootBlocks[i], 1);
	}

	// Don't follow any hkpShape derived classes when traversing from the world
	hkArray<const Block*> ignoreBlocks;
	hkScanReportUtil::appendBlocksWithTypeIndex(scanSnapshot, typeIndexMap, CATEGORY_SHAPE, ignoreBlocks);

	for (int i = 0; i < ignoreBlocks.getSize(); i++)
	{
		// Don't allow following from world to shape
		dontFollowMap.insert(ignoreBlocks[i], 1);
	}

	// Do types which connect alot of objects first (whilst ignoring the main other ones)

	{
		const int worldTypeIndices[] = { CATEGORY_WORLD, CATEGORY_CONTACT_MGR }; // CATEGORY_DESTRUCTION_WORLD, 
		hkArray<const Block*> worldBlocks;
		hkScanReportUtil::appendBlocksWithTypeIndices(scanSnapshot, typeIndexMap, worldTypeIndices, HK_COUNT_OF(worldTypeIndices), worldBlocks);

		hkScanReportUtil::appendParentAndDontFollowMap(scanSnapshot, worldBlocks, dontFollowMap, filter, parentMap);
	}

	// Allow tracking to ignored derived classes
	for (int i = 0; i < ignoreBlocks.getSize(); i++)
	{
		// Don't allow following from world to shape
		dontFollowMap.remove(ignoreBlocks[i]);
	}

	// Now do it for any of the other roots
	hkScanReportUtil::appendParentAndDontFollowMap(scanSnapshot, rootBlocks, dontFollowMap, filter, parentMap);

	// Find any unknown types

	{
		const hkArray<hkTrackerScanSnapshot::Block*>& blocks = scanSnapshot->getBlocks();
		//hkArray<const Block*> remainingBlocks;
		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			// Its already owned
			if (parentMap.hasKey(block))
			{
				continue;
			}
			// If it doesn't have a type - its not a root
			const RttiNode* rttiNode = block->m_type;
			if (!rttiNode || block->m_arraySize >= 0)
			{
				continue;
			}

			int typeIndex = typeIndexMap.getWithDefault(rttiNode, -1);
			if (typeIndex < 0)
			{
				// Add type to the type map
				typeIndexMap.insert(rttiNode, CATEGORY_REMAINING);
				typeIndex = CATEGORY_REMAINING;
			}

			if (typeIndex == CATEGORY_REMAINING)
			{
				// Add as a root
				rootBlocks.pushBack(block);
			}
		}

		// Find the map for the remaining blocks
		hkScanReportUtil::appendParentAndDontFollowMap(scanSnapshot, rootBlocks, dontFollowMap, filter, parentMap);
	}
}

/* static */const char* HK_CALL hkCategoryReportUtil::getCategoryName(Category type)
{
	// Mapping categories to names
	static const NameTypePair pairs [] = 
	{
		{"Physics", CATEGORY_PHYSICS},

		{"Physics 2012 World", CATEGORY_WORLD} ,
		//{"hkdWorld", CATEGORY_DESTRUCTION_WORLD},
		{"Contact Manager", CATEGORY_CONTACT_MGR},
		{"Rigid Bodies", CATEGORY_RIGID_BODY},
		{"Physics Phantoms", CATEGORY_PHANTOM},
		{"Constraints", CATEGORY_CONSTRAINT}, 
		{"Packfile Data", CATEGORY_PACK_FILE_DATA},
		{"Physics Action", CATEGORY_ACTION},
		{"Collision", CATEGORY_COLLISION},

		{"Graphics", CATEGORY_GRAPHICS},
		{"Destruction", CATEGORY_DESTRUCTION},
		{"Ai", CATEGORY_AI},
		{"Animation", CATEGORY_ANIMATION},
		{"Behavior", CATEGORY_BEHAVIOR},
		{"Cloth", CATEGORY_CLOTH},
		{"Geometry processing", CATEGORY_GEOMETRY_PROCESSING},

		{"Remaining", CATEGORY_REMAINING},	
		{"Unassigned remaining", CATEGORY_UNASSIGNED_REMAINING},
	};

	for (int i = 0; i < (int)HK_COUNT_OF(pairs); i++)
	{
		const NameTypePair& pair = pairs[i];

		if (pair.m_typeIndex == type)
		{
			return pair.m_name;
		}
	}

	return "(Unknown)";
}

HK_FORCE_INLINE hkBool _orderSummaries(const hkTypeSummaryReportUtil::TypeSummary* a, const hkTypeSummaryReportUtil::TypeSummary* b)
{
	return a->m_totalSize > b->m_totalSize;
}

/* static */const char* HK_CALL hkCategoryReportUtil::getCategoryTypeName(Category type)
{
	switch (type)
	{
		case CATEGORY_WORLD: return "hkpWorld";
		case CATEGORY_CONTACT_MGR: return "hkpContactMgr";
		case CATEGORY_RIGID_BODY: return "hkpRigidBody";
		case CATEGORY_PHANTOM: return "hkpPhantom";
		case CATEGORY_CONSTRAINT: return "hkpConstraintInstance";
		case CATEGORY_PACK_FILE_DATA: return "hkPackfileData";
		case CATEGORY_ACTION: return "hkpAction";
		default: return HK_NULL;
	}
}


void HK_CALL hkCategoryReportUtil::report(hkTrackerScanSnapshot* scanSnapshot, FollowFilter* filter, hkOstream& stream)
{
	stream << 
		"Havok Memory Report Release 2010.1\n"
		"===================================\n"
		"(hkCategoryReportUtil::report)\n\n"
		"Report puts blocks into categories, and then reports in summary the amount \n"
		"of blocks and memory usage by type. \n"
		"Blocks listed as '(Unknown)' have indeterminate type.\n"
		"'Allocated' memory is actual allocated memory from the memory system. 'Used' is the \n"
		"used portion of the allocated memory. An example of unused but allocated memory is \n"
		"the unused capacity in an array. All memory values are 'used' unless specified otherwise.\n\n";
	
	ParentMap parentMap;
	TypeIndexMap typeIndexMap;
	hkArray<const Block*> rootBlocks;

	calcCategories(scanSnapshot, filter, parentMap, typeIndexMap, rootBlocks);

	// Work out the child map
	ChildMultiMap childMultiMap;
	hkScanReportUtil::calcChildMap(parentMap, childMultiMap);

	hk_size_t usedSize;
	{
		const hkArray<hkTrackerScanSnapshot::Block*>& blocks = scanSnapshot->getBlocks();
		usedSize = hkTrackerScanSnapshot::calcTotalUsed(blocks.begin(), blocks.getSize());

		stream << "Total Memory Used     : " << MemorySize(usedSize) << "\n\n";
	}


	// Find blocks that are not categorized
	hkArray<const Block*> remainingBlocks;

	{
		const hkArray<hkTrackerScanSnapshot::Block*>& blocks = scanSnapshot->getBlocks();
		for (int i = 0; i < blocks.getSize(); i++)
		{
			const Block* block = blocks[i];
			if (!parentMap.hasKey(block))
			{
				remainingBlocks.pushBack(block);
			}
		}
	}

	hkArray<TypeSummary> summaries;
	hkArray<int> summaryStartIndices;
	hkArray<int> blockStartIndices;
	hkArray<TypeSummary> categorySummaries;
	hkArray<const Block*> blocks;

	// Okay I'm now ready for reporting... 

	{
		hkArray<TypeSummary> workSummaries;
		
		HK_COMPILE_TIME_ASSERT(CATEGORY_WORLD == 0);

		for (int i = CATEGORY_WORLD; i < CATEGORY_LAST; i++)
		{
			// Store the start index
			summaryStartIndices.pushBack(summaries.getSize());
			const int startBlockIndex = blocks.getSize();
			blockStartIndices.pushBack(startBlockIndex);

			workSummaries.clear();
		
			if (i == CATEGORY_UNASSIGNED_REMAINING)
			{
				// Add unassigned blocks
				blocks.insertAt(blocks.getSize(), remainingBlocks.begin(), remainingBlocks.getSize());
			}
			else
			{
				// Find root blocks that belong to the category
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
			}

			// Set up an array just holding the blocks for this category
			const int numBlocks = blocks.getSize() - startBlockIndex;
			hkArray<const Block*> categoryBlocks(blocks.begin() + startBlockIndex, numBlocks, numBlocks);
		
			// Calculate the summaries
			hkTypeSummaryReportUtil::calcTypeSummaries(categoryBlocks, parentMap, workSummaries);
			// Order largest to smallest
			hkTypeSummaryReportUtil::orderTypeSummariesBySize(workSummaries.begin(), workSummaries.getSize());
			// 
			summaries.insertAt(summaries.getSize(), workSummaries.begin(), workSummaries.getSize());

			// Work out the summary of all the summaries
			hkTypeSummaryReportUtil::calcTotalSummary(workSummaries.begin(), workSummaries.getSize(), categorySummaries.expandOne()); 
		}

		// Store so the end one has the right size
		summaryStartIndices.pushBack(summaries.getSize());
		blockStartIndices.pushBack(blocks.getSize());
	}

	hkArray<const TypeSummary*> orderedSummaries;
	{
		orderedSummaries.setSize(categorySummaries.getSize());
		for (int i = 0; i < categorySummaries.getSize(); i++)
		{
			orderedSummaries[i] = &categorySummaries[i];
		}
	
		hkSort(orderedSummaries.begin(), orderedSummaries.getSize(), _orderSummaries);
	}

	{
		stream <<
			"Memory usage overview\n"
			"=====================\n\n"
			"All units are in Kb\n"
			"% of Total - percentage of total memory\n"
			"Mem Used   - total memory used in category\n"
			"# Self     - number of allocated objects of this type (or derived)\n"
			"# Objs     - number of child objects\n"
			"Objs       - total allocated memory on child objects\n"
			"# Non-Objs - number of non object allocations\n"
			"Non-Objs   - total non object allocations (arrays and contained data)\n"
			"\n";

		const int numCols = 8;
		
		const char* titles = "Category|% of Total|Mem Used|# Self|# Objs|Objs|# Non-Objs|Non-Objs";
		int widths[numCols] = { 26, -1, -1, 6, 10, -1, -1, -1 };
		hkTextReportLayoutUtil::calcColumnWidths(titles, '|', numCols, widths, widths);
		hkArray<hkTextReportLayoutUtil::Column> columns;
		hkTextReportLayoutUtil::calcColumns(widths, numCols, 2, 0, columns);

		// Write the titles
		hkTextReportLayoutUtil::writeColumns(columns, "c", titles, '|', stream);
		stream << "\n";
		hkTextReportLayoutUtil::writeCharColumns(columns, '-', stream);
		stream << "\n";

		// Dump out the category summaries
		{
			hkStringBuf buffer;

			for (int i = 0; i< orderedSummaries.getSize(); i++) 
			{
				const TypeSummary& summary = *orderedSummaries[i];
				if (summary.m_numBlocks <= 0)
				{
					continue;
				}

				Category category = Category(&summary - categorySummaries.begin());

				const char* typeName = getCategoryTypeName(category);

				int numObjs = 0;
				int numSelf = 0;
				hk_size_t totalObjs = 0;
				hk_size_t totalNonObjs = 0;

				const Block*const* categoryBlocks = blocks.begin() + blockStartIndices[category];
				const int numBlocks = blockStartIndices[category + 1] - blockStartIndices[category];

				for (int j = 0; j < numBlocks; j++)
				{
					const Block* block = categoryBlocks[j];

					// If it has a type, its not an array of the type, there is a typename, and its set up in the type indexmap 
					// as the category, then its a self
					if (block->m_type && block->m_arraySize < 0 && typeName && typeIndexMap.getWithDefault(block->m_type, -1) == category)
					{
						numSelf++;
					}
					if (block->m_type && block->m_arraySize < 0)
					{
						numObjs++;
						totalObjs += block->m_size;
					}
					else
					{
						totalNonObjs += block->m_size;
					}
				}

				// Category
				buffer = "";
				buffer.append(getCategoryName(category));
				buffer.append("|");

				// & of total
				buffer.appendPrintf("%0.1f%%|", (float(summary.m_totalSize) / usedSize) * 100.0f);
				// Mem used
				buffer.appendPrintf("%i|", int((summary.m_totalSize + 512) / 1024));
				// # Self
				if (typeName)
				{
					buffer.appendPrintf("%i|", int(numSelf));
				}
				else
				{
					buffer.append("|");
				}
				// Num objects
				buffer.appendPrintf("%i|", int(numObjs));
				// Objs
				buffer.appendPrintf("%i|", int(totalObjs + 512) / 1024);

				// Num non objects
				buffer.appendPrintf("%i|", int(numBlocks - numObjs));
				// Non objs
				buffer.appendPrintf("%i", int(totalNonObjs + 512) / 1024);
				
				// Write it all out
				hkTextReportLayoutUtil::writeColumns(columns, "lr", buffer.cString(), '|', stream);
				stream << "\n";
			}
		}
	}

	// Now lets do the detail with the types
	{
		stream << 
			"\n"
			"Memory breakdown by category\n"
			"============================\n\n"
			"All units in bytes\n"
			"% of Category  - percentage of total category memory used on type\n"
			"Mem Used       - total memory used on the type including owned allocations\n"
			"# Self         - number of allocated objects of this type\n"
			"Self           - amount of memory used on this type\n"
			"# Non-Self     - total non self allocations (arrays and contained data)\n"
			"Non-Self       - number of non self allocations\n"
			"\n";

		const int numCols = 7;
		
		const char* titles = "Category/Type|% of Category|Mem Used|# Self|Self|# Non-Self|Non Self";
		int widths[numCols] = { 34, -1, 10, -1, 10, -1, 10 };

		hkTextReportLayoutUtil::calcColumnWidths(titles, '|', numCols, widths, widths);
		hkArray<hkTextReportLayoutUtil::Column> columns;
		hkTextReportLayoutUtil::calcColumns(widths, numCols, 2, 0, columns);

		

		// Dump out the category summaries
		{
			hkStringBuf buffer;

			for (int i = 0; i< orderedSummaries.getSize(); i++) 
			{
				const TypeSummary& catSummary = *orderedSummaries[i];
				if (catSummary.m_numBlocks <= 0)
				{
					continue;
				}

				// Write the titles
				hkTextReportLayoutUtil::writeColumns(columns, "c", titles, '|', stream);
				stream << "\n";
				hkTextReportLayoutUtil::writeCharColumns(columns, '-', stream);
				stream << "\n";

				Category category = Category(&catSummary - categorySummaries.begin());

				stream << getCategoryName(category) << "\n";

				const TypeSummary* typeSummaries = summaries.begin() + summaryStartIndices[category];
				int numSummaries = summaryStartIndices[category + 1] - summaryStartIndices[category];

				for (int j = 0; j < numSummaries; j++)
				{
					const TypeSummary& summary = typeSummaries[j];

					char typeName[256];
					hkOstream typeStream(typeName, (int)sizeof(typeName), true);
					hkTrackerTypeTreeNode::dumpType(summary.m_type, typeStream);
					// Category
					buffer = "    ";
					buffer.append(typeName);
					buffer.append("|");

					// % of category
					buffer.appendPrintf("%0.1f%%|", (float(summary.m_totalSize) / catSummary.m_totalSize) * 100.0f);
					// Total
					buffer.appendPrintf("%i|", int(summary.m_totalSize));
					
					// Num objects
					buffer.appendPrintf("%i|", int(summary.m_numInstances));
					// Size of the instances
					buffer.appendPrintf("%i|", int(summary.m_totalInstanceSize));

					// Num non self
					buffer.appendPrintf("%i|", int(summary.m_numBlocks - summary.m_numInstances));
					// non self
					buffer.appendPrintf("%i", int(summary.m_totalSize - summary.m_totalInstanceSize));

					// Write it all out
					hkTextReportLayoutUtil::writeColumns(columns, "lr", buffer.cString(), '|', stream);
					stream << "\n";
				}

				stream << "\n";
			}
		}
	}

	{
		hkStackTracer tracer;

		if (remainingBlocks.getSize() > 0)
		{
			stream << 
				"\n"
				"Unassigned remaining blocks call stacks\n"
				"=======================================\n\n";

			const int maxBlocks = 30;
			int numReport = remainingBlocks.getSize();
			if (numReport > maxBlocks)
			{
				numReport = maxBlocks;
				stream << "There are " << remainingBlocks.getSize() << " unaccounted for blocks. Only dumping the first " << numReport << "\n\n";
			}

			for (int i = 0; i < numReport; i++)
			{
				const Block* block = remainingBlocks[i];
				hkScanReportUtil::appendBlockType(block, stream);
				stream << "@" << (void*)block->m_start << " ";
				stream << " Size: " << MemorySize(block->m_size);
				stream << "\n";
				hkScanReportUtil::dumpAllocationCallStack(&tracer, scanSnapshot, (void*)block->m_start, stream);
				stream << "\n\n";
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
