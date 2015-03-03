/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkScanReportUtil.h>
#include <Common/Base/Container/Queue/hkQueue.h>
#include <Common/Base/Container/String/hkStringBuf.h>

static void HK_CALL _dumpStackTrace(const char* text, void* context)
{
	hkOstream& stream = *(hkOstream*)context;
	stream << text;
}

/* static */void HK_CALL hkScanReportUtil::writeTraceText(hkStackTracer* tracer, hkTrackerScanSnapshot* scanSnapshot, hkUlong addr, hkOstream& stream)
{
	if (scanSnapshot->hasTraceText())
	{
		const char* text = scanSnapshot->getTraceText(addr);
		if (text)
		{
			stream << text;
			return;
		}
	}

	if (tracer)
	{
		tracer->dumpStackTrace( &addr, 1, _dumpStackTrace, &stream);
		return;
	}

	stream << "(" << (void*)addr << ")\n";
}

/* static */void HK_CALL hkScanReportUtil::dumpAllocationCallStack(hkStackTracer* tracer, hkTrackerScanSnapshot* scanSnapshot, void* ptr, hkOstream& stream)
{
	const hkMemorySnapshot& rawSnapshot = scanSnapshot->getRawSnapshot();
	const hkArrayBase<hkMemorySnapshot::Allocation>& allocs = rawSnapshot.getAllocations();

	// We do not sort the allocation list anymore, sorting is not needed for most
	// reports. We proceed with linear search for every report that needs it.

	hkStackTracer::CallTree::TraceId traceId = -1;
	for(int i = 0; i < allocs.getSize(); ++i)
	{
		const hkMemorySnapshot::Allocation& alloc = allocs[i];
		if(alloc.m_start == ptr && alloc.m_traceId != -1)
		{
			traceId = alloc.m_traceId;
			break;
		}
	}

	if(traceId != -1)
	{
		HK_WARN(0x23434a32, "Allocation not found or found without trace id");
		stream << "Callstack for allocation (" << ptr << ") not found!\n";
		return;
	}

	// Get the trace
	dumpCallStack(tracer, scanSnapshot, traceId, stream);
}

/* static */void HK_CALL hkScanReportUtil::dumpCallStack(hkStackTracer* tracer, hkTrackerScanSnapshot* scanSnapshot, hkStackTracer::CallTree::TraceId traceId, hkOstream& stream)
{
	const hkStackTracer::CallTree& callTree = scanSnapshot->getRawSnapshot().getCallTree();

	// 
	hkArray<hkUlong> trace;
	int traceSize = callTree.getCallStackSize(traceId);

	if (traceSize <= 0)
	{
		return;
	}
	trace.setSize(traceSize);
	callTree.getCallStack(traceId, trace.begin(), traceSize);

	for (int i = 0; i < traceSize; i++)
	{
		writeTraceText(tracer, scanSnapshot, trace[i], stream);
	}
}

int HK_CALL hkScanReportUtil::calcMemberIndex(const hkTrackerScanSnapshot* scanSnapshot, const hkTrackerTypeLayout* layout, const Block* block, const Block* childBlock)
{
	if (layout->m_fullScan )
	{
		return -1;
	}

	int memberIndex = scanSnapshot->findReferenceIndex(block, childBlock);
	HK_ASSERT(0x2432432, memberIndex >= 0);

	if (block->m_arraySize >= 0)
	{
		return memberIndex % layout->m_members.getSize();
	}
	else
	{
		return memberIndex;
	}
}

const char* HK_CALL hkScanReportUtil::calcMemberName(const hkTrackerScanSnapshot* scanSnapshot, const Block* block, const Block* childBlock)
{
	hkTrackerLayoutCalculator* layoutCalc = scanSnapshot->getLayoutCalculator();

	if (block->m_type == HK_NULL)
	{
		return HK_NULL;
	}
	const hkTrackerTypeLayout* layout = layoutCalc->getLayout(block->m_type);
	if (!layout)
	{
		return HK_NULL;
	}
	const int memberIndex = calcMemberIndex(scanSnapshot, layout, block, childBlock);
	if (memberIndex < 0)
	{
		return HK_NULL;
	}

	return layout->m_members[memberIndex].m_name;
}

void HK_CALL hkScanReportUtil::appendDepthFirstParentMap(const hkTrackerScanSnapshot* scanSnapshot, const Block* rootBlock, DontFollowMap* dontFollowMap, FollowFilter* filter, ParentMap& parentMap)
{
	/* if (dontFollowMap && dontFollowMap->hasKey(rootBlock))
	{
		return;
	} */

	parentMap.insert(rootBlock, HK_NULL);

	hkArray<const Block*> stack;
	stack.pushBack(rootBlock);

	hkTrackerLayoutCalculator* layoutCalc = scanSnapshot->getLayoutCalculator();
	// 
	while (stack.getSize() > 0)
	{
		const Block* block = stack.back();
		stack.popBack();
		if (block == HK_NULL)
		{
			continue;
		}

		const hkTrackerTypeLayout* layout = HK_NULL;
		if (block->m_type)
		{
			layout = layoutCalc->getLayout(block->m_type);
		}

		// Add the members which are not visited

		const int numRefs = block->m_numReferences;
		Block*const* refs = scanSnapshot->getBlockReferences(block);

		for (int j = numRefs - 1; j >= 0; j--)
		{
			Block* childBlock = refs[j];
			if (childBlock == HK_NULL || parentMap.hasKey(childBlock))
			{
				continue;
			}

			if (dontFollowMap && dontFollowMap->hasKey(childBlock))
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

			// Check the relationship is correct
			HK_ASSERT(0x3242423, scanSnapshot->findReferenceIndex(block, childBlock) >= 0);

			parentMap.insert(childBlock, block);

			// Put on the stack
			stack.pushBack(childBlock);
		}
	}
}

void HK_CALL hkScanReportUtil::appendBreadthFirstParentMap(const hkTrackerScanSnapshot* scanSnapshot, const Block* rootBlock, DontFollowMap* dontFollowMap, FollowFilter* filter, ParentMap& parentMap)
{
	/* if (dontFollowMap && dontFollowMap->hasKey(rootBlock))
	{
		return;
	} */

	parentMap.insert(rootBlock, HK_NULL);

	hkQueue<const Block*> queue;
	queue.enqueue(rootBlock);

	hkTrackerLayoutCalculator* layoutCalc = scanSnapshot->getLayoutCalculator();
	// 
	while (queue.isEmpty() == false)
	{
		const Block* block;
		queue.dequeue(block);
		if (block == HK_NULL)
		{
			continue;
		}

		const hkTrackerTypeLayout* layout = HK_NULL;
		if (block->m_type)
		{
			layout = layoutCalc->getLayout(block->m_type);
		}

		// Add the members which are not visited

		const int numRefs = block->m_numReferences;
		Block*const* refs = scanSnapshot->getBlockReferences(block);

		for (int j = numRefs - 1; j >= 0; j--)
		{
			Block* childBlock = refs[j];
			if (childBlock == HK_NULL || parentMap.hasKey(childBlock))
			{
				continue;
			}

			if (dontFollowMap && dontFollowMap->hasKey(childBlock))
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

			// Check the relationship is correct
			HK_ASSERT(0x3242423, scanSnapshot->findReferenceIndex(block, childBlock) >= 0);

			parentMap.insert(childBlock, block);

			// Put on the stack
			queue.enqueue(childBlock);
		}
	}
}

void HK_CALL hkScanReportUtil::appendParentMap(const hkTrackerScanSnapshot* scanSnapshot, const Block* rootBlock, DontFollowMap* dontFollowMap, FollowFilter* filter, ParentMap& parentMap, Traversal traversal)
{

	switch (traversal)
	{
		case TRAVERSAL_BREADTH_FIRST:
		{
			appendBreadthFirstParentMap(scanSnapshot, rootBlock, dontFollowMap, filter, parentMap);
			break;
		}
		case TRAVERSAL_DEPTH_FIRST:
		{
			appendDepthFirstParentMap(scanSnapshot, rootBlock, dontFollowMap, filter, parentMap);
			break;
		}
	}
}

void HK_CALL hkScanReportUtil::calcParentMap(const hkTrackerScanSnapshot* scanSnapshot, const Block* rootBlock, FollowFilter* filter, ParentMap& parentMap, Traversal traversal)
{
	parentMap.clear();
	appendParentMap(scanSnapshot, rootBlock, HK_NULL, filter, parentMap, traversal);
}

void HK_CALL hkScanReportUtil::appendParentAndDontFollowMap(const hkTrackerScanSnapshot* scanSnapshot, const hkArray<const Block*>& rootBlocks, DontFollowMap& dontFollowMap, FollowFilter* filter, ParentMap& parentMap)
{
	ParentMap localParentMap;
	for (int i = 0; i < rootBlocks.getSize(); i++)
	{
		const Block* rootBlock = rootBlocks[i];
		
		localParentMap.clear();
		appendParentMap(scanSnapshot, rootBlock, &dontFollowMap, filter, localParentMap);

		// I need to add to the parentMap
		ParentMap::Iterator iter = localParentMap.getIterator();
		for (; localParentMap.isValid(iter); iter = localParentMap.getNext(iter))
		{
			const Block* block = localParentMap.getKey(iter);
			const Block* parentBlock = localParentMap.getValue(iter);

			// This is now owned, so stop it being followed
			dontFollowMap.insert(block, 1);
			// Add to the mapping
			parentMap.insert(block, parentBlock);
		}
	}
}

void HK_CALL hkScanReportUtil::calcTypeRootBlocks(const hkTrackerScanSnapshot* scanSnapshot, DontFollowMap& dontFollowMap, FollowFilter* filter, ParentMap& parentMap, hkArray<const Block*>& rootBlocks)
{
	dontFollowMap.clear();
	rootBlocks.clear();
	parentMap.clear();

	calcReferencedObjectRootBlocks(scanSnapshot, dontFollowMap, rootBlocks);

	// Set up the parent map for these blocks (and update dontFollowMap)
	appendParentAndDontFollowMap(scanSnapshot, rootBlocks, dontFollowMap, filter, parentMap);

	// Find any blocks that haven't been reached yet
	hkArray<const Block*> nonVirtualRoots;
	const hkArray<Block*>& blocks = scanSnapshot->getBlocks();
	for (int i = 0; i < blocks.getSize(); i++)
	{
		const Block* block = blocks[i];
		if (!dontFollowMap.hasKey(block) && block->m_arraySize < 0 && block->m_type && block->m_type->isNamedType())
		{
			nonVirtualRoots.pushBack(block);
		}
	}

	appendParentAndDontFollowMap(scanSnapshot, nonVirtualRoots, dontFollowMap, filter, parentMap);
	rootBlocks.insertAt(0, nonVirtualRoots.begin(), nonVirtualRoots.getSize());
}

hkOstream& operator<<(hkOstream& stream, const hkScanReportUtil::MemorySize& size)
{
    char buffer[128];

    hkStringBuf buf;
    buf.getArray().setDataUserFree(buffer, sizeof(buffer), sizeof(buffer));

    hkScanReportUtil::memorySizeToText(size.m_size, size.m_flags, buf);
    stream << buf;
    return stream;
}

/* static */void HK_CALL hkScanReportUtil::appendSpaces(hkOstream& stream, int size)
{
	for (int i = 0; i < size; i++)
	{
		stream << " ";
	}
}

/* static */void HK_CALL hkScanReportUtil::alignRight(hkOstream& stream, const hkSubString& text, int totalSize)
{
	int len = text.length();
	if (len < totalSize)
	{
		appendSpaces(stream, totalSize - len);
	}
	stream << text;
}

/* static */void HK_CALL hkScanReportUtil::alignLeft(hkOstream& stream, const hkSubString&  text, int totalSize)
{
	stream << text;

	int len = text.length();
	if (len < totalSize)
	{
		appendSpaces(stream, totalSize - len);
	}
}

/* static */void HK_CALL hkScanReportUtil::memorySizeToText(hk_size_t size, int flags, hkStringBuf& string)
{
	static const char spaces[] = "                    ";

	if (flags & MemorySize::FLAG_RAW)
	{
		string.printf("%i", size);
		if (flags & MemorySize::FLAG_PAD_LEFT)
		{
			string.prepend(spaces, MemorySize::MAX_FULL_DIGITS - string.getLength());
		}
		return;
	}

    if (size < 1024)
    {
        string.printf("%i Bytes", int(size));
    } 
	else if (size < 10 * 1024)
    {
        string.printf("%.1f Kb", ((float)size) / 1024.0f);
    }
	else if (size < 1024 * 1024)
    {
        string.printf("%i Kb", int(size / 1024));
    }
	else
	{
		string.printf("%.2f Mb", float(size / (1024 * 1024.0f)));
	}

	int point = 0;
	for (; string[point] != ' ' && string[point] != '.'; point++) ;

	if (flags & MemorySize::FLAG_PAD_RIGHT)
	{

		// Find the length
		int addSpaces = point + MemorySize::TAIL_SIZE - string.getLength();
		string.append(spaces, addSpaces);
	}

	if (flags & MemorySize::FLAG_PAD_LEFT)
	{
		string.prepend(spaces, MemorySize::MAX_FULL_DIGITS - point);
	}
}

/* static */ void HK_CALL hkScanReportUtil::appendChildren(const Block* block, const ChildMultiMap& childMap, hkArray<const Block*>& children)
{
	int start = children.getSize();
	// Add all the children of the root
	{
		ChildMultiMap::Iterator iter = childMap.findKey(block);
		for (; childMap.isValid(iter); iter = childMap.getNext(iter, block))
		{
			children.pushBack(childMap.getValue(iter));
		}
	}

	for (int i = start; i < children.getSize(); i++)
	{
		const Block* cur = children[i];

		// Add all its children
		ChildMultiMap::Iterator iter = childMap.findKey(cur);
		for (; childMap.isValid(iter); iter = childMap.getNext(iter, cur))
		{
			children.pushBack(childMap.getValue(iter));
		}
	}
}


/* static */ void HK_CALL hkScanReportUtil::findChildren(const Block* block, const ChildMultiMap& childMap, hkArray<const Block*>& children)
{
	children.clear();
	appendChildren(block, childMap, children);
}

/* static */ void HK_CALL hkScanReportUtil::calcChildMap(const ParentMap& parentMap, ChildMultiMap& childMap)
{
	childMap.clear();

	hkPointerMap<const Block*, const Block*>::Iterator iter = parentMap.getIterator();
	for (; parentMap.isValid(iter); iter = parentMap.getNext(iter))
	{
		const Block* block = parentMap.getKey(iter);
		const Block* parentBlock = parentMap.getValue(iter);

		if (parentBlock)
		{
			childMap.insert(parentBlock, block);
		}
	}
}


void HK_CALL hkScanReportUtil::calcReferencedObjectRootBlocks(const hkTrackerScanSnapshot* scanSnapshot, DontFollowMap& dontFollowMap, hkArray<const Block*>& rootBlocks)
{
	dontFollowMap.clear();
	rootBlocks.clear();

	const hkArray<Block*>& blocks = scanSnapshot->getBlocks();

	for (int i = 0; i < blocks.getSize(); i++)
	{
		const Block* block = blocks[i];
		if (block->m_type == HK_NULL)
		{
			continue;
		}

		const hkTrackerTypeLayout* typeLayout = scanSnapshot->getLayoutCalculator()->getLayout(block->m_type);

		if (typeLayout && typeLayout->m_isVirtual)
		{
			// Add it to the blockers
			dontFollowMap.insert(block, 1);

			// Add the block... they will be the roots
			rootBlocks.pushBack(block);
		}
	}
}

HK_FORCE_INLINE static hkBool _orderBlocksByStart(const hkScanReportUtil::Block* a, const hkScanReportUtil::Block* b)
{
	if (a->m_start == b->m_start)
	{
		// If same address order with largest first
		return a->m_size > b->m_size;
	}

	return a->m_start < b->m_start;
}

/* static */void HK_CALL hkScanReportUtil::calcRootBlocks(const hkArray<const Block*>& blocks, hkArray<const Block*>& rootBlocks)
{
	rootBlocks.clear(); 
	if (blocks.getSize() <= 0)
	{
		return;
	}

	rootBlocks.insertAt(0, blocks.begin(), blocks.getSize());

	hkSort(rootBlocks.begin(), rootBlocks.getSize(), _orderBlocksByStart);

	const Block* prevBlock = rootBlocks[0];
	int prevIndex = 1;
	for (int i = 1; i < rootBlocks.getSize(); i++)
	{
		const Block* block = rootBlocks[i];

		if (block->m_start >= prevBlock->m_start && block->m_start < prevBlock->m_start + prevBlock->m_size)
		{
			// Its inside the previous
			//int z =0;
		}
		else
		{
			// Add to the end
			rootBlocks[prevIndex++] = block;
			prevBlock = block;
		}
	}

	// Set the size
	rootBlocks.setSizeUnchecked(prevIndex);
}

/* static */hk_size_t HK_CALL hkScanReportUtil::calcTotalSize(const hkArray<const Block*>& blocks)
{
	hk_size_t totalSize = 0;
	for (int i = 0; i < blocks.getSize(); i++)
	{
		totalSize += blocks[i]->m_size;
	}
	return totalSize;
}

/* static */void HK_CALL hkScanReportUtil::calcParentMapBlocks(const ParentMap& parentMap, hkArray<const Block*>& blocks)
{
	blocks.clear();

	ParentMap::Iterator iter = parentMap.getIterator();
	for (; parentMap.isValid(iter); iter = parentMap.getNext(iter))
	{
		const Block* block = parentMap.getKey(iter);
		HK_ASSERT(0x23423432, block);
		blocks.pushBack(block);
	}
}

/* static */void HK_CALL hkScanReportUtil::findDerivedTypes(hkMemoryTracker& memoryTracker, const hkMemoryTracker::ClassDefinition* clsDef, hkTrackerTypeTreeCache* typeCache, TypeIndexMap& typeIndexMap)
{
	hkInplaceArray<const RttiNode*, 16> path;

	{
		// Add the leaf
		const Node* rttiNode = typeCache->getNamedNode(clsDef->m_typeName);
		path.pushBack(rttiNode);
	} 

	while (clsDef)
	{
		const char* parentTypeName = clsDef->m_parentTypeName;
		if (!parentTypeName)
		{
			break;
		}

		// Get the rtti node
		const Node* rttiNode = typeCache->getNamedNode(parentTypeName);
		if (rttiNode)
		{
			// Look up to see if it 
			TypeIndexMap::Iterator iter = typeIndexMap.findKey(rttiNode);

			if (typeIndexMap.isValid(iter))
			{
				// Found it..
				const int typeIndex = typeIndexMap.getValue(iter);

				for (int i = 0; i < path.getSize(); i++)
				{
					typeIndexMap.insert(path[i], typeIndex);
				}

				return;
			}

			// Add it to the path
			path.pushBack(rttiNode);
		}

		clsDef = memoryTracker.findClassDefinition(parentTypeName);
	}
}

/* static */void HK_CALL hkScanReportUtil::findAllDerivedTypes(hkMemoryTracker& memoryTracker, hkTrackerTypeTreeCache* typeCache, TypeIndexMap& typeIndexMap)
{
	int numTypes = (int)memoryTracker.getTypeDefinitions(HK_NULL);
	
	hkArray<const hkMemoryTracker::TypeDefinition*> typeDefs;
	typeDefs.setSize(numTypes);

	memoryTracker.getTypeDefinitions(typeDefs.begin());

	for (int i = 0; i < typeDefs.getSize(); i++)
	{
		const hkMemoryTracker::TypeDefinition* typeDef = typeDefs[i];

		if (typeDef->m_type == hkMemoryTracker::TypeDefinition::TYPE_CLASS)
		{
			// Its a class.. set this and its derived types to the type of the first one that has 
			// a type defined
			findDerivedTypes(memoryTracker, static_cast<const hkMemoryTracker::ClassDefinition*>(typeDef), typeCache, typeIndexMap);
		}
	}
}

/* static */void HK_CALL hkScanReportUtil::addTypes(const hkTrackerScanSnapshot* scanSnapshot, const NameTypePair* pairs, int numPairs, TypeIndexMap& typeIndexMap)
{
	for (int i = 0; i < numPairs; i++)
	{
		const NameTypePair& pair = pairs[i];

		const RttiNode* rttiNode = findTypeFromName(scanSnapshot, pair.m_name);

		if (rttiNode)
		{
			typeIndexMap.insert(rttiNode, pair.m_typeIndex);
		}
	}
}

/* static */void HK_CALL hkScanReportUtil::appendBlocksWithTypeIndex(const hkTrackerScanSnapshot* scanSnapshot, const TypeIndexMap& typeIndexMap, int typeIndex, hkArray<const Block*>& blocksOut)
{
	HK_ASSERT(0x2342432, typeIndex >= 0);
	const hkArray<Block*>& blocks = scanSnapshot->getBlocks();

	for (int i = 0; i < blocks.getSize(); i++)
	{
		const Block* block = blocks[i];

		if (block->m_type)
		{
			if (typeIndexMap.getWithDefault(block->m_type, -1) == typeIndex)
			{
				blocksOut.pushBack(block);
			}
		}
	}
}

/* static */void HK_CALL hkScanReportUtil::appendBlocksWithTypeIndex(const hkTrackerScanSnapshot* scanSnapshot, const TypeIndexMap& typeIndexMap, hkArray<const Block*>& blocksOut)
{
	const hkArray<Block*>& blocks = scanSnapshot->getBlocks();

	for (int i = 0; i < blocks.getSize(); i++)
	{
		const Block* block = blocks[i];
		if (block->m_type)
		{
			if (typeIndexMap.hasKey(block->m_type))
			{
				blocksOut.pushBack(block);
			}
		}
	}
}

/* static */void HK_CALL hkScanReportUtil::appendBlocksWithTypeIndices(const hkTrackerScanSnapshot* scanSnapshot, const TypeIndexMap& typeIndexMap, const int* typeIndices, int numTypeIndices, hkArray<const Block*>& blocksOut)
{
	for (int i = 0; i < numTypeIndices; i++)
	{
		appendBlocksWithTypeIndex(scanSnapshot, typeIndexMap, typeIndices[i], blocksOut);
	}
}

/* static */int HK_CALL hkScanReportUtil::getTypeIndex(const Block* block, const TypeIndexMap& typeIndexMap)
{
	return block->m_type ? typeIndexMap.getWithDefault(block->m_type, -1) : -1;
}

/* static */const hkScanReportUtil::RttiNode* hkScanReportUtil::getTypeTreeNode(const hkMemoryTracker::TypeDefinition* typeDef, hkTrackerTypeTreeCache* typeCache)
{
	typedef hkMemoryTracker::TypeDefinition TypeDefinition;

	hkTrackerTypeTreeNode::Type type = hkTrackerTypeTreeNode::TYPE_UNKNOWN;

	switch (typeDef->m_type)
	{
		case TypeDefinition::TYPE_BASIC:
		{
			type = hkTrackerTypeTreeNode::TYPE_NAMED;
			break;
		}
		case TypeDefinition::TYPE_CLASS:
		case TypeDefinition::TYPE_SCAN:
		{
			type = hkTrackerTypeTreeNode::TYPE_CLASS;
			break;
		}
	}

	return typeCache->newNamedNode(type, typeDef->m_typeName);
}

const hkScanReportUtil::RttiNode* hkScanReportUtil::findTypeFromName(const hkTrackerScanSnapshot* scanSnapshot, const char* name)
{
	hkTrackerTypeTreeCache* typeCache = scanSnapshot->getLayoutCalculator()->getTypeCache();

	// See if its in the cache
	{
		const Node* node = typeCache->getNamedNode(name);
		if (node)
		{
			return node;
		}
	}

	// Look up a definition
	{
		hkMemoryTracker& memoryTracker = hkMemoryTracker::getInstance();
		const hkMemoryTracker::TypeDefinition* typeDef = memoryTracker.findTypeDefinition(name);
		// Add it
		if (typeDef)
		{
			return getTypeTreeNode(typeDef, typeCache);
		}
	}

	HK_WARN(0x2442342, "Couldn't find type '" << name << "'");
	return HK_NULL;
}

void HK_CALL hkScanReportUtil::setTypeIndexByNamePrefix(hkTrackerScanSnapshot* scanSnapshot, const char* prefix, hkBool prefixFollowedByCapital, int typeIndex, TypeIndexMap& typeIndexMap)
{
	// Find types that are prefixed 'hkg' as graphics
	const hkArray<hkTrackerScanSnapshot::Block*>& blocks = scanSnapshot->getBlocks();
	const int len = hkString::strLen(prefix);

	for (int i = 0; i < blocks.getSize(); i++)
	{
		const Block* block = blocks[i];
		const RttiNode* rttiNode = block->m_type;

		if (rttiNode == HK_NULL || !rttiNode->isNamedType() || typeIndexMap.hasKey(rttiNode))
		{
			continue;
		}

		const hkSubString& name = rttiNode->m_name;
		if (name.length() < len)
		{
			continue;
		}

		hkSubString subString(name.m_start, name.m_start + len);
		
		if (subString == prefix)
		{
			if (prefixFollowedByCapital)
			{
				// Check its followed by a capital (ie its the end of a havok style prefix)
				if (name.length() < len + 1)
				{
					continue;
				}
				char c = name.m_start[len];

				hkBool isCapitalOrDigit = (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');

				if ( !isCapitalOrDigit)
				{
					continue;
				}
			}
	
			typeIndexMap.insert(rttiNode, typeIndex);
		}
	}
}

/* static */void HK_CALL hkScanReportUtil::calcPath(const ParentMap& parentMap, const Block* block, hkArray<const Block*>& path)
{
	path.clear();

	while (block)
	{
		path.pushBack(block);
		block = parentMap.getWithDefault(block, HK_NULL);
	}

	// Invert the order
	if (path.getSize() >= 2)
	{
		const Block** end = &path.back();
		int size = path.getSize() / 2;
		for (int i = 0; i < size; i++)
		{
			hkAlgorithm::swap(path[i], end[-i]);
		}
	}
}

/* static */void HK_CALL hkScanReportUtil::calcTextPath(const hkTrackerScanSnapshot* scanSnapshot, const ParentMap& parentMap, const Block* topBlock, hkStringBuf& pathOut)
{
	hkInplaceArray<const Block*, 16> path;
	calcPath(parentMap, topBlock, path);

	pathOut.clear();

	for (int i = 0; i < path.getSize() - 1; i++)
	{
		const Block* parentBlock = path[i];
		const Block* block = path[i + 1];

		if (i > 0)
		{
			pathOut.append(".");
		}

		const char* name = hkScanReportUtil::calcMemberName(scanSnapshot, parentBlock, block);
		if (name)
		{
			pathOut.append(name);
		}
		else
		{
			pathOut.append("#");
		}
	}
}

/* static */void HK_CALL hkScanReportUtil::appendBlockType(const Block* block, hkOstream& stream)
{
	const hkTrackerTypeTreeNode* type = block->m_type;
	if (type)
	{
		type->dumpType(stream);	
	}
	else
	{
		stream << "Unknown";
	}

	if (block->m_arraySize >= 0)
	{
		stream << "[" << block->m_arraySize << "] ";
	}
}

/* static */void HK_CALL hkScanReportUtil::getBlockTypeAsText(const Block* block, hkStringBuf& string)
{
	char buffer[256];

	hkOstream stream(buffer, HK_COUNT_OF(buffer), true);
	hkScanReportUtil::appendBlockType(block, stream);
	
	string = buffer;
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
