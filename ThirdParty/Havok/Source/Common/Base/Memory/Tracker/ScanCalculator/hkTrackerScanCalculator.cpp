/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerScanCalculator.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeParser.h>
#include <Common/Base/Memory/Tracker/CurrentFunction/hkCurrentFunction.h>
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

hkTrackerScanSnapshot* hkTrackerScanCalculator::createSnapshot(const hkTrackerSnapshot* snapshot, hkTrackerLayoutCalculator* layoutCalc)
{
	// Register the reflected types for the snapshot
	hkTrackerTypeInit::registerTypes(hkMemoryTracker::getInstancePtr());
	hkTrackerScanSnapshot* scanSnapshot = _createSnapshot(snapshot, layoutCalc);
	if(hkMemoryTracker::getInstancePtr())
	{
		hkMemoryTracker::getInstancePtr()->clearTypeDefinitions();
	}

	return scanSnapshot;
}

hkTrackerScanSnapshot* hkTrackerScanCalculator::_createSnapshot(const hkTrackerSnapshot* snapshotIn, hkTrackerLayoutCalculator* layoutCalc)
{
	// construction of hkTrackerScanSnapshot will:
	// - copy all allocations from the hkTrackerSnapshot (TODO - remove)
	// - get all the provider allocations at all level inside the m_rawSnapshot member
	// - copy the statistcs string obtained from the memory system into m_memSysStatistics
	m_scanSnapshot = new hkTrackerScanSnapshot(snapshotIn, layoutCalc);
	// The rest of this function is just about processing memory tracker blocks and adding
	// references between them.

	// map used as a type cache to associate a type object with a type name
	const hkArrayBase<hkTrackerSnapshot::ClassAlloc>& trackerAllocs = snapshotIn->getClassAllocs();
	hkTrackerTypeTreeCache* typeCache = layoutCalc->getTypeCache();
	HK_ASSERT(0x5b23fc7c, typeCache != HK_NULL); // make sure that the type cache is not null

	// 1 : Compute type for every block in the tracker snapshot and add the blocks
	// with their type to the scan snapshot
	{
		hkStringMap<const hkTrackerTypeTreeNode*> typeFromName;
		for(int i = 0; i < trackerAllocs.getSize(); ++i)
		{
			const hkTrackerSnapshot::ClassAlloc& trackerAlloc = trackerAllocs[i];
			const char* typeName = trackerAlloc.m_typeName;
			HK_ASSERT(0x25ed7bf, typeName != HK_NULL); // make sure that the type name is not null

			const hkTrackerTypeTreeNode* typeNode = typeFromName.getWithDefault(typeName, HK_NULL);
			if(typeNode == HK_NULL)
			{
				// type not in the map, build a new type and insert it in the map
				if(typeName[0] == '!')
				{
					typeName = typeCache->newText(&typeName[1]);
				}
				else
				{
					hkLocalBuffer<char> buffer(128);
					hkCurrentFunctionUtil::getClassName(typeName, buffer.begin());
					typeName = typeCache->newText(buffer.begin());
				}

				typeNode = hkTrackerTypeTreeParser::parseType(typeName, *typeCache);
				HK_ASSERT(0xafdd104, typeNode != HK_NULL); // typename parsing should always succeed

				typeFromName.insert(typeName, typeNode);
			}

			// The layout calculator is capable of computing the size of a given type
			// autonomously, but registring it now will allow rapid lookup by looking
			// for an entry in the typeNode map. For template types this will also work.
			layoutCalc->setTypeSize(typeNode, trackerAlloc.m_size);

			m_scanSnapshot->addBlock(typeNode, trackerAlloc.m_ptr, trackerAlloc.m_size);
		}
	}

	// 2 : Compute references going out of every block to other blocks, in this
	// phase new blocks might be created for stuff the memory tracker does not know
	// about (like array buffers). Every reference will specify whether the pointed
	// entity is owned by the pointing entity or not (by default ownership is not
	// enforced). Finding outgoing references procedes as follows:
	//     a) If we have a handler that matches the type of the block, we just 
	//        use the user-provided handler to get all the references going out 
	//        of the block, and then we match them with the other blocks available.
	//     b) If the block comes from a non-template class instance and a layout 
	//        is available for it (describing the types of the members) we just 
	//        run through the members looking for pointers.
	//     c) NOT IMPLEMENTED YET
	//        In all other cases, we attempt a scan of the content of the block 
	//        (being it a class instantiation or a generic buffer allocation) 
	//        and match the result with the list of available memory tracker blocks.
	// When looking for references using a user-provided handler, new blocks might
	// be created, these will be added at the end of the block list
	{
		const hkArray<hkTrackerScanSnapshot::Block*>& blocks = m_scanSnapshot->getBlocks();
		hkArray<const hkTrackerLayoutBlock*>::Temp newBlocks;
		hkArray<const hkTrackerLayoutBlock*>::Temp allBlocks;
		// I) Find all references and new blocks from current blocks 		
		for(int i = 0; i < blocks.getSize(); ++i)
		{
			const hkTrackerScanSnapshot::Block* block = blocks[i];
			
			// This function is used to retrieve outgoing references from a specific
			// block as raw pointers. They will be matched with other memory tracker
			// blocks only after all the references have been collected. If a block
			// type is treated specially using a layout handler, the user might also
			// decide to create new blocks inside the handler which are memory areas
			// the tracker wasn't notified about that belong to the containing block.
			// The handler writer should also recur on the new block created finding
			// references going out of those. In the end all the new Blocks will be added
			// to the scan snapshot, but only after the references have all been collected.
			hkTrackerLayoutBlock* layoutBlock = new hkTrackerLayoutBlock(block->m_type, block->m_start, block->m_size, block->m_arraySize);
			allBlocks.pushBack(layoutBlock);
			layoutCalc->getReferences(layoutBlock, newBlocks);
		}
		// II) Add all new blocks to the scan snapshot and the references going out
		//     of them to the references map
		for(int i = 0; i < newBlocks.getSize(); ++i)
		{
			const hkTrackerLayoutBlock* block = newBlocks[i];
			m_scanSnapshot->addBlock(block->m_type, block->m_start, block->m_size);
		}
		allBlocks.append(newBlocks);
		newBlocks.clearAndDeallocate();
		// III) Remove all references which are not pointing to other blocks and add references to the scan snapshot
		//
		// iterate over the references found and get rid of all the references that are not pointing
		// to any block (might be pointing to static or non-heap allocated memory, or to non memory
		// tracked memory.
		hkArray<hkTrackerScanSnapshot::Block*>& snapReferences = m_scanSnapshot->getReferences();
		for(int i = 0; i < allBlocks.getSize(); ++i)
		{
			const hkTrackerLayoutBlock* layoutBlock = allBlocks[i];
			hkTrackerScanSnapshot::Block* block = m_scanSnapshot->getBlock(layoutBlock->m_start);
			HK_ASSERT(0x43ff3a5c, block != HK_NULL); // should always find a corresponding block
			block->m_startReferenceIndex = snapReferences.getSize();
			for(int j = 0; j < layoutBlock->m_references.getSize(); ++j)
			{
				const void* pointedBlockAddr = layoutBlock->m_references[j];
				hkTrackerScanSnapshot::Block* pointedBlock = m_scanSnapshot->getBlock(pointedBlockAddr);
				if(pointedBlock != HK_NULL)
				{
					snapReferences.pushBack(pointedBlock);
				}
			}
			block->m_numReferences = snapReferences.getSize() - block->m_startReferenceIndex;
			delete layoutBlock;
		}
	}

	return m_scanSnapshot;
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
