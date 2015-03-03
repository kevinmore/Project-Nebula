/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerStringMapLayoutHandler.h>

void hkTrackerStringMapLayoutHandler::getReferences(
	hkTrackerLayoutBlock* curBlock,
	const void* curData,
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc,
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	bool isStorageStringMap = (curType->m_name == "hkStorageStringMap");
	const hkTrackerTypeTreeNode* contentType = curType->m_contains;
	bool containsPointers = (contentType->m_type == hkTrackerTypeTreeNode::TYPE_POINTER);
	const hkStringMap<void*>* map = static_cast< const hkStringMap<void*>* >(curData);
	hkTrackerTypeTreeCache* typeCache = layoutCalc->getTypeCache();
	const int mapCapacity = map->getCapacity();

	if(mapCapacity > 0)
	{
		const char* bufferTypeName = isStorageStringMap ? "buffer_hkStorageStringMap" : "buffer_hkStringMap";
		const void* bufferPtr = map->m_map.m_elem;
		hk_size_t bufferCapacity = mapCapacity * sizeof(hkCachedHashMap<hkStringMapOperations, hkContainerHeapAllocator>::Elem);

		// add reference to buffer to the current block
		curBlock->m_references.pushBack(bufferPtr);
		// add a new block corresponding to the body of the map
		const hkTrackerTypeTreeNode* bufferType = 
			typeCache->newNamedNode(hkTrackerTypeTreeNode::TYPE_NAMED, bufferTypeName, false);
		hkTrackerLayoutBlock* bufferBlock = new hkTrackerLayoutBlock(bufferType, bufferPtr, bufferCapacity);
		newBlocks.pushBack(bufferBlock);
	
		// add references from the map body (and new blocks in case of storage map)
		for ( hkStringMap<void*>::Iterator it = map->getIterator(); 
		      map->isValid(it); 
		      it = map->getNext(it) )
		{
			const char* key = map->getKey(it);

			// add references
			bufferBlock->m_references.pushBack(key);
			if(containsPointers)
			{
				void* value = map->getValue(it);
				bufferBlock->m_references.pushBack(value);
			}
			// add new block for keys in case of storage string map
			if(isStorageStringMap)
			{
				const hkTrackerTypeTreeNode* keyType = 
					typeCache->newNamedNode(hkTrackerTypeTreeNode::TYPE_NAMED, "buffer_hkStorageStringMap_key", false);
				hkTrackerLayoutBlock* keyBlock = new hkTrackerLayoutBlock(keyType, key, hkString::strLen(key)+1);
				newBlocks.pushBack(keyBlock);
			}
		}
	}
}

hk_size_t hkTrackerStringMapLayoutHandler::getSize(
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc )
{
	// same size for string maps and storage string maps
	return sizeof(hkStringMap<void*>);
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
