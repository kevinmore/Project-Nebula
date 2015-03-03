/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerQueueLayoutHandler.h>
#include <Common/Base/Container/Queue/hkQueue.h>

void hkTrackerQueueLayoutHandler::getReferences(
	hkTrackerLayoutBlock* curBlock,
	const void* curData,
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc,
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	hkTrackerTypeTreeCache* typeCache = layoutCalc->getTypeCache();
	const hkTrackerTypeTreeNode* contentType = curType->m_contains;
	hk_size_t contentSize = layoutCalc->calcTypeSize(contentType);
	const hkQueue<hkUint8>* queue = static_cast< const hkQueue<hkUint8>* >(curData);

	const int queueCapacity = queue->getCapacity();
	if(queueCapacity > 0)
	{
		const void* bufferPtr = queue->getData();
		const int queueSize = queue->getSize();
		// add new block and reference to the buffer

		hk_size_t bufferCapacity = queueCapacity * contentSize;
		// add reference to buffer to the current block
		curBlock->m_references.pushBack(bufferPtr);
		// add a new block corresponding to the body
		const hkTrackerTypeTreeNode* bufferType = 
			typeCache->newNamedNode(hkTrackerTypeTreeNode::TYPE_NAMED, "buffer_hkQueue", false);
		hkTrackerLayoutBlock* bufferBlock = new hkTrackerLayoutBlock(bufferType, bufferPtr, bufferCapacity);
		newBlocks.pushBack(bufferBlock);
	
		const int queueHead = queue->_getHead();
		// recur on the contained elements
		for(int i = queueHead, count = 0; count < queueSize; i = (i+1)%queueCapacity, ++count)
		{
			const void* elemPtr = hkAddByteOffsetConst(bufferPtr, i*contentSize);
			layoutCalc->getReferencesRecursive(bufferBlock, elemPtr, contentType, newBlocks);
		}
	}
}

hk_size_t hkTrackerQueueLayoutHandler::getSize(
	const hkTrackerTypeTreeNode*, // unused
	hkTrackerLayoutCalculator* ) // unused
{
	return sizeof(hkQueue<hkUint8>);
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
