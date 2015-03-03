/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerArrayLayoutHandler.h>

void hkTrackerArrayLayoutHandler::getReferences(
	hkTrackerLayoutBlock* curBlock,
	const void* curData,
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc,
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	hkTrackerTypeTreeCache* typeCache = layoutCalc->getTypeCache();
	const hkTrackerTypeTreeNode* contentType = curType->m_contains;
	hk_size_t contentSize = layoutCalc->calcTypeSize(contentType);
	
	const void* bufferPtr = HK_NULL;
	int arraySize = 0;
	int arrayCapacity = 0;
	hkTrackerLayoutBlock* bufferBlock = HK_NULL; 
	if(curType->m_name == "hkInplaceArray")
	{
		const hkInplaceArray<hkUint8, 4>* ar = static_cast<const hkInplaceArray<hkUint8, 4>*>(curData);
		bufferPtr = ar->begin();
		arraySize = ar->getSize();
		arrayCapacity = ar->getCapacity();
		if(arrayCapacity > 0)
		{
			bufferBlock = curBlock;
			if(bufferPtr != &ar->m_storage[0])
			{
				hk_size_t bufferCapacity = arrayCapacity * contentSize;
				// add reference to buffer to the current block
				curBlock->m_references.pushBack(bufferPtr);
				// add a new block corresponding to the body
				const hkTrackerTypeTreeNode* bufferType = 
					typeCache->newNamedNode(hkTrackerTypeTreeNode::TYPE_NAMED, "buffer_hkInplaceArray", false);
				bufferBlock = new hkTrackerLayoutBlock(bufferType, bufferPtr, bufferCapacity);
				newBlocks.pushBack(bufferBlock);
			}
		}
	}
	else if(curType->m_name == "hkSmallArray")
	{
		const hkSmallArray<hkUint8>* ar = static_cast<const hkSmallArray<hkUint8>*>(curData);
		bufferPtr = ar->begin();
		arraySize = ar->getSize();
		arrayCapacity = ar->getCapacity();

		if(arrayCapacity > 0)
		{
			hk_size_t bufferCapacity = arrayCapacity * contentSize;
			// add reference to buffer to the current block
			curBlock->m_references.pushBack(bufferPtr);
			// add a new block corresponding to the body
			const hkTrackerTypeTreeNode* bufferType = 
				typeCache->newNamedNode(hkTrackerTypeTreeNode::TYPE_NAMED, "buffer_hkSmallArray", false);
			bufferBlock = new hkTrackerLayoutBlock(bufferType, bufferPtr, bufferCapacity);
			newBlocks.pushBack(bufferBlock);
		}
	}
	else // hkArray<> or hkArrayBase<>
	{
		HK_WARN_ON_DEBUG_IF(curType->m_name == "hkArrayBase", 0, "Tracking references from an hkArrayBase<> object, assuming hkArray<> memory layout.");

		const hkArray<hkUint8>* ar = static_cast<const hkArray<hkUint8>*>(curData);
		bufferPtr = ar->begin();
		arraySize = ar->getSize();
		arrayCapacity = ar->getCapacity();

		if(arrayCapacity > 0)
		{
			hk_size_t bufferCapacity = arrayCapacity * contentSize;
			// add reference to buffer to the current block
			curBlock->m_references.pushBack(bufferPtr);
			// add a new block corresponding to the body
			const hkTrackerTypeTreeNode* bufferType = 
				typeCache->newNamedNode(hkTrackerTypeTreeNode::TYPE_NAMED, "buffer_hkArray", false);
			bufferBlock = new hkTrackerLayoutBlock(bufferType, bufferPtr, bufferCapacity);
			newBlocks.pushBack(bufferBlock);
		}
	}

	// recur on buffer block (even if it's inplace)
	for(int i = 0; i < arraySize; ++i)
	{
		const void* elemPtr = hkAddByteOffsetConst(bufferPtr, i*contentSize);
		layoutCalc->getReferencesRecursive(bufferBlock, elemPtr, contentType, newBlocks);
	}
}

hk_size_t hkTrackerArrayLayoutHandler::getSize(
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc )
{
	if(curType->m_name == "hkInplaceArray")
	{
		const hkTrackerTypeTreeNode* contentType = curType->m_contains;
		hk_size_t contentSize = layoutCalc->calcTypeSize(contentType);
		const int inplaceSize = curType->m_contains->m_next->m_dimension;
		return sizeof(hkArray<hkUint8>) + inplaceSize*contentSize;
	}
	if(curType->m_name == "hkSmallArray")
	{
		return sizeof(hkSmallArray<hkUint8>);
	}
	else // hkArray<> or hkArrayBase<>
	{
		return sizeof(hkArray<hkUint8>);
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
