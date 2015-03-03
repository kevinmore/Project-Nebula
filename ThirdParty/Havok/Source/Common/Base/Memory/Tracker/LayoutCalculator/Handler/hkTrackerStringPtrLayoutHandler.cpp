/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerStringPtrLayoutHandler.h>

void hkTrackerStringPtrLayoutHandler::getReferences(
	hkTrackerLayoutBlock* curBlock,
	const void* curData,
	const hkTrackerTypeTreeNode*, // unused
	hkTrackerLayoutCalculator* layoutCalc,
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	hkTrackerTypeTreeCache* typeCache = layoutCalc->getTypeCache();
	const hkStringPtr* strPtr = static_cast<const hkStringPtr*>(curData);
	if(strPtr != HK_NULL)
	{
		// first register the reference
		curBlock->m_references.pushBack(strPtr->cString());
		// register a new block if owned
		if((hkUlong(strPtr->m_stringAndFlag) & hkStringPtr::OWNED_FLAG) == hkStringPtr::OWNED_FLAG)
		{
			const hkTrackerTypeTreeNode* bufferType = 
				typeCache->newNamedNode(hkTrackerTypeTreeNode::TYPE_NAMED, "buffer_hkStringPtr", false);
			hkTrackerLayoutBlock* bufferBlock = new hkTrackerLayoutBlock(bufferType, strPtr->cString(), strPtr->getLength()+1);
			newBlocks.pushBack(bufferBlock);
		}
	}
}

hk_size_t hkTrackerStringPtrLayoutHandler::getSize(
	const hkTrackerTypeTreeNode*, // unused
	hkTrackerLayoutCalculator* ) // unused
{
	return sizeof(hkStringPtr);
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
