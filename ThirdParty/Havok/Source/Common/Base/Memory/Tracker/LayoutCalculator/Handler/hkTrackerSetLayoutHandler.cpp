/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerSetLayoutHandler.h>
#include <Common/Base/Container/Set/hkSet.h>

void hkTrackerSetLayoutHandler::getReferences(
	hkTrackerLayoutBlock* curBlock,
	const void* curData,
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc,
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	hkTrackerTypeTreeCache* typeCache = layoutCalc->getTypeCache();
	const hkTrackerTypeTreeNode* contentType = curType->m_contains;

	// create nested array type
	hkTrackerTypeTreeNode* arrayType = typeCache->newNode(hkTrackerTypeTreeNode::TYPE_CLASS_TEMPLATE);
	arrayType->m_contains = contentType;
	const char* name = typeCache->newText("hkArray");
	arrayType->m_name = name;

	const hkSet<hkUint8>* set = static_cast<const hkSet<hkUint8>*>(curData);
	const void* arrayData = &set->m_elem;

	layoutCalc->getReferencesRecursive(curBlock, arrayData, arrayType, newBlocks);
}

hk_size_t hkTrackerSetLayoutHandler::getSize(
	const hkTrackerTypeTreeNode*, // unused
	hkTrackerLayoutCalculator* ) // unused
{
	return sizeof(hkSet<hkUint8>);
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
