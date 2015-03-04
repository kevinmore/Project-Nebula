/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerJobQueueLayoutHandler.h>
#include <Common/Base/Thread/JobQueue/hkJobQueue.h>

void hkTrackerJobQueueLayoutHandler::getReferences(
	hkTrackerLayoutBlock* curBlock,
	const void* curData,
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc,
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	hkTrackerTypeTreeCache* typeCache = layoutCalc->getTypeCache();
	const hkTrackerTypeLayout* layout = layoutCalc->getLayout(curType);
	HK_ASSERT(0x2ccfb8e7, layout != HK_NULL); // should always find a layout
	
	// run through the members and find additional references
	// fix the type of the members depending on the platform
	for (int i = 0; i < layout->m_members.getSize(); i++)
	{
		hkTrackerTypeTreeNode* memberType = typeCache->newNode(hkTrackerTypeTreeNode::TYPE_ARRAY);

		const hkTrackerTypeLayout::Member& member = layout->m_members[i];
		*memberType = *(member.m_type); // copy the type

		if(hkString::strCmp(member.m_name, "m_queueSemaphores") == 0)
		{
			HK_ASSERT(0x462431f3, memberType->m_type == hkTrackerTypeTreeNode::TYPE_ARRAY);
			memberType->m_dimension = MAX_NUM_THREAD_TYPES;
		}
		else if(hkString::strCmp(member.m_name, "m_jobFuncs") == 0)
		{
			HK_ASSERT(0x2f6c3782, memberType->m_type == hkTrackerTypeTreeNode::TYPE_ARRAY);
			memberType->m_dimension = HK_JOB_TYPE_MAX;	
		}

		const void* memberData = static_cast<const hkUint8*>(curData) + member.m_offset;

		// Recur
		layoutCalc->getReferencesRecursive(curBlock, memberData, memberType, newBlocks);
	}
}

hk_size_t hkTrackerJobQueueLayoutHandler::getSize(
	const hkTrackerTypeTreeNode*, // unused
	hkTrackerLayoutCalculator* ) // unused
{
	return sizeof(hkJobQueue);
}

void hkTrackerJobQueueDynamicDataLayoutHandler::getReferences(
	hkTrackerLayoutBlock* curBlock,
	const void* curData,
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc,
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	hkTrackerTypeTreeCache* typeCache = layoutCalc->getTypeCache();
	const hkTrackerTypeLayout* layout = layoutCalc->getLayout(curType);
	HK_ASSERT(0x2e272f5, layout != HK_NULL); // should always find a layout
	
	// run through the members and find additional references
	// fix the type of the member depending on the platform
	for (int i = 0; i < layout->m_members.getSize(); i++)
	{
		hkTrackerTypeTreeNode* memberType = typeCache->newNode(hkTrackerTypeTreeNode::TYPE_ARRAY);

		const hkTrackerTypeLayout::Member& member = layout->m_members[i];
		*memberType = *(member.m_type); // copy the type

		if(hkString::strCmp(member.m_name, "m_jobQueue") == 0)
		{
			HK_ASSERT(0x1150649c, memberType->m_type == hkTrackerTypeTreeNode::TYPE_ARRAY);
			memberType->m_dimension = MAX_NUM_QUEUES;
		}

		const void* memberData = static_cast<const hkUint8*>(curData) + member.m_offset;

		// Recur
		layoutCalc->getReferencesRecursive(curBlock, memberData, memberType, newBlocks);
	}
}

hk_size_t hkTrackerJobQueueDynamicDataLayoutHandler::getSize(
	const hkTrackerTypeTreeNode*, // unused
	hkTrackerLayoutCalculator* ) // unused
{
	return sizeof(hkJobQueue::DynamicData);
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
