/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Query/Multithreaded/hknpRaycastTask.h>
#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>
#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>
#include <Physics/Physics/Collide/Shape/hknpShapeQueryInterface.h>
#include <Physics/Physics/Collide/Filter/Group/hknpGroupCollisionFilter.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>

#if !defined(HK_PLATFORM_SPU)

hknpRaycastTask::hknpRaycastTask() : hkTask()
{
#if defined(HK_PLATFORM_HAS_SPU)
	hkString::memCpy4(m_shapeVTablesPpu, hknpShapeVirtualTableUtil::getVTables(), sizeof(m_shapeVTablesPpu) >> 2);
#endif

	reset();
}

void hknpRaycastTask::reset()
{
	m_maxGrab = 5; // reasonable default grab
	m_currentSubTaskIndex = 0;
	m_subTasks.setSize(0);
}

#if defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_SIM_PPU)

void* hknpRaycastTask::getElf()
{
#if defined(HK_PLATFORM_PS3_PPU)
	extern char _binary_hknpSpursRaycast_elf_start[];
	return _binary_hknpSpursRaycast_elf_start;
#else
	return (void*)hkTaskType::HKNP_RAYCAST_TASK;
#endif
}
#endif

hknpShapeRaycastSubTask* hknpRaycastTask::allocateShapeRaycastSubTask()
{
	SubTask* pNewTask = &m_subTasks.expandOne();
	pNewTask->m_subTaskType = SHAPE_RAYCAST;
	hknpShapeRaycastSubTask* pTypedJob = pNewTask->asShapeRaycastSubTask();
	pTypedJob = new(pTypedJob) hknpShapeRaycastSubTask;
	return pTypedJob;
}

hknpWorldRaycastSubTask* hknpRaycastTask::allocateWorldRaycastSubTask()
{
	SubTask* pNewTask = &m_subTasks.expandOne();
	pNewTask->m_subTaskType = WORLD_RAYCAST;
	hknpWorldRaycastSubTask* pTypedJob = pNewTask->asWorldRaycastSubTask();
	pTypedJob = new(pTypedJob) hknpWorldRaycastSubTask;
	return pTypedJob;
}

void hknpRaycastTask::process()
{
	for(int i=0; i<m_subTasks.getSize(); i++)
	{
		switch(m_subTasks[i].m_subTaskType)
		{
		case hknpRaycastTask::SHAPE_RAYCAST:
			m_subTasks[i].asShapeRaycastSubTask()->process(this);
			break;
		case hknpRaycastTask::WORLD_RAYCAST:
			m_subTasks[i].asWorldRaycastSubTask()->process(this);
			break;
		default:
			HK_ASSERT2(0x5ddaf6d, 0, "Unknown job sub-type");
		}
	}
}

#endif

#if !defined(HK_PLATFORM_SPU)

hknpShapeRaycastSubTask::hknpShapeRaycastSubTask()
{

}

void hknpShapeRaycastSubTask::process(hknpRaycastTask* task)
{
	hknpClosestHitCollector collector;
	{
		m_query.m_filter = task->m_querySharedData.m_collisionFilter;

		hknpCollisionQueryContext queryContext;
		queryContext.m_shapeTagCodec = task->m_querySharedData.m_shapeTagCodec;

		m_targetShapeInfo.m_shapeToWorld = &m_targetShapeToWorld;
		m_targetShapeInfo.m_body = HK_NULL;

		hknpShapeQueryInterface::castRay(&queryContext, m_query, *m_targetShape, m_targetShapeFilterData, m_targetShapeInfo, &collector);
	}

	m_numHits = 0;

	if(collector.hasHit())
	{
		m_numHits = 1;
		m_resultArray[0] = collector.getHits()[0];
	}
}

hknpWorldRaycastSubTask::hknpWorldRaycastSubTask()
{

}

void hknpWorldRaycastSubTask::process(hknpRaycastTask* task)
{
	hknpClosestHitCollector collector;
	m_pWorld->castRay(m_query, &collector);

	m_numHits = 0;

	if(collector.hasHit())
	{
		m_numHits = 1;
		m_resultArray[0] = collector.getHits()[0];
	}
}

#endif

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
