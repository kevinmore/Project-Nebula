/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Query/Multithreaded/hknpCollisionQueryTask.h>

#if defined(HK_PLATFORM_SPU)
hknpCollisionQueryTask::subTaskFunctions hknpCollisionQueryTask::g_subTaskFunctions[hknpCollisionQuerySubTask::NUM_SUB_TASK_TYPES];
#endif

#if !defined(HK_PLATFORM_SPU)

hknpCollisionQueryTask::hknpCollisionQueryTask() : hkTask()
{
#if defined(HK_PLATFORM_PPU)
	hkString::memCpy4(m_shapeVTablesPpu, hknpShapeVirtualTableUtil::getVTables(), sizeof(m_shapeVTablesPpu) >> 2);
#endif

	reset();
}

void hknpCollisionQueryTask::reset()
{
	m_maxGrab = 5; // reasonable default grab
	m_currentSubTaskIndex = 0;
	m_subTasks.setSize(0);
}

#endif

#if defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_SIM_PPU)

void* hknpCollisionQueryTask::getElf()
{
#if defined(HK_PLATFORM_PS3_PPU)
	extern char _binary_hknpSpursQuery_elf_start[];
	return _binary_hknpSpursQuery_elf_start;
#else
	return (void*)hkTaskType::HKNP_COLLISION_QUERY_TASK;
#endif
}
#endif

#if !defined(HK_PLATFORM_SPU)

hknpAabbQuerySubTask* hknpCollisionQueryTask::allocateAabbQuerySubTask()
{
	hknpCollisionQuerySubTask* pNewTask = &m_subTasks.expandOne();
	pNewTask->m_subTaskType = hknpCollisionQuerySubTask::COLLISION_QUERY_AABB;
	hknpAabbQuerySubTask* pTypedJob = pNewTask->asAabbQuerySubTask();
	pTypedJob = new(pTypedJob) hknpAabbQuerySubTask;
	return pTypedJob;
}

hknpPairGetClosestPointsSubTask* hknpCollisionQueryTask::allocatePairGetClosestPointsSubTask()
{
	hknpCollisionQuerySubTask* pNewTask = &m_subTasks.expandOne();
	pNewTask->m_subTaskType = hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS;
	hknpPairGetClosestPointsSubTask* pTypedJob = pNewTask->asPairGetClosestPointsSubTask();
	pTypedJob = new(pTypedJob) hknpPairGetClosestPointsSubTask;
	return pTypedJob;
}

hknpWorldGetClosestPointsSubTask* hknpCollisionQueryTask::allocateWorldGetClosestPointsSubTask()
{
	hknpCollisionQuerySubTask* pNewTask = &m_subTasks.expandOne();
	pNewTask->m_subTaskType = hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS;
	hknpWorldGetClosestPointsSubTask* pTypedJob = pNewTask->asWorldGetClosestPointsSubTask();
	pTypedJob = new(pTypedJob) hknpWorldGetClosestPointsSubTask;
	return pTypedJob;
}

hknpPairShapeCastSubTask* hknpCollisionQueryTask::allocatePairShapeCastSubTask()
{
	hknpCollisionQuerySubTask* pNewTask = &m_subTasks.expandOne();
	pNewTask->m_subTaskType = hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_SHAPE_CAST;
	hknpPairShapeCastSubTask* pTypedJob = pNewTask->asPairShapeCastSubTask();
	pTypedJob = new(pTypedJob) hknpPairShapeCastSubTask;
	return pTypedJob;
}

hknpWorldShapeCastSubTask* hknpCollisionQueryTask::allocateWorldShapeCastSubTask()
{
	hknpCollisionQuerySubTask* pNewTask = &m_subTasks.expandOne();
	pNewTask->m_subTaskType = hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_SHAPE_CAST;
	hknpWorldShapeCastSubTask* pTypedJob = pNewTask->asWorldShapeCastSubTask();
	pTypedJob = new(pTypedJob) hknpWorldShapeCastSubTask;
	return pTypedJob;
}

void hknpCollisionQueryTask::process()
{
	for(int i=0; i<m_subTasks.getSize(); i++)
	{
		switch(m_subTasks[i].m_subTaskType)
		{
		case hknpCollisionQuerySubTask::COLLISION_QUERY_AABB:
			m_subTasks[i].asAabbQuerySubTask()->process(this);
			break;
		case hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS:
			m_subTasks[i].asPairGetClosestPointsSubTask()->process(this);
			break;
		case hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS:
			m_subTasks[i].asWorldGetClosestPointsSubTask()->process(this);
			break;
		case hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_SHAPE_CAST:
			m_subTasks[i].asPairShapeCastSubTask()->process(this);
			break;
		case hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_SHAPE_CAST:
			m_subTasks[i].asWorldShapeCastSubTask()->process(this);
			break;
		default:
			HK_ASSERT2(0x5ddaf6d, 0, "Unknown job sub-type");
		}
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
