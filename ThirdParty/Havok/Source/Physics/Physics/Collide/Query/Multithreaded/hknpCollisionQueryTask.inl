/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#if defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE void hknpCollisionQueryTask::resetSubtasks()
{
	for(hkUint32 i=0; i<hknpCollisionQuerySubTask::NUM_SUB_TASK_TYPES; i++)
	{
		g_subTaskFunctions[i].m_preloadFP = HK_NULL;
		g_subTaskFunctions[i].m_processFP = HK_NULL;
	}
}

HK_FORCE_INLINE void hknpCollisionQueryTask::registerSubTask(hknpCollisionQuerySubTask::SubTaskType subTaskType)
{
	switch(subTaskType)
	{
	case hknpCollisionQuerySubTask::COLLISION_QUERY_AABB:
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_AABB].m_preloadFP = &hknpAabbQuerySubTask::preload;
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_AABB].m_processFP = &hknpAabbQuerySubTask::process;
		break;
	case hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS:
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS].m_preloadFP = &hknpPairGetClosestPointsSubTask::preload;
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS].m_processFP = &hknpPairGetClosestPointsSubTask::process;
		break;
	case hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS:
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS].m_preloadFP = &hknpWorldGetClosestPointsSubTask::preload;
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS].m_processFP = &hknpWorldGetClosestPointsSubTask::process;
		break;
	case hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_SHAPE_CAST:
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_SHAPE_CAST].m_preloadFP = &hknpPairShapeCastSubTask::preload;
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_PAIR_SHAPE_CAST].m_processFP = &hknpPairShapeCastSubTask::process;
		break;
	case hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_SHAPE_CAST:
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_SHAPE_CAST].m_preloadFP = &hknpWorldShapeCastSubTask::preload;
		g_subTaskFunctions[hknpCollisionQuerySubTask::COLLISION_QUERY_WORLD_SHAPE_CAST].m_processFP = &hknpWorldShapeCastSubTask::process;
		break;
	default:
		HK_ASSERT2(0x313939b8, 0, "Unknown subTask type");
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
