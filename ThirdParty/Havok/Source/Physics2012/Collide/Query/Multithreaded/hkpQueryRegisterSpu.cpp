/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobQueueUtils.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobQueueUtils.h>

#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuPairGetClosestPointsJob.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuWorldGetClosestPointsJob.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuMoppAabbJob.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuPairLinearCastJob.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuWorldLinearCastJob.h>

#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Cpu/hkpCpuShapeRaycastJob.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Cpu/hkpCpuWorldRaycastJob.h>


static hkJobQueue::ProcessJobFunc s_collisionQueryProcessFuncs[hkpCollisionQueryJob::COLLISION_QUERY_JOB_END];

void HK_CALL hkpCollisionQueryJobQueueUtils::registerWithJobQueue(hkJobQueue* queue)
{
#if defined(HK_PLATFORM_MULTI_THREAD) && (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)
	hkJobQueue::hkJobHandlerFuncs jobHandlerFuncs;
	jobHandlerFuncs.m_popJobFunc	= hkpCollisionQueryJobQueueUtils::popCollisionJob;
	jobHandlerFuncs.m_finishJobFunc = hkpCollisionQueryJobQueueUtils::finishCollisionJob;

	jobHandlerFuncs.initProcessJobFuncs( s_collisionQueryProcessFuncs, HK_COUNT_OF(s_collisionQueryProcessFuncs) );
	jobHandlerFuncs.registerProcessJobFunc( hkpCollisionQueryJob::COLLISION_QUERY_PAIR_LINEAR_CAST, hkCpuPairLinearCastJob );
	jobHandlerFuncs.registerProcessJobFunc( hkpCollisionQueryJob::COLLISION_QUERY_WORLD_LINEAR_CAST, hkCpuWorldLinearCastJob );
	jobHandlerFuncs.registerProcessJobFunc( hkpCollisionQueryJob::COLLISION_QUERY_MOPP_AABB, hkCpuMoppAabbQueryJob );
	jobHandlerFuncs.registerProcessJobFunc( hkpCollisionQueryJob::COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS, hkCpuPairGetClosestPointsJob );
	jobHandlerFuncs.registerProcessJobFunc( hkpCollisionQueryJob::COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS, hkCpuWorldGetClosestPointsJob );
	queue->registerJobHandler( HK_JOB_TYPE_COLLISION_QUERY, jobHandlerFuncs );

#	if defined(HK_PLATFORM_HAS_SPU)
#		if defined(HK_PLATFORM_PS3_PPU) 
	extern char _binary_hkpSpursCollisionQuery_elf_start[];
	void* elf =	_binary_hkpSpursCollisionQuery_elf_start;
#		else
	void* elf = (void*)HK_JOB_TYPE_COLLISION_QUERY;
#		endif
	queue->registerSpuElf( HK_JOB_TYPE_COLLISION_QUERY, elf );
#	endif
#endif
}


static hkJobQueue::ProcessJobFunc s_raycastQueryProcessFuncs[hkpRayCastQueryJob::RAYCAST_QUERY_JOB_END];

void HK_CALL hkpRayCastQueryJobQueueUtils::registerWithJobQueue(hkJobQueue* queue)
{
#ifdef HK_PLATFORM_MULTI_THREAD
	hkJobQueue::hkJobHandlerFuncs jobHandlerFuncs;
	jobHandlerFuncs.m_popJobFunc	= hkpRayCastQueryJobQueueUtils::popRayCastQueryJob;
	jobHandlerFuncs.m_finishJobFunc = hkpRayCastQueryJobQueueUtils::finishRayCastQueryJob;

	jobHandlerFuncs.initProcessJobFuncs( s_raycastQueryProcessFuncs, HK_COUNT_OF(s_raycastQueryProcessFuncs) );

	jobHandlerFuncs.registerProcessJobFunc(hkpRayCastQueryJob::RAYCAST_QUERY_SHAPE_RAYCAST, hkCpuShapeRayCastJob );
	jobHandlerFuncs.registerProcessJobFunc(hkpRayCastQueryJob::RAYCAST_QUERY_WORLD_RAYCAST, hkCpuWorldRayCastJob );

	queue->registerJobHandler( HK_JOB_TYPE_RAYCAST_QUERY, jobHandlerFuncs );

#	if defined(HK_PLATFORM_HAS_SPU)
#		ifdef HK_PLATFORM_PS3_PPU
	extern char _binary_hkpSpursRayCastQuery_elf_start[];
	void* elf =	_binary_hkpSpursRayCastQuery_elf_start;
#		else // Win32 SPU Simulator
	void* elf = (void*)HK_JOB_TYPE_RAYCAST_QUERY;
#		endif

	queue->registerSpuElf( HK_JOB_TYPE_RAYCAST_QUERY, elf );
#	endif
#endif
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
