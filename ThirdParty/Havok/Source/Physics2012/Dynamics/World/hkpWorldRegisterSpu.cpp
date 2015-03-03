/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpMultithreadedSimulation.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobQueueUtils.h>

#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobQueueUtils.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobQueueUtils.h>

#include <Common/Base/Thread/JobQueue/hkJobQueue.h>

static hkJobQueue::ProcessJobFunc s_dynamicsProcessFuncs[hkpDynamicsJob::DYNAMICS_JOB_END];

void HK_CALL hkpWorld::registerWithJobQueue( hkJobQueue* jobQueue )
{
	hkJobQueue::hkJobHandlerFuncs jobHandlerFuncs;
	jobHandlerFuncs.initProcessJobFuncs( s_dynamicsProcessFuncs, HK_COUNT_OF( s_dynamicsProcessFuncs) );

	for (hkJobSubType jobId = 0; jobId < hkpDynamicsJob::DYNAMICS_JOB_END; jobId++)
	{
		jobHandlerFuncs.registerProcessJobFunc( jobId, hkpMultiThreadedSimulation::processNextJob );
	}

	// Physics only works if we have a custom job specified for the broad phase for internal synchronization
	// If we have only one thread, we still allow for 2, and specify the broad phase job to be associated with the
	/// never used thread 1 - this effectively de-prioritizes the broad phase job for thread 0,
	// and avoids deadlocks
	if (jobQueue->m_hwSetup.m_numCpuThreads < 2)
	{
		jobQueue->m_hwSetup.m_numCpuThreads = 2;
	}
	jobQueue->registerJobWithCpuThread( HK_JOB_TYPE_DYNAMICS, hkpDynamicsJob::DYNAMICS_JOB_BROADPHASE, HK_BROAD_PHASE_THREAD_AFFINITY);

	// Handle integration jobs
	jobHandlerFuncs.m_popJobFunc	 = hkpJobQueueUtils::popIntegrateJob;
	jobHandlerFuncs.m_finishJobFunc  = hkpJobQueueUtils::finishIntegrateJob;
	jobQueue->registerJobHandler( HK_JOB_TYPE_DYNAMICS, jobHandlerFuncs );

	// Handle all types of collide job
	jobHandlerFuncs.m_popJobFunc	 = hkpJobQueueUtils::popCollideJob;
	jobHandlerFuncs.m_finishJobFunc  = hkpJobQueueUtils::finishCollideJob;
	jobQueue->registerJobHandler( HK_JOB_TYPE_COLLIDE, jobHandlerFuncs );
	jobQueue->registerJobHandler( HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND, jobHandlerFuncs );

	// Register all ELFs
#if defined(HK_PLATFORM_HAS_SPU) && !defined( HK_PLATFORM_SPU )
#ifdef HK_PLATFORM_PS3_PPU
	extern char _binary_hkpSpursIntegrate_elf_start[];
	jobQueue->registerSpuElf( HK_JOB_TYPE_DYNAMICS, _binary_hkpSpursIntegrate_elf_start );

	extern char _binary_hkpSpursCollide_elf_start[];
	jobQueue->registerSpuElf( HK_JOB_TYPE_COLLIDE, _binary_hkpSpursCollide_elf_start );

	extern char _binary_hkpSpursCollideStaticCompound_elf_start[];
	jobQueue->registerSpuElf( HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND, _binary_hkpSpursCollideStaticCompound_elf_start );
#else
	jobQueue->registerSpuElf( HK_JOB_TYPE_DYNAMICS, (void*)HK_JOB_TYPE_DYNAMICS );
	jobQueue->registerSpuElf( HK_JOB_TYPE_COLLIDE, (void*)HK_JOB_TYPE_COLLIDE );
	jobQueue->registerSpuElf( HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND, (void*)HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND );
#endif
#endif

	hkpCollisionQueryJobQueueUtils::registerWithJobQueue( jobQueue );
	hkpRayCastQueryJobQueueUtils::registerWithJobQueue( jobQueue );
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
