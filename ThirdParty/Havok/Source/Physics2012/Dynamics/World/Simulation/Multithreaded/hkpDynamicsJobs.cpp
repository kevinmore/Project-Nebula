/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>


#include <Physics2012/Dynamics/Constraint/Setup/hkpConstraintSolverSetup.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobQueueUtils.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Simulation/hkpSimulation.h>

#if defined(HK_PLATFORM_SPU)
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#endif

#if defined (HK_ENABLE_DETERMINISM_CHECKS)
hkCheckDeterminismUtil::Fuid hkpDynamicsJob::getFuid() const
{
	hkCheckDeterminismUtil::Fuid fuid;
	fuid.setPackedJobId(*this);
	fuid.m_0 = m_island->m_uTag;
	fuid.m_2 = m_jobSid;
	fuid.m_3 = m_island->getWorld()->m_simulation->m_determinismCheckFrameCounter;
	fuid.m_4 = m_island->m_determinismFrameCounterFromCreationTime;
	return fuid;
}
#endif


hkJobQueue::JobPopFuncResult hkpBroadPhaseJob::popJobTask( hkArray<hkpSimulationIsland*>& islands, hkpBroadPhaseJob& out )
{
	hkpBuildJacobianTaskHeader* taskHeader = this->m_taskHeader;

	// check for pending split simulation island jobs
	if ( taskHeader && taskHeader->m_newSplitIslands.getSize() )
	{
		HK_ASSERT2(0xad786654, out.m_island->m_entities.begin() != taskHeader->m_allEntities, "Old entities not destroyed");
		int numNewIslands = taskHeader->m_newSplitIslands.getSize();
		taskHeader->m_referenceCount += numNewIslands;
		taskHeader->m_numIslandsAfterSplit += numNewIslands;


		// give the new islands to the job on the queue
		this->m_islandIndex = hkObjectIndex(islands.getSize());
		this->m_numIslands  = hkObjectIndex(numNewIslands);

		hkArray<hkpSimulationIsland*>& newIslands = taskHeader->m_newSplitIslands;
		hkArray<hkpSimulationIsland*>& activeIslands = out.m_island->getWorld()->m_activeSimulationIslands;

		// still need to add the new islands
		for (int i =0 ; i <  numNewIslands; i++)
		{
			hkpSimulationIsland* island = newIslands[i];
			island->m_storageIndex = hkObjectIndex(activeIslands.getSize());
			activeIslands.pushBack(island);
		}
		newIslands.clearAndDeallocate();

		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}

	// if possible split the job into two parts
	if ( m_numIslands > 1 )
	{
		m_numIslands    -= 1;
		m_islandIndex   += 1;
		out.m_numIslands = 1;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
	return hkJobQueue::POP_QUEUE_ENTRY;
}

HK_COMPILE_TIME_ASSERT( sizeof(hkpFireJacobianSetupCallback) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpBuildJacobiansJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveConstraintsJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveApplyGravityJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveConstraintBatchJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveIntegrateVelocitiesJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveExportResultsJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpPostCollideJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpBroadPhaseJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpCreateJacobianTasksJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpSplitSimulationIslandJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpIntegrateMotionJob) <= sizeof(hkJobQueue::JobQueueEntry) );
HK_COMPILE_TIME_ASSERT( sizeof(hkpAgentSectorJob) <= sizeof(hkJobQueue::JobQueueEntry) );

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
