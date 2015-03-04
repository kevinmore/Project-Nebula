/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuBuildAccumulatorsJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>


hkJobQueue::JobStatus HK_CALL hkCpuBuildAccumulatorsJob(hkpMtThreadStructure&		tl,
														hkJobQueue&					jobQueue,
														hkJobQueue::JobQueueEntry&	nextJobOut )
{
	const hkpBuildAccumulatorsJob& job = reinterpret_cast<hkpBuildAccumulatorsJob&>(nextJobOut);

	HK_TIMER_BEGIN_LIST("Integrate", "BuildAccumulators" );

	hkpSimulationIsland* island = job.m_island;
	island->markAllEntitiesReadOnly();

	{
		hkpMotion*const* motions				= (hkpMotion*const*)( job.m_islandEntitiesArray + job.m_firstEntityIdx );
		int numMotions						= job.m_numEntities;
		int motionsOffset					= HK_OFFSET_OF(hkpEntity, m_motion);
		hkpVelocityAccumulator* accumulators	= job.m_taskHeader->m_accumulatorsBase + 1 + job.m_firstEntityIdx;

		hkRigidMotionUtilApplyForcesAndBuildAccumulators( tl.m_collisionInput.m_stepInfo, motions, numMotions, motionsOffset, accumulators );
	}

	island->unmarkAllEntitiesReadOnly();

	HK_TIMER_END_LIST();

	return jobQueue.finishJobAndGetNextJob( (const hkJobQueue::JobQueueEntry*)&job, nextJobOut );
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
