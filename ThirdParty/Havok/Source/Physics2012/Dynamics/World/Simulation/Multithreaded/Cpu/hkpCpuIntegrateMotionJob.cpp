/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>
#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuIntegrateMotionJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>


hkJobQueue::JobStatus HK_CALL hkCpuIntegrateMotionJob(	hkpMtThreadStructure&		tl,
														hkJobQueue&					jobQueue,
														hkJobQueue::JobQueueEntry&	jobInOut)
{
	HK_TIMER_BEGIN_LIST("Integrate", "IntegrateMotions");

	hkpIntegrateMotionJob& job = reinterpret_cast<hkpIntegrateMotionJob&>(jobInOut);

	int firstEntityIdx = job.m_firstEntityIdx;
	int numEntities    = job.m_numEntities;

	hkpVelocityAccumulator* accumulatorBatch;
	{
		hkpVelocityAccumulator* accumulatorsBase = job.m_taskHeader->m_accumulatorsBase;
		accumulatorBatch = &accumulatorsBase[1+firstEntityIdx];
	}
	hkpEntity*const* entityBatch = &job.m_taskHeader->m_allEntities[firstEntityIdx];

	HK_ON_DEBUG_MULTI_THREADING( {	 for ( int i=0; i < numEntities; i++ ){ entityBatch[i]->markForWrite(); }	} );

	//
	// apply accumulators to all entities in batch
	//
	if(job.m_applyForcesAndStepMotionOnly)
	{
		job.m_numInactiveFrames = hkRigidMotionUtilApplyForcesAndStep( tl.m_world->m_dynamicsStepInfo.m_solverInfo, tl.m_world->m_dynamicsStepInfo.m_stepInfo, tl.m_world->m_dynamicsStepInfo.m_solverInfo.m_globalAccelerationPerStep, (hkpMotion*const*)entityBatch, numEntities, HK_OFFSET_OF(hkpEntity,m_motion) );
	}
	else
	{
		job.m_numInactiveFrames = hkRigidMotionUtilApplyAccumulators( tl.m_world->m_dynamicsStepInfo.m_solverInfo, tl.m_world->m_dynamicsStepInfo.m_stepInfo, accumulatorBatch, (hkpMotion*const*)entityBatch, numEntities, HK_OFFSET_OF(hkpEntity, m_motion));
	}

	//
	// calculate AABBs for all entities in batch
	//
	hkpEntityAabbUtil::entityBatchRecalcAabb(tl.m_world->getCollisionInput(), entityBatch, numEntities);

	HK_ON_DEBUG_MULTI_THREADING( {	 for ( int i=0; i < numEntities; i++ ){ entityBatch[i]->unmarkForWrite(); }	} );

	HK_TIMER_END_LIST();

	return jobQueue.finishJobAndGetNextJob( &jobInOut, jobInOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
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
