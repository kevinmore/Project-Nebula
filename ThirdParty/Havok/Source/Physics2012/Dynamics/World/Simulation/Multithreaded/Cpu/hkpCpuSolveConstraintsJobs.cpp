/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuSolveConstraintsJobs.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>


hkJobQueue::JobStatus HK_CALL hkCpuSolveConstraintsJob( hkpMtThreadStructure& tl, hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& jobInOut )
{
	HK_TIMER_BEGIN_LIST("Integrate", "Solve");

	const hkpSolveConstraintsJob& job = reinterpret_cast<hkpSolveConstraintsJob&>(jobInOut);

	hkpVelocityAccumulator* accumulators = reinterpret_cast<hkpVelocityAccumulator*>( hkAddByteOffset(job.m_buffer, job.m_accumulatorsOffset) );
	hkpJacobianSchema*	   schemas      = reinterpret_cast<hkpJacobianSchema*>     ( hkAddByteOffset(job.m_buffer, job.m_schemasOffset)      );
	hkpSolverElemTemp*	   solverTemp   = reinterpret_cast<hkpSolverElemTemp*>     ( hkAddByteOffset(job.m_buffer, job.m_solverTempOffset)   );

	//
	//	zero solver results
	//
	{
		const unsigned int elemTempSize = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, job.m_numSolverElemTemps * sizeof(hkpSolverElemTemp));
		HK_ASSERT2(0xad7855dd, hkAddByteOffset(solverTemp, elemTempSize) <= hkAddByteOffset(job.m_buffer, job.m_bufferSize), "Internal Error: elemTemp doesn't fit into buffer.");
		HK_ASSERT2(0xad8766dd, (hkUlong(solverTemp) & (HK_REAL_ALIGNMENT-1)) == 0, "SolverElemTemp buffer not aligned !");
		HK_ALIGN16( const hkUint32 zero[4] ) = { 0, 0, 0, 0 };
		hkString::memSet16(solverTemp, zero, elemTempSize >> 4);
	}

	{
		const hkpWorldDynamicsStepInfo& stepInfo = tl.m_world->m_dynamicsStepInfo;
		hkSolveConstraints( stepInfo.m_solverInfo, schemas, accumulators, solverTemp );
		//HK_MONITOR_ADD_VALUE( "NumJacobians", float(island->m_numSolverResults), HK_MONITOR_TYPE_INT );
		//HK_MONITOR_ADD_VALUE( "NumEntities",  float(island->getEntities().getSize()), HK_MONITOR_TYPE_INT );
	}

	// save some of the data in the job (as job is modified below when transforming to broadphase job)
	hkpBuildJacobianTaskHeader* taskHeader = job.m_taskHeader;

	{
		hkJobQueue::JobQueueEntry jobBuffer;
		new (&jobBuffer) hkpIntegrateMotionJob(job, *taskHeader);
		HK_ASSERT2(0xad876555, taskHeader->m_solveTasks.m_firstSolveJacobiansTask == HK_NULL, "This must be zero to prevent this job from morphing into a SolverResultsExport job.");
		jobQueue.addJob( jobBuffer, hkJobQueue::JOB_HIGH_PRIORITY );
	}

	HK_TIMER_SPLIT_LIST("SolverExport");

	// export solver data
#	if ! defined (HK_PLATFORM_HAS_SPU)
	hkExportImpulsesAndRhs(tl.m_world->m_dynamicsStepInfo.m_solverInfo, taskHeader->m_solverTempBase, taskHeader->m_schemasBase, taskHeader->m_accumulatorsBase );
#	else
	hkExportImpulsesAndRhs(tl.m_world->m_dynamicsStepInfo.m_solverInfo, taskHeader->m_solverTempBase, taskHeader->m_schemasBase, taskHeader->m_accumulatorsBase, HK_NULL );
#	endif

	HK_TIMER_END_LIST();

	return jobQueue.finishJobAndGetNextJob( &jobInOut, jobInOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
}


hkJobQueue::JobStatus HK_CALL hkCpuSolveApplyGravityJob(	hkpMtThreadStructure& tl, hkJobQueue& jobQueue,
																			   hkJobQueue::JobQueueEntry& jobInOut)
{
	HK_TIMER_BEGIN_LIST("Integrate", "AppGrav");

	const hkpSolveApplyGravityJob& job = reinterpret_cast<hkpSolveApplyGravityJob&>(jobInOut);
	const hkpWorldDynamicsStepInfo& stepInfo = tl.m_world->m_dynamicsStepInfo;

	//
	//	apply initial gravity
	//
	hkSolveApplyGravityByTheSteps( stepInfo.m_solverInfo, job.m_accumulators, job.m_accumulatorsEnd );

	HK_TIMER_END_LIST();

	return jobQueue.finishJobAndGetNextJob( &jobInOut, jobInOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
}

#ifdef USE_SOLVER_DEBUG_OSTREAM
//#	define USE_SOLVER_DEBUG_INTERNAL_STEPS
extern hkOstream* debugOstream;
#endif

hkJobQueue::JobStatus HK_CALL hkCpuSolveConstraintBatchJob(	hkpMtThreadStructure& tl, hkJobQueue& jobQueue,
																				  hkJobQueue::JobQueueEntry& jobInOut)
{
	HK_TIMER_BEGIN_LIST("Integrate", "Solve");

	hkpSolveConstraintBatchJob& job = reinterpret_cast<hkpSolveConstraintBatchJob&>(jobInOut);

	hkpVelocityAccumulator* accumulators = job.m_solveConstraintBatchTask->m_accumulators;
	const hkpJacobianSchema*schemas      = job.m_solveConstraintBatchTask->m_schemas;
	hkpSolverElemTemp*	   solverTemp   = job.m_solveConstraintBatchTask->m_solverElemTemp;

	job.m_numSolverMicroSteps = tl.m_world->m_dynamicsStepInfo.m_solverInfo.m_numMicroSteps;

#ifdef USE_SOLVER_DEBUG_OSTREAM
	//
	// Debug print jacobians
	//
	hkBool isFirst0 = job.m_taskHeader->m_solveTasks.m_firstSolveJacobiansTask == job.m_solveConstraintBatchTask;
	if (isFirst0 )
	{
		if ( !debugOstream )
		{
#		if defined(HK_PLATFORM_SPU)
			debugOstream = new hkOfstream("SPU" USE_SOLVER_DEBUG_OSTREAM);
#		else
			debugOstream = new hkOfstream(USE_SOLVER_DEBUG_OSTREAM);
#		endif
		}

		if (job.m_currentSolverStep == 0)
		{
			static int frameCounter = 0;
			if (debugOstream)
			{
				debugOstream->printf("\n\n\n***************************\n");
				debugOstream->printf("*******Frame %i     ******\n", frameCounter++);
				debugOstream->printf("***************************\n\n");
			}

			hkpSolveConstraintBatchTask* task = job.m_solveConstraintBatchTask;
			do
			{
				hkDebugPrintfJacobians(task->m_schemas);
				task = task->m_next;
			}
			while( task );
		}

		{
			// Print all accumulators.
			hkDebugPrintfAccumulators(job.m_currentSolverStep, tl.m_world->m_dynamicsStepInfo.m_solverInfo, accumulators, HK_NULL);
		}
	}
#endif

	//
	//	zero solver results
	//
	if ( (job.m_currentSolverStep == 0) && (job.m_currentSolverMicroStep == 0) )
	{
		const unsigned int elemTempSize = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, job.m_solveConstraintBatchTask->m_sizeOfSolverElemTempBuffer);
		HK_ASSERT2(0xad8766de, (hkUlong(solverTemp) & (HK_REAL_ALIGNMENT-1)) == 0, "SolverElemTemp buffer not aligned");
		HK_ALIGN16( const hkUint32 zero[4] ) = { 0, 0, 0, 0 };
		hkString::memSet16(solverTemp, zero, elemTempSize >> 4);
	}

	{
		const hkpWorldDynamicsStepInfo& stepInfo = tl.m_world->m_dynamicsStepInfo;

#		if ! defined (HK_PLATFORM_HAS_SPU)
		hkSolveConstraintsByTheSteps( stepInfo.m_solverInfo, job.m_currentSolverStep, job.m_currentSolverMicroStep, schemas, accumulators, solverTemp );
#		else
		hkSolveConstraintsByTheSteps( stepInfo.m_solverInfo, job.m_currentSolverStep, job.m_currentSolverMicroStep, schemas, accumulators, solverTemp, job.m_solveConstraintBatchTask->m_accumulatorInterIndices, HK_NULL, HK_NULL );
#		endif
	}

	HK_TIMER_END_LIST();

	return jobQueue.finishJobAndGetNextJob( &jobInOut, jobInOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
}

hkJobQueue::JobStatus HK_CALL hkCpuSolveIntegrateVelocitiesJob(	hkpMtThreadStructure& tl, hkJobQueue& jobQueue,
																					  hkJobQueue::JobQueueEntry& jobInOut)
{
	HK_TIMER_BEGIN_LIST("Integrate", "IntVel");

	hkpSolveIntegrateVelocitiesJob& job = reinterpret_cast<hkpSolveIntegrateVelocitiesJob&>(jobInOut);

	// other one will be
	hkBool isSolvingFinished = false;
	const hkpWorldDynamicsStepInfo& stepInfo = tl.m_world->m_dynamicsStepInfo;

	hkSolveIntegrateVelocitiesByTheSteps( stepInfo.m_solverInfo, job.m_currentSolverStep, job.m_accumulators, job.m_accumulatorsEnd, isSolvingFinished );

	job.m_solvingFinished = isSolvingFinished;

	HK_TIMER_END_LIST();

	return jobQueue.finishJobAndGetNextJob( &jobInOut, jobInOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
}

hkJobQueue::JobStatus HK_CALL hkCpuSolveExportResultsJob(	hkpMtThreadStructure& tl, hkJobQueue& jobQueue,
																				hkJobQueue::JobQueueEntry& jobInOut)
{
	HK_TIMER_BEGIN_LIST("Integrate", "SolverExport");
	hkpSolveExportResultsJob& job = reinterpret_cast<hkpSolveExportResultsJob&>(jobInOut);
#	if ! defined (HK_PLATFORM_HAS_SPU)
	hkExportImpulsesAndRhs(tl.m_world->m_dynamicsStepInfo.m_solverInfo, job.m_solveConstraintBatchTask->m_solverElemTemp, job.m_solveConstraintBatchTask->m_schemas, job.m_solveConstraintBatchTask->m_accumulators );
#	else
	hkExportImpulsesAndRhs(tl.m_world->m_dynamicsStepInfo.m_solverInfo, job.m_solveConstraintBatchTask->m_solverElemTemp, job.m_solveConstraintBatchTask->m_schemas, job.m_solveConstraintBatchTask->m_accumulators, job.m_solveConstraintBatchTask->m_accumulatorInterIndices );
#	endif
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
