/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */



static void HK_FORCE_INLINE hkMemCpy16Single( void* dst, const void* src)
{
#if defined (HK_PLATFORM_PS3_PPU) || defined (HK_PLATFORM_PS3_SPU)
	const vector signed int* srcQuad = reinterpret_cast<const vector signed int*>(src);
	vector signed int* dstQuad = reinterpret_cast<vector signed int*>(dst);
	dstQuad[0] = srcQuad[0];
#else
	HK_ASSERT2( 0xf021d445, (hkUlong(dst) & HK_NATIVE_ALIGN_CHECK) == 0, "Unaligned address" );
	HK_ASSERT2( 0xf021d446, (hkUlong(src) & HK_NATIVE_ALIGN_CHECK) == 0, "Unaligned address" );
	const hkUint32* src32 = reinterpret_cast<const hkUint32*>(src);
	hkUint32* dst32 = reinterpret_cast<      hkUint32*>(dst);
	{
		dst32[0] = src32[0];
		dst32[1] = src32[1];
		dst32[2] = src32[2];
		dst32[3] = src32[3];
	}
#endif
}

	//
	//	********** Dynamics Job ***************
	//

hkpDynamicsJob::hkpDynamicsJob( hkpDynamicsJob::JobSubType subType, hkUint16 newSize, const hkpDynamicsJob& srcJob, hkJobSpuType jobSpuType )
#if defined (HK_PLATFORM_SPU) 
		{
			hkMemCpy16Single( this, &srcJob );
			hkMemCpy16Single( hkAddByteOffset(this, 16), hkAddByteOffsetConst(&srcJob, 16) );

			m_jobSubType = hkJobSubType(subType);
			m_size = newSize;
			m_jobSpuType = jobSpuType;
			HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid = 0);
		}
#else
	: hkJob( HK_JOB_TYPE_DYNAMICS, subType, (hkUint8)newSize, jobSpuType ), m_islandIndex(srcJob.m_islandIndex), m_island(srcJob.m_island), m_taskHeader(srcJob.m_taskHeader), m_mtThreadStructure( srcJob.m_mtThreadStructure ) { HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid = 0); }
#endif

hkpDynamicsJob::hkpDynamicsJob( hkpDynamicsJob::JobSubType subType, hkUint16 size, NoJob noSrcJob, hkJobSpuType jobSpuType ) 
: hkJob( HK_JOB_TYPE_DYNAMICS, subType, (hkUint8)size, jobSpuType ), m_islandIndex(HK_INVALID_OBJECT_INDEX), m_island(HK_NULL), m_taskHeader(HK_NULL) { HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid = 0); }


hkJobQueue::JobPopFuncResult hkpDynamicsJob::popDynamicsJobTask( hkpDynamicsJob& out )
{
	reinterpret_cast<hkJobQueue::JobQueueEntry&>(out) = reinterpret_cast<hkJobQueue::JobQueueEntry&>(*this);

	return hkJobQueue::POP_QUEUE_ENTRY;
}


//
//	********** Dynamics Job ***************
//
hkpIntegrateJob::hkpIntegrateJob(int numIslands) : hkpDynamicsJob(DYNAMICS_JOB_INTEGRATE, sizeof(hkpIntegrateJob), NO_SRC_JOB, HK_JOB_SPU_TYPE_DISABLED )
{
	m_islandIndex = 0;
	m_numIslands  = numIslands;
}

hkJobQueue::JobPopFuncResult hkpIntegrateJob::popJobTask( hkpIntegrateJob& out )
{
	// if possible split the job into two parts
	if ( m_numIslands > 1 )
	{
		m_islandIndex++;
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_numIslands--;
		out.m_numIslands = 1;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
	return hkJobQueue::POP_QUEUE_ENTRY;
}


hkJobQueue::JobPopFuncResult hkpBuildAccumulatorsJob::popJobTask( hkpBuildAccumulatorsJob& out )
{
	// if possible split the job into two parts
	if ( m_numEntities > hkpBuildAccumulatorsJob::ACCUMULATORS_PER_JOB )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_numEntities    -= hkpBuildAccumulatorsJob::ACCUMULATORS_PER_JOB;
		m_firstEntityIdx += hkpBuildAccumulatorsJob::ACCUMULATORS_PER_JOB;
		out.m_numEntities = hkpBuildAccumulatorsJob::ACCUMULATORS_PER_JOB;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
	return hkJobQueue::POP_QUEUE_ENTRY;
}

hkJobQueue::JobPopFuncResult hkpSplitSimulationIslandJob::popJobTask( hkpSplitSimulationIslandJob& out )
{
	return hkJobQueue::POP_QUEUE_ENTRY;
}

hkpIntegrateMotionJob::hkpIntegrateMotionJob( const hkpDynamicsJob& job, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy )
: hkpSolveExportResultsJob( job, sizeof(hkpIntegrateMotionJob), localTaskHeaderCopy, DYNAMICS_JOB_INTEGRATE_MOTION )
{
	m_firstEntityIdx	= 0;
	m_numEntities		= localTaskHeaderCopy.m_numAllEntities;
	m_buffer			= localTaskHeaderCopy.m_buffer;
	m_applyForcesAndStepMotionOnly		= false;
}


hkJobQueue::JobPopFuncResult hkpIntegrateMotionJob::popJobTask( hkpIntegrateMotionJob& out )
{

	// if possible split the job into two parts
	if ( m_numEntities > hkpIntegrateMotionJob::ACCUMULATORS_PER_JOB )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_numEntities    -= hkpIntegrateMotionJob::ACCUMULATORS_PER_JOB;
		m_firstEntityIdx += hkpIntegrateMotionJob::ACCUMULATORS_PER_JOB;
		out.m_numEntities = hkpIntegrateMotionJob::ACCUMULATORS_PER_JOB;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}

	if ( m_solveConstraintBatchTask )
	{
		// When solving on multiple thereads, the above member variable points to the first solver task ( built in Crate-BuildJacobians-Tasks);
		// and we morph this job into solver export job.
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid = 0);
		m_jobSubType = DYNAMICS_JOB_SOLVE_EXPORT_RESULTS;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}

	return hkJobQueue::POP_QUEUE_ENTRY;
}


hkpBroadPhaseJob::hkpBroadPhaseJob( const hkpDynamicsJob& job, hkpBuildJacobianTaskHeader* newTaskHeader ) : hkpDynamicsJob( DYNAMICS_JOB_BROADPHASE, sizeof(hkpBroadPhaseJob), job, HK_JOB_SPU_TYPE_DISABLED)
{
	m_taskHeader = newTaskHeader;
	m_numIslands  = 1;
}

hkpAgentBaseJob::hkpAgentBaseJob(const hkpDynamicsJob& job, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf) 

:	hkpDynamicsJob( type, size, job ), 
	m_stepInfo(stepInfo)
{
	// Check to make sure the job is properly aligned
	HK_CHECK_ALIGN_NATIVE(&m_stepInfo); 
	
	m_islandIndex			= job.m_islandIndex;
	m_taskIndex				= 0;
	m_elements				= elements;
	m_numElements			= hkUint16(numElements);
	m_numElementsPerTask	= hkUint16(maxNumElementsPerTask);
	m_agentNnTrackType		= nnTrackType;
	m_header				= 0;
}

hkpAgentBaseJob::hkpAgentBaseJob(hkpDynamicsJob::NoJob job, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf) 
:	hkpDynamicsJob( type, size, job ),
	m_stepInfo(stepInfo)
{
	// Check to make sure the job is properly aligned
	HK_CHECK_ALIGN_NATIVE(&m_stepInfo); 

	m_islandIndex			= 0;
	m_taskIndex				= 0;
	m_elements				= elements;
	m_numElements			= hkUint16(numElements);
	m_numElementsPerTask	= hkUint16(maxNumElementsPerTask);
	m_agentNnTrackType		= nnTrackType;
	m_header				= 0;
}

hkpAgentSectorBaseJob::hkpAgentSectorBaseJob(const hkpDynamicsJob& job, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf)
:	hkpAgentBaseJob( job, size, stepInfo, elements, numElements, maxNumElementsPerTask, type, nnTrackType, useStaticCompoundElf )
{
	m_shapeKeyTrack			= HK_NULL;
}


hkpAgentSectorJob::hkpAgentSectorJob(const hkpBroadPhaseJob& job, const hkStepInfo& stepInfo, struct hkpAgentNnSector*const* sectors, int numSectors, int maxNumSectorsPerTask, int bytesInLastSector, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf) 
:	hkpAgentSectorBaseJob( job, sizeof(hkpAgentSectorJob), stepInfo, reinterpret_cast<void*const*>(sectors), numSectors, maxNumSectorsPerTask, DYNAMICS_JOB_AGENT_SECTOR, nnTrackType, useStaticCompoundElf )
{
	m_jobType = useStaticCompoundElf ? HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND : HK_JOB_TYPE_COLLIDE;
	m_bytesUsedInLastSector	= hkUint16(bytesInLastSector);
}

hkpAgentNnEntryBaseJob::hkpAgentNnEntryBaseJob( const hkpDynamicsJob& job, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf )
:	hkpAgentBaseJob( job, size, stepInfo, elements, numElements, maxNumElementsPerTask, type, nnTrackType, useStaticCompoundElf )
{
	m_shapeKeyTrack			= HK_NULL;
	m_jobType = useStaticCompoundElf ? HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND : HK_JOB_TYPE_COLLIDE;
	
}

hkpAgentNnEntryBaseJob::hkpAgentNnEntryBaseJob( hkpDynamicsJob::NoJob noJob, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf )
:	hkpAgentBaseJob( noJob, size, stepInfo, elements, numElements, maxNumElementsPerTask, type, nnTrackType, useStaticCompoundElf )
{
	m_jobType = useStaticCompoundElf ? HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND : HK_JOB_TYPE_COLLIDE;
	
}

hkpAgentNnEntryJob::hkpAgentNnEntryJob(const hkStepInfo& stepInfo, struct hkpAgentNnEntry*const* entries, int numEntries, int maxNumAgentNnEntriesPerTask, hkpContinuousSimulation::CollisionQualityOverride collisionQualityOverride, bool useStaticCompoundElf, hkpAgentNnTrackType nnTrackType )
:	hkpAgentNnEntryBaseJob( NO_SRC_JOB, sizeof(hkpAgentNnEntryJob), stepInfo, reinterpret_cast<void*const*>(entries), numEntries, maxNumAgentNnEntriesPerTask, DYNAMICS_JOB_AGENT_NN_ENTRY, nnTrackType, useStaticCompoundElf )
{
	m_jobType = useStaticCompoundElf ? HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND : HK_JOB_TYPE_COLLIDE;
	
	m_collisionQualityOverride = collisionQualityOverride;
}

inline hkJobQueue::JobPopFuncResult hkpAgentSectorJob::popJobTask( hkpAgentSectorJob& out )
{
	// if possible split the job into two parts
	if ( m_numElements > m_numElementsPerTask )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_numElements  = m_numElements - m_numElementsPerTask;
		m_taskIndex   += 1;
		m_elements += m_numElementsPerTask;
		out.m_numElements = m_numElementsPerTask;
		out.m_bytesUsedInLastSector = HK_AGENT3_SECTOR_SIZE;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
	return hkJobQueue::POP_QUEUE_ENTRY;
}

hkJobQueue::JobPopFuncResult hkpAgentNnEntryJob::popJobTask( hkpAgentNnEntryJob& out )
{
	// if possible split the job into two parts
	if ( m_numElements > m_numElementsPerTask )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_numElements  = m_numElements - m_numElementsPerTask;
		m_taskIndex += 1;
		m_elements += m_numElementsPerTask;
		out.m_numElements = m_numElementsPerTask;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
	return hkJobQueue::POP_QUEUE_ENTRY;
}

hkpBuildJacobiansJob::hkpBuildJacobiansJob(const hkpDynamicsJob& job, struct hkpBuildJacobianTask* firstTaskInMainMemory, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy)
			: hkpDynamicsJob( DYNAMICS_JOB_BUILD_JACOBIANS, sizeof(hkpBuildJacobiansJob), job) 
{
	m_constraintQueryIn				= localTaskHeaderCopy.m_constraintQueryIn; // in main memory
	m_finishSchemasWithGoto         = localTaskHeaderCopy.m_solveInSingleThread;
	m_buildJacobianTaskInMainMemory = firstTaskInMainMemory;// in main memory
}


hkpSolveApplyGravityJob::hkpSolveApplyGravityJob(const hkpDynamicsJob& job, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy )
: hkpDynamicsJob( DYNAMICS_JOB_SOLVE_APPLY_GRAVITY, sizeof(hkpSolveApplyGravityJob), job ) 
{
#if defined(HK_PLATFORM_SPU)
	{
		HK_ASSERT2(0XAD8754AA, ((hkUlong(&localTaskHeaderCopy.m_accumulatorsBase) & 0xf) == 0) && ((hkUlong(&m_accumulators) & 0xf) == 0), "Members not aligned.");
		hkMemCpy16Single( &m_accumulators, &localTaskHeaderCopy.m_accumulatorsBase);
	}
#else
	m_accumulators        = localTaskHeaderCopy.m_accumulatorsBase;
	m_accumulatorsEnd     = localTaskHeaderCopy.m_accumulatorsEnd;
#endif
	HK_ASSERT2(0xad7866dd, m_accumulatorsEnd == hkAddByteOffset(m_accumulators, (1+localTaskHeaderCopy.m_numAllEntities) * sizeof(hkpVelocityAccumulator)), "Error in estimated accumulators end");
}


hkJobQueue::JobPopFuncResult hkpSolveApplyGravityJob::popJobTask( hkpSolveApplyGravityJob& out )
{

	if (m_accumulators + MAX_NUM_ACCUMULATORS_FOR_APPLY_GRAVITY_JOB < m_accumulatorsEnd)
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_accumulators = m_accumulators + MAX_NUM_ACCUMULATORS_FOR_APPLY_GRAVITY_JOB;
		out.m_accumulatorsEnd = m_accumulators;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}

	return hkJobQueue::POP_QUEUE_ENTRY;
}

hkpSolveConstraintBatchJob::hkpSolveConstraintBatchJob(const hkpSolveApplyGravityJob& sagj, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy )
: hkpDynamicsJob( DYNAMICS_JOB_SOLVE_CONSTRAINT_BATCH, sizeof(hkpSolveConstraintBatchJob), sagj ) 
{
	m_solveConstraintBatchTask = localTaskHeaderCopy.m_solveTasks.m_firstSolveJacobiansTask;

	m_currentSolverStep = 0; 
	m_currentSolverMicroStep = 0;
}


hkpSolveConstraintBatchJob::hkpSolveConstraintBatchJob(const hkpSolveConstraintBatchJob& scbij)
: hkpDynamicsJob( DYNAMICS_JOB_SOLVE_CONSTRAINT_BATCH, sizeof(hkpSolveConstraintBatchJob), scbij ) 
{
#if defined(HK_PLATFORM_SPU)
	hkMemCpy16Single( hkAddByteOffset(this, 32), hkAddByteOffsetConst(&scbij,32));
#else
	*this = scbij; // this is actually ~0x100 bytes of instruction code on spu
#endif
		
	m_solveConstraintBatchTask = scbij.m_solveConstraintBatchTask->m_firstTaskInNextBatch;
}

hkpSolveConstraintBatchJob::hkpSolveConstraintBatchJob(const hkpSolveIntegrateVelocitiesJob& sivbij, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy)
: hkpDynamicsJob( DYNAMICS_JOB_SOLVE_CONSTRAINT_BATCH, sizeof(hkpSolveConstraintBatchJob), sivbij ) 
{
	m_solveConstraintBatchTask = localTaskHeaderCopy.m_solveTasks.m_firstSolveJacobiansTask;

	m_currentSolverStep = sivbij.m_currentSolverStep + 1;
	m_currentSolverMicroStep = 0;
}

hkpSolveIntegrateVelocitiesJob::hkpSolveIntegrateVelocitiesJob(const hkpSolveConstraintBatchJob& scbij, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy )
: hkpDynamicsJob( DYNAMICS_JOB_SOLVE_INTEGRATE_VELOCITIES, sizeof(hkpSolveIntegrateVelocitiesJob), scbij ) 
{
	HK_ASSERT2(0xad785544, scbij.m_solveConstraintBatchTask->m_accumulators == localTaskHeaderCopy.m_accumulatorsBase, "Internal error.");

#if defined(HK_PLATFORM_SPU)
	{
		HK_ASSERT2(0XAD8754AA, ((hkUlong(&localTaskHeaderCopy.m_accumulatorsBase) & 0xf) == 0) && ((hkUlong(&m_accumulators) & 0xf) == 0), "Members not aligned.");
		hkMemCpy16Single( &m_accumulators, &localTaskHeaderCopy.m_accumulatorsBase );
	}
#else
	m_accumulators        = localTaskHeaderCopy.m_accumulatorsBase;
	m_accumulatorsEnd     = localTaskHeaderCopy.m_accumulatorsEnd;
#endif
	HK_ASSERT2(0xad7866de, m_accumulatorsEnd == hkAddByteOffset(m_accumulators, (1+localTaskHeaderCopy.m_numAllEntities) * sizeof(hkpVelocityAccumulator)), "Error in estimated accumulators end");

	m_currentSolverStep = scbij.m_currentSolverStep;
	m_solvingFinished = false;
}

hkpSolveExportResultsJob::hkpSolveExportResultsJob(const hkpDynamicsJob& sivbij, hkUint16 size, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy, JobSubType type )
: hkpSplitSimulationIslandJob( sivbij, size, type ) 
{
	m_solveConstraintBatchTask = localTaskHeaderCopy.m_solveTasks.m_firstSolveJacobiansTask; // zero in 
}


hkpSolveConstraintsJob::hkpSolveConstraintsJob(const hkpDynamicsJob& job, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy )
		: hkpDynamicsJob( DYNAMICS_JOB_SOLVE_CONSTRAINTS, sizeof(hkpSolveConstraintsJob), job ) 
{
	m_buffer              = localTaskHeaderCopy.m_buffer;

	// offsets should be calculated earlier.  before buildJacobians
	m_accumulatorsOffset  = hkGetByteOffsetCpuPtr(m_buffer, localTaskHeaderCopy.m_accumulatorsBase);
	m_schemasOffset       = hkGetByteOffsetCpuPtr(m_buffer, localTaskHeaderCopy.m_schemasBase);
	m_solverTempOffset    = hkGetByteOffsetCpuPtr(m_buffer, localTaskHeaderCopy.m_solverTempBase);
	m_bufferSize          = localTaskHeaderCopy.m_bufferSize;

	m_numSolverResults    = localTaskHeaderCopy.m_numSolverResults;
	m_numSolverElemTemps  = localTaskHeaderCopy.m_numSolverElemTemps;
}

hkJobQueue::JobPopFuncResult hkpBuildJacobiansJob::popJobTask( hkpBuildJacobiansJob& out )
{

#if defined (HK_PLATFORM_SPU)
	// dma in the _OLD_ hkpBuildJacobianTask into local memory & assign its reference to job
	// Note: this task memory is deallocated in the finishDynamicsJob()
	struct hkpBuildJacobianTask* task = hkAllocateStack<struct hkpBuildJacobianTask>(1, "hkpBuildJacobianTask");
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( task, m_buildJacobianTaskInMainMemory, sizeof(hkpBuildJacobianTask), hkSpuDmaManager::READ_ONLY);
	out.m_buildJacobianTask = task;

	HK_ASSERT(0xaf34ef22, task->m_numAtomInfos > 0);

	// if possible split the job into two parts
	if ( task->m_next )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		// the job left on the queue now points to the next task on the list
		m_buildJacobianTaskInMainMemory = task->m_next;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
#else
	// if possible split the job into two parts
	if ( m_buildJacobianTask->m_next  )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_buildJacobianTask = m_buildJacobianTask->m_next;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
#endif

	return hkJobQueue::POP_QUEUE_ENTRY;
}

hkJobQueue::JobPopFuncResult hkpSolveConstraintBatchJob::popJobTask( hkpSolveConstraintBatchJob& out )
{

#if defined (HK_PLATFORM_SPU)
	// dma in the _OLD_ hkpBuildJacobianTask into local memory & assign its reference to job
	// Note: this task memory is deallocated in the finishDynamicsJob()
	struct hkpSolveConstraintBatchTask* task = hkAllocateStack<struct hkpSolveConstraintBatchTask>(1, "hkpSolveConstraintBatchTask");
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( task, m_solveConstraintBatchTaskInMainMemory, sizeof(hkpSolveConstraintBatchTask), hkSpuDmaManager::READ_ONLY);
	out.m_solveConstraintBatchTask = task;

//	HK_ASSERT(0xaf34ef22, task->m_numAtomInfos > 0);
	// check for solver buffer pointers ??

	// if possible split the job into two parts
	if ( !task->m_isLastTaskInBatch )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		// the job left on the queue now points to the next task on the list
		m_solveConstraintBatchTaskInMainMemory = task->m_next;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
#else
	// this code must exist on all platforms which can execute this task without dma
	// if possible split the job into two parts
	if ( !m_solveConstraintBatchTask->m_isLastTaskInBatch  )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_solveConstraintBatchTask = m_solveConstraintBatchTask->m_next;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
#endif
	return hkJobQueue::POP_QUEUE_ENTRY;
}

hkJobQueue::JobPopFuncResult hkpSolveIntegrateVelocitiesJob::popJobTask( hkpSolveIntegrateVelocitiesJob& out )
{
	if (m_accumulators + MAX_NUM_ACCUMULATORS_FOR_INTEGRATE_VELOCITIES_JOB < m_accumulatorsEnd)
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_accumulators = m_accumulators + MAX_NUM_ACCUMULATORS_FOR_INTEGRATE_VELOCITIES_JOB;
		out.m_accumulatorsEnd = m_accumulators;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}

	return hkJobQueue::POP_QUEUE_ENTRY;
}


hkJobQueue::JobPopFuncResult hkpSolveExportResultsJob::popJobTask( hkpSolveExportResultsJob& out )
{

#if defined (HK_PLATFORM_SPU)
	// dma in the _OLD_ hkpBuildJacobianTask into local memory & assign its reference to job
	// Note: this task memory is deallocated in the finishDynamicsJob()
	struct hkpSolveConstraintBatchTask* task = hkAllocateStack<struct hkpSolveConstraintBatchTask>(1, "hkpSolveConstraintBatchTask");
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( task, m_solveConstraintBatchTaskInMainMemory, sizeof(hkpSolveConstraintBatchTask), hkSpuDmaManager::READ_ONLY);
	out.m_solveConstraintBatchTask = task;

//	HK_ASSERT(0xaf34ef22, task->m_numAtomInfos > 0);

	// if possible split the job into two parts
	if ( task->m_next )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		// the job left on the queue now points to the next task on the list
		m_solveConstraintBatchTaskInMainMemory = task->m_next;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
#else
	// this code must exist on all platforms which can execute this task without dma
	// if possible split the job into two parts
	if ( m_solveConstraintBatchTask->m_next  )
	{
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED(m_jobSid++);
		m_solveConstraintBatchTask = m_solveConstraintBatchTask->m_next;
		return hkJobQueue::DO_NOT_POP_QUEUE_ENTRY;
	}
#endif
	return hkJobQueue::POP_QUEUE_ENTRY;
}


hkpPostCollideJob::hkpPostCollideJob( const hkpAgentBaseJob& job ): hkpDynamicsJob( DYNAMICS_JOB_POST_COLLIDE,  sizeof(hkpPostCollideJob), job, HK_JOB_SPU_TYPE_DISABLED )
{
	m_islandIndex  = job.m_islandIndex;
	m_header       = job.m_header; 
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
