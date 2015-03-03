/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>

#include <Physics/ConstraintSolver/Solve/hkpSolve.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuSingleThreadedJobsOnIsland.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpMultithreadedSimulation.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpContactImpulseLimitBreachedListener.h>


static void HK_FORCE_INLINE deleteTaskHeaderResources( hkpBuildJacobianTaskHeader* taskHeader )
{
	#if defined(HK_PLATFORM_HAS_SPU)
	while ( taskHeader->m_tasks.m_ppuOnlyBuildJacobianTasks )
	{
		hkpBuildJacobianTask* task = taskHeader->m_tasks.m_ppuOnlyBuildJacobianTasks;
		taskHeader->m_tasks.m_ppuOnlyBuildJacobianTasks = task->m_next;
		delete task;
	}
	#endif
	while ( taskHeader->m_tasks.m_buildJacobianTasks )
	{
		hkpBuildJacobianTask* task = taskHeader->m_tasks.m_buildJacobianTasks;
		taskHeader->m_tasks.m_buildJacobianTasks = task->m_next;
		delete task;
	}
	while ( taskHeader->m_solveTasks.m_firstSolveJacobiansTask )
	{
		hkpSolveConstraintBatchTask* task = taskHeader->m_solveTasks.m_firstSolveJacobiansTask;
		taskHeader->m_solveTasks.m_firstSolveJacobiansTask = task->m_next;
		delete task;
	}
	if ( taskHeader->m_buffer)
	{
		hkMemSolverBufFree<char>( (char*)taskHeader->m_buffer, taskHeader->m_bufferCapacity);
		taskHeader->m_buffer = HK_NULL;
	}

	if ( taskHeader->m_numIslandsAfterSplit > 1 )
	{
		hkMemHeapBufFree( (void**)taskHeader->m_allEntities, taskHeader->m_entitiesCapacity );
		taskHeader->m_allEntities = HK_NULL;
	}
	delete taskHeader;
}

hkJobQueue::JobStatus HK_CALL hkpSingleThreadedJobsOnIsland::cpuBroadPhaseJob(	hkpMtThreadStructure&		tl,
																				hkJobQueue&					jobQueue,
																				hkJobQueue::JobQueueEntry&	nextJobOut )
{
	const hkpBroadPhaseJob& job = reinterpret_cast<hkpBroadPhaseJob&>(nextJobOut);

	// Just do the broadphase for one island here, for now
	hkpSimulationIsland* island = job.m_island;

	hkCheckDeterminismUtil::checkMt(0xf0000080, island->m_entities.getSize() ? island->m_entities[0]->getUid() : 0);
	hkCheckDeterminismUtil::checkMt(0xf0000081, island->m_entities.getSize());
	hkCheckDeterminismUtil::checkMt(0xf0000082, island->m_entities.getSize());

#ifdef HK_DEBUG_MULTI_THREADING
	HK_ASSERT( 0xf0323454, island->m_inIntegrateJob );
	island->m_inIntegrateJob = false;
#endif
	HK_ASSERT(0xf04321e3, job.m_taskHeader->m_numUnfinishedJobsForBroadphase == 0);

	HK_TIMER_BEGIN("Broadphase", HK_NULL);

	//
	//	Deactivation check
	//
	if ( job.m_taskHeader->m_islandShouldBeDeactivated )
	{
		if ( tl.m_world->m_wantDeactivation && island->m_activeMark && !island->m_isSparse)
		{
			hkpWorldOperationUtil::markIslandInactiveMt( tl.m_world, island );
		}
	}

#if defined(HK_ENABLE_DETERMINISM_CHECKS)
	for (int i =0; i < island->m_entities.getSize(); i++)
	{
		hkCheckDeterminismUtil::checkMt(0xf0000083, island->m_entities[i]->getUid());
		hkCheckDeterminismUtil::checkMt(0xf0000084, island->m_entities[i]->m_storageIndex);
	}
#endif

	hkChar* exportFinished = &job.m_taskHeader->m_exportFinished;
	hkLocalArray<hkpBroadPhaseHandlePair> newPairs( tl.m_world->m_broadPhaseUpdateSize );
	hkLocalArray<hkpBroadPhaseHandlePair> delPairs( tl.m_world->m_broadPhaseUpdateSize );

	hkpBuildJacobianTaskHeader* taskHeader = job.m_taskHeader;
	HK_ON_DEBUG_MULTI_THREADING( { for (int e = 0; e < island->m_entities.getSize(); e++) { island->m_entities[e]->markForRead(); } } );

#if defined(HK_ENABLE_DETERMINISM_CHECKS)
	for (int i =0; i < island->m_entities.getSize(); i++)
	{
		hkCheckDeterminismUtil::checkMt(0xad000004, island->m_entities[i]->getUid());
		hkCheckDeterminismUtil::checkMt(0xad000005, island->m_entities[i]->m_storageIndex);
	}
#endif

	tl.m_simulation->collideEntitiesBroadPhaseContinuousFindPairs( island->m_entities.begin(), island->m_entities.getSize(), tl.m_world, newPairs, delPairs );

	//
	// Fire callbacks.
	//
	hkInt32 splitIslandId;

	const int finalRefCount = 1-taskHeader->m_numIslandsAfterSplit;	// reference count when it is allowed to free the taskheader
	int refCount = 0;	

	if ( taskHeader->m_numIslandsAfterSplit == 1)
	{
		splitIslandId = taskHeader->m_referenceCount--;	// in case of a single island there is no need for an atomic operation
	}
	else
	{
		splitIslandId = hkCriticalSection::atomicExchangeAdd( (hkUint32*)&taskHeader->m_referenceCount, -1);
	}

	// if we are the first island of a group of split island entering here, lets fire the solver callbacks
	if ( splitIslandId == taskHeader->m_numIslandsAfterSplit )
	{
		tl.m_simulation->waitForSolverExport(exportFinished);

		// fire solver callbacks:	export limit breached 
		if ( taskHeader->m_impulseLimitsBreached )
		{
			HK_ON_DEBUG_MULTI_THREADING( { for (int e = 0; e < taskHeader->m_numAllEntities; e++) {
				if ( taskHeader->m_allEntities[e]->getSimulationIsland() != island ) {  taskHeader->m_allEntities[e]->markForRead(); } } } );

			// Calculate number of all breached points.
			int numBreachedPoints = 0;
			for ( hkpImpulseLimitBreachedHeader* h = taskHeader->m_impulseLimitsBreached; h; h = h->m_next )
			{
				numBreachedPoints += h->m_numBreached;
			}

			// Create a big buffer to hold the info for the callback.
			hkLocalBuffer<hkpContactImpulseLimitBreachedListenerInfo> infoBuffer(numBreachedPoints);

			// Copy all data
			int nextIndex = 0;
			for ( hkpImpulseLimitBreachedHeader* h = taskHeader->m_impulseLimitsBreached; h; h = h->m_next)
			{
				hkpContactImpulseLimitBreachedListenerInfo* bi = reinterpret_cast<hkpContactImpulseLimitBreachedListenerInfo*>(&h->getElem(0));

				HK_ASSERT2(0XAD64433A, (sizeof(hkpContactImpulseLimitBreachedListenerInfo) & 0x03) == 0, "Size of hkpContactImpulseLimitBreachedListenerInfo is expected to be a multiple of 4.");
				hkString::memCpy4(infoBuffer.begin()+nextIndex, bi, h->m_numBreached * (sizeof(hkpContactImpulseLimitBreachedListenerInfo) >> 2) );
				nextIndex += h->m_numBreached;
			}
			HK_ASSERT(0xad864a33, numBreachedPoints == nextIndex);

			// Trigger the callback
			hkpWorldCallbackUtil::fireContactImpulseLimitBreached( tl.m_world, infoBuffer.begin(), numBreachedPoints );

			HK_ON_DEBUG_MULTI_THREADING( { for (int e = 0; e < taskHeader->m_numAllEntities; e++) {
				if ( taskHeader->m_allEntities[e]->getSimulationIsland() != island ) {  taskHeader->m_allEntities[e]->unmarkForRead(); } } } );
		}

		*exportFinished = 2;
	}
	else
	{
		//
		//	We cannot continue until we are sure that all solver callbacks have been fired, so lets check for our export finished flag
		//
		if ( *exportFinished != hkChar(2) )
		{
			HK_TIME_CODE_BLOCK("WaitForSolverCallbacks", HK_NULL);
			volatile hkChar* flag = exportFinished;
			while ( *flag != hkChar(2) ) { };
		}

		if ( taskHeader->m_numIslandsAfterSplit <= 2 )
		{	
			refCount--;	// we are the only thread accessing m_referenceCount, no need to do an atomic operation
		}
		else
		{
			refCount = hkCriticalSection::atomicExchangeAdd( (hkUint32*)&taskHeader->m_referenceCount, -1) - 1;
		}
	}

#if !defined(HK_ENABLE_DETERMINISM_CHECKS)
	if ( newPairs.getSize() + delPairs.getSize() > 0)
#endif
	{
		HK_TIMER_BEGIN_LIST( "AddRemoveAgnts", "Init" );
		HK_ON_DEBUG_MULTI_THREADING( island->m_allowIslandLocking = true );
		tl.m_simulation->removeAndAddPairs( tl.m_world, island->m_entities.begin(), delPairs, newPairs);
		HK_ON_DEBUG_MULTI_THREADING( island->m_allowIslandLocking = false );
		HK_TIMER_END_LIST();
	}

	HK_ON_DEBUG_MULTI_THREADING( { for( int e = 0; e < island->m_entities.getSize(); e++ ) { island->m_entities[e]->unmarkForRead(); } } );

	// We may only free the task header once ALL broadphase jobs (i.e. all split islands) have reached this point.
	if ( refCount == finalRefCount )
	{
		deleteTaskHeaderResources( taskHeader );
	}


	//
	//	create midphase/narrowphase agent sector jobs
	//

	HK_ASSERT3(0xafe14230, tl.m_world->m_maxSectorsPerMidphaseCollideTask <= ( hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK / HK_AGENT3_MIDPHASE_AGENTS_PER_NN_SECTOR ), "hkpWorld::m_maxElementsPerAgentNnEntryTask may not exceed " << hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK << ".");
	HK_ASSERT3(0xafe14230, tl.m_world->m_maxSectorsPerNarrowphaseCollideTask <= ( hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK / HK_AGENT3_NARROWPHASE_AGENTS_PER_NN_SECTOR ), "hkpWorld::m_maxElementsPerAgentNnEntryTask may not exceed " << hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK << ".");

#if !defined(HK_PLATFORM_HAS_SPU)
	hkJobQueue::JobQueueEntry jobBuffers[1];
	// We reuse the current jobQueueEntry
	hkJobQueue::JobQueueEntry *const newJobs[2] = { &nextJobOut, &jobBuffers[0] };
#else
	hkJobQueue::JobQueueEntry jobBuffers[3];
	// We reuse the current jobQueueEntry
	hkJobQueue::JobQueueEntry *const newJobs[4] = { &nextJobOut, &jobBuffers[0], &jobBuffers[1], &jobBuffers[2] };
#endif
	int numJobs = 0;
	// The jobs may split, so we calculate the eventual number of task.
	int numTasks = 0;
	int numMidphaseTasks = 0;
	
	hkpAgentNnTrack *const tracks[2] = { &island->m_narrowphaseAgentTrack, &island->m_midphaseAgentTrack };	
	for ( int j = 0; j < 2; ++j )
	{
		hkpAgentNnTrack& track = *tracks[j];
		const int maxSectorsPerAgentSectorTask = j ? tl.m_world->m_maxSectorsPerMidphaseCollideTask : tl.m_world->m_maxSectorsPerNarrowphaseCollideTask;
		hkpAgentNnSector** sectors = track.m_sectors.begin();

#if !defined(HK_PLATFORM_HAS_SPU)
		const int numSectors = track.m_sectors.getSize();
		if (numSectors > 0)
		{
			const int numNewTasks = ( ( numSectors - 1 ) / maxSectorsPerAgentSectorTask ) + 1;

			hkpAgentSectorBaseJob* asJob = new(newJobs[numJobs]) hkpAgentSectorJob(job, tl.m_world->m_dynamicsStepInfo.m_stepInfo, sectors, numSectors, maxSectorsPerAgentSectorTask, track.m_bytesUsedInLastSector, ( j ? HK_AGENT3_MIDPHASE_TRACK : HK_AGENT3_NARROWPHASE_TRACK ), tl.m_world->m_useCompoundSpuElf );
			HK_ON_DETERMINISM_CHECKS_ENABLED(asJob->m_jobSid += 0x1000);  // Need to allow space for lower jobs to be split (and have their sud incremented)
			
			// We leave room for the previous jobs to split into indices below this job's index.
			asJob->m_taskIndex = hkUint16( numTasks );
			++numJobs;
			numTasks += numNewTasks;
		}
#else
		const int numSpuSectors = track.m_spuNumSectors;
		const int numPpuSectors = track.getNumPpuSectors();
		
		if ( numSpuSectors )
		{
			const int numNewTasks = ( ( numSpuSectors - 1 ) / maxSectorsPerAgentSectorTask ) + 1;
			hkpAgentSectorBaseJob* asJob = new(newJobs[numJobs]) hkpAgentSectorJob(job, tl.m_world->m_dynamicsStepInfo.m_stepInfo, sectors, numSpuSectors, maxSectorsPerAgentSectorTask, track.m_spuBytesUsedInLastSector, ( j ? HK_AGENT3_MIDPHASE_TRACK : HK_AGENT3_NARROWPHASE_TRACK ), tl.m_world->m_useCompoundSpuElf );
			HK_ON_DETERMINISM_CHECKS_ENABLED(asJob->m_jobSid += 0x4000);  // Need to allow space for lower jobs to be split (and have their sud incremented)
			
			// We leave room for the previous jobs to split into indices below this job's index.
			asJob->m_taskIndex = hkUint16( numTasks );
			asJob->m_taskHeader = HK_NULL;
			sectors += numSpuSectors;
			++numJobs;
			numTasks += numNewTasks;
		}

		if ( numPpuSectors )
		{
			const int numNewTasks = ( ( numPpuSectors - 1 ) / maxSectorsPerAgentSectorTask ) + 1;
			hkpAgentSectorBaseJob* asJob = new(newJobs[numJobs]) hkpAgentSectorJob(job, tl.m_world->m_dynamicsStepInfo.m_stepInfo, sectors, numPpuSectors, maxSectorsPerAgentSectorTask, track.m_ppuBytesUsedInLastSector, ( j ? HK_AGENT3_MIDPHASE_TRACK : HK_AGENT3_NARROWPHASE_TRACK ), tl.m_world->m_useCompoundSpuElf );
			HK_ON_DETERMINISM_CHECKS_ENABLED(asJob->m_jobSid += 0x7000);  // Need to allow space for lower jobs to be split (and have their sud incremented)
			
			// We leave room for the previous jobs to split into indices below this job's index.
			asJob->m_taskIndex = hkUint16( numTasks );
			// force job onto ppu
			asJob->m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
			++numJobs;
			numTasks += numNewTasks;
		}
#endif
	}
	// Did we add any jobs?
	if ( numJobs )
	{
		hkpShapeKeyTrack* shapeKeyTracks = HK_NULL;
		if ( numMidphaseTasks )
		{
			shapeKeyTracks = hkAllocateChunk<hkpShapeKeyTrack>( numMidphaseTasks, HK_MEMORY_CLASS_COLLIDE );
			for ( int i = 0; i < numMidphaseTasks; ++i )
			{
				new ( &shapeKeyTracks[i] ) hkpShapeKeyTrack;
			}
		}
#if !defined(HK_PLATFORM_HAS_SPU)
		// If the jobs correspond to more than one task, we create a header.
		// A single collide job can use quite some shortcuts as it can assume it's the only job accessing data structures
		
		if ( numTasks > 1 )
#endif
		{
			hkpAgentSectorHeader *const header = hkpAgentSectorHeader::allocate(numTasks, hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK);
			// The header keeps the shapeKeyTracks so we can clean them up in a post-collide job.
			header->m_shapeKeyTracks = shapeKeyTracks;
			header->m_numShapeKeyTracks = numMidphaseTasks;
			for ( int i = 0; i < numJobs; ++i )
			{
				hkpAgentSectorBaseJob *const newJob = reinterpret_cast<hkpAgentSectorBaseJob*>( newJobs[i] );
				newJob->m_header = header;

				// Job 0 is stored in nextJobOut and is added in the finishAddAndGetNextJob call below.
				if ( i )
				{
					jobQueue.addJob( *newJobs[i], hkJobQueue::JOB_LOW_PRIORITY );
				}
			}
		}
#if !defined(HK_PLATFORM_HAS_SPU)
		else
		{
			reinterpret_cast<hkpAgentSectorBaseJob*>( newJobs[0] )->m_shapeKeyTrack = shapeKeyTracks;
		}
#endif
		HK_TIMER_END();
		hkJobQueue::JobStatus status = jobQueue.finishAddAndGetNextJob( HK_JOB_TYPE_DYNAMICS, hkJobQueue::JOB_LOW_PRIORITY, nextJobOut );
		return status;
	}
	else
	{
		HK_TIMER_END();
		hkCheckDeterminismUtil::checkMt(0xf0000089, 0xdedededeul);
		return jobQueue.finishJobAndGetNextJob( (const hkJobQueue::JobQueueEntry*)&job, nextJobOut );
	}
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
