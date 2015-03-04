/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

#include <Common/Base/Config/hkConfigMonitors.h>

#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/Spu/Dma/Utils/hkSpuDmaUtils.h>

#include <Physics2012/Dynamics/Constraint/Setup/hkpConstraintSolverSetup.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobQueueUtils.h>
#if defined(HK_PLATFORM_SPU)
#   include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#	include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>
#endif

#include <Physics2012/Dynamics/World/CommandQueue/hkpPhysicsCommandQueue.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#include <Common/Base/Config/hkConfigSolverLog.h>

#if defined(HK_PLATFORM_SPU)
	HK_COMPILE_TIME_ASSERT( HK_POINTER_SIZE == 4 );
	HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveIntegrateVelocitiesJob) == 48 );
	HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpSolveIntegrateVelocitiesJob, m_accumulators) == 32 );
	HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpSolveIntegrateVelocitiesJob, m_accumulatorsEnd) == 32 + 4 );
#endif 

#ifdef USE_SOLVER_DEBUG_OSTREAM
#	if defined(HK_PLATFORM_SPU)
		hkSpuSimulator::Client* debugOstream = HK_NULL;
#	else
		hkArray<char>* debugBuffer = HK_NULL;
		hkOstream* debugOstream = HK_NULL;
		typedef hkOstream hkOfstream;
		hkOfstream* debugOutputFile = HK_NULL;
#	endif
#endif


//Debug Printf's of jobs popped/finished
#if 0 && defined(HK_PLATFORM_SPU) && defined(HK_PLATFORM_WIN32)
#	include <iostream>

#	define HK_SPU_DEBUG1(text) printToScreen(text)
#	define HK_SPU_DEBUG(text) printToScreen(text, job.m_islandIndex)

		extern int g_hkSpuId;

		static void printToScreen(const char* text)
		{
			const char* indent = "            ";
			const char* endline = "\n";

			for (int i = -1; i < g_hkSpuId; i++) 
			{
				hkSpuSimulator::Client::getInstance()->printfToScreen(indent);
			} 

			hkSpuSimulator::Client::getInstance()->printfToScreen(text);

			hkSpuSimulator::Client::getInstance()->printfToScreen(endline);
		}

		static void printToScreen(const char* text, int islandNo)
		{
			const char* indent = "            ";
			const char* endline = "\n";

			for (int i = -1; i < g_hkSpuId; i++) 
			{
				hkSpuSimulator::Client::getInstance()->printfToScreen(indent);
			} 

			hkSpuSimulator::Client::getInstance()->printfToScreen(text);

			const char* numbers[] = {"0", "1", "2", "3", "4" };
			hkSpuSimulator::Client::getInstance()->printfToScreen(":");
			hkSpuSimulator::Client::getInstance()->printfToScreen(numbers[islandNo]);

			hkSpuSimulator::Client::getInstance()->printfToScreen(endline);
		}
#else
//#	if defined(HK_PLATFORM_WIN32)
//#		include <iostream>
//#		define HK_SPU_DEBUG1(text) printToScreen(text)
//#		define HK_SPU_DEBUG(text) printToScreen(text, job.m_islandIndex)
//
//		static void printToScreen(const char* text)
//		{
//			const char* endline = "\n";
//			std::cout << text;
//			std::cout << endline;
//		}
//		static void printToScreen(const char* text, int islandNo)
//		{
//			const char* endline = "\n";
//			std::cout << text << ":" << islandNo;
//			std::cout << endline;
//		}
//#	else
#		define HK_SPU_DEBUG(text) 
#		define HK_SPU_DEBUG1(text) 
//#	endif
#endif


#if !defined (HK_PLATFORM_SPU)
hkpAgentSectorHeader* hkpAgentSectorHeader::allocate(int numTasks, int numAgentNnEntriesPerTask)
{
	int size = hkpAgentSectorHeader::getAllocatedSize( numTasks );
	hkpAgentSectorHeader* sh = static_cast<hkpAgentSectorHeader*>(hkAllocateChunk<void>(size, HK_MEMORY_CLASS_DYNAMICS));

	// We will allocate JobInfo in a way that the command queue memory is actually appended to the JobInfo struct itself.
	int sizeOfCommandQueueInBytes  = hkpPhysicsCommandQueue::BYTES_PER_COMMAND * numAgentNnEntriesPerTask;
	int sizeOfJobInfoInBytes       = sizeof(JobInfo) + sizeOfCommandQueueInBytes;
	HK_ASSERT(0xaf241e23, (sizeOfJobInfoInBytes & 0xf) == 0);

	sh->m_openJobs      = numTasks;
	sh->m_numTotalTasks = numTasks;
	sh->m_sizeOfJobInfo = sizeOfJobInfoInBytes;

	JobInfo** jobInfos = reinterpret_cast<JobInfo**>(sh+1);


//	if ( numTasks < 4 )
	{
		for ( int i =0; i < numTasks; i++)
		{
			jobInfos[i] = reinterpret_cast<JobInfo*>( hkAllocateChunk<hkUint8>(sizeOfJobInfoInBytes, HK_MEMORY_CLASS_DYNAMICS) );
			new (jobInfos[i]) JobInfo(sizeOfCommandQueueInBytes);
		}
	}
// 	else
// 	{
// 		hkMemoryRouter::getInstance().allocateChunkBatch( (void**)&jobInfos[0], numTasks, sizeOfJobInfoInBytes, HK_MEMORY_CLASS_DYNAMICS);
// 		for ( int i = 0; i < numTasks; i++)
// 		{
// 			new (jobInfos[i]) JobInfo(sizeOfCommandQueueInBytes);
// 		}
// 
// 	}

	return sh;
}

void hkpAgentSectorHeader::deallocate()
{
	JobInfo** jobInfos = reinterpret_cast<JobInfo**>(this+1);

//	if ( m_numTotalTasks < 4 )
	{
		for ( int i =0; i < m_numTotalTasks; i++)
		{
			hkDeallocateChunk<hkUint8>( reinterpret_cast<hkUint8*>( jobInfos[i] ), m_sizeOfJobInfo, HK_MEMORY_CLASS_DYNAMICS );
		}
	}
// 	else
// 	{
// 		hkMemoryRouter::getInstance().deallocateChunkBatch( (void**)&jobInfos[0], m_numTotalTasks, m_sizeOfJobInfo, HK_MEMORY_CLASS_DYNAMICS);
// 	}

	int size = hkpAgentSectorHeader::getAllocatedSize( m_numTotalTasks );
	hkDeallocateChunk<void>( this, size, HK_MEMORY_CLASS_DYNAMICS );
}
#endif



////////////////////////////////////////////////////////////////////////
//
// Callbacks from job queue
//
////////////////////////////////////////////////////////////////////////


hkJobQueue::JobPopFuncResult HK_CALL hkpJobQueueUtils::popIntegrateJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntry& jobOut )
{
	// onQueue is the job that stays on the queue. It is modified so that it points to the next task after the one being extracted.
	// out     is the job that points to the task being released. This is the image/clone to be used locally by the thread.

	hkpDynamicsJob& job = reinterpret_cast<hkpDynamicsJob&>(jobIn);
	hkString::memCpy16NonEmpty(&jobOut, &jobIn, sizeof(hkJobQueue::JobQueueEntry)>>4);

#if !defined (HK_PLATFORM_SPU)
	hkpWorld* world = job.m_mtThreadStructure->m_world;
	{
		hkpDynamicsJob& dynamicsJobOut = reinterpret_cast<hkpDynamicsJob&>(jobOut);

		HK_ASSERT2(0xad903191, dynamicsJobOut.m_islandIndex != HK_INVALID_OBJECT_INDEX, "This function cannot process the fixed island!");
		hkArray<hkpSimulationIsland*>& islands = world->m_activeSimulationIslands;
		dynamicsJobOut.m_island = islands[dynamicsJobOut.m_islandIndex];
	}
#endif

	switch ( job.m_jobSubType )
	{ 
		case hkpDynamicsJob::DYNAMICS_JOB_INTEGRATE:
			{
				HK_SPU_DEBUG1("integrate.pop");
				hkpIntegrateJob& onQueue = static_cast<hkpIntegrateJob&>(job);
				hkpIntegrateJob& out = reinterpret_cast<hkpIntegrateJob&>(jobOut);
				return onQueue.popJobTask( out );
			}

		case hkpDynamicsJob::DYNAMICS_JOB_BUILD_ACCUMULATORS:
			{
				HK_SPU_DEBUG1("buildAcc.pop");
				hkpBuildAccumulatorsJob& onQueue = static_cast<hkpBuildAccumulatorsJob&>(job);
				hkpBuildAccumulatorsJob& out     = reinterpret_cast<hkpBuildAccumulatorsJob&>(jobOut);
				return onQueue.popJobTask( out );
			}

		case hkpDynamicsJob::DYNAMICS_JOB_BUILD_JACOBIANS:
			{
				HK_SPU_DEBUG1("buildJac.pop");
				hkpBuildJacobiansJob& onQueue = reinterpret_cast<hkpBuildJacobiansJob&>(job);
				hkpBuildJacobiansJob& out = reinterpret_cast<hkpBuildJacobiansJob&>(jobOut);
				return onQueue.popJobTask( out );
			}

#if !defined (HK_PLATFORM_SPU)
		case hkpDynamicsJob::DYNAMICS_JOB_BROADPHASE:
			{
				hkpBroadPhaseJob& onQueue = reinterpret_cast<hkpBroadPhaseJob&>(job);
				hkpBroadPhaseJob& out = reinterpret_cast<hkpBroadPhaseJob&>(jobOut);
				hkJobQueue::JobPopFuncResult result = onQueue.popJobTask( world->m_activeSimulationIslands, out );
#if defined (HK_PLATFORM_HAS_SPU)
				int numTotalIslands = out.m_island->getWorld()->m_activeSimulationIslands.getSize();
				if ( result == hkJobQueue::DO_NOT_POP_QUEUE_ENTRY && data->m_jobQueue[HK_JOB_TYPE_DYNAMICS].getCapacity() < numTotalIslands )
				{
					data->m_jobQueue[HK_JOB_TYPE_DYNAMICS].setCapacity(numTotalIslands);
					data->m_jobQueue[HK_JOB_TYPE_COLLIDE].setCapacity(numTotalIslands);
					data->m_jobQueue[HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND].setCapacity(numTotalIslands);
				}
#endif
				return result;
			}
#endif

		case hkpDynamicsJob::DYNAMICS_JOB_SPLIT_ISLAND:
			{
				HK_SPU_DEBUG1("split.pop");
				hkpSplitSimulationIslandJob& onQueue = reinterpret_cast<hkpSplitSimulationIslandJob&>(job);
				hkpSplitSimulationIslandJob& out = reinterpret_cast<hkpSplitSimulationIslandJob&>(jobOut);
				return onQueue.popJobTask( out );
			}

		case hkpDynamicsJob::DYNAMICS_JOB_INTEGRATE_MOTION:
			{
				HK_SPU_DEBUG1("intMotion.pop");
				hkpIntegrateMotionJob& onQueue = static_cast<hkpIntegrateMotionJob&>(job);
				hkpIntegrateMotionJob& out     = reinterpret_cast<hkpIntegrateMotionJob&>(jobOut);
				return onQueue.popJobTask( out );
			}

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_APPLY_GRAVITY:
			{
				HK_SPU_DEBUG1("appGrav.pop");
				hkpSolveApplyGravityJob& onQueue = static_cast<hkpSolveApplyGravityJob&>(job);
				hkpSolveApplyGravityJob& out = reinterpret_cast<hkpSolveApplyGravityJob&>(jobOut);
				return onQueue.popJobTask( out );
			}
		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_CONSTRAINT_BATCH:
			{
				HK_SPU_DEBUG1("solveBatch.pop");
				hkpSolveConstraintBatchJob& onQueue = static_cast<hkpSolveConstraintBatchJob&>(job);
				hkpSolveConstraintBatchJob& out     = reinterpret_cast<hkpSolveConstraintBatchJob&>(jobOut);
				return onQueue.popJobTask( out );
			}
		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_INTEGRATE_VELOCITIES:
			{
				HK_SPU_DEBUG1("intVel.pop");
				hkpSolveIntegrateVelocitiesJob& onQueue = static_cast<hkpSolveIntegrateVelocitiesJob&>(job);
				hkpSolveIntegrateVelocitiesJob& out     = reinterpret_cast<hkpSolveIntegrateVelocitiesJob&>(jobOut);
				return onQueue.popJobTask( out );
			}
		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_EXPORT_RESULTS:
			{
				HK_SPU_DEBUG1("export.pop");
				hkpSolveExportResultsJob& onQueue = static_cast<hkpSolveExportResultsJob&>(job);
				hkpSolveExportResultsJob& out     = reinterpret_cast<hkpSolveExportResultsJob&>(jobOut);
				return onQueue.popJobTask( out );

			}

		case hkpDynamicsJob::DYNAMICS_JOB_FIRE_JACOBIAN_SETUP_CALLBACK:
			HK_SPU_DEBUG1("bjCallack.pop");
			goto commonFireJacSetupCallbackAndSolveConstraintStAndPostCollide;

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_CONSTRAINTS:			
			HK_SPU_DEBUG1("solveST.pop");
			goto commonFireJacSetupCallbackAndSolveConstraintStAndPostCollide;

		case hkpDynamicsJob::DYNAMICS_JOB_POST_COLLIDE:
			{
				HK_SPU_DEBUG1("postCollide.pop");
commonFireJacSetupCallbackAndSolveConstraintStAndPostCollide:
				hkpDynamicsJob& onQueue = static_cast<hkpDynamicsJob&>(job);
				hkpDynamicsJob& out     = reinterpret_cast<hkpDynamicsJob&>(jobOut);
				return onQueue.popDynamicsJobTask( out );
			}

		default:
			{
				HK_ASSERT2(0xad789ddd, false, "Can't pop this job!!!");
				break;
			}

	}
	return hkJobQueue::POP_QUEUE_ENTRY;
}

hkJobQueue::JobPopFuncResult HK_CALL hkpJobQueueUtils::popCollideJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntry& jobOut )
{
	// onQueue is the job that stays on the queue. It is modified so that it points to the next task after the one being extracted.
	// out     is the job that points to the task being released. This is the image/clone to be used locally by the thread.

	hkpDynamicsJob& job = reinterpret_cast<hkpDynamicsJob&>(jobIn);
	hkString::memCpy16NonEmpty(&jobOut, &jobIn, sizeof(hkJobQueue::JobQueueEntry)>>4);

#if !defined (HK_PLATFORM_SPU)
	hkpWorld* world = job.m_mtThreadStructure->m_world;
	{
		hkpDynamicsJob& dynamicsJobOut = reinterpret_cast<hkpDynamicsJob&>(jobOut);

		HK_ASSERT2(0xad903191, dynamicsJobOut.m_islandIndex != HK_INVALID_OBJECT_INDEX, "This function cannot process the fixed island!");
		hkArray<hkpSimulationIsland*>& islands = world->m_activeSimulationIslands;
		dynamicsJobOut.m_island = islands[dynamicsJobOut.m_islandIndex];
	}
#endif

	switch ( job.m_jobSubType )
	{ 
		case hkpDynamicsJob::DYNAMICS_JOB_POST_COLLIDE:
			{
				HK_SPU_DEBUG1("postCollide.pop");
				hkpDynamicsJob& onQueue = static_cast<hkpDynamicsJob&>(job);
				hkpDynamicsJob& out     = reinterpret_cast<hkpDynamicsJob&>(jobOut);
				return onQueue.popDynamicsJobTask( out );
			}

		case hkpDynamicsJob::DYNAMICS_JOB_AGENT_SECTOR:
			{
				HK_SPU_DEBUG1("agentSec.pop");
				hkpAgentSectorJob& onQueue = static_cast<hkpAgentSectorJob&>(job);
				hkpAgentSectorJob& out = reinterpret_cast<hkpAgentSectorJob&>(jobOut);
				return onQueue.popJobTask( out );
			}

		case hkpDynamicsJob::DYNAMICS_JOB_AGENT_NN_ENTRY:
			{
				HK_SPU_DEBUG1("agentTrackCollection.pop");
				hkpAgentNnEntryJob& onQueue = static_cast<hkpAgentNnEntryJob&>(job);
				hkpAgentNnEntryJob& out = reinterpret_cast<hkpAgentNnEntryJob&>(jobOut);
				return onQueue.popJobTask( out );			
			}

		default:
			{
				HK_ASSERT2(0xad789ddd, false, "Can't pop this job!!!");
				break;
			}

	}
	return hkJobQueue::POP_QUEUE_ENTRY;
}

#if defined (HK_PLATFORM_SPU)
static HK_LOCAL_INLINE const hkpBuildJacobianTaskHeader* downloadTaskHeader(const hkpBuildJacobianTaskHeader* taskHeaderInMainMemory, hkpBuildJacobianTaskHeader* buffer)
{
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( buffer, taskHeaderInMainMemory, sizeof(hkpBuildJacobianTaskHeader), hkSpuDmaManager::READ_COPY );
	HK_SPU_DMA_PERFORM_FINAL_CHECKS( taskHeaderInMainMemory, buffer, sizeof(hkpBuildJacobianTaskHeader) );
	return buffer;
}
#endif

static HK_LOCAL_INLINE void uploadOpenJobsVariable( const hkpBuildJacobianTaskHeader* localTaskHeader, const hkpBuildJacobianTaskHeader* taskHeaderInMainMemory )
{
#if defined (HK_PLATFORM_SPU)
	HK_CPU_PTR(hkpBuildJacobianTaskHeader*) dest = const_cast<HK_CPU_PTR(hkpBuildJacobianTaskHeader*)>(taskHeaderInMainMemory);
	hkSpuDmaManager::putToMainMemorySmallAndWaitForCompletion( &dest->m_openJobs, &localTaskHeader->m_openJobs, sizeof(int), hkSpuDmaManager::WRITE_NEW );
	HK_SPU_DMA_PERFORM_FINAL_CHECKS( &dest->m_openJobs, &localTaskHeader->m_openJobs, sizeof(int) );
#endif
}

static HK_LOCAL_INLINE void spawnSplitSimulationIslandJob( const hkpDynamicsJob* dynamicsJob, const hkpBuildJacobianTaskHeader& localTaskHeader, hkJobQueue& queue, hkJobQueue::DynamicData* data )
{
	if ( localTaskHeader.m_splitCheckRequested )
	{ 
		hkpSplitSimulationIslandJob splitJob( *dynamicsJob, sizeof(hkpSplitSimulationIslandJob) );
		queue.addJobQueueLocked( data, (const hkJobQueue::JobQueueEntry&)splitJob, hkJobQueue::JOB_HIGH_PRIORITY);
	}
}

static /*HK_LOCAL_INLINE*/ void spawnBuildJacobiansJobs(const hkpDynamicsJob& dynamicsJob, const hkpBuildJacobianTaskHeader& taskHeader, hkJobQueue& queue, hkJobQueue::DynamicData* data, hkJobQueue::JobQueueEntryInput& newJobCreated)
{
#	if defined (HK_PLATFORM_HAS_SPU)
	{
		HK_ASSERT2(0XAD5643DA, taskHeader.m_tasks.m_numBuildJacobianTasks + taskHeader.m_tasks.m_numPpuOnlyBuildJacobianTasks, "Internal error");

		taskHeader.m_openJobs = taskHeader.m_tasks.m_numPpuOnlyBuildJacobianTasks + taskHeader.m_tasks.m_numBuildJacobianTasks;

		hkpBuildJacobianTask* firstTaskForNewJobCreated = taskHeader.m_tasks.m_buildJacobianTasks;
		hkJobSpuType typeForNewJobCreated = HK_JOB_SPU_TYPE_ENABLED;

		if (taskHeader.m_tasks.m_numPpuOnlyBuildJacobianTasks)
		{
			// ppu-only constraints
			firstTaskForNewJobCreated = taskHeader.m_tasks.m_ppuOnlyBuildJacobianTasks;
			typeForNewJobCreated = HK_JOB_SPU_TYPE_DISABLED;

			//
			// there's still some setup left that can be done on spu -> convert job into a high priority build jacobians job
			//
			if ( taskHeader.m_tasks.m_numBuildJacobianTasks > 0 ) 
			{
				// non-ppu-only constraints
				hkpBuildJacobiansJob buildNonPpuJacobiansJob( dynamicsJob, taskHeader.m_tasks.m_buildJacobianTasks, taskHeader );
				queue.addJobQueueLocked(data, (const hkJobQueue::JobQueueEntry&)buildNonPpuJacobiansJob, hkJobQueue::JOB_HIGH_PRIORITY );
			}
		}

		new (&newJobCreated.m_job) hkpBuildJacobiansJob( dynamicsJob, firstTaskForNewJobCreated, taskHeader );
		newJobCreated.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;
		newJobCreated.m_job.m_jobSpuType = typeForNewJobCreated;
	}
#	else
	{
		HK_ASSERT2(0XAD5643DA, taskHeader.m_tasks.m_numBuildJacobianTasks, "Internal error");
		new (&newJobCreated.m_job) hkpBuildJacobiansJob( dynamicsJob, taskHeader.m_tasks.m_buildJacobianTasks, taskHeader );
		taskHeader.m_openJobs = taskHeader.m_tasks.m_numBuildJacobianTasks;
		newJobCreated.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;
		newJobCreated.m_job.m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
	}
#	endif
}

hkJobQueue::JobCreationStatus HK_CALL hkpJobQueueUtils::finishIntegrateJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, const hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntryInput& newJobCreated )
{
	int numJobsToFinish = 1;
	flushSolverDebugOstream();

	const hkpDynamicsJob& job = reinterpret_cast<const hkpDynamicsJob&>(jobIn);
	
#if defined (HK_PLATFORM_SPU)
	HK_ALIGN16( char taskHeaderBufferOnSpu[sizeof(hkpBuildJacobianTaskHeader)] );
	const hkpBuildJacobianTaskHeader* localTaskHeader = HK_NULL;
	if (job.m_taskHeader)
	{
		HK_ASSERT2(0xad7855dd, job.m_jobSubType != hkpDynamicsJob::DYNAMICS_JOB_AGENT_SECTOR, "Collision detection jobs, should not have a pointer to the taskHeader any more. The task header has been already destroyed."); 
		localTaskHeader = downloadTaskHeader(job.m_taskHeader, reinterpret_cast<hkpBuildJacobianTaskHeader*>(taskHeaderBufferOnSpu));
	}
#else
	const hkpBuildJacobianTaskHeader* localTaskHeader = job.m_taskHeader;
#endif


	switch( job.m_jobSubType )
	{
		case hkpDynamicsJob::DYNAMICS_JOB_CREATE_JACOBIAN_TASKS:
			createSolverDebugOstream();
			goto commonForCreateJacAndBuildAcc;			
			// fall through
		case hkpDynamicsJob::DYNAMICS_JOB_BUILD_ACCUMULATORS:
			{
				HK_SPU_DEBUG("buildAcc.fin");
commonForCreateJacAndBuildAcc:
				const hkpBuildAccumulatorsJob& baj = reinterpret_cast<const hkpBuildAccumulatorsJob&>(job);

				//
				// decrease m_openJobs in taskHeader and check for creating a new job
				//
				localTaskHeader->m_openJobs--;
				if ( localTaskHeader->m_openJobs == 0 )
				{
					int numElemsForPpu = localTaskHeader->m_tasks.m_numCallbackConstraints;
					HK_ON_PLATFORM_HAS_SPU( numElemsForPpu += localTaskHeader->m_tasks.m_numPpuOnlyBuildJacobianTasks; )
					if ( numElemsForPpu > 0 )
					{
						// last job finished && callback requests -> convert job
						new (&newJobCreated.m_job) hkpFireJacobianSetupCallback(baj);
						newJobCreated.m_jobPriority     = hkJobQueue::JOB_HIGH_PRIORITY;
					}
					else
					{
						// last job finished && !callback requests -> convert job
						spawnBuildJacobiansJobs(baj, *localTaskHeader, queue, data, newJobCreated);
					}
					goto yesJobCreated;
				}
				goto noJobCreated;
			}

#if !defined (HK_PLATFORM_SPU)
		case hkpDynamicsJob::DYNAMICS_JOB_FIRE_JACOBIAN_SETUP_CALLBACK:
			{
				const hkpFireJacobianSetupCallback& fjscb = reinterpret_cast<const hkpFireJacobianSetupCallback&>(job);
				spawnBuildJacobiansJobs(fjscb, *fjscb.m_taskHeader, queue, data, newJobCreated);
				return hkJobQueue::JOB_CREATED;
			}
#endif

		case hkpDynamicsJob::DYNAMICS_JOB_BUILD_JACOBIANS:
			{
				HK_SPU_DEBUG("buildJac.fin");

				localTaskHeader->m_openJobs--;

				const hkpBuildJacobiansJob& bjj = reinterpret_cast<const hkpBuildJacobiansJob&>(job);

#if defined (HK_PLATFORM_SPU)
				{
					// deallocate job-specific memory
					HK_SPU_DMA_PERFORM_FINAL_CHECKS( HK_NULL, bjj.m_buildJacobianTask, sizeof(hkpBuildJacobianTask) );
					hkDeallocateStack(sizeof(hkpBuildJacobianTask));
				}  
#endif
				if ( localTaskHeader->m_openJobs == 0 )
				{
					hkpDynamicsJob* newDynamicsJob;
					HK_ASSERT2(0xad6775da, localTaskHeader->m_solveInSingleThread || !localTaskHeader->m_solveInSingleThreadOnPpuOnly, "Check: solveOnPpu => solveInSingleThread.");
					if (localTaskHeader->m_solveInSingleThread)
					{
						//
						// Create single solve job.
						//
						newDynamicsJob = new (&newJobCreated.m_job) hkpSolveConstraintsJob(bjj, *localTaskHeader);
						newJobCreated.m_jobPriority = hkJobQueue::JOB_HIGH_PRIORITY;
						newJobCreated.m_job.m_jobSpuType = localTaskHeader->m_solveInSingleThreadOnPpuOnly ? HK_JOB_SPU_TYPE_DISABLED : HK_JOB_SPU_TYPE_ENABLED;
					}
					else
					{
						newDynamicsJob = new (&newJobCreated.m_job) hkpSolveApplyGravityJob(job, *localTaskHeader );
						newJobCreated.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;
						newJobCreated.m_job.m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
						localTaskHeader->m_openJobs   = localTaskHeader->m_numApplyGravityJobs;
					}

					spawnSplitSimulationIslandJob( newDynamicsJob, *localTaskHeader, queue, data );

					goto yesJobCreated;
				}

				break;
			}


		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_CONSTRAINTS:
			{
				HK_SPU_DEBUG("solveST.fin");

				// allow broadphase to continue with stage 2 (remove & add pairs)
				// important: the last job to finish must be on the ppu (so that it can release the job queue); if there are no agent sector jobs, this last job
				// has to be the broadphase job; to assure that the broadphase job (which got started by this solve job) does not overtake the solve job we need
				// to wait until this solve job is finished (i.e. we get here) before letting the (currently blocking) broadphase to continue
				hkSpuDmaUtils::setChar8InMainMemory(&job.m_taskHeader->m_exportFinished, hkChar(1));

				return hkJobQueue::NO_JOB_CREATED;
			}

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_APPLY_GRAVITY:
			{
				HK_SPU_DEBUG("applyGrav.fin");

				localTaskHeader->m_openJobs--;
				if ( localTaskHeader->m_openJobs == 0)
				{
					const hkpSolveApplyGravityJob& sagj = reinterpret_cast<const hkpSolveApplyGravityJob&>(job);
					// Create solver job
					new (&newJobCreated.m_job) hkpSolveConstraintBatchJob(sagj, *localTaskHeader );
					newJobCreated.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;
					newJobCreated.m_job.m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
					localTaskHeader->m_openJobs = localTaskHeader->m_solveTasks.m_firstBatchSize;

					HK_ON_DETERMINISM_CHECKS_ENABLED( hkUint16 valueToUpload = hkUint16(localTaskHeader->m_openJobs); )
					HK_ON_DETERMINISM_CHECKS_ENABLED( HK_ASSERT2(0xad875add, sizeof(job.m_taskHeader->m_sidForNextJobType) == 2, "m_sidForNextJobType must be a hkUint16."); )
					HK_ON_DETERMINISM_CHECKS_ENABLED( hkSpuDmaUtils::setChar8InMainMemory(reinterpret_cast<hkChar*>(&job.m_taskHeader->m_sidForNextJobType)+0, reinterpret_cast<hkChar*>(&valueToUpload)[0]); )
					HK_ON_DETERMINISM_CHECKS_ENABLED( hkSpuDmaUtils::setChar8InMainMemory(reinterpret_cast<hkChar*>(&job.m_taskHeader->m_sidForNextJobType)+1, reinterpret_cast<hkChar*>(&valueToUpload)[1]); )


					goto yesJobCreated;
				}
				goto noJobCreated;
			}
		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_CONSTRAINT_BATCH:
			{
				HK_SPU_DEBUG("solveBatch.fin");

				const hkpSolveConstraintBatchJob& scbij = reinterpret_cast<const hkpSolveConstraintBatchJob&>(job);
				hkJobQueue::JobCreationStatus status = hkJobQueue::NO_JOB_CREATED;

				localTaskHeader->m_openJobs--;
				if ( localTaskHeader->m_openJobs == 0 )
				{
					// if that was not the last iteration then start another one.
					const unsigned int sizeOfNextBatch = scbij.m_solveConstraintBatchTask->m_sizeOfNextBatch;

					int numOpenJobsForNewJobCreated;

					if (sizeOfNextBatch != 0)
					{
						// Create a new job for the next solving batch
						HK_ON_DETERMINISM_CHECKS_ENABLED( hkpSolveConstraintBatchJob* newJob =) new (&newJobCreated.m_job) hkpSolveConstraintBatchJob( scbij ); 
						HK_ON_DETERMINISM_CHECKS_ENABLED( newJob->m_jobSid = localTaskHeader->m_sidForNextJobType; )
						numOpenJobsForNewJobCreated = sizeOfNextBatch;
					}
					else
					{
						if (scbij.m_currentSolverMicroStep + 1 < scbij.m_numSolverMicroSteps )
						{
							// Create a new job for the next solving batch
							hkpSolveConstraintBatchJob* newJob = new (&newJobCreated.m_job) hkpSolveConstraintBatchJob( scbij );
							HK_ON_DETERMINISM_CHECKS_ENABLED( newJob->m_jobSid = localTaskHeader->m_sidForNextJobType; )
							newJob->m_currentSolverMicroStep++;
							newJob->m_solveConstraintBatchTask = localTaskHeader->m_solveTasks.m_firstSolveJacobiansTask;
							numOpenJobsForNewJobCreated = localTaskHeader->m_solveTasks.m_firstBatchSize;
						}
						else
						{
							// create an integrateVelocites job
							HK_ON_DETERMINISM_CHECKS_ENABLED( hkpSolveIntegrateVelocitiesJob* newJob =) new (&newJobCreated.m_job) hkpSolveIntegrateVelocitiesJob( scbij, *localTaskHeader );
							HK_ON_DETERMINISM_CHECKS_ENABLED( newJob->m_jobSid = localTaskHeader->m_sidForNextJobType; );
							numOpenJobsForNewJobCreated = localTaskHeader->m_numIntegrateVelocitiesJobs;
						}

					}
					newJobCreated.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;
					newJobCreated.m_job.m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
					localTaskHeader->m_openJobs = numOpenJobsForNewJobCreated;
					HK_ON_DETERMINISM_CHECKS_ENABLED( hkUint16 valueToUpload = hkUint16(localTaskHeader->m_sidForNextJobType + localTaskHeader->m_openJobs); )
					HK_ON_DETERMINISM_CHECKS_ENABLED( HK_ASSERT2(0xad875add, sizeof(job.m_taskHeader->m_sidForNextJobType) == 2, "m_sidForNextJobType must be a hkUint16."); )
					HK_ON_DETERMINISM_CHECKS_ENABLED( hkSpuDmaUtils::setChar8InMainMemory(reinterpret_cast<hkChar*>(&job.m_taskHeader->m_sidForNextJobType)+0, reinterpret_cast<hkChar*>(&valueToUpload)[0]); )
					HK_ON_DETERMINISM_CHECKS_ENABLED( hkSpuDmaUtils::setChar8InMainMemory(reinterpret_cast<hkChar*>(&job.m_taskHeader->m_sidForNextJobType)+1, reinterpret_cast<hkChar*>(&valueToUpload)[1]); )

					status = hkJobQueue::JOB_CREATED;
				}

#if defined(HK_PLATFORM_SPU)
				{
					// deallocate job-specific memory
					hkSpuDmaManager::performFinalChecks( HK_NULL, scbij.m_solveConstraintBatchTask, sizeof(hkpSolveConstraintBatchTask) );
					hkDeallocateStack(sizeof(hkpSolveConstraintBatchTask)); 
				}  
#endif
				uploadOpenJobsVariable(localTaskHeader, job.m_taskHeader);
				return status;
			}

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_INTEGRATE_VELOCITIES:
			{
				HK_SPU_DEBUG("intVel.fin");

				const hkpSolveIntegrateVelocitiesJob& sivbij = reinterpret_cast<const hkpSolveIntegrateVelocitiesJob&>(job);

				localTaskHeader->m_openJobs--;

				if ( localTaskHeader->m_openJobs == 0 )
				{
					int numOpenJobsForNewJobCreated;
					if (!sivbij.m_solvingFinished)
					{
						// if that was not the last iteration then start another one.
						HK_ON_DEBUG( hkUint32 currentSolverStep = sivbij.m_currentSolverStep );

						//
						// Create a new job for the next solving step
						//

					#	if defined (HK_DEBUG) || defined (HK_ENABLE_DETERMINISM_CHECKS)
						hkpSolveConstraintBatchJob* newJob =
					#	endif
						new (&newJobCreated.m_job) hkpSolveConstraintBatchJob( sivbij, *localTaskHeader );
						HK_ON_DETERMINISM_CHECKS_ENABLED( newJob->m_jobSid = localTaskHeader->m_sidForNextJobType; )
						HK_ASSERT2(0xad79d8d7, newJob->m_currentSolverStep == currentSolverStep+1, "internal error");
						numOpenJobsForNewJobCreated = localTaskHeader->m_solveTasks.m_firstBatchSize;
					}
					else
					{
						//
						// Morph in to the integrate motions job (which in turn morphs into solver export) here and set the open jobs count
						//
						HK_ASSERT2(0xad8754aa, localTaskHeader->m_solveTasks.m_firstSolveJacobiansTask, "You must have solve tasks to do export");

						new (&newJobCreated.m_job) hkpIntegrateMotionJob(sivbij, *localTaskHeader); 
						numOpenJobsForNewJobCreated = localTaskHeader->m_tasks.m_numBuildJacobianTasks; // this will be used by the export ...
						HK_ON_PLATFORM_HAS_SPU( numOpenJobsForNewJobCreated += localTaskHeader->m_tasks.m_numPpuOnlyBuildJacobianTasks; )
					}

					newJobCreated.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;
					newJobCreated.m_job.m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;					
					localTaskHeader->m_openJobs = numOpenJobsForNewJobCreated;

					HK_ON_DETERMINISM_CHECKS_ENABLED( hkUint16 valueToUpload = hkUint16(localTaskHeader->m_sidForNextJobType + localTaskHeader->m_openJobs); )
					HK_ON_DETERMINISM_CHECKS_ENABLED( HK_ASSERT2(0xad875bdd, sizeof(job.m_taskHeader->m_sidForNextJobType) == 2, "m_sidForNextJobType must be a hkUint16."); )
					HK_ON_DETERMINISM_CHECKS_ENABLED( hkSpuDmaUtils::setChar8InMainMemory(reinterpret_cast<hkChar*>(&job.m_taskHeader->m_sidForNextJobType)+0, reinterpret_cast<hkChar*>(&valueToUpload)[0]); )
					HK_ON_DETERMINISM_CHECKS_ENABLED( hkSpuDmaUtils::setChar8InMainMemory(reinterpret_cast<hkChar*>(&job.m_taskHeader->m_sidForNextJobType)+1, reinterpret_cast<hkChar*>(&valueToUpload)[1]); )
					goto yesJobCreated;
				}

				goto noJobCreated;
			}

		case hkpDynamicsJob::DYNAMICS_JOB_SOLVE_EXPORT_RESULTS:
			{
				HK_SPU_DEBUG("export.fin");

				const hkpSolveExportResultsJob& serbij = reinterpret_cast<const hkpSolveExportResultsJob&>(job);
				HK_CPU_PTR(hkpBuildJacobianTaskHeader*) taskHeaderInMainMemory = serbij.m_taskHeader;
				localTaskHeader->m_openJobs--;

#if !defined(HK_PLATFORM_SPU)
				hkpImpulseLimitBreachedHeader* ilbh = (hkpImpulseLimitBreachedHeader*) (serbij.m_solveConstraintBatchTask->m_schemas );
				if ( ilbh->m_numBreached)
				{
					ilbh->m_next = taskHeaderInMainMemory->m_impulseLimitsBreached;
					taskHeaderInMainMemory->m_impulseLimitsBreached = ilbh;
				}
#else
				if ( serbij.m_numImpulseLimitsBreached )
				{
					hkpImpulseLimitBreachedHeader  h; 
					hkpImpulseLimitBreachedHeader* hOnPpu = serbij.m_impulseLimitsBreached;
					h.m_next = localTaskHeader->m_impulseLimitsBreached;
					h.m_numBreached = serbij.m_numImpulseLimitsBreached;
					localTaskHeader->m_impulseLimitsBreached = hOnPpu;

					// send back task header and impulse limit header
					hkSpuDmaManager::putToMainMemory( hOnPpu, &h, sizeof(h), hkSpuDmaManager::WRITE_NEW );
					hkSpuDmaManager::putToMainMemorySmall( &taskHeaderInMainMemory->m_impulseLimitsBreached, &localTaskHeader->m_impulseLimitsBreached, sizeof(localTaskHeader->m_impulseLimitsBreached), hkSpuDmaManager::WRITE_NEW );
					hkSpuDmaManager::waitForAllDmaCompletion();
					HK_SPU_DMA_PERFORM_FINAL_CHECKS( hOnPpu, &h, sizeof(h) );
					HK_SPU_DMA_PERFORM_FINAL_CHECKS( &taskHeaderInMainMemory->m_impulseLimitsBreached, &localTaskHeader->m_impulseLimitsBreached, sizeof(localTaskHeader->m_impulseLimitsBreached) );
				}
					// deallocate job-specific memory
				hkSpuDmaManager::performFinalChecks( HK_NULL, serbij.m_solveConstraintBatchTask, sizeof(hkpSolveConstraintBatchTask) );
				hkDeallocateStack(sizeof(hkpSolveConstraintBatchTask));
#endif

				if ( localTaskHeader->m_openJobs == 0 )
				{
					// allow broadphase to continue with stage 2 (remove & add pairs)
					// important: the last job to finish must be on the ppu (so that it can release the job queue); if there are no agent sector jobs, this last job
					// has to be the broadphase job; to assure that the broadphase job (which got started by this solve job) does not overtake the solve job we need
					// to wait until this solve job is finished (i.e. we get here) before letting the (currently blocking) broadphase to continue
					hkSpuDmaUtils::setChar8InMainMemory(&taskHeaderInMainMemory->m_exportFinished, hkChar(1));

					// The above destroys the task header in the main memory!
					localTaskHeader = HK_NULL;
					return hkJobQueue::NO_JOB_CREATED;
				}

				HK_ASSERT2(0xad8765da, localTaskHeader->m_openJobs != 0, "Cannot upload localTaskHeader->m_openJobs, after export is finished -- the task header gets immediately destroyed.");
				// Export not finished -- still need to upload openJobs.
				goto noJobCreated; 
			}


		case hkpDynamicsJob::DYNAMICS_JOB_INTEGRATE_MOTION:
			{
				HK_SPU_DEBUG("intMotion.fin");

				const hkpIntegrateMotionJob& imj = reinterpret_cast<const hkpIntegrateMotionJob&>(job);
				if ( imj.m_numInactiveFrames <= hkpMotion::NUM_INACTIVE_FRAMES_TO_DEACTIVATE)
				{
					hkSpuDmaUtils::setInt32InMainMemory( &imj.m_taskHeader->m_islandShouldBeDeactivated, 0);
				}
				HK_ASSERT( 0xf0002123, imj.m_numEntities>0);
				numJobsToFinish = 1 + ( unsigned(imj.m_numEntities-1)/hkpIntegrateMotionJob::ACCUMULATORS_PER_JOB);
				// no break here
				goto commonForIntegrateMotinoAndSplitIsland;
			}
		case hkpDynamicsJob::DYNAMICS_JOB_SPLIT_ISLAND:
			{
				HK_SPU_DEBUG("split.fin");
commonForIntegrateMotinoAndSplitIsland:
				const hkpSplitSimulationIslandJob& sij = reinterpret_cast<const hkpSplitSimulationIslandJob&>(job);

				int openJobs = hkSpuDmaUtils::incrementInt32InMainMemory( &sij.m_taskHeader->m_numUnfinishedJobsForBroadphase, -numJobsToFinish );
				HK_ASSERT( 0xf0343212, openJobs >= 0 );
				if ( openJobs == 0 )
				{
					new (&newJobCreated.m_job) hkpBroadPhaseJob(sij, sij.m_taskHeader);
					newJobCreated.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;
					return hkJobQueue::JOB_CREATED;
				}
				return hkJobQueue::NO_JOB_CREATED;
			}
		case hkpDynamicsJob::DYNAMICS_JOB_AGENT_SECTOR:
			{
				HK_SPU_DEBUG("agentSec.fin");

				const hkpAgentSectorJob& asj = reinterpret_cast<const hkpAgentSectorJob&>(job);
				if ( !asj.m_header)
				{
					// without the header, there is only one agent sector job for this island, no need for the PostCollide Job
					return hkJobQueue::NO_JOB_CREATED;
				}

				int openJobs = hkSpuDmaUtils::incrementInt32InMainMemory( &asj.m_header->m_openJobs, -1 );
				if ( openJobs == 0 )
				{
					new (&newJobCreated.m_job) hkpPostCollideJob(asj);
					newJobCreated.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;
					return hkJobQueue::JOB_CREATED;
				}
				return hkJobQueue::NO_JOB_CREATED;
			}
		case hkpDynamicsJob::DYNAMICS_JOB_AGENT_NN_ENTRY:
			{
				return hkJobQueue::NO_JOB_CREATED;
			}
		case hkpDynamicsJob::DYNAMICS_JOB_DUMMY:
			{
				return hkJobQueue::NO_JOB_CREATED;
			}
		case hkpDynamicsJob::DYNAMICS_JOB_POST_COLLIDE:
			{
				HK_SPU_DEBUG("postCollide.fin");
				break;
			}

		default:
			{
				break;
			}
	}

noJobCreated:
	uploadOpenJobsVariable( localTaskHeader, job.m_taskHeader );
	return hkJobQueue::NO_JOB_CREATED;

yesJobCreated:
	uploadOpenJobsVariable( localTaskHeader, job.m_taskHeader );
	return hkJobQueue::JOB_CREATED;
}

hkJobQueue::JobCreationStatus HK_CALL hkpJobQueueUtils::finishCollideJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, const hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntryInput& newJobCreated )
{
	flushSolverDebugOstream();

	const hkpDynamicsJob& job = reinterpret_cast<const hkpDynamicsJob&>(jobIn);

#if defined (HK_PLATFORM_SPU)
	HK_ALIGN16( char taskHeaderBufferOnSpu[sizeof(hkpBuildJacobianTaskHeader)] );
	const hkpBuildJacobianTaskHeader* localTaskHeader = HK_NULL;
	if (job.m_taskHeader)
	{
		HK_ASSERT2(0xad7855dd, job.m_jobSubType != hkpDynamicsJob::DYNAMICS_JOB_AGENT_SECTOR, "Collision detection jobs, should not have a pointer to the taskHeader any more. The task header has been already destroyed."); 
		localTaskHeader = downloadTaskHeader(job.m_taskHeader, reinterpret_cast<hkpBuildJacobianTaskHeader*>(taskHeaderBufferOnSpu));
	}
#else
	const hkpBuildJacobianTaskHeader* localTaskHeader = job.m_taskHeader;
#endif


	switch( job.m_jobSubType )
	{
	case hkpDynamicsJob::DYNAMICS_JOB_AGENT_SECTOR:
		{
			HK_SPU_DEBUG("agentSec.fin");

			const hkpAgentSectorJob& asj = reinterpret_cast<const hkpAgentSectorJob&>(job);
			if ( !asj.m_header)
			{
				// without the header, there is only one agent sector job for this island, no need for the PostCollide Job
				return hkJobQueue::NO_JOB_CREATED;
			}

			int openJobs = hkSpuDmaUtils::incrementInt32InMainMemory( &asj.m_header->m_openJobs, -1 );
			if ( openJobs == 0 )
			{
				new (&newJobCreated.m_job) hkpPostCollideJob(asj);
				newJobCreated.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;
				return hkJobQueue::JOB_CREATED;
			}
			return hkJobQueue::NO_JOB_CREATED;
		}
	case hkpDynamicsJob::DYNAMICS_JOB_AGENT_NN_ENTRY:
		{
			return hkJobQueue::NO_JOB_CREATED;
		}
	case hkpDynamicsJob::DYNAMICS_JOB_DUMMY:
		{
			return hkJobQueue::NO_JOB_CREATED;
		}
	case hkpDynamicsJob::DYNAMICS_JOB_POST_COLLIDE:
		{
			HK_SPU_DEBUG("postCollide.fin");
			break;
		}

	default:
		{
			break;
		}
	}

	uploadOpenJobsVariable( localTaskHeader, job.m_taskHeader );
	return hkJobQueue::NO_JOB_CREATED;
}


HK_COMPILE_TIME_ASSERT( sizeof( hkpIntegrateJob )			<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof( hkpBuildAccumulatorsJob )	<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpFireJacobianSetupCallback) <= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpBuildJacobiansJob)			<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveConstraintsJob)		<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveApplyGravityJob)		<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveConstraintBatchJob)	<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT(sizeof(hkpSolveIntegrateVelocitiesJob)<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveExportResultsJob)		<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpPostCollideJob)			<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpBroadPhaseJob)				<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpAgentSectorJob)			<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpCreateJacobianTasksJob)	<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpSplitSimulationIslandJob)	<= sizeof(hkJobQueue::JobQueueEntry)) ;
HK_COMPILE_TIME_ASSERT( sizeof(hkpIntegrateMotionJob)		<= sizeof(hkJobQueue::JobQueueEntry)) ;

// When changing sizes of jobs here, make sure the PlayStation(R)3's versions for job constructors 
// are updated appropriately. They use hkMemCpy16Single() for copying constructors.
#if HK_POINTER_SIZE == 4
		HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveConstraintBatchJob) == 48 );
		HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveApplyGravityJob) == 48 );
#if (HK_NATIVE_ALIGNMENT==16)
		HK_COMPILE_TIME_ASSERT( sizeof(hkpDynamicsJob) == 32 );
#endif
		HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpSolveApplyGravityJob, m_accumulators) == 32 );
#else // HK_POINTER_SIZE == 8
#	if defined (HK_PLATFORM_HAS_SPU) // we only care about this for playstation3
		HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveConstraintBatchJob) == 64 );
		HK_COMPILE_TIME_ASSERT( sizeof(hkpSolveApplyGravityJob) == 48 );
		HK_COMPILE_TIME_ASSERT( sizeof(hkpDynamicsJob) == 32 );
		HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpSolveApplyGravityJob, m_accumulators) == 32 );
#	endif
#endif

#if (HK_NATIVE_ALIGNMENT==16)
		HK_COMPILE_TIME_ASSERT( (HK_OFFSET_OF(hkpBuildJacobianTaskHeader, m_accumulatorsBase) & 0xf) == 0 );
#endif
HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpSolveApplyGravityJob, m_accumulators) + HK_POINTER_SIZE == HK_OFFSET_OF(hkpSolveApplyGravityJob, m_accumulatorsEnd) );
HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpBuildJacobianTaskHeader, m_accumulatorsBase) + HK_POINTER_SIZE == HK_OFFSET_OF(hkpBuildJacobianTaskHeader, m_accumulatorsEnd) );

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
