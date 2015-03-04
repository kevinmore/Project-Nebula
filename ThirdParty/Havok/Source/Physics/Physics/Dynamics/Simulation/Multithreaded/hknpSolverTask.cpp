/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpSolverTask.h>

#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>
#include <Common/Base/Thread/Task/hkTaskGraphUtil.h>

#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverUtil.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>

extern HK_THREAD_LOCAL( int ) hkThreadNumber;

// Uncomment this to print the solver task graph to a .gv file every frame.
// You can use Graphviz's dot.exe to generate an image from the file.
//#define HKNP_PRINT_SOLVER_TASK_GRAPH

#if defined(HKNP_PRINT_SOLVER_TASK_GRAPH)

#define HKNP_SOLVER_TASK_GRAPH_FILE	"SolverTaskGraph.gv""

namespace
{
	class SolverJobPrinter : public hkTaskGraphUtil::TaskPrinter
	{
	public:

		SolverJobPrinter(const hkArray<hknpSolverSchedulerTask>& tasks) : m_tasks(tasks) {}

		virtual void print(const hkTask* job, hkStringBuf& nodeNameOut, hkStringBuf& nodeAttributesOut) HK_OVERRIDE
		{
			hknpSolverScheduler::TaskJob taskJob(job);
			const hknpSolverSchedulerTask& task = m_tasks[taskJob.getTaskIndex()];
			char* type;
			nodeAttributesOut = "[shape=box]";
			switch (task.m_processType)
			{
				case hknpSolverSchedulerTask::PROCESS_ADD_ACTIVE_BODY_PAIRS:
				{
					type = "AB";
					break;
				}

				case hknpSolverSchedulerTask::PROCESS_GC_INACTIVE_CACHES:
				{
					type = "GC";
					break;
				}

				case hknpSolverSchedulerTask::PROCESS_STEP_JACOBIAN_CONSTRAINT:
				{
					switch (task.m_grid)
					{
						case hknpJacobianGridType::JOINT_CONSTRAINT:
						{
							type = "JC";
							nodeAttributesOut = "[shape=diamond, style=rounded]";
							break;
						}
						case hknpJacobianGridType::MOVING_CONTACT:
						{
							type = "MC";
							nodeAttributesOut = "[shape=ellipse]";
							break;
						}
						case hknpJacobianGridType::FIXED_CONTACT:
						{
							type = "FC";
							nodeAttributesOut = "[shape=ellipse]";
							break;
						}
						default:
						{
							type = "?C";
						}
					}
					break;
				}

				case hknpSolverSchedulerTask::PROCESS_STEP_SUBINTEGRATE:
				{
					type = "IN";
					break;
				}

				case hknpSolverSchedulerTask::PROCESS_CLOSE_QUEUE:
				{
					type = "CQ";
					break;
				}

				default:
				{
					type = "??";
				}
			}

			nodeNameOut.printf("%s(%d,%d) %d", type, task.m_cellA, task.m_cellB, taskJob.getSubStep());
		}

	protected:

		const hkArray<hknpSolverSchedulerTask>& m_tasks;
	};
}

#endif // if defined(HKNP_PRINT_SOLVER_TASK_GRAPH)


hknpSolverTask::hknpSolverTask( hknpSimulationContext& simulationContext, hknpSolverData* solverData, hkTaskQueue* taskQueue )
:	m_taskQueue( taskQueue )
{
	HK_CHECK_FLUSH_DENORMALS();
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	HK_TIMER_BEGIN_LIST2( timerStream, "CreateSolverTask", "SetupSolverIdTable" );

	const int currentThreadId = simulationContext.getCurrentThreadNumber();
	hknpSimulationThreadContext& threadContext = *simulationContext.getThreadContext( currentThreadId );
	hknpWorld* world = threadContext.m_world;
	hknpSpaceSplitter* splitter = world->m_spaceSplitter;

	// Prepare motions
	{
		// Build a reindex id list
		hkArray<hknpMotionId> globalSolverIdToMotion;
		m_cellIdxToGlobalSolverId.setSize( splitter->getNumCells() );
		world->m_motionManager.buildSolverIdToMotionIdMap( m_cellIdxToGlobalSolverId, globalSolverIdToMotion );

		HK_TIMER_SPLIT_LIST("SetupSolverVelocities");

		// Build solver velocities (first fixed motion will get converted correctly by this also)
		const int numSolverVels = globalSolverIdToMotion.getSize();
		world->m_solverVelocities.setSize( numSolverVels );
		world->m_solverSumVelocities.setSize( numSolverVels );
		hknpMotionUtil::gatherSolverVelocities(
			&threadContext, world, globalSolverIdToMotion.begin(), globalSolverIdToMotion.getSize(),
			world->m_solverVelocities.begin(), world->m_solverSumVelocities.begin() );
	}

	if( world->areConsistencyChecksEnabled() )
	{
		hknpSimulation::checkConsistencyOfJacobians( world, hknpSimulation::CHECK_ALL, &solverData->m_jacMovingGrid );
		hknpSimulation::checkConsistencyOfJacobians( world, hknpSimulation::CHECK_ALL, &solverData->m_jacFixedGrid );
		
	}

	// Prepare deactivation data
	{
		m_deactivationStepInfo = HK_NULL;
		if( world->isDeactivationEnabled() )
		{
			HK_TIMER_SPLIT_LIST2( timerStream, "SetupDeactivationData");

			const int numAllActiveMotionsPlusPadding = world->m_solverVelocities.getSize();
			m_deactivationStepInfo = new hknpDeactivationStepInfo( numAllActiveMotionsPlusPadding, simulationContext.getNumThreads() );
			m_deactivationStepInfo->addAllBodyLinks( *world, world->m_deactivationManager->getBodyLinks(), m_cellIdxToGlobalSolverId );

			// Set per-thread data
			const int bitFieldSize = HK_NEXT_MULTIPLE_OF(4,numAllActiveMotionsPlusPadding) + 4;
			for( int ti = 0; ti < simulationContext.getNumThreads(); ti++ )
			{
				hknpDeactivationThreadData* data = &m_deactivationStepInfo->getThreadData(ti);
				hknpSimulationThreadContext* tl = simulationContext.getThreadContext(ti);
				tl->m_deactivationData = data;
				data->m_solverVelOkToDeactivate.setSizeAndFill(0, bitFieldSize, 0);

#if HKNP_ENABLE_SOLVER_PARALLEL_TASKS == 1
				tl->m_cellIdxToGlobalSolverId = &( m_cellIdxToGlobalSolverId );
				tl->m_deactivationStepInfo = m_deactivationStepInfo;
#endif
			}
		}
	}

	// Prepare space splitter data
	{
		m_spaceSplitterData = hkAllocateChunk<hknpSpaceSplitterData>( simulationContext.getNumThreads(), HK_MEMORY_CLASS_PHYSICS );

		// Set per-thread data
		for( int ti = 0; ti < simulationContext.getNumThreads(); ti++ )
		{
			m_spaceSplitterData[ti].reset();
			simulationContext.getThreadContext(ti)->m_spaceSplitterData = &m_spaceSplitterData[ti];
		}
	}

	// Setup the array of grids to be passed to the scheduler.
	m_jacobianGrids[hknpJacobianGridType::JOINT_CONSTRAINT] = &solverData->m_jacConstraintsGrid;
	m_jacobianGrids[hknpJacobianGridType::MOVING_CONTACT] = &solverData->m_jacMovingGrid;
	m_jacobianGrids[hknpJacobianGridType::FIXED_CONTACT] = &solverData->m_jacFixedGrid;

	// Fill scheduler infos.
	m_jacobianGridInfos[hknpJacobianGridType::JOINT_CONSTRAINT].setIsLinkGrid();
	m_jacobianGridInfos[hknpJacobianGridType::JOINT_CONSTRAINT].setPriority(hknpDefaultConstraintSolverPriority::JOINTS);
	m_jacobianGridInfos[hknpJacobianGridType::MOVING_CONTACT].setIsLinkGrid();
	m_jacobianGridInfos[hknpJacobianGridType::MOVING_CONTACT].setPriority(hknpDefaultConstraintSolverPriority::MOVING_CONTACTS);
	m_jacobianGridInfos[hknpJacobianGridType::FIXED_CONTACT].setIsCellArray();
	m_jacobianGridInfos[hknpJacobianGridType::FIXED_CONTACT].setPriority(hknpDefaultConstraintSolverPriority::FIXED_CONTACTS);

	//
	// Let solvers that need it allocate their temps.
	//

	HK_TIMER_SPLIT_LIST2(timerStream, "SetupTempImpulses" );
	hknpSolverUtil::allocateSolverTemps(
		threadContext, m_jacobianGridInfos, m_jacobianGrids, hknpJacobianGridType::NUM_TYPES,
		&solverData->m_threadData[currentThreadId].m_solverTempsStream,
		&solverData->m_contactSolverTempsStream );

	//
	// setup step dependencies
	//

	HK_TIMER_SPLIT_LIST2( timerStream, "SetupScheduler");
#if defined(HKNP_ENABLE_LOCKLESS_SOLVER)
	hknpSolverScheduler::TaskAddContext taskAddContext( splitter->getNumCells() );
	m_taskBuilder = new hkSimpleSchedulerTaskBuilder();
#else
	const int numCells = splitter->getNumCells();
	hkLocalArray<hkDefaultTaskGraph::TaskId> lastTaskInCell(numCells);
	lastTaskInCell.setSize(numCells, hkDefaultTaskGraph::TaskId::invalid());
	m_subTasks.reserve(32);
#endif

	// Setup non-solver tasks to be run in parallel to solver tasks.
#if HKNP_ENABLE_SOLVER_PARALLEL_TASKS == 1

	hknpSolverSchedulerTask::ProcessType deactivationTasks[2] = {
		hknpSolverSchedulerTask::PROCESS_GC_INACTIVE_CACHES,
		hknpSolverSchedulerTask::PROCESS_ADD_ACTIVE_BODY_PAIRS,
	};
	const int deactivationTaskCount = HK_COUNT_OF( deactivationTasks );

	
	if( world->isDeactivationEnabled() )
	{
	#if defined(HKNP_ENABLE_LOCKLESS_SOLVER)
		hknpSolverScheduler::addNonSolverTasks(
			taskAddContext, m_taskBuilder, deactivationTaskCount, deactivationTasks );
	#else
		hknpSolverScheduler::addNonSolverTasks(
			&m_taskGraph, m_subTasks, deactivationTaskCount, deactivationTasks );
	#endif
	}

#endif

	const hknpSolverInfo& solverInfo = world->m_solverInfo;
	const hknpSolverScheduler::ScheduleType scheduleType = world->m_enableSolverDynamicScheduling ?
														   hknpSolverScheduler::SCHEDULE_DYNAMIC :
														   hknpSolverScheduler::SCHEDULE_STATIC;

#if defined(HKNP_ENABLE_LOCKLESS_SOLVER)

	hknpSolverScheduler::addSolverTasks(
		taskAddContext, splitter, m_jacobianGridInfos, m_jacobianGrids,
		hknpJacobianGridType::NUM_TYPES, &solverData->m_liveJacInfoGrid,
		&m_cellIdxToGlobalSolverId, m_taskBuilder, scheduleType );

	m_taskBuilder->finalize();

#if 0 && defined(HK_DEBUG)
	taskBuilder->printSimulatedRun<hknpSolverSchedulerTask>();
#endif

#else

	// Setup solver tasks
	hknpSolverScheduler::addSolverTasks(
		lastTaskInCell, &m_taskGraph, m_subTasks, solverInfo.m_numSteps,
		solverInfo.m_numMicroSteps, splitter, m_jacobianGridInfos, m_jacobianGrids,
		hknpJacobianGridType::NUM_TYPES, &solverData->m_liveJacInfoGrid,
		&m_cellIdxToGlobalSolverId, scheduleType );

	// Add a task to close the queue
	{
		const int taskIndex = m_subTasks.getSize();
		hknpSolverSchedulerTask& task = m_subTasks.expandOne();
		task = hknpSolverSchedulerTask(hknpSolverSchedulerTask::PROCESS_CLOSE_QUEUE, 0, 0, 0, 0);
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
		task.m_pad = (hkUint16)taskIndex;
#endif
		hkDefaultTaskGraph::TaskId taskId = m_taskGraph.addTask(hknpSolverScheduler::TaskJob(taskIndex, 0, 0).asJob());
		for (int i = 0; i < lastTaskInCell.getSize(); ++i)
		{
			if (lastTaskInCell[i].isValid())
			{
				m_taskGraph.addDependency(lastTaskInCell[i], taskId);
			}
		}

#if HKNP_ENABLE_SOLVER_PARALLEL_TASKS == 1
		// Add the deactivation jobs as parents of the closing job
		if( world->isDeactivationEnabled() )
		{
			for (int i = 0; i < deactivationTaskCount; ++i)
			{
				m_taskGraph.addDependency( hkDefaultTaskGraph::TaskId(i), taskId );
			}
		}
#endif
	}

	m_taskGraph.finish(MAX_AVAILABLE_TASKS);

	m_taskQueue->open();
	m_graphId = m_taskQueue->addGraph(&m_taskGraph, 0);

#if defined(HKNP_PRINT_SOLVER_TASK_GRAPH)
	hkOstream file(HKNP_SOLVER_TASK_GRAPH_FILE);
	SolverJobPrinter jobPrinter(m_subTasks);
	hkTaskGraphUtil::print(&m_taskGraph, &jobPrinter, file);
#endif

#endif // #if defined(HKNP_ENABLE_LOCKLESS_SOLVER)

	HK_TIMER_SPLIT_LIST2( timerStream, "SetupSolverStepInfo");

	m_solverStepInfo = new hknpSolverStepInfo;
	threadContext.m_solverStepInfo = m_solverStepInfo;
	{
		hknpSolverStepInfo* HK_RESTRICT solverManager = m_solverStepInfo;

		// management stuff
		solverManager->m_tempAllocator = threadContext.m_tempAllocator->m_blockStreamAllocator;
		solverManager->m_heapAllocator = threadContext.m_heapAllocator->m_blockStreamAllocator;
		solverManager->m_simulationContext = &simulationContext;
		solverManager->m_solverData = solverData;

	#if defined(HKNP_ENABLE_LOCKLESS_SOLVER)
		// jobs data
		solverManager->m_taskList = m_taskBuilder;
	#endif

		// const solver setup data
		solverManager->m_solverInfo = &solverInfo;
		solverManager->m_spaceSplitter = world->m_spaceSplitter;
		solverManager->m_spaceSplitterSize = world->m_spaceSplitter->getSize();
		solverManager->m_intSpaceUtil = &world->m_intSpaceUtil;
		solverManager->m_motionGridEntriesStart = m_cellIdxToGlobalSolverId.m_entries.begin();

		// general working data
		solverManager->m_bodies = world->m_bodyManager.accessBodyBuffer();
		solverManager->m_motions = world->m_motionManager.accessMotionBuffer();
		solverManager->m_qualities = world->getBodyQualityLibrary()->getBuffer();
		solverManager->m_solverVelocities = world->m_solverVelocities.begin();

		// specific data for hknpStepConstraintJacobianProcess
		solverManager->m_jacGridEntries[hknpJacobianGridType::JOINT_CONSTRAINT] = m_jacobianGrids[hknpJacobianGridType::JOINT_CONSTRAINT]->m_entries.begin();
		solverManager->m_jacGridEntries[hknpJacobianGridType::MOVING_CONTACT] = m_jacobianGrids[hknpJacobianGridType::MOVING_CONTACT]->m_entries.begin();
		solverManager->m_jacGridEntries[hknpJacobianGridType::FIXED_CONTACT] = m_jacobianGrids[hknpJacobianGridType::FIXED_CONTACT]->m_entries.begin();
		solverManager->m_liveJacInfoGridEntries = solverData->m_liveJacInfoGrid.m_entries.begin();

		// specific data for hknpSubIntegrateProcess
		solverManager->m_solverSumVelocities = world->m_solverSumVelocities.begin();
		solverManager->m_deactivationStates = world->m_deactivationManager->getAllDeactivationStates();
		solverManager->m_motionProperties = world->getMotionPropertiesLibrary()->getBuffer();
		solverManager->m_numMotionProperties = world->getMotionPropertiesLibrary()->getCapacity();

		// specific data for hknpUpdateBodiesProcess
		{
			const hkArray<hknpBodyId>& activeBodies = world->getActiveBodies();
			solverManager->m_dynamicBodyIds = activeBodies.begin();
			solverManager->m_numDynamicBodyIds = HK_NEXT_MULTIPLE_OF(hk4xVector4::mxLength, activeBodies.getSize());
		}

		solverManager->m_numBodyChunksUpdatedVar = 0;
		solverManager->m_numBodyChunksUpdated = &solverManager->m_numBodyChunksUpdatedVar;
	}

	// Calculate the best case number of threads that can work in parallel
	int numWorkerThreads = simulationContext.getNumThreads();
#if defined(HK_PLATFORM_HAS_SPU)
		numWorkerThreads = simulationContext.getNumSpuThreads();
#endif

	// Initialize remaining data members
	{
		m_numThreadsToRun = numWorkerThreads;
		m_orgNumThreadsToRun = numWorkerThreads;

		m_syncBufferSize = solverInfo.m_numSteps * solverInfo.m_numMicroSteps;
		m_syncBuffer = hkAllocateChunk<hkUint32>(m_syncBufferSize, HK_MEMORY_CLASS_PHYSICS);
		for (int i = 0; i < m_syncBufferSize; i++)
		{
			m_syncBuffer[i] = m_orgNumThreadsToRun + 1;
		}

	#if defined(HK_PLATFORM_PPU)
		hkString::memCpy4(m_shapeVTablesPpu, hknpShapeVirtualTableUtil::getVTables(), sizeof(m_shapeVTablesPpu) >> 2);
	#endif
	}

	HK_TIMER_END_LIST2(timerStream);	// "CreateSolverTask"
}

hknpSolverTask::~hknpSolverTask()
{
	// Deallocate in inverse order of allocation because it may be more allocator friendly
	hkDeallocateChunk(m_syncBuffer, m_syncBufferSize, HK_MEMORY_CLASS_PHYSICS);
	delete m_solverStepInfo;
#if defined(HKNP_ENABLE_LOCKLESS_SOLVER)
	delete m_taskBuilder;
#else
	m_taskQueue->removeGraph( m_graphId );
#endif
	int numThreads = m_orgNumThreadsToRun;
	HK_ON_PLATFORM_HAS_SPU(numThreads++);
	hkDeallocateChunk(m_spaceSplitterData, numThreads, HK_MEMORY_CLASS_PHYSICS);
	delete m_deactivationStepInfo;
}

void hknpSolverTask::process()
{
	HK_CHECK_FLUSH_DENORMALS();
	const int threadIdx = HK_THREAD_LOCAL_GET(hkThreadNumber);

	// Get thread monitor stream just once as access to TLS may be expensive
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	// If this is being processed in a worker thread, set up the same timer path as the master thread
	if( threadIdx )
	{
		HK_TIMER_BEGIN2( timerStream, "Physics", HK_NULL );
		HK_TIMER_BEGIN2( timerStream, "Solve", HK_NULL );
	}

	HK_TIMER_BEGIN_LIST2( timerStream, "SolverTask", "Init" );

	hknpSimulationThreadContext* threadContext = m_solverStepInfo->m_simulationContext->getThreadContext();

#if defined(HKNP_ENABLE_LOCKLESS_SOLVER)

	hknpSolverScheduler::solve(*m_solverStepInfo, *threadContext, threadIdx, *this);

#else

	// Process subtasks
	hkTaskQueue::PrioritizedTask prioritizedTask;
	bool closeQueue = false;
	hkTaskQueue::GetNextTaskResult result = m_taskQueue->finishTaskAndGetNext(&prioritizedTask, hkTaskQueue::WAIT_UNTIL_QUEUE_CLOSED);
	while (result == hkTaskQueue::TASK_OBTAINED)
	{
		HK_TIMER_SPLIT_LIST2( timerStream, "SolverSubTask" );

		hknpSolverScheduler::TaskJob taskJob(m_taskQueue->getTask(prioritizedTask));
		hknpSolverSchedulerTask& subtask = m_subTasks[taskJob.getTaskIndex()];

		closeQueue = hknpSolverScheduler::executeTask(
			*threadContext, m_solverStepInfo, taskJob.getSubStep(), taskJob.getMicroStep(), threadIdx, subtask );

		if( !closeQueue )
		{
			result = m_taskQueue->finishTaskAndGetNext( &prioritizedTask, hkTaskQueue::WAIT_UNTIL_QUEUE_CLOSED, &prioritizedTask );
		}
		else
		{
			m_taskQueue->finishTask( prioritizedTask );
			break;
		}
	}

	// Close the queue if the last task we got told us to do so
	if( closeQueue )
	{
		m_taskQueue->close();
	}

#endif

	HK_TIMER_SPLIT_LIST2( timerStream, "UpdateBodies" );
	hknpUpdateBodiesProcess(*threadContext, m_solverStepInfo);

	HK_TIMER_END_LIST2( timerStream );

	if( threadIdx )
	{
		HK_TIMER_END2( timerStream );
		HK_TIMER_END2( timerStream );
	}
}


void* hknpSolverTask::getElf()
{
#if defined(HK_PLATFORM_PS3_PPU)
	extern char _binary_hknpSpursSolver_elf_start[];
	return _binary_hknpSpursSolver_elf_start;
#else
	return (void*)hkTaskType::HKNP_SOLVER_TASK;
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
