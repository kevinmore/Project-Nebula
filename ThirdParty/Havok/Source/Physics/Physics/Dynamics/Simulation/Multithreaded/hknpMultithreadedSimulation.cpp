/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpMultithreadedSimulation.h>

#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>
#include <Common/Base/Container/String/hkStringBuf.h>

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Dynamics/Action/Manager/hknpActionManager.h>
#include <Physics/Physics/Dynamics/World/CacheManager/hknpCollisionCacheManager.h>
#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpSolverTask.h>
#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpConstraintSetupTask.h>
#include <Physics/Physics/Dynamics/Simulation/Utils/hknpCacheSorter.h>

#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>


hknpMultithreadedSimulation::hknpMultithreadedSimulation()
:	m_nextStage(STAGE_COLLIDE_1)
{
}


void HK_CALL hknpMultithreadedSimulation::printGridSizes( const hknpSimulationThreadContext& tl, hknpCdCacheGrid* grid )
{
	HK_TIMER_BEGIN("DebugPringGridSizes", HK_NULL );
	hknpSpaceSplitter* splitter = tl.m_world->m_spaceSplitter;
	for (int a = 0; a < splitter->getNumCells(); a++ )
	{
		hkStringBuf s;
		for (int b = 0; b < splitter->getNumCells(); b++ )
		{
			int linkIdx = splitter->getLinkIdx( a, b );
			int numElems = grid->m_entries[linkIdx].getNumElements();
			if ( splitter->isLinkFlipped(a, b) )
			{
				numElems = 0;	// only upper diagonal is set
			}

			char buf[256]; hkString::sprintf( buf, "% 4i ", numElems );
			s.append( buf );
		}
		HK_REPORT( s.cString() );
	}
	HK_REPORT( "\n\n" );
	HK_TIMER_END();
}

void hknpMultithreadedSimulation::checkConsistencyOfCollisionCaches( hknpWorld* world, hkInt32 checkFlags, bool allowNewGridsToBeInvalid )
{
	hknpCdCacheGrid* ppuGrid = HK_NULL;
	hknpCdCacheGrid* newPpuGrid = HK_NULL;
	HK_ON_PLATFORM_HAS_SPU(ppuGrid = &world->m_collisionCacheManager->m_cdCachePpuGrid);
	HK_ON_PLATFORM_HAS_SPU(newPpuGrid = &world->m_collisionCacheManager->m_newCdCachePpuGrid);

	checkConsistencyOfCdCacheGrids(
		world, checkFlags,
		world->m_collisionCacheManager->m_cdCacheStream, world->m_collisionCacheManager->m_childCdCacheStream,
		&world->m_collisionCacheManager->m_cdCacheGrid, ppuGrid );

	if( allowNewGridsToBeInvalid )
	{
		checkConsistencyOfCdCacheStream(
			world, checkFlags,
			world->m_collisionCacheManager->m_newCdCacheStream, world->m_collisionCacheManager->m_newChildCdCacheStream );
	}
	else
	{
		checkConsistencyOfCdCacheGrids( world, checkFlags,
			world->m_collisionCacheManager->m_newCdCacheStream, world->m_collisionCacheManager->m_newChildCdCacheStream,
			&world->m_collisionCacheManager->m_newCdCacheGrid, newPpuGrid );
	}
}

void hknpMultithreadedSimulation::checkConsistency( hknpWorld* world, hkInt32 checkFlags )
{
	checkConsistencyOfCollisionCaches( world, checkFlags, false );
}

/// This class collides two grid of caches into:
///   - a single grid of caches
///   - a stream of inactive caches
///   - a stream of caches which crossed a grid cell
///   - a grid of jacobians
void hknpMultithreadedSimulation::createNarrowPhaseTask(
	hknpSimulationContext& simulationContext, hknpCollisionCacheManager& cdCacheManager, hknpSolverData& solverData,
	bool processOnlyNew )
{
	// Create the task
	m_narrowPhaseTask.setAndDontIncrementRefCount( new hknpNarrowPhaseTask(
		simulationContext, cdCacheManager, solverData, processOnlyNew, m_inactiveCdCacheGrid, m_crossGridCdCacheGrid ) );

	// Add it to the task queue
	int multiplicity = simulationContext.getNumCpuThreads();
#if defined(HK_PLATFORM_HAS_SPU)
	multiplicity = simulationContext.getNumSpuThreads();
#endif
	multiplicity = hkMath::min2( multiplicity, m_narrowPhaseTask->m_subTasks.getSize() );
	if (multiplicity)
	{
		hkTask* task = m_narrowPhaseTask;
		simulationContext.m_taskGraph->addTasks( &task, 1, &multiplicity, HK_NULL );
	}

#if defined(HK_PLATFORM_HAS_SPU)
	// Create a separate narrow phase task for PPU caches
	if (!cdCacheManager.m_newCdCachePpuGrid.isEmpty() || (!processOnlyNew && !cdCacheManager.m_cdCachePpuGrid.isEmpty()))
	{
		m_narrowPhaseTaskPpu.setAndDontIncrementRefCount(new hknpNarrowPhaseTask(
			simulationContext, cdCacheManager, solverData, processOnlyNew, m_inactiveCdCachePpuGrid,
			m_crossGridCdCachePpuGrid, true));

		// Add it to the task queue
		multiplicity = hkMath::min2( simulationContext.getNumCpuThreads(), m_narrowPhaseTaskPpu->m_subTasks.getSize() );
		if (multiplicity)
		{
			hkTask* task = m_narrowPhaseTaskPpu;
			simulationContext.m_taskGraph->addTasks( &task, 1, &multiplicity, HK_NULL );
		}
	}
#endif
}


void hknpMultithreadedSimulation::processNarrowPhaseResults(
	hknpSimulationContext& simulationContext, hknpCollisionCacheManager& cdCacheManager, hknpSolverData& solverData,
	bool processOnlyNew )
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& mStream = hkMonitorStream::getInstance();
#endif
	HK_TIMER_BEGIN_LIST2(mStream, "ProcessNarrowPhaseResults", "MergeCacheStreams");

	// Clear the consumed input data (the data is now copied to in hknpNarrowPhaseTask::ThreadOutput)
	// Note that we are not clearing the grids, as this is already done by the collision process.
	hknpSimulationThreadContext* HK_RESTRICT threadContext = simulationContext.getThreadContext();
	hkThreadLocalBlockStreamAllocator* heapAllocator = threadContext->m_heapAllocator;
	if (!processOnlyNew)
	{
		// Clear grid of new caches if we have merged them with the old ones (first round of narrow phase)
		if (!cdCacheManager.m_newCdCacheStream.isEmpty())
		{
			cdCacheManager.m_newCdCacheGrid.clearGrid();
		#if defined(HK_PLATFORM_HAS_SPU)
			cdCacheManager.m_newCdCachePpuGrid.clearGrid();
		#endif
		}
		cdCacheManager.m_cdCacheStream.reset(heapAllocator);
		cdCacheManager.m_childCdCacheStream.reset(heapAllocator);
	}
	cdCacheManager.m_newCdCacheStream.reset(heapAllocator);
	cdCacheManager.m_newChildCdCacheStream.reset(heapAllocator);

	// Create array of narrow phase task pointers
#if !defined(HK_PLATFORM_HAS_SPU)
	hknpNarrowPhaseTask* tasks[] = { m_narrowPhaseTask };
	const int numTasks = 1;
#else
	hknpNarrowPhaseTask* tasks[] = { m_narrowPhaseTask, m_narrowPhaseTaskPpu };
	const int numTasks = m_narrowPhaseTaskPpu ? 2 : 1;
#endif

	// Merge inactive and cross grid pairs into a single stream. The data was owned by hknpNarrowPhaseTask::ThreadOutput
	// and we are making a copy here
	if (!processOnlyNew)
	{
		// Merge inactive caches
		hknpDeactivatedIsland* deactivatedIsland = threadContext->m_world->m_deactivationManager->m_newlyDeactivatedIsland;
		if (deactivatedIsland)
		{
			//HK_TIMER_SPLIT_LIST2( mStream, "MergeInactiveCaches");

			// Merge per-thread inactive cache streams into a single one
			hknpCdCacheStream mergedInactiveChildCdCacheStream;
			mergedInactiveChildCdCacheStream.initBlockStream(heapAllocator);

			// Append threads' deactivated child cache streams
			for (int i = 0; i < numTasks; ++i)
			{
				hknpNarrowPhaseTask* HK_RESTRICT task = tasks[i];
				const int numThreadsInTask = task->m_cdCacheStreamsOut.getSize();
				for (int j = 0; j < numThreadsInTask; j++)
				{
					hknpNarrowPhaseTask::ThreadOutput& threadOutput = task->m_cdCacheStreamsOut[j];
					if (threadOutput.m_isInitialized)
					{
						mergedInactiveChildCdCacheStream.append(heapAllocator, &threadOutput.m_inactiveChildCdCacheStream);
					}
				}
			}

		#if !defined(HK_PLATFORM_HAS_SPU)
			hknpCdCacheGrid* inactiveCdCacheGrids[] = { &m_inactiveCdCacheGrid };
			const int numGrids = 1;
		#else
			hknpCdCacheGrid* inactiveCdCacheGrids[] = { &m_inactiveCdCacheGrid, &m_inactiveCdCachePpuGrid };
			const int numGrids = m_narrowPhaseTaskPpu ? 2 : 1;
		#endif

			// Copy inactive caches into the cache stream of the deactivated island
			hknpCacheSorter::deactivateCdCacheRanges(
				*threadContext, m_narrowPhaseTask->m_sharedData, deactivatedIsland->m_bodyIds, inactiveCdCacheGrids, numGrids,
				mergedInactiveChildCdCacheStream, deactivatedIsland->m_useExclusiveCdCacheRanges,
				cdCacheManager.m_inactiveCdCacheStream, cdCacheManager.m_inactiveChildCdCacheStream,
				deactivatedIsland->m_deletedCaches, deactivatedIsland->m_cdCaches, deactivatedIsland->m_cdChildCaches );

			// Clean up
			mergedInactiveChildCdCacheStream.clear(heapAllocator);
			for (int i = 0; i < numGrids; ++i)
			{
				inactiveCdCacheGrids[i]->m_entries.clear();
			}
		}

		// Copy cross grid caches to the stream of new ones
		//HK_TIMER_SPLIT_LIST2( mStream, "MergeCrossGridCaches");
		//crossGridCvxCacheStreamOut->reset( heapAllocator );
		hknpCacheSorter::mergeCdCacheRanges(*threadContext, m_crossGridCdCacheGrid, cdCacheManager.m_newCdCacheStream);
		m_crossGridCdCacheGrid.m_entries.clear();
	#if defined(HK_PLATFORM_HAS_SPU)
		hknpCacheSorter::mergeCdCacheRanges(*threadContext, m_crossGridCdCachePpuGrid, cdCacheManager.m_newCdCacheStream);
		m_crossGridCdCachePpuGrid.m_entries.clear();
	#endif
	}

	//HK_TIMER_SPLIT_LIST2( mStream, "MergeNormalCaches");

	// Select destination streams based on whether we are in the first or second round of narrow phase
	hknpCdCacheStream* cdCacheStreamOut;
	hknpCdCacheStream* childCdCacheStreamOut;
	if (processOnlyNew)
	{
		cdCacheStreamOut = &cdCacheManager.m_newCdCacheStream;
		childCdCacheStreamOut = &cdCacheManager.m_newChildCdCacheStream;
	}
	else
	{
		cdCacheStreamOut = &cdCacheManager.m_cdCacheStream;
		childCdCacheStreamOut = &cdCacheManager.m_childCdCacheStream;
	}

	// Process output streams
	for (int i = 0; i < numTasks; ++i)
	{
		hknpNarrowPhaseTask* HK_RESTRICT task = tasks[i];
		const int numThreadsInTask = task->m_cdCacheStreamsOut.getSize();

		for (int j = 0; j < numThreadsInTask; j++)
		{
			hknpNarrowPhaseTask::ThreadOutput& threadOutput = task->m_cdCacheStreamsOut[j];
			if (!threadOutput.m_isInitialized)
			{
				continue;
			}

			// Append thread output streams to main ones. The caches in these streams are owned by either m_cdCacheGrid
			// or m_newCdCacheGrid in the cache manager.
			cdCacheStreamOut->append(heapAllocator, &threadOutput.m_cdCacheStream);
			childCdCacheStreamOut->append(heapAllocator, &threadOutput.m_childCdCacheStream);

			// Append cross grid caches to the stream of new child caches
			if (!processOnlyNew)
			{
				cdCacheManager.m_newChildCdCacheStream.append(heapAllocator, &threadOutput.m_crossChildCdCacheStream);
			}
			threadOutput.exitData(*threadContext);
		}
	}

	// Release tasks
	m_narrowPhaseTask = HK_NULL;
	HK_ON_PLATFORM_HAS_SPU(m_narrowPhaseTaskPpu = HK_NULL);

	//HK_ON_DEBUG(printGridSizes(tl, cdCacheGridOut));

	HK_TIMER_END_LIST2(mStream);	// "ProcessNarrowPhaseResults"
}


hknpSolverData* hknpMultithreadedSimulation::collideStage1(
	const hknpSimulationThreadContext& threadContext, hknpSimulationContext& simulationContext )
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	hknpWorld* world = threadContext.m_world;

	// Allocate the output structure
	{
		HK_TIMER_BEGIN2( timerStream, "CreateSolverData", HK_NULL );
		const int numThreads = simulationContext.getNumThreads();
		const int numLinks = world->m_spaceSplitter->getNumLinks();
		const int numCells = world->m_spaceSplitter->getNumCells();
		m_solverData = new hknpSolverData( threadContext.m_tempAllocator, numThreads, numCells, numLinks );
		HK_TIMER_END2( timerStream );
	}

	// Process actions
	if( !world->m_actionManager->m_activeActions.isEmpty() )
	{
		HK_TIMER_BEGIN2( timerStream, "ApplyActions", HK_NULL );
		const int currentThreadId = simulationContext.getCurrentThreadNumber();
		hknpCdPairWriter pairWriter;
		pairWriter.setToEndOfStream(
			threadContext.m_tempAllocator, &m_solverData->m_threadData[currentThreadId].m_activePairStream );
		world->m_actionManager->executeActions(
			threadContext, world->m_solverInfo, &pairWriter, world->m_deactivationManager->m_newlyDeactivatedIsland );
		pairWriter.finalize();
		HK_TIMER_END2( timerStream );
	}

	// Create broad phase tasks
	{
		HK_TIMER_BEGIN2( timerStream, "CreateBroadPhaseTasks", HK_NULL );
		m_newPairsStream.initBlockStream( threadContext.m_tempAllocator );
		world->m_broadPhase->buildTaskGraph( world, &simulationContext, &m_newPairsStream, simulationContext.m_taskGraph );
		HK_TIMER_END2( timerStream );
	}

	// Create constraint gathering task
	int gatherConstraintsTaskId = -1;
	if( world->m_constraintAtomSolver->getNumConstraints() > 0 )
	{
		m_gatherConstraintsTask.setAndDontIncrementRefCount( new hknpGatherConstraintsTask( simulationContext ) );
		simulationContext.m_taskGraph->addTasks( (hkTask**)&m_gatherConstraintsTask, 1, HK_NULL, &gatherConstraintsTaskId );
	}

	// Create first round of narrow phase tasks
	// This will process all collisions and merge the previous newCvxCacheStream into the main cvxCacheStream/cvxCacheGrid
	{
		hknpCollisionCacheManager* collisionCacheManager = world->m_collisionCacheManager;
		createNarrowPhaseTask( simulationContext, *collisionCacheManager, *m_solverData, false );
	}

	// Create constraint setup tasks, dependent on the results of the gather constraints task.
	// We don't yet know if or how many of these we will need. We create enough for the worst case.
	if( m_gatherConstraintsTask != HK_NULL )
	{
		m_constraintSetupTask.setAndDontIncrementRefCount( new hknpConstraintSetupTask( simulationContext, *m_solverData ) );
		m_constraintSetupTask->m_constraintStates = &m_gatherConstraintsTask->m_constraintStates;
		m_constraintSetupTask->m_subTasks = &m_gatherConstraintsTask->m_subTasks;

		// This task runs only on CPU threads
		const int numCpuThreads = simulationContext.getNumCpuThreads();
		const int multiplicity = hkMath::min2( numCpuThreads, world->m_constraintAtomSolver->getNumConstraints() );

		if (multiplicity)
		{
			// Add task
			hkTask* task = m_constraintSetupTask;
			int taskId;
			simulationContext.m_taskGraph->addTasks(&task, 1, &multiplicity, &taskId);

			// Add dependency with gather constraints task
			hkTaskGraph::Dependency dependency;
			dependency.m_parentId = gatherConstraintsTaskId;
			dependency.m_childId = taskId;
			simulationContext.m_taskGraph->addDependencies(&dependency, 1);
		}
	}

	m_nextStage = STAGE_COLLIDE_2;

	return HK_NULL;
}


hknpSolverData* hknpMultithreadedSimulation::collideStage2(
	const hknpSimulationThreadContext& threadContext, hknpSimulationContext& simulationContext )
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	hknpWorld* world = threadContext.m_world;
	hknpCollisionCacheManager* cdCacheManager = world->m_collisionCacheManager;

	processNarrowPhaseResults(simulationContext, *cdCacheManager, *m_solverData, false);
	if (world->areConsistencyChecksEnabled())
	{
		HK_TIME_CODE_BLOCK2(timerStream, "CheckConsistency", HK_NULL);
		checkConsistencyOfCollisionCaches(world, hknpSimulation::CHECK_ALL, true);
	}


	//HK_TIMER_SPLIT_LIST2( mStream, "Debug");	hknpGridUtils::showEntries( this, &m_unifiedAgent->m_cvxCacheGrid );

	// Run functions which would generate new caches (adds the new caches in m_unifiedAgent->m_newCvxCacheStream)
	if( world->m_deactivationEnabled )
	{
		// Put all single threaded commands into grid entry 0
		threadContext.beginCommands(0);

		// Find activated bodies & activate
		hknpDeactivationManager* deactivationManager = world->m_deactivationManager;
		{
			HK_TIME_CODE_BLOCK2(timerStream, "ActivateIslands", HK_NULL);
			hkBlockStream<hknpBodyIdPair>::Reader newPairsReader;
			newPairsReader.setToStartOfStream( &m_newPairsStream );
			deactivationManager->markIslandsForActivationFromNewPairs( &newPairsReader );

			deactivationManager->activateMarkedIslands(false);	// don't clear activated island list yet
			deactivationManager->moveActivatedCaches(
				threadContext, &cdCacheManager->m_newCdCacheStream, &cdCacheManager->m_newChildCdCacheStream, &m_newPairsStream );
		}

		threadContext.endCommands(0);

		// Create the 2nd round of constraint setup tasks, for any newly activated pairs
		if( world->m_constraintAtomSolver->getNumConstraints() > 0 &&
			!world->m_deactivationManager->m_islandsMarkedForActivation.isEmpty() )
		{
			HK_TIMER_BEGIN2( timerStream, "AddTasksForActivatedConstraints", HK_NULL );
			world->m_deactivationManager->sortIslandsMarkedForActivation();
			addTasksForActivatedConstraints( threadContext, simulationContext );
			HK_TIMER_END2( timerStream );
		}

		world->m_deactivationManager->clearIslandsMarkedForDeactivation();
	}

	// Add new body pairs
	{
		if( !cdCacheManager->m_newUserCollisionPairs.isEmpty() )
		{
			HK_TIMER_BEGIN2(timerStream, "AppendNewUserPairs", HK_NULL);
			cdCacheManager->filterDeletedPairs(world, cdCacheManager->m_newUserCollisionPairs);
			cdCacheManager->appendPairsToStream( threadContext, &m_newPairsStream, cdCacheManager->m_newUserCollisionPairs.begin(), cdCacheManager->m_newUserCollisionPairs.getSize() );
			HK_MONITOR_ADD_VALUE("NumNewUserPairs", (float)cdCacheManager->m_newUserCollisionPairs.getSize(), HK_MONITOR_TYPE_INT);
			cdCacheManager->m_newUserCollisionPairs.clear();
			HK_TIMER_END2(timerStream);
		}

		if( !m_newPairsStream.isEmpty() )
		{
			HK_TIMER_BEGIN2(timerStream, "AppendNewBpPairs", HK_NULL);
			hkBlockStream<hknpBodyIdPair>::Reader newPairReader;
			newPairReader.setToStartOfStream( &m_newPairsStream );
			cdCacheManager->addNewPairs( threadContext, &newPairReader, m_newPairsStream.getTotalNumElems() );
			HK_MONITOR_ADD_VALUE("NumNewPairs", (float)m_newPairsStream.getTotalNumElems(), HK_MONITOR_TYPE_INT);
			HK_TIMER_END2(timerStream);
		}

		m_newPairsStream.clear( threadContext.m_tempAllocator );
	}

	// Append the new user collision cache streams
	if( !cdCacheManager->m_newUserCdCacheStream.isEmpty() )
	{
		cdCacheManager->m_newCdCacheStream.append(threadContext.m_heapAllocator, &cdCacheManager->m_newUserCdCacheStream);
		cdCacheManager->m_newChildCdCacheStream.append(threadContext.m_heapAllocator, &cdCacheManager->m_newUserChildCdCacheStream);
		cdCacheManager->m_newUserCdCacheStream.reset(threadContext.m_heapAllocator);
		cdCacheManager->m_newUserChildCdCacheStream.reset(threadContext.m_heapAllocator);
	}

	// Create the 2nd round of narrow phase tasks.
	// If there are collision caches produced by the broad phase or collision caches which changed the grid cell,
	// we need to rerun collision detection on those caches.
	if( !cdCacheManager->m_newCdCacheStream.isEmpty() )
	{
		// Sort caches into grid cells
		HK_TIMER_BEGIN2( timerStream, "SortCaches", HK_NULL );
	#if !defined(HK_PLATFORM_HAS_SPU)
		hknpCacheSorter::sortCaches(
			threadContext, &cdCacheManager->m_newCdCacheStream, &cdCacheManager->m_newCdCacheGrid, HK_NULL);
	#else
		hknpCacheSorter::sortCaches(
			threadContext, &cdCacheManager->m_newCdCacheStream, &cdCacheManager->m_newCdCacheGrid,
			&cdCacheManager->m_newCdCachePpuGrid);
	#endif
		HK_TIMER_END2( timerStream );

		// Create narrow phase tasks.
		// Don't merge with the previous collision detection caches yet, this will be done in the next frame.
		createNarrowPhaseTask( simulationContext, *cdCacheManager, *m_solverData, true );
	}

	m_nextStage = STAGE_COLLIDE_3;

	return HK_NULL;
}

hknpSolverData* hknpMultithreadedSimulation::collideStage3(
	const hknpSimulationThreadContext& threadContext, hknpSimulationContext& simulationContext )
{
	hknpWorld* world = threadContext.m_world;
	hknpCollisionCacheManager* cdCacheManager = world->m_collisionCacheManager;

	// Any constraint tasks will have been processed by now, so can be freed
	{
		m_constraintSetupTask = HK_NULL;
		m_gatherConstraintsTask = HK_NULL;
	}

	// If we had a 2nd round of narrow phase, process those results
	if( m_narrowPhaseTask )
	{
		processNarrowPhaseResults(simulationContext, *cdCacheManager, *m_solverData, true);
	}

#if defined(HK_PLATFORM_HAS_SPU)
	// Merge grids of jacobians for PPU caches into normal grids
	{
		hknpConstraintSolverJacobianGrid* grids[] = { &m_solverData->m_jacMovingGrid, &m_solverData->m_jacFixedGrid };
		hknpConstraintSolverJacobianGrid* gridsPpu[] =
			{ &m_solverData->m_jacMovingPpuGrid, &m_solverData->m_jacFixedPpuGrid };
		hknpConstraintSolverJacobianStream* streams[] =
			{ &m_solverData->m_threadData[0].m_jacMovingStream, &m_solverData->m_threadData[0].m_jacFixedStream };

		for (int i = 0; i < 2; ++i)
		{
			// Use the jacobian stream for the ranges too
			hknpConstraintSolverJacobianWriter writer;
			writer.setToEndOfStream(threadContext.m_tempAllocator, streams[i]);

			hknpConstraintSolverJacobianGrid* HK_RESTRICT gridPpu = gridsPpu[i];
			hknpConstraintSolverJacobianGrid* HK_RESTRICT grid = grids[i];
			for (int entryIndex = 0; entryIndex < gridPpu->getSize(); ++entryIndex)
			{
				hknpConstraintSolverJacobianRange2* range = &(*gridPpu)[entryIndex];
				while (range && !range->isEmpty())
				{
					hknpConstraintSolverJacobianRange2* next = (hknpConstraintSolverJacobianRange2*) range->m_next;
					range->m_next = HK_NULL;
					grid->addRange(writer, entryIndex, *range);
					range = next;
				}
			}
			writer.finalize();
		}
	}
#endif

	if( world->areConsistencyChecksEnabled() )
	{
		checkConsistencyOfCollisionCaches(world, hknpSimulation::CHECK_ALL, false);
	}

	m_nextStage = STAGE_SOLVE_1;

	return m_solverData;
}

void hknpMultithreadedSimulation::collide( hknpSimulationContext& simulationContext, hknpSolverData*& solverDataOut )
{
	HK_ASSERT( 0x707d69cb, simulationContext.m_taskGraph->getNumTasks() == 0 );

	hknpSimulationThreadContext& threadContext = *simulationContext.getThreadContext();

	hknpSolverData* solverData = HK_NULL;
	switch (m_nextStage)
	{
		case STAGE_COLLIDE_1:
		{
			solverData = collideStage1(threadContext, simulationContext);
			break;
		}

		case STAGE_COLLIDE_2:
		{
			solverData = collideStage2(threadContext, simulationContext);

			// If there are no pending tasks we will fall through to stage 3.
			if (simulationContext.m_taskGraph->getNumTasks())
			{
				break;
			}
		}

		case STAGE_COLLIDE_3:
		{
			solverData = collideStage3(threadContext, simulationContext);
			break;
		}

		default:
		{
			HK_ASSERT2(0x2d174f5f, 0, "Collide step cannot be executed in the current simulation state");
		}
	}

	if( solverData )
	{
		solverDataOut = solverData;
	}
}

void hknpMultithreadedSimulation::addTasksForActivatedConstraints(
	const hknpSimulationThreadContext& threadContext, hknpSimulationContext& simulationContext )
{
	hknpWorld* world = threadContext.m_world;

	// Update the subtasks stored in the existing gather constraints task
	
	{
		HK_ASSERT( 0x717d69ca, m_gatherConstraintsTask != HK_NULL );
		HK_ASSERT( 0x717d69cb, &m_constraintSetupTask->m_simulationContext == &simulationContext );

		// Regroup any newly activated constraints
		hknpConstraintAtomSolverSetup::ConstraintStates& states = m_gatherConstraintsTask->m_constraintStates;
		states.regroupReactivatedConstraints(
			world, world->m_constraintAtomSolver->getConstraints(),
			world->m_deactivationManager->m_islandsMarkedForActivation );

		// Create subtasks for the reactivated groups
		m_gatherConstraintsTask->m_subTasks.create(
			world->m_constraintAtomSolver->getConstraints(), states,
			states.getFirstReactivatedIndex(), states.getLastReactivatedIndex() );
	}

	// Check if we need to do anything
	const int multiplicity = hkMath::min2( simulationContext.getNumCpuThreads(), m_gatherConstraintsTask->m_subTasks.getSize() );
	if( multiplicity > 0 )
	{
		// Reset the existing constraint setup task
		{
			HK_ASSERT( 0x717d69cc, m_constraintSetupTask != HK_NULL );
			HK_ASSERT( 0x717d69cd, &m_constraintSetupTask->m_simulationContext == &simulationContext );
			HK_ASSERT( 0x717d69ce, &m_constraintSetupTask->m_solverData == m_solverData );

			m_constraintSetupTask->m_currentSubTaskIndex = 0;
		}

		// Add it to the task graph
		hkTask* task = m_constraintSetupTask;
		simulationContext.m_taskGraph->addTasks( &task, 1, &multiplicity );
	}
	else
	{
		// We don't need these task any more, so free them
		m_constraintSetupTask = HK_NULL;
		m_gatherConstraintsTask = HK_NULL;
	}
}

void hknpMultithreadedSimulation::solveStage1(
	hknpSimulationThreadContext* threadContext, hknpSimulationContext& simulationContext, hknpSolverData* solverData )
{
	// Create the solver task
	m_solverTask.setAndDontIncrementRefCount( new hknpSolverTask( simulationContext, solverData, &m_solverTaskQueue ) );

	// Add it to the task graph
	const int multiplicity = m_solverTask->m_numThreadsToRun;
	hkTask* task = m_solverTask;
	simulationContext.m_taskGraph->addTasks( &task, 1, &multiplicity, HK_NULL );

	m_nextStage = STAGE_SOLVE_2;
}

void hknpMultithreadedSimulation::solveStage2(
	hknpSimulationThreadContext* threadContext, hknpSimulationContext& simulationContext, hknpSolverData* solverDataIn )
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	hknpSolverData* solverData = static_cast<hknpSolverData*>(solverDataIn);
	hknpWorld* world = threadContext->m_world;
	world->m_bodyManager.resetPreviousAabbsAndFlagsOfStaticBodies();

#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	{
		HK_TIMER_BEGIN2( timerStream, "Statistics", HK_NULL );

		int numConstraintJac = 0;
		int numMovingContactJac = 0;
		int numFixedContactJac = 0;
		for( int i = 0; i < solverData->m_threadData.getSize(); i++ )
		{
			const hknpSolverData::ThreadData& td = solverData->m_threadData[i];
			numConstraintJac += td.m_jacConstraintsStream.getTotalNumElems();
			numMovingContactJac += HKNP_NUM_MX_JACOBIANS * td.m_jacMovingStream.getTotalNumElems();
			numFixedContactJac  += HKNP_NUM_MX_JACOBIANS * td.m_jacFixedStream.getTotalNumElems();
		}

		int numActiveBodies = world->m_bodyManager.getNumActiveBodies();
		hkBlockStreamAllocator* tempAlloc = threadContext->m_tempAllocator->m_blockStreamAllocator;
		hkBlockStreamAllocator* heapAlloc = threadContext->m_heapAllocator->m_blockStreamAllocator;

		HK_MONITOR_ADD_VALUE( "NumActiveBodies",		hkFloat32(numActiveBodies), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "NumActiveConstraints",	hkFloat32(numConstraintJac), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "NumActiveMovingContacts",hkFloat32(numMovingContactJac), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "NumActiveFixedContacts",	hkFloat32(numFixedContactJac), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "StepLocalBytesUsed",		hkFloat32(tempAlloc->getBytesUsed()), HK_MONITOR_TYPE_INT );
		//HK_MONITOR_ADD_VALUE( "StepLocalBytesPeakUsed",	hkFloat32(tempAlloc->getMaxBytesUsed()), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "StepLocalBytesUnused",	hkFloat32(tempAlloc->getCapacity() - tempAlloc->getBytesUsed()), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "HeapBytesUsed",			hkFloat32(heapAlloc->getBytesUsed()), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "HeapBytesPeakUsed",		hkFloat32(heapAlloc->getMaxBytesUsed()), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "HeapBytesPeakUnused",	hkFloat32(heapAlloc->getCapacity() - heapAlloc->getMaxBytesUsed()), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "InactiveCdCacheBytesUsed", hkFloat32(world->m_collisionCacheManager->m_inactiveCdCacheStream.getTotalBytesAllocated()), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "InactiveChildCdCacheBytesUsed", hkFloat32(world->m_collisionCacheManager->m_inactiveChildCdCacheStream.getTotalBytesAllocated()), HK_MONITOR_TYPE_INT );

		HK_TIMER_END2( timerStream );
	}
#endif

	HK_TIMER_BEGIN2( timerStream, "PostSolve", HK_NULL );

	// Deactivation
	if( world->isDeactivationEnabled() )
	{
		hknpDeactivationStepInfo* deactivationStepInfo = m_solverTask->m_deactivationStepInfo;
		hknpIdxRangeGrid& cellIdxToGlobalSolverId = m_solverTask->m_cellIdxToGlobalSolverId;

		// Put all single threaded commands into grid entry 0
		threadContext->beginCommands( 0 );

#if HKNP_ENABLE_SOLVER_PARALLEL_TASKS == 0
		{
			HK_TIME_CODE_BLOCK2(timerStream, "GarbageCollectInactive", HK_NULL);
			world->m_deactivationManager->garbageCollectInactiveCaches(
				*threadContext, &world->m_collisionCacheManager->m_inactiveCdCacheStream, &world->m_collisionCacheManager->m_inactiveChildCdCacheStream );
		}
#endif

		HK_TIMER_BEGIN_LIST2( timerStream, "Deactivation", "MergeBitFields" );
		hkBitField solverVelOkToDeactivate;
		deactivationStepInfo->combineActivityBitFields( solverVelOkToDeactivate );

#if HKNP_ENABLE_SOLVER_PARALLEL_TASKS == 0
		HK_TIMER_SPLIT_LIST2( timerStream, "UnionFind");
		hknpAddActiveBodyPairsProcess( solverData, deactivationStepInfo, &cellIdxToGlobalSolverId );
#endif

		HK_TIMER_SPLIT_LIST2( timerStream, "FindIsland");
		hknpDeactivatedIsland* deactivatedIsland;
		deactivatedIsland = deactivationStepInfo->createDeactivatedIsland(
			*world, solverVelOkToDeactivate, &cellIdxToGlobalSolverId, true );

		HK_TIMER_SPLIT_LIST2( timerStream, "Deactivate" );
		world->m_deactivationManager->deactivateIsland( *threadContext, deactivatedIsland );
		HK_TIMER_END_LIST2( timerStream ); // Deactivation

		threadContext->endCommands( 0 );
	}

	// Update space splitter
	{
		HK_TIMER_BEGIN2( timerStream, "UpdateSpaceSplitter", HK_NULL );
		world->m_spaceSplitter->applyThreadData(
			m_solverTask->m_spaceSplitterData, simulationContext.getNumThreads(), &world->m_intSpaceUtil );
		HK_TIMER_END2( timerStream );
	}

	m_solverTask = HK_NULL;

	m_nextStage = STAGE_COLLIDE_1;

	HK_TIMER_END2( timerStream );	// "PostSolve"
}


void hknpMultithreadedSimulation::solve( hknpSimulationContext& simulationContext, hknpSolverData* solverData )
{
	HK_ASSERT( 0x707d69cc, simulationContext.m_taskGraph->getNumTasks() == 0 );

	hknpSimulationThreadContext* threadContext = simulationContext.getThreadContext();

	switch( m_nextStage )
	{
		case STAGE_SOLVE_1:
		{
			solveStage1( threadContext, simulationContext, solverData );
			break;
		}

		case STAGE_SOLVE_2:
		{
			solveStage2( threadContext, simulationContext, solverData );
			break;
		}

		default:
		{
			HK_ASSERT2( 0x2d174f5f, 0, "Solve step cannot be executed in the current simulation state" );
		}
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
