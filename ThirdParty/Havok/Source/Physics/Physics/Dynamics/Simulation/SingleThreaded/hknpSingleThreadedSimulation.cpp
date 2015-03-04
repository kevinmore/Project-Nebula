/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Simulation/SingleThreaded/hknpSingleThreadedSimulation.h>

#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>

#include <Physics/Physics/Collide/NarrowPhase/hknpCollidePipeline.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>
#include <Physics/Physics/Dynamics/World/CacheManager/hknpCollisionCacheManager.h>
#include <Physics/Physics/Dynamics/World/Deactivation/CdCacheFilter/hknpDeactiveCdCacheFilter.h>
#include <Physics/Physics/Dynamics/Simulation/Utils/hknpCacheSorter.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverUtil.h>
#include <Physics/Physics/Dynamics/Solver/MxJacobianSorter/hknpMxJacobianSorter.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>

#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>

#include <Physics/Physics/Dynamics/Action/Manager/hknpActionManager.h>


namespace
{
	void hknpSingleThreadedSimulation_collideInternal(
		const hknpSimulationThreadContext& tl,
		hknpCollisionCacheManager* agent,
		hknpCdPairStream* activePairStreamOut, hknpLiveJacobianInfoStream* liveJacInfoStream,
		hknpConstraintSolverJacobianStream* jacMovingStreamOut, hknpConstraintSolverJacobianStream* jacFixedStreamOut )
	{
		if( agent->m_cdCacheStream.isEmpty() && agent->m_newCdCacheStream.isEmpty() )
		{
			return;
		}

		hknpWorld* world = tl.m_world;
		int currentLinkIndex = 0;
		HK_ASSERT2(0xf0dfc1c5, world->m_spaceSplitter->getNumCells() == 1, "You cannot run single threaded composite collisions using multiple grid cells");

		// remember all pairs for the deactivation
		hknpCdPairWriter activePairWriter; activePairWriter.setToEndOfStream( tl.m_tempAllocator, activePairStreamOut );
		hknpLiveJacobianInfoWriter liveJacInfoWriter; liveJacInfoWriter.setToEndOfStream( tl.m_tempAllocator, liveJacInfoStream );

		// writer for CdCaches
		hknpCdCacheStream updateCdCacheStream; updateCdCacheStream.initBlockStream( tl.m_heapAllocator );	// temp stream to put our dest bodyMesh caches
		hknpCdCacheWriter cdCacheWriter; cdCacheWriter.setToStartOfStream( tl.m_heapAllocator, &updateCdCacheStream );

		hknpCdCacheStream updatedChildCdCacheStream; updatedChildCdCacheStream.initBlockStream(tl.m_heapAllocator); // temp stream to put cvx caches

		// writer for child caches
		hknpCdCacheStream	  inactiveCdCacheStream;  inactiveCdCacheStream.initBlockStream( tl.m_heapAllocator );
		hknpCdCacheStream inactiveChildCdCacheStream; inactiveChildCdCacheStream.initBlockStream( tl.m_heapAllocator );

		hknpInternalCollideSharedData collideSharedData( world );

		{
			hknpCdCacheConsumer cdCacheConsumer;  cdCacheConsumer. setToStartOfStream( tl.m_heapAllocator, &agent->m_cdCacheStream,    HK_NULL /*stream ppu*/ );
			hknpCdCacheConsumer cdCacheConsumer2; cdCacheConsumer2.setToStartOfStream( tl.m_heapAllocator, &agent->m_newCdCacheStream, HK_NULL /*stream ppu*/ );

			hknpConstraintSolverJacobianWriter jacMovingWriter; jacMovingWriter.setToEndOfStream( tl.m_tempAllocator, jacMovingStreamOut );
			hknpConstraintSolverJacobianWriter fixedJacWriter;  fixedJacWriter. setToEndOfStream( tl.m_tempAllocator, jacFixedStreamOut );

			hknpCdCacheWriter inactiveCdCacheWriter; inactiveCdCacheWriter. setToStartOfStream( tl.m_heapAllocator, &inactiveCdCacheStream );
			hknpCdCacheWriter inactiveChildCdCacheWriter; inactiveChildCdCacheWriter.setToStartOfStream( tl.m_heapAllocator, &inactiveChildCdCacheStream );

			hknpCdCacheWriter childCdCacheWriter; childCdCacheWriter.setToStartOfStream( tl.m_heapAllocator, &updatedChildCdCacheStream );

			hknpMxJacobianSorter jacMovingMxSorter(&jacMovingWriter);
			hknpMxJacobianSorter fixedJacMxSorter(&fixedJacWriter);

			hknpCollidePipeline::mergeAndCollide2Streams(
				tl, collideSharedData, currentLinkIndex,
				cdCacheConsumer,  agent->m_childCdCacheStream, HK_NULL,
				&cdCacheConsumer2, &agent->m_newChildCdCacheStream, HK_NULL,
				cdCacheWriter, childCdCacheWriter,
				&inactiveCdCacheWriter, &inactiveChildCdCacheWriter,
				HK_NULL, HK_NULL,		// no cross grid for single threaded simulation
				activePairWriter, &liveJacInfoWriter,
				&jacMovingMxSorter, &fixedJacMxSorter );

			childCdCacheWriter.finalize();

			inactiveCdCacheWriter.finalize();
			inactiveChildCdCacheWriter.finalize();

			jacMovingWriter.finalize();
			fixedJacWriter.finalize();
			cdCacheWriter.finalize();
			liveJacInfoWriter.finalize();
			activePairWriter.finalize();

		}
		agent->m_childCdCacheStream.clearAndSteal( tl.m_heapAllocator, &updatedChildCdCacheStream );
		updatedChildCdCacheStream.clear( tl.m_heapAllocator );

		agent->m_cdCacheStream.clearAndSteal( tl.m_heapAllocator, &updateCdCacheStream );
		updateCdCacheStream.clear( tl.m_heapAllocator );

		agent->m_newCdCacheStream.reset( tl.m_heapAllocator );
		agent->m_newChildCdCacheStream.reset( tl.m_heapAllocator );

		//
		//	Deactivate
		//
		hknpDeactivationManager* deactMgr = world->m_deactivationManager;
		hknpDeactivatedIsland* island = deactMgr->m_newlyDeactivatedIsland;
		if (island)
		{
			hknpCdCacheReader reader; reader.setToStartOfStream( &inactiveCdCacheStream );
			hknpCdCacheWriter islandCdCacheWriter; islandCdCacheWriter.setToEndOfStream(tl.m_heapAllocator, &agent->m_inactiveCdCacheStream );
			hknpCdCacheWriter islandChildCdCacheWriter; islandChildCdCacheWriter.setToEndOfStream(tl.m_heapAllocator, &agent->m_inactiveChildCdCacheStream);

			if (island->m_useExclusiveCdCacheRanges)
			{
				island->m_cdCaches.setStartPointExclusive(&islandCdCacheWriter);
				island->m_cdChildCaches.setStartPointExclusive(&islandChildCdCacheWriter);
			}
			else
			{
				island->m_cdCaches.setStartPoint(&islandCdCacheWriter);
				island->m_cdChildCaches.setStartPoint(&islandChildCdCacheWriter);
			}

			{
				hknpDeactiveCdCacheFilter* filter = world->m_deactiveCdCacheFilter;
				filter->deactivateCaches(tl, collideSharedData, island->m_bodyIds, reader, inactiveChildCdCacheStream, islandCdCacheWriter, islandChildCdCacheWriter, island->m_deletedCaches );
			}
			if (island->m_useExclusiveCdCacheRanges)
			{
				island->m_cdCaches.setEndPointExclusive(&islandCdCacheWriter);
				island->m_cdChildCaches.setEndPointExclusive(&islandChildCdCacheWriter);
			}
			else
			{
				island->m_cdCaches.setEndPoint(&islandCdCacheWriter);
				island->m_cdChildCaches.setEndPoint(&islandChildCdCacheWriter);
			}
			islandCdCacheWriter.finalize();
			islandChildCdCacheWriter.finalize();
		}
		else
		{
			HK_ASSERT( 0xf03dfdef, inactiveCdCacheStream.isEmpty() );
		}

		inactiveChildCdCacheStream.clear( tl.m_heapAllocator );
		inactiveCdCacheStream.clear( tl.m_heapAllocator );
	}

}	// anonymous namespace


void hknpSingleThreadedSimulation::collide( hknpSimulationContext& simulationContext, hknpSolverData*& solverDataOut )
{
	const hknpSimulationThreadContext& tl = *simulationContext.getThreadContext();

	hknpWorld* world = tl.m_world;
	tl.beginCommands( 0 );

	//
	//	Find new active collision pairs
	//		- broad phase
	//		- activated islands
	//
	HK_TIMER_BEGIN("BroadPhase", HK_NULL );
	{
		hkBlockStream<hknpBodyIdPair> newPairsStream;
		newPairsStream.initBlockStream( tl.m_tempAllocator );

		//
		// Update and query the broad phase for new pairs
		//
		{
			world->m_bodyManager.prefetchActiveBodies();

			world->m_broadPhase->update( world->m_bodyManager.accessBodyBuffer(), hknpBroadPhase::UPDATE_DYNAMIC );


			hkBlockStream<hknpBodyIdPair>::Writer pairWriter;
			pairWriter.setToEndOfStream( tl.m_tempAllocator, &newPairsStream );
			world->m_broadPhase->findNewPairs(
				world->m_bodyManager.accessBodyBuffer(), world->m_bodyManager.getPreviousAabbs().begin(), &pairWriter);
			pairWriter.finalize();

			// update the old AABBs to the current AABBs. This ensures that only new pairs will be reported.
			world->m_bodyManager.updatePreviousAabbsOfActiveBodies();
		}
		//world->m_broadPhase->debugDisplay( world->m_intSpaceUtil, world->m_bodyManager.getBroadPhaseBodies().begin() );



		hknpCollisionCacheManager* cdCacheMgr = world->m_collisionCacheManager;

		if (world->m_deactivationEnabled)
		{
			HK_TIMER_SPLIT_LIST("ActivateIslands");
			hknpDeactivationManager* mgr = world->m_deactivationManager;

			// iterate over all new pairs and if an inactive body is found which is not marked for activation, mark island for activation
			hkBlockStream<hknpBodyIdPair>::Reader newPairReader;
			newPairReader.setToStartOfStream( &newPairsStream );
			mgr->markIslandsForActivationFromNewPairs( &newPairReader );

			// activate islands. This will copy the caches from the island to the agents
			// and if a cache was deleted it will copy the bodyIdPair to the newPairs output stream
			mgr->activateMarkedIslands();
			mgr->moveActivatedCaches(tl, &cdCacheMgr->m_newCdCacheStream, &cdCacheMgr->m_newChildCdCacheStream, &newPairsStream );
		}

		HK_TIMER_SPLIT_LIST("AppendNewPairs");

		// Append the new user collision pairs to the stream
		if ( !cdCacheMgr->m_newUserCollisionPairs.isEmpty() )
		{
			cdCacheMgr->filterDeletedPairs( world, cdCacheMgr->m_newUserCollisionPairs );
			hknpCollisionCacheManager::appendPairsToStream(tl, &newPairsStream,  cdCacheMgr->m_newUserCollisionPairs.begin(), cdCacheMgr->m_newUserCollisionPairs.getSize());
			cdCacheMgr->m_newUserCollisionPairs.clear();
		}

		//
		// from new broad phase pairs and from activation pairs: turn bodyId pairs into proper collision caches in the collide agent
		//
		hkBlockStream<hknpBodyIdPair>::Reader newPairReader;
		newPairReader.setToStartOfStream( &newPairsStream );
		cdCacheMgr->addNewPairs( tl, &newPairReader, newPairsStream.getTotalNumElems() );
		newPairsStream.clear( tl.m_tempAllocator );
	}

	//
	//	Run narrow phase collision detection
	//
	hknpSingleThreadedSolverData* solverData = new hknpSingleThreadedSolverData( tl.m_tempAllocator );
	hknpSolverData::ThreadData* solverThreadData = &solverData->m_threadData[0];
	{
		//
		// Handle collisions
		//
		{
			HK_TIMER_SPLIT_LIST("NarrowPhase");
			hknpCollisionCacheManager* agent = world->m_collisionCacheManager;

			// Append the new user cd cache streams
			if (!agent->m_newUserCdCacheStream.isEmpty())
			{
				agent->m_newCdCacheStream.append(tl.m_heapAllocator, &agent->m_newUserCdCacheStream);
				agent->m_newChildCdCacheStream.append(tl.m_heapAllocator, &agent->m_newUserChildCdCacheStream);
				agent->m_newUserCdCacheStream.reset(tl.m_heapAllocator);
				agent->m_newUserChildCdCacheStream.reset(tl.m_heapAllocator);
			}

			// sort new pairs
			if (!agent->m_newCdCacheStream.isEmpty())
			{
				HK_TIMER_BEGIN("SortNewCaches", HK_NULL);
				hknpCacheSorter::sortCaches( tl, &agent->m_newCdCacheStream, HK_NULL, HK_NULL );
				HK_TIMER_END();
			}

			//
			//	Perform collide
			//
			hknpSingleThreadedSimulation_collideInternal(tl, agent, &solverThreadData->m_activePairStream, &solverThreadData->m_liveJacInfoStream, &solverThreadData->m_jacMovingStream, &solverThreadData->m_jacFixedStream);

			const int cell = 0;
			solverData->m_jacMovingGrid[cell].initRange( hknpConstraintSolverType::CONTACT_CONSTRAINT_SOLVER, hknpConstraintSolverJacobianRange2::LIVE_JACOBIANS | hknpConstraintSolverJacobianRange2::SOLVER_TEMPS);
			solverData->m_jacMovingGrid[cell].setEntireStream( &solverThreadData->m_jacMovingStream );
			solverData->m_jacFixedGrid[cell].initRange( hknpConstraintSolverType::CONTACT_CONSTRAINT_SOLVER, hknpConstraintSolverJacobianRange2::LIVE_JACOBIANS | hknpConstraintSolverJacobianRange2::SOLVER_TEMPS);
			solverData->m_jacFixedGrid[cell].setEntireStream( &solverThreadData->m_jacFixedStream );
			if ( world->areConsistencyChecksEnabled() )
			{
				checkConsistencyOfCdCacheStream(world, hknpSimulation::CHECK_ALL, agent->m_cdCacheStream, agent->m_childCdCacheStream );
			}
		}
	}

	//
	//	Actions
	//
	{
		hknpCdPairWriter pairWriter; pairWriter.setToEndOfStream( tl.m_tempAllocator, &solverThreadData->m_activePairStream  );
		world->m_actionManager->executeActions( tl, world->m_solverInfo, &pairWriter, world->m_deactivationManager->m_newlyDeactivatedIsland );
		pairWriter.finalize();
	}


	HK_TIMER_END();
	tl.endCommands( 0 );

	solverDataOut = solverData;
}

void hknpSingleThreadedSimulation::solve( hknpSimulationContext& simulationContext, hknpSolverData* solverData )
{
	hknpSimulationThreadContext* tl = simulationContext.getThreadContext();

#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	hknpSolverData::ThreadData* solverThreadData = &solverData->m_threadData[0];
	hknpWorld* world = tl->m_world;

	//
	//	Pre solve.
	//	Prepare deactivation data, prepare constraints, etc.
	//
	hknpDeactivationStepInfo* deactivationStepInfo = HK_NULL;
	{
		HK_TIMER_BEGIN_LIST2( timerStream, "PreSolve", "SetupDeactivation" );

		world->m_deactivationManager->clearAndTrackIslandForDeactivatingCaches();

		//
		// Init deactivation info. This is filled in during collision detection & Jacobian building stage.
		// The m_solverVelOkToDeactivate array will be filled the the integrator.
		//
		if (world->isDeactivationEnabled())
		{
			const int numThreads = 1;
			const int numAllActiveMotionsPlusOne = world->m_motionManager.getSolverIdToMotionIdForCell(0).getSize();
			deactivationStepInfo = new hknpDeactivationStepInfo(numAllActiveMotionsPlusOne, numThreads);
			deactivationStepInfo->addAllBodyLinksSt(*world, world->m_deactivationManager->getBodyLinks());

			const int bitFieldSize = HK_NEXT_MULTIPLE_OF(4,numAllActiveMotionsPlusOne)+4;
			deactivationStepInfo->getThreadData(0).m_solverVelOkToDeactivate.setSizeAndFill(0, bitFieldSize, 0);

			// Update ThreadLocal with new deactivation info
			tl->m_deactivationData = &deactivationStepInfo->getThreadData(0);
		}

		//hknpSpaceSplitterData spaceSplitterData; spaceSplitterData.reset();
		//tl->m_spaceSplitterData = &spaceSplitterData;

		//
		//	Simply append fixed (=high priority) contact constraints to normal constraints.
		//  This ensures that those fixed contacts get higher priority than moving moving interactions.
		//
		{
			solverThreadData->m_jacMovingStream.append( tl->m_tempAllocator, &solverThreadData->m_jacFixedStream );
		}

		//
		//	Atom constraints
		//
		HK_TIMER_SPLIT_LIST2( timerStream, "SetupConstraints" );
		{
			const bool isDeactivationEnabled = world->isDeactivationEnabled();

			hknpConstraintSolverJacobianWriter schemaWriter; schemaWriter.setToEndOfStream( tl->m_tempAllocator, &solverThreadData->m_jacConstraintsStream );
			hknpConstraintSolverJacobianWriter solverTempsWriter; solverTempsWriter.setToEndOfStream( tl->m_tempAllocator, &solverThreadData->m_solverTempsStream );
			hknpCdPairWriter activePairWriter;
			if (isDeactivationEnabled)
			{
				activePairWriter.setToEndOfStream( tl->m_tempAllocator, &solverThreadData->m_activePairStream );
			}

			hknpConstraintSolverJacobianRange2 range;
			range.initRange( hknpConstraintSolverType::ATOM_CONSTRAINT_SOLVER, 0);
			range.setStartPoint( &schemaWriter );
			range.m_solverTempRange.setStartPoint( &solverTempsWriter );

			world->m_constraintAtomSolver->setupConstraints( tl, schemaWriter, solverTempsWriter, isDeactivationEnabled ? &activePairWriter : HK_NULL);

			range.setEndPoint( &schemaWriter );
			range.m_solverTempRange.setEndPoint( &solverTempsWriter );

			if (!range.isEmpty())
			{
				solverData->m_jacConstraintsGrid.addRange( schemaWriter, 0, range );
			}

			if (isDeactivationEnabled)
			{
				activePairWriter.finalize();
			}
			solverTempsWriter.finalize();
			schemaWriter.finalize();
		}

		HK_TIMER_END_LIST2( timerStream );	// PreSolve
	}


	tl->beginCommands( 0 );
	//
	//	Solver
	//
	{
		//
		//	prepare, resort, and compact hknpMotion -- this needs to be done after jacobians are built because we zero 'lastLinear/AngularVelocities'
		//
		{
			HK_TIMER_BEGIN_LIST2(timerStream, "PreSolve", "PrepareMotions");
			HK_ASSERT( 0xf0345f4f, world->m_spaceSplitter->getNumCells() == 1);
			const hkArray<hknpMotionId>& solverIdToMotionId = world->m_motionManager.getSolverIdToMotionIdForCell(0);
			hknpMotionUtil::buildSolverVelocities(tl, world, solverIdToMotionId, world->m_solverVelocities, world->m_solverSumVelocities);

			HK_TIMER_END_LIST2(timerStream);
		}

		//
		// solve constraints, and contacts
		//
		{
			// Setup the array of grids to be passed to the solving code.
			hknpConstraintSolverJacobianGrid* jacGrids[hknpJacobianGridType::NUM_TYPES];
			jacGrids[hknpJacobianGridType::JOINT_CONSTRAINT] = &solverData->m_jacConstraintsGrid;
			jacGrids[hknpJacobianGridType::MOVING_CONTACT] = &solverData->m_jacMovingGrid;
			jacGrids[hknpJacobianGridType::FIXED_CONTACT] = &solverData->m_jacFixedGrid;

			// Fill scheduler infos.
			hknpConstraintSolverSchedulerGridInfo jacGridInfos[hknpJacobianGridType::NUM_TYPES];
			jacGridInfos[hknpJacobianGridType::JOINT_CONSTRAINT].setIsLinkGrid();
			jacGridInfos[hknpJacobianGridType::JOINT_CONSTRAINT].setPriority(hknpDefaultConstraintSolverPriority::JOINTS);
			jacGridInfos[hknpJacobianGridType::MOVING_CONTACT].setIsLinkGrid();
			jacGridInfos[hknpJacobianGridType::MOVING_CONTACT].setPriority(hknpDefaultConstraintSolverPriority::MOVING_CONTACTS);
			jacGridInfos[hknpJacobianGridType::FIXED_CONTACT].setIsCellArray();
			jacGridInfos[hknpJacobianGridType::FIXED_CONTACT].setPriority(hknpDefaultConstraintSolverPriority::FIXED_CONTACTS);

			hknpSolverUtil::solveSt(
				simulationContext, *tl, jacGridInfos, jacGrids, &solverThreadData->m_liveJacInfoStream,
				world->m_solverVelocities.begin(), world->m_solverSumVelocities.begin(), world->m_solverVelocities.getSize(),
				&solverThreadData->m_solverTempsStream, &solverData->m_contactSolverTempsStream );
		}
	}


	//
	// update body transforms and AABB
	//
	{
		HK_TIMER_BEGIN_LIST2(timerStream, "PostSolve", "UpdateBodies");
		const hkArray<hknpBodyId>& activeBodies = world->getActiveBodies( );
		int numPaddedActiveBodies = HK_NEXT_MULTIPLE_OF(hk4xVector4::mxLength, activeBodies.getSize());
		hknpBodyManager* bodyMgr = &world->m_bodyManager;
		hknpMotion* motions = world->m_motionManager.accessMotionBuffer();

		hknpMotionUtil::updateAllBodies(
			tl, activeBodies.begin(), numPaddedActiveBodies,
			bodyMgr->accessBodyBuffer(), motions, world->getBodyQualityLibrary()->getBuffer(),
			world->m_solverInfo, world->m_intSpaceUtil );

		bodyMgr->resetPreviousAabbsAndFlagsOfStaticBodies();

		if (world->m_deactivationEnabled)
		{
			//	Removing caches from the inactive cache stream will produce holes, which need to be garbage collected
			HK_TIMER_SPLIT_LIST("GarbageCollectInactive");
			world->m_deactivationManager->garbageCollectInactiveCaches(
					*tl, &world->m_collisionCacheManager->m_inactiveCdCacheStream, &world->m_collisionCacheManager->m_inactiveChildCdCacheStream );
		}
		HK_TIMER_END_LIST2(timerStream);
	}


	//
	// Perform deactivation using the solver results
	//
	{
		if (world->isDeactivationEnabled())
		{
			HK_TIMER_BEGIN_LIST2( timerStream, "PostSolve", "UnionFind" );
			deactivationStepInfo->addActiveBodyPairsBegin();
			deactivationStepInfo->addActiveBodyPairs(solverThreadData->m_activePairStream);
			deactivationStepInfo->addActiveBodyPairsEnd();

			HK_TIMER_SPLIT_LIST2( timerStream, "FindIsland");
			hknpDeactivatedIsland* deactivatedIsland = deactivationStepInfo->createDeactivatedIsland(*world, tl->m_deactivationData->m_solverVelOkToDeactivate, HK_NULL, false);
			if ( deactivatedIsland )
			{
				HK_TIMER_SPLIT_LIST("Deactivate island");
				world->m_deactivationManager->deactivateIsland(*tl, deactivatedIsland);
			}
			delete deactivationStepInfo;
			HK_TIMER_END_LIST2( timerStream ); //
		}
	}
	tl->endCommands( 0 );

	//
	//	Cleanup
	//
	{
		tl->m_tempAllocator->clear();
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
