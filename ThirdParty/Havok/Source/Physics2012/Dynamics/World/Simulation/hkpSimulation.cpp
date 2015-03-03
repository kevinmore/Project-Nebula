/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Container/PointerMap/hkMap.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseHandle.h>

#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>

#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>

#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgent.h>
#include <Physics2012/Collide/Agent/hkpCollisionInput.h>
#include <Physics2012/Collide/Agent/Query/hkpLinearCastCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>

#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>

#include <Physics/ConstraintSolver/Solve/hkpSolve.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>

#include <Physics2012/Internal/Solver/Atom/hkpBuildJacobianFromAtoms.h>

#include <Physics2012/Dynamics/Action/hkpAction.h>
#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>
#include <Physics2012/Dynamics/Motion/hkpMotion.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>
#include <Physics2012/Dynamics/Constraint/Setup/hkpConstraintSolverSetup.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldMemoryUtil.h>

#include <Physics2012/Dynamics/World/Simulation/hkpSimulation.h>

#include <Physics2012/Dynamics/Collide/hkpDynamicsContactMgr.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>

#include <Physics2012/Dynamics/World/Maintenance/hkpWorldMaintenanceMgr.h>

#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/hkpToiEvent.h>
#include <Physics2012/Internal/Solver/SimpleConstraints/hkpSimpleConstraintUtil.h>
#include <Physics2012/Dynamics/Constraint/Response/hkpSimpleCollisionResponse.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpContactImpulseLimitBreachedListener.h>

#include <Common/Base/Config/hkOptionalComponent.h>

hkpSimulation::hkpSimulation( hkpWorld* world )
:	m_world(world),
	m_currentTime( 0.0f ),
	m_currentPsiTime( 0.0f ),
	m_simulateUntilTime(-1.f),
	m_frameMarkerPsiSnap( .0001f ),
	m_previousStepResult(HK_STEP_RESULT_SUCCESS)
{
	m_lastProcessingStep = COLLIDE;
	m_determinismCheckFrameCounter = 0;
}

hkpSimulation::~hkpSimulation()
{
}

hkpSimulation::CreateSimulationFunction hkpSimulation::createDiscrete = HK_NULL;
hkpSimulation::CreateSimulationFunction hkpSimulation::createContinuous = HK_NULL;
hkpSimulation::CreateSimulationFunction hkpSimulation::createMultithreaded = HK_NULL;

hkpSimulation* HK_CALL hkpSimulation::create( hkpWorld* world )
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkpSimulation);
	return new hkpSimulation( world );
}
HK_OPTIONAL_COMPONENT_DEFINE(hkpSimulation, hkpSimulation::createDiscrete, hkpSimulation::create);


void hkpSimulation::setFrameTimeMarker( hkReal frameDeltaTime )
{
	// This function is also used in multithreading  asynchronous stepping, so we call mark / unmark for write
	m_world->markForWrite();
	m_simulateUntilTime = m_currentTime + frameDeltaTime;
	m_world->unmarkForWrite();
}

bool hkpSimulation::isSimulationAtMarker()
{
	return( m_currentTime == m_simulateUntilTime );
}

bool hkpSimulation::isSimulationAtPsi() const
{
	return ( m_currentPsiTime == m_currentTime );
}


hkpStepResult hkpSimulation::integrate( hkReal physicsDeltaTime )
{
	HK_TIME_CODE_BLOCK( "Physics 2012", HK_NULL );

	HK_ASSERT2(0x9764ea25, m_lastProcessingStep == COLLIDE, "You must not call integrate twice without calling collide");
	HK_ASSERT2(0xadfefed7, hkGetOutOfMemoryState() == hkMemoryAllocator::MEMORY_STATE_OK, "All memory exceptions must be handled and the memory state flag must be set back to hkMemoryRouter::MEMORY_STATE_OK");
	HK_ASSERT2(0xad000070, !m_world->areCriticalOperationsLocked(), "The m_world cannot be locked when calling integrate()");
	HK_ASSERT2(0xadef876d, !m_world->m_pendingOperationsCount, "No operations may be pending on the hkpWorld::m_pendingOperations queue when calling integrate");
	HK_ASSERT2(0xa0750079, isSimulationAtPsi(), "You may only call integrate when the simulation is at a PSI. Use isSimulationAtPsi() to check for this. Common error: advanceTime() not called.");
	HK_ASSERT2(0x5e4543e4, hkMemorySystem::getInstance().solverCanAllocSingleBlock(1), "Solver allocator has not been initialized");

	HK_ASSERT2(0xcba47962, (m_previousStepResult == HK_STEP_RESULT_SUCCESS) || (physicsDeltaTime == m_physicsDeltaTime), "When recovering from a step failure, you must step with the same delta time" );

	#ifdef HK_DEBUG
		checkDeltaTimeIsOk( physicsDeltaTime );
	#endif

	m_physicsDeltaTime = physicsDeltaTime;

	hkStepInfo physicsStepInfo( m_currentPsiTime, m_currentPsiTime + m_physicsDeltaTime );

	m_world->m_dynamicsStepInfo.m_stepInfo = physicsStepInfo;
	m_world->m_collisionInput->m_stepInfo = physicsStepInfo;


	m_world->m_maintenanceMgr->performMaintenance( m_world, physicsStepInfo );
	m_previousStepResult = integrateInternal( physicsStepInfo );

	if (m_previousStepResult == HK_STEP_RESULT_SUCCESS)
	{
		m_lastProcessingStep = INTEGRATE;
	}

	return (hkpStepResult)m_previousStepResult;
}




hkpStepResult hkpSimulation::collide()
{
	HK_TIME_CODE_BLOCK( "Physics 2012", HK_NULL );
	HK_ASSERT2( 0xadfefed7, hkGetOutOfMemoryState() == hkMemoryAllocator::MEMORY_STATE_OK, "All memory exceptions must be handled and the memory state flag must be set back to hkMemoryRouter::MEMORY_STATE_OK");
	HK_ASSERT2(0xad000070, !m_world->areCriticalOperationsLocked(), "The m_world cannot be locked when calling collide()");
	HK_ASSERT2(0xadef876d, !m_world->m_pendingOperationsCount, "No operations may be pending on the hkpWorld::m_pendingOperations queue when calling collide()");

	if ( m_previousStepResult != HK_STEP_RESULT_SUCCESS )
	{
		return reCollideAfterStepFailure();
	}
	else
	{
		HK_ASSERT2(0x9764ea25, m_lastProcessingStep == INTEGRATE, "You must call call collideSt after integrateSt");
		HK_ASSERT2(0xa0750079, isSimulationAtPsi(), "You may only call collide when the simulation is at a PSI. Use isSimulationAtPsi() to check for this.");

		hkStepInfo stepInfo(  m_currentPsiTime, m_currentPsiTime + m_physicsDeltaTime );

		collideInternal( stepInfo );

		// Check memory
		if (hkMemoryStateIsOutOfMemory(10)  )
		{
			m_previousStepResult = HK_STEP_RESULT_MEMORY_FAILURE_DURING_COLLIDE;
			return hkpStepResult(m_previousStepResult);
		}

		// Increment the current psi time by the delta time
		m_currentPsiTime += m_physicsDeltaTime;
		HK_ON_DETERMINISM_CHECKS_ENABLED(m_determinismCheckFrameCounter++);

		m_world->checkConstraintsViolated();

		//
		// Fire post detection callbacks
		//
		if ( m_world->m_worldPostCollideListeners.getSize() )
		{
			HK_TIMER_BEGIN("PostCollideCB", HK_NULL);
			hkpWorldCallbackUtil::firePostCollideCallback( m_world, stepInfo );
			HK_TIMER_END();
		}

		HK_ASSERT(0xad000070, hkpDebugInfoOnPendingOperationQueues::areEmpty(m_world) );

		m_lastProcessingStep = COLLIDE;
		m_previousStepResult = HK_STEP_RESULT_SUCCESS;
		return HK_STEP_RESULT_SUCCESS;
	}
}



void hkpSimulation::checkDeltaTimeIsOk( hkReal deltaTime )
{
	HK_ASSERT2(0x7486d67e, deltaTime > HK_REAL_EPSILON, 
		"You are trying to step the simulation with a 0 delta time - this will lead to numerical problems, and is not allowed.");

	const hkReal factor = 4.f;
	if(deltaTime <  m_world->m_dynamicsStepInfo.m_stepInfo.m_deltaTime / factor )
	{
		HK_WARN(0x2a2cde91, "Simulation may become unstable, the time step has decreased by more than a factor of " << factor << " from the previous step");
	}
}


// This function is largely the same as the collide() function, but re-written, because of some differences in updating filters,
// deciding when to update m_current time and fire collide callbacks, and what stepInfo to use
hkpStepResult hkpSimulation::reCollideAfterStepFailure()
{
	HK_ASSERT2( 0xadfefed7, hkGetOutOfMemoryState() == hkMemoryAllocator::MEMORY_STATE_OK, "All memory exceptions must be handled and the memory state flag must be set back to hkMemoryRouter::MEMORY_STATE_OK");
	HK_ASSERT2(0xad000070, !m_world->areCriticalOperationsLocked(), "The m_world cannot be locked when calling collide()");
	HK_ASSERT2(0xadef876d, !m_world->m_pendingOperationsCount, "No operations may be pending on the hkpWorld::m_pendingOperations queue when calling collide()");

	HK_ASSERT2( 0xeee97545, m_previousStepResult != HK_STEP_RESULT_MEMORY_FAILURE_BEFORE_INTEGRATION, "If integrate() fails, you must call integrate() again");

	// Do a full re-evaluation of the broadphase and all shape collections. This is very slow.  
	// The last step may have failed after broadphase update but before
	// agent update, so we re-calc everything.  We could instead store the last broadphase results to avoid this.
	// Also, because the bvTree agent caches the query AABB, need to similarly re-do all the bvTree queries.
	m_world->updateCollisionFilterOnWorld(HK_UPDATE_FILTER_ON_WORLD_FULL_CHECK, HK_UPDATE_COLLECTION_FILTER_PROCESS_SHAPE_COLLECTIONS);

	hkStepInfo stepInfo;
	if ( m_previousStepResult == HK_STEP_RESULT_MEMORY_FAILURE_DURING_TOI_SOLVE )
	{
		// If the previous step failed during TOI solve, collide is re-called, from the current time the TOI solve had reached.
		stepInfo.set( m_currentTime, m_currentPsiTime );
	}
	else
	{
		// For normal simulation the simulation must always be at the PSI.
		HK_ASSERT2(0xa0750079, isSimulationAtPsi(), "You may only call collide when the simulation is at a PSI. Use isSimulationAtPsi() to check for this.");
		stepInfo.set( m_currentPsiTime, m_currentPsiTime + m_physicsDeltaTime );
	}

	collideInternal( stepInfo );

	// Check memory
	if (hkMemoryStateIsOutOfMemory(11)  )
	{
		// Note: do not change the previous step result here - could be COLLIDE or TOI
		HK_TIMER_END();
		return (hkpStepResult)m_previousStepResult;

	}

	// If we failed in collide() then re-do the collide step
	if ( m_previousStepResult == HK_STEP_RESULT_MEMORY_FAILURE_DURING_COLLIDE )
	{
		// Increment the current psi time by the delta time
		m_currentPsiTime += m_physicsDeltaTime;
		HK_ON_DETERMINISM_CHECKS_ENABLED(m_determinismCheckFrameCounter++);


		//
		// Fire post detection callbacks
		//
		if ( m_world->m_worldPostCollideListeners.getSize() )
		{
			HK_TIMER_BEGIN("PostCollideCB", HK_NULL);
			hkpWorldCallbackUtil::firePostCollideCallback( m_world, stepInfo );
			HK_TIMER_END();
		}


		HK_ASSERT(0xad000070, hkpDebugInfoOnPendingOperationQueues::areEmpty(m_world) );

		m_lastProcessingStep = COLLIDE;
	}

	m_previousStepResult = HK_STEP_RESULT_SUCCESS;
	return (hkpStepResult)m_previousStepResult;
}


hkReal hkpSimulation::snapSimulateTimeAndGetTimeToAdvanceTo()
{
	m_world->markForWrite();
	// If we are doing asynchronous stepping, then snap the simulate time to the current PSI time if close enough
	if ( m_simulateUntilTime != -1 )
	{
		if ( hkMath::fabs( m_simulateUntilTime - m_currentPsiTime ) < m_frameMarkerPsiSnap )
		{
			m_simulateUntilTime = m_currentPsiTime;
		}
	}
	m_world->unmarkForWrite();

	hkReal timeToAdvanceTo;

	// If the user has scheduled an asynchronous step, then only advance the current time to the frame time
	// specified.

	if ( m_simulateUntilTime == -1.f)
	{
		timeToAdvanceTo = m_currentPsiTime;
	}
	else
	{
		HK_ASSERT2(0xaf687532, m_simulateUntilTime >= m_currentTime, "Once you start calling setStepMarkerSt you must continue to do so." );
		timeToAdvanceTo = hkMath::min2( m_currentPsiTime, m_simulateUntilTime);
	}

	return timeToAdvanceTo;
}

hkpStepResult hkpSimulation::advanceTime()
{
	m_currentTime = snapSimulateTimeAndGetTimeToAdvanceTo();

	if ( (m_currentTime >= m_simulateUntilTime) && ( m_world->m_worldPostSimulationListeners.getSize() ) )
	{
		//
		// Fire post simulate callbacks --- this must be fired here in order for visualization to be updated
		//
		HK_TIME_CODE_BLOCK("PostSimCb", HK_NULL);
		hkpWorldCallbackUtil::firePostSimulationCallback( m_world );
	}

	m_previousStepResult = HK_STEP_RESULT_SUCCESS;
	return (hkpStepResult)m_previousStepResult;
}


hkpStepResult hkpSimulation::stepDeltaTime( hkReal physicsDeltaTime )
{
	// Initially m_previousStepResult will be the value returned from the last call to stepDeltaTime.
	// Each of these functions sets m_previousStepResult to their return value.
		
	if (	( m_previousStepResult == HK_STEP_RESULT_SUCCESS ) || 
			( m_previousStepResult == HK_STEP_RESULT_MEMORY_FAILURE_BEFORE_INTEGRATION ))
	{
		integrate( physicsDeltaTime );
	}

	if (	( m_previousStepResult == HK_STEP_RESULT_SUCCESS ) || 
			( m_previousStepResult == HK_STEP_RESULT_MEMORY_FAILURE_DURING_COLLIDE ))
	{
		collide();
	}

	if (	( m_previousStepResult == HK_STEP_RESULT_SUCCESS ) || 
			( m_previousStepResult == HK_STEP_RESULT_MEMORY_FAILURE_DURING_TOI_SOLVE ))
	{
		advanceTime();
	}

	// Attempt recovering world simulation after memory failure.
	if (m_previousStepResult != HK_STEP_RESULT_SUCCESS && m_world->getMemoryWatchDog())
	{
		hkpWorldMemoryUtil::tryToRecoverFromMemoryErrors(m_world);
	}

	return (hkpStepResult)m_previousStepResult;
}



void hkpSimulation::collideEntitiesBroadPhaseDiscrete( hkpEntity** entities, int numEntities, hkpWorld* world )
{
	HK_TIMER_BEGIN_LIST("BroadPhase", "InitMem");

	HK_ASSERT2(0xad63ee38, numEntities > 0, "No entities?");
	HK_ASSERT(0xad63ee37, world->areCriticalOperationsLocked());

	HK_ON_DEBUG(world->m_simulation->assertThereIsNoCollisionInformationForEntities(entities, numEntities, world));

	hkLocalArray<hkpBroadPhaseHandlePair> newPairs( world->m_broadPhaseUpdateSize );
	hkLocalArray<hkpBroadPhaseHandlePair> delPairs( world->m_broadPhaseUpdateSize );
	{
		hkAabbUint32* aabbs32 = hkAllocateStack<hkAabbUint32>( numEntities );
		hkpBroadPhaseHandle** broadPhaseHandles = hkAllocateStack<hkpBroadPhaseHandle*>(numEntities);

		HK_TIMER_SPLIT_LIST("CalcAabbs");
		int numAabbs = numEntities;
		{
			hkpEntity **e = entities;

			hkAabbUint32* aabb32 = aabbs32;
			hkpBroadPhaseHandle **handle = broadPhaseHandles;

			for( int i = numAabbs-1; i >=0 ; e++, i-- )
			{
				hkpEntity* entity = *e;

				hkpCollidable* collidable = entity->getCollidableRw();

				handle[0] = collidable->getBroadPhaseHandle();

				//
				// Calculate AABB if not yet done (e.g. no IntegrateMotion job was processed yet or (on PlayStation(R)3) the shape was not allowed to go onto SPU).
				//
				if ( !collidable->m_boundingVolumeData.isValid() )
				{
					hkpEntityAabbUtil::entityBatchRecalcAabb(world->getCollisionInput(), &entity, 1);
				}

					// Copy the already calculated AABB into our local contiguous array. (only non-continuous)
				#ifdef HK_ARCH_ARM
					HK_ASSERT2(0x46aefcee, (((hkUlong)&collidable->m_boundingVolumeData) & 0x3) == 0, "Unaligned bounding volume data!");
				#else
					HK_ASSERT2(0x46aefcee, (((hkUlong)&collidable->m_boundingVolumeData) & 0xF) == 0, "Unaligned bounding volume data!");
				#endif
				*aabb32 = reinterpret_cast<hkAabbUint32&>(collidable->m_boundingVolumeData);

				aabb32++;
				handle++;
			}
		}

		HK_TIMER_SPLIT_LIST("3AxisSweep");

		{
			world->m_broadPhase->updateAabbsUint32( broadPhaseHandles, aabbs32, numAabbs, newPairs, delPairs );
		}

		hkDeallocateStack(broadPhaseHandles, numEntities);
		hkDeallocateStack(aabbs32, numEntities);
	}

#if !defined(HK_ENABLE_DETERMINISM_CHECKS)
	if ( newPairs.getSize() + delPairs.getSize() > 0)
#endif
	{
		HK_TIMER_SPLIT_LIST("RemoveDup");
		hkpTypedBroadPhaseDispatcher::removeDuplicates(newPairs, delPairs);

		HK_TIMER_SPLIT_LIST("RemoveAgt");
		world->m_broadPhaseDispatcher->removePairs( static_cast<hkpTypedBroadPhaseHandlePair*>(delPairs.begin()), delPairs.getSize() );

		// be careful about overflows here
		int memoryEstimatePerPair = 128 /*agent*/ + 192 /* contactMgr */ + 256 /*contact points*/ + 64 /* links */;
		if ( !hkHasMemoryAvailable(12, newPairs.getSize() * memoryEstimatePerPair ))
		{
			hkSetOutOfMemoryState( hkMemoryAllocator::MEMORY_STATE_OUT_OF_MEMORY );
			HK_TIMER_END_LIST();
			return;
		}

		HK_TIMER_SPLIT_LIST("AddAgt");
		world->m_broadPhaseDispatcher->addPairs(    static_cast<hkpTypedBroadPhaseHandlePair*>(newPairs.begin()), newPairs.getSize(), world->getCollisionFilter() );
	}
	HK_TIMER_END_LIST();
}

void hkpSimulation::collideIslandNarrowPhaseDiscrete( hkpSimulationIsland* island, const hkpProcessCollisionInput& input)
{
	HK_TIMER_BEGIN( "NarrowPhase", HK_NULL);

	input.m_collisionQualityInfo = input.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::COLLISION_QUALITY_PSI );
	input.m_createPredictiveAgents = false;

	hkpAgentNnTrack& agentTrack = island->m_narrowphaseAgentTrack;
	hkAgentNnMachine_ProcessTrack( island, agentTrack, input );

	hkpAgentNnTrack& midphaseAgentTrack = island->m_midphaseAgentTrack;
	hkAgentNnMachine_ProcessTrack( island, midphaseAgentTrack, input );

	HK_TIMER_END();
}


void hkpSimulation::collideInternal( const hkStepInfo& stepInfoIn )
{
	HK_TIME_CODE_BLOCK( "Collide", HK_NULL );

	//
	// Initialize all parameters of the dynamics step into that depend on the stepInfo
	//
	{
		// Step Info
		m_world->m_dynamicsStepInfo.m_stepInfo = stepInfoIn;
		m_world->m_collisionInput->m_stepInfo   = stepInfoIn;

		// Update Solver Info
		hkpSolverInfo& solverInfo = m_world->m_dynamicsStepInfo.m_solverInfo;
		solverInfo.m_deltaTime	 = stepInfoIn.m_deltaTime    * solverInfo.m_invNumSteps;
		solverInfo.m_invDeltaTime= stepInfoIn.m_invDeltaTime * solverInfo.m_numSteps;
	}

	// validateWorld();

	//
	//	Broadphase
	//
	{
		// enable delay operations, as we cannot allow for the islands to merge now.
		m_world->lockCriticalOperations();
		const hkArray<hkpSimulationIsland*>& activeIslands = m_world->getActiveSimulationIslands();

		for ( int i = 0; i < activeIslands.getSize(); ++i )
		{
			hkpSimulationIsland* island = activeIslands[i];
			collideEntitiesBroadPhaseDiscrete( &island->m_entities[0], island->m_entities.getSize(), m_world );

			if (hkMemoryStateIsOutOfMemory(13)  )
			{
				m_world->unlockAndAttemptToExecutePendingOperations();
				return;
			}
		}
		m_world->unlockAndAttemptToExecutePendingOperations();
	}

	//
	//	Narrowphase
	//
	{
		m_world->lockCriticalOperations();
		const hkArray<hkpSimulationIsland*>& activeIslands = m_world->getActiveSimulationIslands();
		for ( int i = 0; i < activeIslands.getSize(); ++i )
		{
			hkpSimulationIsland* island = activeIslands[i];

			// Can remove constraints here but not entities
			collideIslandNarrowPhaseDiscrete( island, *m_world->m_collisionInput );

			if ( hkMemoryStateIsOutOfMemory(14)  )
			{
				break;
			}

			if ( m_world->m_islandPostCollideListeners.getSize() )
			{
				HK_TIMER_BEGIN("IslandPostCollideCb", HK_NULL);
				hkpWorldCallbackUtil::fireIslandPostCollideCallback( m_world, island, stepInfoIn );
				HK_TIMER_END();
			}
		}
		m_world->unlockAndAttemptToExecutePendingOperations();
	}
}

void hkpSimulation::integrateIsland( hkpSimulationIsland* island, const hkpWorldDynamicsStepInfo& stepInfo, hkpConstraintQueryIn& constraintQueryIn )
{										
	// ApplyIslandActions is now called explicitly from integrate
	int numInactiveFrames;
	if ( island->m_constraintInfo.m_sizeOfSchemas == 0 )
	{
		HK_TIMER_BEGIN("SingleObj", HK_NULL);
		numInactiveFrames = hkRigidMotionUtilApplyForcesAndStep( stepInfo.m_solverInfo, stepInfo.m_stepInfo, stepInfo.m_solverInfo.m_globalAccelerationPerStep, (hkpMotion*const*)island->m_entities.begin(), island->m_entities.getSize(), HK_OFFSET_OF(hkpEntity,m_motion) );
		hkpEntityAabbUtil::entityBatchRecalcAabb(island->getWorld()->getCollisionInput(), island->m_entities.begin(), island->m_entities.getSize());
		HK_TIMER_END();
	}
	else
	{
		numInactiveFrames = hkpConstraintSolverSetup::solve( stepInfo.m_stepInfo, stepInfo.m_solverInfo,
			constraintQueryIn, *island, HK_NULL, 0,
			&island->m_entities[0],    island->m_entities.getSize() );
	}

	if ( numInactiveFrames > hkpMotion::NUM_INACTIVE_FRAMES_TO_DEACTIVATE )
	{
		if (island->m_activeMark)
		{
			if ( island->m_world->m_wantDeactivation )
			{
				hkpWorldOperationUtil::markIslandInactive( island->m_world, island );
			}
		}
	}

}


hkpStepResult hkpSimulation::integrateInternal( const hkStepInfo& stepInfoIn )
{
	HK_ASSERT(0xf0ff0061, !m_world->m_pendingOperationsCount);

	HK_TIMER_BEGIN_LIST("Integrate", "Init");

	//
	// Initialize all parameters of the dynamics step into that depend on the stepInfo
	//
	{
		// Step Info
		m_world->m_dynamicsStepInfo.m_stepInfo = stepInfoIn;
		
		// Solver Info
		hkpSolverInfo& solverInfo = m_world->m_dynamicsStepInfo.m_solverInfo;
		solverInfo.m_deltaTime	 = stepInfoIn.m_deltaTime    * solverInfo.m_invNumSteps;
		solverInfo.m_invDeltaTime= stepInfoIn.m_invDeltaTime * solverInfo.m_numSteps;

		solverInfo.m_globalAccelerationPerSubStep.setMul( hkSimdReal::fromFloat(solverInfo.m_deltaTime), m_world->m_gravity );
		hkSimdReal deltaTime; deltaTime.setFromFloat( stepInfoIn.m_deltaTime );
		solverInfo.m_globalAccelerationPerStep.setMul( deltaTime, m_world->m_gravity );
	}
	
	//
	// Integrate islands
	//
	{
		// We allow whatever operations here (from activation/deactivation callbacks.		

		// Perform memory checks prior to integration but after the dirty list and actions are processed.
		hkWorldMemoryAvailableWatchDog::MemUsageInfo memInfo;
		m_world->calcRequiredSolverBufferSize(memInfo);

		if( !hkMemorySystem::getInstance().solverCanAllocSingleBlock( memInfo.m_maxRuntimeBlockSize ) )
		{
			if (m_world->getMemoryWatchDog())
			{
				m_world->unmarkForWrite();
				hkpWorldMemoryUtil::checkMemoryForIntegration(m_world);	// hkMemory::getInstance().getRuntimeMemorySize()
				m_world->markForWrite();

				HK_ON_DEBUG( m_world->calcRequiredSolverBufferSize( memInfo ) );
				HK_ASSERT2(0xad907071, hkMemorySystem::getInstance().solverCanAllocSingleBlock(memInfo.m_maxRuntimeBlockSize), "Still not enough memory for integration.");
			}
			else
			{
				m_world->unmarkForWrite();
				m_previousStepResult = HK_STEP_RESULT_MEMORY_FAILURE_BEFORE_INTEGRATION;
				return HK_STEP_RESULT_MEMORY_FAILURE_BEFORE_INTEGRATION;
			}
		}


		// Execute all actions. Note, that we only allow operations on phantoms here (i.e. entities/actions/constrainsts are still blocked)
		// TODO consider allowing collisionFilterUpdate too.
		{
			m_world->lockCriticalOperations();
			m_world->unlockCriticalOperationsForPhantoms();

			HK_TIMER_SPLIT_LIST("Actions");
			applyActions( );

			m_world->lockCriticalOperationsForPhantoms();
			m_world->unlockAndAttemptToExecutePendingOperations();
		}

		m_world->lockCriticalOperations();

		m_world->m_dynamicsStepInfo.m_solverInfo.incrementDeactivationFlags();

			// Constraint Query Info In
		hkpConstraintQueryIn constraintQueryIn(HK_ON_CPU(&hkpBeginConstraints));	
		constraintQueryIn.set( m_world->m_dynamicsStepInfo.m_solverInfo, stepInfoIn, m_world->m_violatedConstraintArray);

		HK_TIMER_SPLIT_LIST("Integrate");
		const hkArray<hkpSimulationIsland*>& activeIslands = m_world->getActiveSimulationIslands();
		for (int i = activeIslands.getSize()-1; i>=0; i--)
		{
			hkpSimulationIsland* activeIsland = activeIslands[i];
			hkCheckDeterminismUtil::checkMt( 0xf00000f0, activeIsland->m_storageIndex);
			HK_ON_DETERMINISM_CHECKS_ENABLED( hkCheckDeterminismUtil::checkMt( 0xf00000f1, activeIsland->m_uTag) );
			hkCheckDeterminismUtil::checkMt( 0xf00000f2, activeIsland->m_entities[0]->getUid());
			HK_ASSERT(0x3b3ca726,  activeIsland->m_storageIndex == i );

			integrateIsland( activeIsland, m_world->m_dynamicsStepInfo, constraintQueryIn );

			// 
			//	fire island post integrate listener
			//
			if ( m_world->m_islandPostIntegrateListeners.getSize() )
			{
				HK_TIMER_SPLIT_LIST("IslandPostIntegrateCb");
				hkpWorldCallbackUtil::fireIslandPostIntegrateCallback( m_world, activeIsland, stepInfoIn );
			}
		}

		// required as actions may change the m_world; we need to apply mouseAction and other such things
		m_world->unlockAndAttemptToExecutePendingOperations();
	}


	//
	// Fire post integrate callbacks
	//
	if ( m_world->m_worldPostIntegrateListeners.getSize() )
	{
		HK_TIMER_BEGIN("WorldPostIntegrateCb", HK_NULL);
		hkpWorldCallbackUtil::firePostIntegrateCallback( m_world, stepInfoIn );
		HK_TIMER_END();
	}


	//
	// End the integrate timer list
	//
	HK_TIMER_END_LIST(); // integrate

	return HK_STEP_RESULT_SUCCESS;
}


void hkpSimulation::applyActions()
{
	const hkArray<hkpSimulationIsland*>& activeIslands = m_world->getActiveSimulationIslands();
	for (int i = 0; i < activeIslands.getSize(); i++)
	{
		HK_ASSERT(0x3b3ca726,  activeIslands[i]->m_storageIndex == i );

		hkArray<hkpAction*>& actions = activeIslands[i]->m_actions;
		for (int j = 0; j < actions.getSize(); j++)
		{
			hkpAction* action = actions[j];
			action->applyAction( m_world->m_dynamicsStepInfo.m_stepInfo );
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
			hkInplaceArray<hkpEntity*, 16> entities;
			action->getEntities(entities);
			for (int e = 0; e < entities.getSize(); e++)
			{
				hkpEntity* entity = entities[e];
				hkCheckDeterminismUtil::checkMtCrc(0xf0000092, &entity->getMotion()->getTransform(),1);
				hkCheckDeterminismUtil::checkMtCrc(0xf0000093, &entity->getMotion()->getLinearVelocity(),1);
				hkCheckDeterminismUtil::checkMtCrc(0xf0000094, &entity->getMotion()->getAngularVelocity(),1);

			}
#endif
		}
	}
}

void hkpSimulation::collideEntitiesDiscrete( hkpEntity** entities, int numEntities, hkpWorld* world, const hkStepInfo& stepInfo, FindContacts findExtraContacts )
{
	HK_ASSERT2(0xad45ee3b, numEntities, "Must pass at least one hkpEntity when callling hkpSimulation::collideEntitiesDiscrete.");

	hkpProcessCollisionInput input = *world->getCollisionInput();
	input.m_stepInfo = stepInfo;

	world->lockCriticalOperations();
	{
		collideEntitiesBroadPhaseDiscrete(entities, numEntities, world);

		//if (HK_OUT_OF_MEM_TEST(15)  )
		//{
		//	world->unlockAndAttemptToExecutePendingOperations();
		//	return;
		//}

		collideEntitiesNarrowPhaseDiscrete(entities, numEntities, input, findExtraContacts);
	}
	world->unlockAndAttemptToExecutePendingOperations();
}




void hkpSimulation::collideEntitiesNarrowPhaseDiscrete( hkpEntity** entities, int numEntities, const hkpProcessCollisionInput& input, FindContacts findExtraContacts )
{
	HK_ASSERT2(0xadfe825d, numEntities, "Must call the function with a non-zero number of entities.");
	HK_ASSERT2(0xadfe825d, entities[0]->getWorld()->areCriticalOperationsLocked(), "The hkpWorld must be locked when calling hkpSimulation::collideEntitiesNarrowPhaseDiscrete.");

	processAgentsOfEntities(entities, numEntities, input, &hkpSimulation::processAgentCollideDiscrete, findExtraContacts);
}

void hkpSimulation::processAgentCollideDiscrete(hkpAgentNnEntry* entry, const hkpProcessCollisionInput& processInput, hkpProcessCollisionOutput& processOutput)
{
	processOutput.reset();
	processInput.m_collisionQualityInfo   = processInput.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::CollisionQualityIndex(processInput.m_dispatcher->COLLISION_QUALITY_PSI) );
	processInput.m_createPredictiveAgents = processInput.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex                   )->m_useContinuousPhysics;

	hkAgentNnMachine_ProcessAgent( entry, processInput, processOutput, entry->m_contactMgr );

	if (hkMemoryStateIsOutOfMemory(16)  )
	{
		return;
	}

	if ( !processOutput.isEmpty() )
	{
		hkpCollidable& collA = *entry->getCollidableA();
		hkpCollidable& collB = *entry->getCollidableB();
		entry->m_contactMgr->processContact( collA, collB, processInput, processOutput );
	} 
}


	//	void processAgentClearManifoldsAndTimsAndToisOfEntities( hkpEntity** entities, int numEntities, hkpWorld* world );
void hkpSimulation::processAgentResetCollisionInformation(hkpAgentNnEntry* entry, const hkpProcessCollisionInput& processInput, hkpProcessCollisionOutput& processOutput)
{
	// Invalidate Tims
	hkAgentNnMachine_InvalidateTimInAgent(entry, const_cast<hkpProcessCollisionInput&>(processInput));

	// Clear manifolds
//	hkInplaceArray<hkContactPointId, 10> contactPointIds;
//	hkpDynamicsContactMgr* mgr = static_cast<hkpDynamicsContactMgr*>(entry->m_contactMgr);
//	mgr->getAllContactPointIds(contactPointIds);
//	for (int i = 0; i < contactPointIds.getSize(); i++)
//	{
//		// We clear the manifold by invalidating all present contact points by setting their distance to 
//		// infinity.
//		// This solution does not guarantee determinism though, as the internal agent caches are not reset.
//		mgr->getContactPoint()->setDistance(HK_REAL_MAX);
//		// crashes:mgr->removeContactPoint(contactPointIds[i]);
//	}

	// Info: to guarantee determinism, agents must be recreated.
}

void hkpSimulation::resetCollisionInformationForEntities( hkpEntity** entities, int numEntities, hkpWorld* world, enum ResetCollisionInformation resetInfo )
{
	HK_ASSERT2(0XAD4545DD, world->areCriticalOperationsLocked(), "This is an internal function. It requires the world to be locked.");

	if ( (resetInfo & RESET_TIM) != 0)
	{
		// world->getCollisionInput() is not used in hkpSimulation::processAgentResetCollisionInformation.
		processAgentsOfEntities( entities, numEntities, *world->getCollisionInput(), &hkpSimulation::processAgentResetCollisionInformation, FIND_CONTACTS_DEFAULT);
	}
	if ( (resetInfo & RESET_AABB) != 0)
	{
		hkpEntityAabbUtil::entityBatchInvalidateAabb(entities, numEntities);
	}
}




// NOTE: This function does not alter the swept transform; it only modifies the transform of the rigid body
// as that is what is used in collision detection
static HK_LOCAL_INLINE void setRotationAroundCentreOfMass( hkpRigidBody* rb, hkQuaternion& newRotation )
{
	hkTransform& trans = rb->getRigidMotion()->getMotionState()->getTransform();
	trans.setRotation(newRotation);
	hkVector4 centerShift;
	centerShift._setRotatedDir( trans.getRotation(), rb->getRigidMotion()->getCenterOfMassLocal() );
	trans.getTranslation().setSub( rb->getRigidMotion()->getCenterOfMassInWorld(), centerShift );
}

void hkpSimulation::processAgentsOfEntities( hkpEntity** entities, int numEntities, const hkpProcessCollisionInput& processInput, AgentEntryProcessFunction processingFunction, FindContacts findExtraContacts)
{
	// Have a set of entities which have been processed. To avoid processing agents twice, we don't process agents
	// that link the entity currently being processed to another entity, which already has been processed.
	HK_COMPILE_TIME_ASSERT( sizeof(hkUlong) == sizeof(hkpLinkedCollidable*) );

	hkMap<hkUlong> processedEntities(numEntities);

	//const hkpProcessCollisionInput& processInput = *world->getCollisionInput();
	HK_ALIGN16(typedef hkpProcessCollisionOutput hkAlignedCollisionOutput);
	hkAlignedCollisionOutput processOutput(HK_NULL);
	hkArray<struct hkpLinkedCollidable::CollisionEntry> collisionEntriesTmp;

	// Process each entity from the 'entities' list.
	for (int i = 0; i < numEntities; i++)
	{
		hkpEntity* entity = entities[i];
		HK_ASSERT2(0xade278fd, entity->getWorld() == entities[0]->getWorld(), "All entities must belong to the same hkpWorld");

		
		//HK_ASSERT2(0xadf256fe, entity->getSimulationIsland()->m_isInActiveIslandsArray, "The entity is _inactive_ and some of it's agents might be not present");

		hkpLinkedCollidable* lColl = entity->getLinkedCollidable();

		// Check that every entity is only passed once
		HK_ON_DEBUG( hkMap<hkUlong>::Iterator collIt = processedEntities.findKey(hkUlong(lColl)); )
		HK_ASSERT2(0xad45db3, !processedEntities.isValid(collIt), "Same entity passed more than once to hkpSimulation::processAgentsOfEntities.");

		processedEntities.insert(hkUlong(lColl), 0);
		
		lColl->getCollisionEntriesSorted(collisionEntriesTmp); // order matters for callbacks
		const hkArray<struct hkpLinkedCollidable::CollisionEntry>& collisionEntries = collisionEntriesTmp;

		for (int c = 0; c < collisionEntries.getSize(); c++) 
		{
			const hkpLinkedCollidable::CollisionEntry& cEntry = collisionEntries[c];

			hkpLinkedCollidable* partner = cEntry.m_partner;
			HK_ASSERT2( 0xf0321245, hkpGetRigidBody(cEntry.m_partner), "Internal error, entity expected, something else found");

			hkMap<hkUlong>::Iterator it = processedEntities.findKey(hkUlong(partner));
			if (processedEntities.isValid(it))
			{
				continue;
			}

			{
				// Process agents not linking to processed entities.
				hkpAgentNnEntry* entry = cEntry.m_agentEntry;
				{
					hkpEntity* entityA = static_cast<hkpEntity*>(entry->getCollidableA()->getOwner());
					hkpEntity* entityB = static_cast<hkpEntity*>(entry->getCollidableB()->getOwner());
					hkpSimulationIsland* island = (entityA->isFixed() )? entityB->getSimulationIsland(): entityA->getSimulationIsland();
					processOutput.m_constraintOwner = island;
				}

				hkAgentNnMachine_InvalidateTimInAgent(entry, const_cast<hkpProcessCollisionInput&>(processInput));
				(this->*processingFunction)(entry, processInput, processOutput);

				if (	(findExtraContacts == FIND_CONTACTS_EXTRA) && 
						(processOutput.m_firstFreeContactPoint != processOutput.m_contactPoints ) )
				{
					hkpRigidBody* rb = (hkpRigidBody*)entity;
					hkQuaternion origRot = rb->getRigidMotion()->getRotation();
					hkTransform origTrans = rb->getTransform();

					// Keep this as the "master" normal
					hkVector4 normal = processOutput.m_contactPoints[0].m_contact.getSeparatingNormal();

					if (cEntry.m_agentEntry->getCollidableA() != entity->getCollidable())
					{
						normal.setNeg<4>(normal);
					}

					// Calculate the axis of rotation based on the contact normal and the line from the contact point to the centre of mass
					hkVector4 diff; diff.setSub(rb->getCenterOfMassInWorld(), processOutput.m_contactPoints[0].m_contact.getPosition());
					hkVector4 rotateDir; rotateDir.setCross(normal, diff);
					const hkSimdReal len2 = rotateDir.lengthSquared<3>();

					const hkSimdReal l = normal.dot<3>(diff);
					const hkSimdReal d = diff.length<3>();
					const hkSimdReal rotateSinFrom = (l / d);

					// Only rotate if the contact point is not below the center of mass
					if ( (len2 > hkSimdReal::fromFloat(0.00001f*0.00001f)) && (rotateSinFrom < hkSimdReal::fromFloat(.9999f)) )
					{
						const hkReal toleranceMult = 20.f;

						// Increase tolerances to 10 times the normal, and save old versions for reset
						hkReal savedTolerance = processInput.m_tolerance;
						hkReal searchTolerance = processInput.m_tolerance * toleranceMult;
						(const_cast<hkpProcessCollisionInput&>(processInput)).m_tolerance = searchTolerance;
						hkReal savedCreate4dContact = processInput.m_collisionQualityInfo->m_create4dContact;
						hkReal savedCreateContact = processInput.m_collisionQualityInfo->m_createContact;
						(const_cast<hkpProcessCollisionInput&>(processInput)).m_collisionQualityInfo->m_create4dContact = searchTolerance;
						(const_cast<hkpProcessCollisionInput&>(processInput)).m_collisionQualityInfo->m_createContact = searchTolerance;


						// Calculate angle for rotation = asin(l/d) - asin( (l-x) / d) where x is the max distance we want the closest point to separate by
						hkSimdReal rotateSinTo; rotateSinTo.setMax((l - hkSimdReal::fromFloat(searchTolerance * (1.f / toleranceMult)) ) / d, hkSimdReal_0 );
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
						hkVector4 sines; sines.set(rotateSinFrom, rotateSinTo, hkSimdReal_0, hkSimdReal_0);
						hkVector4 arcsines;
						hkVector4Util::aSin(sines,arcsines);
						hkSimdReal rotateAngle = arcsines.getComponent<0>() - arcsines.getComponent<1>();
#else
						hkSimdReal rotateAngle; rotateAngle.setFromFloat( hkMath::asin( rotateSinFrom.getReal() ) - hkMath::asin( rotateSinTo.getReal() ) );
#endif
						// normalize the rotation direction
						rotateDir.mul(len2.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>());

						// Rotate body by a small rotation to pick up the point at the opposite side to the closest point
						{
							hkQuaternion quat; quat.setAxisAngle( rotateDir, rotateAngle );
							hkQuaternion rbRot;
							rbRot.setMul(quat, rb->getRigidMotion()->getRotation());

							setRotationAroundCentreOfMass( rb, rbRot );

							hkAgentNnMachine_InvalidateTimInAgent(entry, const_cast<hkpProcessCollisionInput&>(processInput));

							(this->*processingFunction)(entry, processInput, processOutput);
						}

						// Rotate the body by a small rotation to pick up the points at 90 degree extremities to the "primary" axis
						{
							hkVector4 rotateCrossDir; rotateCrossDir.setCross( rotateDir, normal );
							rotateCrossDir.normalize<3>();

							hkQuaternion quat; quat.setAxisAngle( rotateCrossDir, rotateAngle );
							hkQuaternion rbRot; rbRot.setMul(quat, origRot);
							setRotationAroundCentreOfMass( rb, rbRot );

							hkAgentNnMachine_InvalidateTimInAgent(entry, const_cast<hkpProcessCollisionInput&>(processInput));

							(this->*processingFunction)(entry, processInput, processOutput);


							rbRot.setInverseMul(quat, origRot);
							setRotationAroundCentreOfMass( rb, rbRot );

							hkAgentNnMachine_InvalidateTimInAgent(entry, const_cast<hkpProcessCollisionInput&>(processInput));

							(this->*processingFunction)(entry, processInput, processOutput);
						}

						// Reset the body and update the manifold
						{
							// Reset the tolerances to their original values
							(const_cast<hkpProcessCollisionInput&>(processInput)).m_tolerance = savedTolerance;
							(const_cast<hkpProcessCollisionInput&>(processInput)).m_collisionQualityInfo->m_create4dContact = savedCreate4dContact;
							(const_cast<hkpProcessCollisionInput&>(processInput)).m_collisionQualityInfo->m_createContact = savedCreateContact;

							rb->getRigidMotion()->setTransform(origTrans);
							hkAgentNnMachine_InvalidateTimInAgent(entry, const_cast<hkpProcessCollisionInput&>(processInput));

							// Ideally we just want to update manifold here, not re-collide
							(this->*processingFunction)(entry, processInput, processOutput);

						}
					}
				}
				if (hkMemoryStateIsOutOfMemory(17)  )
				{
					return;
				}

			}
		}
	}
}

void hkpSimulation::reintegrateAndRecollideEntities( hkpEntity** entities, int numEntities, hkpWorld* world, int reintegrateRecollideMode )
{
	HK_ASSERT2( 0xf03f667d, !world->areCriticalOperationsLocked(), "invalid lockCriticalOperations()");
	world->lockCriticalOperations();

	// Prepare the proper stepInfo. 
	const hkStepInfo physicsStepInfo( world->getCurrentTime(), world->getCurrentPsiTime() );
	world->m_collisionInput->m_stepInfo = physicsStepInfo;


	// Re-integrate selected bodies.
	if  ( reintegrateRecollideMode & hkpWorld::RR_MODE_REINTEGRATE )
	{
		for (int i = 0; i < numEntities; i++)
		{
			hkpRigidBody* body = static_cast<hkpRigidBody*>(entities[i]);
			hkMotionState& ms = *body->getRigidMotion()->getMotionState();

			if ( !body->isFixed())
			{
				hkSweptTransformUtil::backStepMotionState(world->getCurrentTime(), ms);
			}
		}
		hkRigidMotionUtilStep(physicsStepInfo, (hkpMotion*const*)entities, numEntities, HK_OFFSET_OF(hkpEntity, m_motion));
		hkpEntityAabbUtil::entityBatchRecalcAabb(world->getCollisionInput(), entities, numEntities);
	}

	// Re-collide selected bodies.
	if  ( reintegrateRecollideMode & hkpWorld::RR_MODE_RECOLLIDE_BROADPHASE )
	{
		collideEntitiesBroadPhaseDiscrete(entities, numEntities, world);
	}


	if  ( reintegrateRecollideMode & hkpWorld::RR_MODE_RECOLLIDE_NARROWPHASE )
	{
		// recollide all agents, this will invalidate tims
		collideEntitiesNarrowPhaseDiscrete(entities, numEntities, *world->getCollisionInput(), FIND_CONTACTS_DEFAULT);
	}
	else if ( reintegrateRecollideMode & hkpWorld::RR_MODE_REINTEGRATE )
	{
		// Reintegrate TIMs for all agents involved
		resetCollisionInformationForEntities(entities, numEntities, world, hkpSimulation::RESET_TIM);
	}

	world->unlockAndAttemptToExecutePendingOperations();
}

// A specialized code to determine simplified collision response for two isolated physical bodies.
// For details see: hkpSimpleCollisionResponse::solveSingleContact
//
// If it is possible to postpone application of the collision impulse on one of the colliding bodies and avoid penetration
// at the same time, then it is done so.
//
//  Params:
//  toBeActivated array containts one or both bodies. If activating (re-integrating) only one of the bodies
//  suffices to prevent interpenetration, then only that body is returned in the toBeActivated array.
void HK_CALL hkLs_doSimpleCollisionResponse( hkpWorld* world, const hkpToiEvent& event, hkReal rotateNormal, hkArray<hkpEntity*>& toBeActivated )
{
	hkpRigidBody* body[2] = { static_cast<hkpRigidBody*>(event.m_entities[0]) ,
		static_cast<hkpRigidBody*>(event.m_entities[1]) };

	hkpMotion* motion[2];
	motion[0] = body[0]->getRigidMotion();
	motion[1] = body[1]->getRigidMotion();

	hkpSimpleConstraintUtilCollideParams params;
	params.m_externalSeperatingVelocity = event.m_seperatingVelocity;
	params.m_extraSeparatingVelocity = 0.1f;
	params.m_extraSlope = rotateNormal;

	params.m_friction    = event.m_properties.getFriction();
	params.m_restitution = event.m_properties.getRestitution();
	if ( !event.m_properties.m_maxImpulse.m_value)
	{
		params.m_maxImpulse = HK_REAL_MAX;
	}
	else
	{
		params.m_maxImpulse = event.m_properties.m_maxImpulse;
	}

	hkpSimpleCollisionResponse::SolveSingleOutput result;
	hkpSimpleCollisionResponse::solveSingleContact( event.m_contactPoint, event.m_time, params, motion[0], motion[1], event.m_contactMgr, result );

	if ( params.m_contactImpulseLimitBreached )
	{
		hkpSimpleConstraintContactMgr* mgr = (hkpSimpleConstraintContactMgr*)( event.m_contactMgr );
		HK_ASSERT2( 0xf02dfe45, mgr->getType() == hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR, "You can only use limiting of contact impulses when using the default hkpSimpleConstraintContactMgr" );
		hkpContactImpulseLimitBreachedListenerInfo bi;
		bi.set( &mgr->m_constraint, const_cast<hkpContactPointProperties*>(&event.m_properties), const_cast<hkContactPoint*>(&event.m_contactPoint), true );
		hkpWorldCallbackUtil::fireContactImpulseLimitBreached( world, &bi, 1 );
		return;
	}

	HK_ASSERT2(0xad000040, !body[0]->isFixedOrKeyframed() || !body[1]->isFixedOrKeyframed(), "Havok's continuous collision resolution does not allow for TOI events between keyframed body pairs or fixed/keyframed pairs. Change the body quality types to prevent this." );

	if ( result.m_velocityKeyframedA < 0.0f && result.m_velocityKeyframedB < 0.0f)
	{	// activate both bodies
#ifdef HK_ENABLE_TOI_WARNINGS
		// This happens !
		if (body[0]->isFixedOrKeyframed() || body[1]->isFixedOrKeyframed())
		{
			HK_WARN(0xad667667, "Internal check. A body which should be reintegrated in SCR is fixed-or-keyframed. Possible cause: a keyframed/fixed object with inconsistent motion-velocity and swept-transform.");
		}
#endif
		if (!body[0]->isFixedOrKeyframed())	{ toBeActivated.pushBackUnchecked(body[0]); }
		if (!body[1]->isFixedOrKeyframed())	{ toBeActivated.pushBackUnchecked(body[1]); }
	}
	else if (result.m_velocityKeyframedA <  0.0f && result.m_velocityKeyframedB >= 0.0f)
	{	// activate body A, keyframe body B
		
#ifdef HK_ENABLE_TOI_WARNINGS
		HK_ASSERT2(0xad667667, !body[0]->isFixedOrKeyframed(), "Internal check. This assert is safe to ignore. It may cause temporary artifacts, though. A body which should be reintegrated in SCR is fixed-or-keyframed. Possible cause: a keyframed/fixed object with inconsistent motion-velocity and swept-transform.");
#endif
		if (!body[0]->isFixedOrKeyframed())		{	toBeActivated.pushBackUnchecked(body[0]);		}
		else									{	toBeActivated.pushBackUnchecked(body[1]); 		}
	}
	else if (result.m_velocityKeyframedA >= 0.0f && result.m_velocityKeyframedB <  0.0f)
	{
		// activate body B, keyframe body A
		
#ifdef HK_ENABLE_TOI_WARNINGS
		HK_ASSERT2(0xad667667, !body[1]->isFixedOrKeyframed(), "Internal check. This assert is safe to ignore. It may cause temporary artifacts, though. A body which should be reintegrated in SCR is fixed-or-keyframed. Possible cause: a keyframed/fixed object with inconsistent motion-velocity and swept-transform.");
#endif
		if (!body[1]->isFixedOrKeyframed())	{		toBeActivated.pushBackUnchecked(body[1]); 	}
		else								{		toBeActivated.pushBackUnchecked(body[0]); 	}
	}
	else
	{
		HK_ASSERT(0xad000011, result.m_velocityKeyframedA >= 0.0f && result.m_velocityKeyframedB >=  0.0f);
#ifdef HK_ENABLE_TOI_WARNINGS
		HK_WARN(0xad667667, "Internal check. Potentially redundant TOI. This warning is safe to ignore.");
#endif

		// TODO find a better (yielding better efficiency) way to chose between the bodies to activate
		if (  body[0]->isFixedOrKeyframed() ||
			(!body[1]->isFixedOrKeyframed() && result.m_velocityKeyframedA > result.m_velocityKeyframedB) )
		{
			HK_ASSERT(0xad000016, !body[1]->isFixedOrKeyframed());
			toBeActivated.pushBackUnchecked(body[1]);
		}
		else
		{
			HK_ASSERT(0xad000017, !body[0]->isFixedOrKeyframed());
			toBeActivated.pushBackUnchecked(body[0]);
		}
	}
	HK_ON_DEBUG( for (int i = 0; i < toBeActivated.getSize(); i++){	HK_ASSERT(0xad000012, !static_cast<hkpRigidBody*>(toBeActivated[i])->isFixedOrKeyframed());	} );
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
