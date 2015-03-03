/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/World/Simulation/Continuous/hkpContinuousSimulation.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>
#include <Common/Base/DebugUtil/TraceStream/hkTraceStream.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/PointerMap/hkMap.h>
#include <Common/Base/Container/String/hkStringBuf.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseHandle.h>

#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>

#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>

#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>
#include <Physics2012/Collide/Agent/hkpCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>

#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>

#include <Physics2012/Internal/Solver/SimpleConstraints/hkpSimpleConstraintUtil.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>

#include <Physics2012/Dynamics/Collide/hkpDynamicsContactMgr.h>
#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>

#include <Physics2012/Dynamics/Constraint/Response/hkpSimpleCollisionResponse.h>
#include <Physics2012/Dynamics/Constraint/Setup/hkpConstraintSolverSetup.h>

#include <Physics2012/Internal/Solver/Atom/hkpBuildJacobianFromAtoms.h>

#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>
#include <Physics2012/Dynamics/World/Maintenance/hkpWorldMaintenanceMgr.h>

#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>//for debug
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>

#include <Physics/ConstraintSolver/Solve/hkpSolve.h>

#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/ToiResourceMgr/hkpToiResourceMgr.h>
#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/ToiResourceMgr/Default/hkpDefaultToiResourceMgr.h>

#include <Physics2012/Dynamics/Constraint/Contact/hkpSimpleContactConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpContactImpulseLimitBreachedListener.h>

#include <Common/Base/Config/hkConfigSolverLog.h>

#include <Common/Base/Config/hkOptionalComponent.h>

// Must assert that hkpToiEvent::m_properties + hkpToiEvent::m_extendedUserDatas + the class' padding corresponds to hkContactPointPropertiesWithExtendedUserData16
HK_COMPILE_TIME_ASSERT( sizeof(hkpContactPointProperties) == HK_OFFSET_OF(hkpToiEvent, m_extendedUserDatas) - HK_OFFSET_OF(hkpToiEvent, m_properties) );
#if defined (HK_PLATFORM_HAS_SPU)
HK_COMPILE_TIME_ASSERT( sizeof(hkpToiEvent) == HK_OFFSET_OF(hkpToiEvent, m_extendedUserDatas) + sizeof(hkpToiEvent().m_extendedUserDatas));
#endif


//#include <hkvisualize/type/hkColor.h>
//#include <hkvisualize/hkDebugDisplay.h>
#if 0 && defined(HK_DEBUG)
#	define HK_TOI_COLORING(code) code
#else
#	define HK_TOI_COLORING(code)
#endif
//#define HK_ENABLE_TOI_WARNINGS

//#define HK_DEBUG_TOI_DETAIL

#ifdef HK_DEBUG_TOI
#	define STRING_ENTRY_SIZE 1024
#	define STRING_TABLE_SIZE 128
	static char g_hkstringTable[STRING_TABLE_SIZE][STRING_ENTRY_SIZE];
	static int g_hklastStringIndex = -1;

#	define GET_BODY_NAME(str, body)  char str[STRING_ENTRY_SIZE]; body->getName(str, STRING_ENTRY_SIZE);
#	define RETURN_BODY_NAME(body)                            (( body->getName( g_hkstringTable [ g_hklastStringIndex = (g_hklastStringIndex + 1) % STRING_TABLE_SIZE ], STRING_ENTRY_SIZE ), g_hkstringTable[g_hklastStringIndex  ] ))
#	define RETURN_BODY_NAME_VOLATILE(body)                   (( body->getName( g_hkstringTable [ g_hklastStringIndex = (g_hklastStringIndex + 1) % STRING_TABLE_SIZE ], STRING_ENTRY_SIZE ), g_hkstringTable[g_hklastStringIndex--] ))


#	define GET_CONSTRAINT_NAME(str, c)  char str[STRING_ENTRY_SIZE]; \
										{ GET_BODY_NAME(b0, c->getInternal()->m_entities[0]); \
										  GET_BODY_NAME(b1, c->getInternal()->m_entities[1]); \
										  hkpConstraintInstance::ConstraintPriority p = c->getPriority(); char ch = p == hkpConstraintInstance::PRIORITY_TOI_HIGHER ? 'h' : p == hkpConstraintInstance::PRIORITY_TOI ? 't' : p == hkpConstraintInstance::PRIORITY_PSI ? 'p' : p == hkpConstraintInstance::PRIORITY_TOI_FORCED ? 'F' : '!'; hkString::snprintf(str, STRING_ENTRY_SIZE, "%s-%s_%c", b0, b1, ch); }


#	define CONSTRAINT_NAME(c)    hkString( BODY_NAME(c->m_internal->m_entities[0]) + "-" + BODY_NAME(c->getInternal()->m_entities[1]) + (c->getPriority() == hkpConstraintInstance::PRIORITY_TOI_HIGHER ? "h" : c->m_internal->m_priority == hkpConstraintInstance::PRIORITY_TOI ? "t" : c->getPriority() == hkpConstraintInstance::PRIORITY_PSI ? "p" : p == hkpConstraintInstance::PRIORITY_TOI_FORCED ? 'F' : "!") )
#	define BODY_NAME(body) hkString( RETURN_BODY_NAME_VOLATILE(body) )

#else

#	define GET_BODY_NAME(str, body)
#	define GET_CONSTRAINT_NAME(str, c)  char str[1];
#	define RETURN_BODY_NAME(body)    ((char*)HK_NULL)
#	define RETURN_BODY_NAME_VOLATILE(body)  ((char*)HK_NULL)
#	define BODY_NAME(body) RETURN_BODY_NAME_VOLATILE(body)
#endif


#define ENTITY_STATE( entity ) entityState[ entity->m_storageIndex ]

// state of an entity during TOI solving
enum hkContinuousSimulationToiEntityState
{
	// This is the default for each entity, before it is accessed by the TOI subsolver
	HK_TOI_ENTITY_STATE_ZERO              = 0,

	// The entities transform T is updated to the interpolated position at the TOI
	// The entities velocities are still untouched
	HK_TOI_ENTITY_STATE_TRANSFORM_UPDATED = 1,

	// touched means: there is a interesting constraint between this body and an ACTIVE entity
	HK_TOI_ENTITY_STATE_TOUCHED           = 2,

	HK_TOI_ENTITY_STATE_PENDING_FLAG	  = 4,

	// Set when a touched entity is pending to be activated but doesn't have all its manifolds updated yet.
	// Then entity may be not activated eventually if we run out of solver resources. In such case it will change to HK_TOI_ENTITY_STATE_MANIFOLD_UPDATED.
	HK_TOI_ENTITY_STATE_PENDING_AND_TOUCHED           = 2+4,

	// Set if all contact constraints attached to this entity have their manifolds updated.
	HK_TOI_ENTITY_STATE_MANIFOLD_UPDATED   = 8,

	// Set if all contact constraints attached to this entity have their manifolds updated, and the entity is pending to be activated.
	HK_TOI_ENTITY_STATE_PENDING_AND_MANIFOLD_UPDATED   = 8+4,

	// The ACTIVATE entity which will change their motion and therefor will be reintegrated
	HK_TOI_ENTITY_STATE_ACTIVE            = 16,
};




hkpContinuousSimulation::hkpContinuousSimulation( hkpWorld* world )
:	hkpSimulation( world )
{
	m_lastProcessingStep = COLLIDE;

	m_toiResourceMgr = new hkpDefaultToiResourceMgr();
	m_toiCounter = 0;

	//
	// init debug output stream
	//
#	ifdef HK_DEBUG_TOI
	if ( ! hkTraceStream::getInstance().m_stream )
	{
			hkTraceStream::getInstance().m_stream = new hkOfstream(HK_DEBUG_TOI);
	}
		//Toi.sim
	hkTraceStream::getInstance().disableTitle("Gsk");
	hkTraceStream::getInstance().disableTitle("Pts");
	hkTraceStream::getInstance().disableTitle("Tim");
	hkTraceStream::getInstance().disableTitle("Tst");
	//hkTraceStream::getInstance().disableTitle("Toi");
	hkTraceStream::getInstance().disableTitle("Toi.del");
	hkTraceStream::getInstance().disableTitle("Man");

	//hkTraceStream::getInstance().disableTitle("int");

	hkTraceStream::getInstance().disableTitle("sub-add");
	hkTraceStream::getInstance().disableTitle("sub-act");
	hkTraceStream::getInstance().disableTitle("sub-upd");

	hkTraceStream::getInstance().disableTitle("add.toi.event");
	hkTraceStream::getInstance().disableTitle("sub.toi.delay");
	hkTraceStream::getInstance().disableTitle("sub.scr");
	  //hkTraceStream::getInstance().disableTitle("Pre");
	  //hkTraceStream::getInstance().disableTitle("Post");
	hkTraceStream::getInstance().disableTitle("sub.toi.inc");
	  hkTraceStream::getInstance().disableTitle("sub.toi.fulsol");
	hkTraceStream::getInstance().disableTitle("sub.toi.integ");
	//hkTraceStream::getInstance().disableTitle("sub.toi.summary");
	hkTraceStream::getInstance().disableTitle("sub.end");

	//hkTraceStream::getInstance().disableTitle("sub.toi.backstep");
	//hkTraceStream::getInstance().disableTitle("sub.toi.abort");

	hkTraceStream::getInstance().disableTitle("rem.dupl");

	hkTraceStream::getInstance().disableTitle("integr");

	hkTraceStream::getInstance().disableTitle("sub.toi.newcon");
	hkTraceStream::getInstance().disableTitle("sub.toi.entMan");

	hkTraceStream::getInstance().disableTitle("sub.toi.conVer");




	//sub.toi.conVer // Constraint Verification
	//sub.toi.newcon // Activation of entities and adding new constraints and entities

#	endif

	m_toiEvents.reserve( world->m_sizeOfToiEventQueue );
}

hkpContinuousSimulation::~hkpContinuousSimulation()
{
	delete m_toiResourceMgr;


#	if defined HK_DEBUG_TOI
	if (hkTraceStream::getInstance().m_stream)
	{
		hkTraceStream::getInstance().m_stream->flush();
		delete hkTraceStream::getInstance().m_stream;
		hkTraceStream::getInstance().m_stream = HK_NULL;
	}
#	endif //HK_DEBUG_TOI
}

hkpSimulation* HK_CALL hkpContinuousSimulation::create( hkpWorld* world )
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkpContinuousSimulation);
	return new hkpContinuousSimulation( world );
}
HK_OPTIONAL_COMPONENT_DEFINE(hkpContinuousSimulation, hkpSimulation::createContinuous, hkpContinuousSimulation::create);

HK_FORCE_INLINE void hkpContinuousSimulation::fireToiEventRemoved( struct hkpToiEvent& event )
{
	// Get the non fixed island of this pair
	hkpSimulationIsland* island = (!event.m_entities[0]->isFixed()) ? event.m_entities[0]->getSimulationIsland() : event.m_entities[1]->getSimulationIsland();
	event.m_contactMgr->removeToi(*island, event.m_properties);
}

class hkpContinuousSimulation::LockCriticalOperationsScope
{
	public:
		LockCriticalOperationsScope( hkpWorld* world ) : m_world( world )
		{
			world->lockCriticalOperations();
		}

		~LockCriticalOperationsScope()
		{
			m_world->unlockAndAttemptToExecutePendingOperations();
		}

	public:
		hkpWorld* m_world;
};

hkpStepResult hkpContinuousSimulation::handleAllToisTill( hkTime minTime )
{
#ifdef HK_PLATFORM_HAS_SPU
	if( m_toiEvents.getSize() == m_toiEvents.getCapacity() )
	{
		HK_WARN_ONCE(0x301738ac, "TOI events queue is full; It is likely the simulation may behave nondeterministically from now on." \
									"Consider increasing the capacity of the queue"	);
	}
#endif
		// find min time event until frame time and next PSI
	while(1)
	{
		int 	earliestEventIndex = -1;
		hkReal  minTimeSoFar = minTime;

		// Since we fire callbacks at this stage, we need to preserve the value set for the TOI's rotate normal.
		hkReal rotateNormal;

		// Lock critical operations for each phase of continuous physics (i.e. each iteration of the outer loop).
		HK_ASSERT(0xad000070, hkpDebugInfoOnPendingOperationQueues::areEmpty(m_world) );
		LockCriticalOperationsScope lockCriticalOperations( m_world );

		//	find the earliest TOI that isn't rejected by a contactPointCallback.
		while(1)
		{
			// start with the default rotate normal.
			rotateNormal = m_world->m_toiCollisionResponseRotateNormal;

			// get earliest event
			for ( int i = 0; i < m_toiEvents.getSize(); i++ )
			{
				hkpToiEvent& toiEvent = m_toiEvents[i];
				if ( toiEvent.m_time < minTimeSoFar )
				{
					earliestEventIndex = i;
					minTimeSoFar = toiEvent.m_time;
				}
#			if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
				else if ( toiEvent.m_time == minTimeSoFar && earliestEventIndex != -1)
				{
					hkUint32 toiUids[2] = { toiEvent.m_entities[0]->getUid(),
											toiEvent.m_entities[1]->getUid() };
					hkUint32 minUids[2] = { m_toiEvents[earliestEventIndex].m_entities[0]->getUid(),
											m_toiEvents[earliestEventIndex].m_entities[1]->getUid() };

					if (   toiUids[0] <  minUids[0] || ( toiUids[0] == minUids[0] && toiUids[1] <  minUids[1] ) )
					{
						// nothing -- use this TOI
						earliestEventIndex = i;
						minTimeSoFar = toiEvent.m_time;
					}
				}
#			endif
			}

			if ( earliestEventIndex < 0 )
			{
				// There are no more TOIs, so return.
				return HK_STEP_RESULT_SUCCESS;
			}
			else
			{
				hkpToiEvent& event = m_toiEvents[earliestEventIndex];
				event.m_properties.m_flags |= hkContactPointMaterial::CONTACT_IS_NEW;

				if ( event.m_contactMgr->fireCallbacksForEarliestToi( event, rotateNormal ) )
				{
					// There was a TOI, which the callbacks did not reject.
					break;
				}
				else
				{
					// The TOI was rejected by a callback. Remove it and look again for the next earliest event.
					// We assume that rejection is very rare. Otherwise, we could to heap order or sort the 
					// array by time.
					m_toiEvents.removeAt( earliestEventIndex );
					earliestEventIndex = -1;
					minTimeSoFar = minTime;
				}
			}
		}

		{
			hkpToiEvent event = m_toiEvents[earliestEventIndex];
			HK_ASSERT(0x4147c4a1, m_currentTime <= event.m_time); 
			m_toiEvents.removeAt(earliestEventIndex);

			// hack: current time update is moved before simulateToi to allow proper checking of newly generated TOI events

			// We're setting currentTime here already, no further collision detection queries
			// will go before that time (this includes out-of-memory event recovery).
			m_currentTime = event.m_time;

			hkStepInfo physicsStepInfo( m_currentTime, m_currentPsiTime );
			m_world->m_dynamicsStepInfo.m_stepInfo = physicsStepInfo;
			m_world->m_collisionInput->m_stepInfo = physicsStepInfo;

			{
				HK_ON_DETERMINISM_CHECKS_ENABLED( m_determinismCheckFrameCounter++; )
				if (!event.m_useSimpleHandling)
				{
					simulateToi(m_world, event, m_physicsDeltaTime, rotateNormal );
				}
				else
				{
					handleSimpleToi(m_world, event, m_physicsDeltaTime, rotateNormal );
				}
			}
			m_toiCounter++;

			if (hkMemoryStateIsOutOfMemory(20)  )
			{
				return HK_STEP_RESULT_MEMORY_FAILURE_DURING_TOI_SOLVE;
			}
		}
	}
}

hkpStepResult hkpContinuousSimulation::advanceTime()
{

	hkpStepResult result = advanceTimeInternal();

	HK_ACCESS_CHECK_OBJECT( m_world, HK_ACCESS_RW );

	//
	// Perform final collision detection of non-continuous agents that need to be updated
	//
	
	// We do this at every advance, even if we have not reached the target time
	// as if we do not, entities involved in the collision may have been deactivated
	// by a later step but still be in the list
	if (m_entitiesNeedingPsiCollisionDetection.getSize())
	{
		collideEntitiesNeedingPsiCollisionDetectionNarrowPhase_toiOnly(*m_world->getCollisionInput(), m_entitiesNeedingPsiCollisionDetection);
	}
	if ( hkMemoryStateIsOutOfMemory(28) )
	{
		return HK_STEP_RESULT_MEMORY_FAILURE_DURING_TOI_SOLVE;
	}

	if ( m_currentTime >= m_simulateUntilTime )
	{

		//
		//	Fire violated constraints callback
		//
		m_world->checkConstraintsViolated();

		//
		// Fire post simulate callbacks, if we are at the end of a simulation step
		//
		if (m_world->m_worldPostSimulationListeners.getSize())
		{
			HK_TIME_CODE_BLOCK("PostSimCb", HK_NULL);
			hkpWorldCallbackUtil::firePostSimulationCallback( m_world );
		}
	}

	return result;
}

hkpStepResult hkpContinuousSimulation::advanceTimeInternal()
{
	HK_TIME_CODE_BLOCK("Physics 2012", HK_NULL);

	HK_ACCESS_CHECK_OBJECT( m_world, HK_ACCESS_RW );

	{
		if ( m_previousStepResult != HK_STEP_RESULT_SUCCESS )
		{
			HK_TIME_CODE_BLOCK("reCollideAfterStepFailure", HK_NULL);

			// Re-collide (note that this will clear and re-evaluate TOIs)
			hkpStepResult result = hkpSimulation::reCollideAfterStepFailure();
			if (result != HK_STEP_RESULT_SUCCESS)
			{
				return result;
			}
		}

		hkTime toiSimulateTime = snapSimulateTimeAndGetTimeToAdvanceTo();

		if ( m_toiEvents.getSize() )
		{
			HK_TIME_CODE_BLOCK("TOIs", HK_NULL);
			hkpStepResult result = handleAllToisTill( toiSimulateTime );
			if ( result != HK_STEP_RESULT_SUCCESS)
			{
				m_previousStepResult = result;
				return result;
			}
		}
		m_currentTime = toiSimulateTime;
	}

	m_previousStepResult = HK_STEP_RESULT_SUCCESS;
	return HK_STEP_RESULT_SUCCESS;
}

void hkpContinuousSimulation::resetCollisionInformationForEntities( hkpEntity** entities, int numEntities, hkpWorld* world, enum ResetCollisionInformation resetInfo )
{
	HK_ASSERT2(0XAD4545DD, world->areCriticalOperationsLocked(), "This is an internal function. It requires the world to be locked.");

	hkpSimulation::resetCollisionInformationForEntities(entities, numEntities, world, resetInfo);

	// Plus: remove TOI events
	if ( (resetInfo & RESET_TOI) != 0)
	{
		if (numEntities != 1)
		{
			removeToiEventsOfEntities(entities, numEntities);
		}
		else
		{
			removeToiEventsOfEntity(entities[0]);
		}
	}
}

void hkpContinuousSimulation::reintegrateAndRecollideEntities( hkpEntity** entities, int numEntities, hkpWorld* world, int reintegrateRecollideMode )
{
	HK_ACCESS_CHECK_OBJECT( world, HK_ACCESS_RW );
	HK_ASSERT2( 0xf03f667e, !world->areCriticalOperationsLocked(), "invalid lockCriticalOperations()");
	world->lockCriticalOperations();

	// Prepare the proper stepInfo.
	const hkStepInfo originalStepInfo = world->m_collisionInput->m_stepInfo;
	const hkStepInfo physicsStepInfo = hkStepInfo(m_currentTime, m_currentPsiTime);
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
				hkSweptTransformUtil::backStepMotionState(m_currentTime, ms);
			}
		}
		hkRigidMotionUtilStep(physicsStepInfo, (hkpMotion*const*)entities, numEntities, HK_OFFSET_OF(hkpEntity, m_motion));
		hkpEntityAabbUtil::entityBatchRecalcAabb(world->getCollisionInput(), entities, numEntities);
	}

	// Re-collide selected bodies.
	{
		// Remove TOIs only, tims will automatically reset in collideEntitiesNarrowPhaseContinuous, AABBs are already done above
		{
			int resetType = hkpSimulation::RESET_ALL - hkpSimulation::RESET_AABB - hkpSimulation::RESET_TIM;
			HK_ASSERT( 0xf0df2323, resetType == hkpSimulation::RESET_TOI );
			resetCollisionInformationForEntities(entities, numEntities, world, hkpSimulation::ResetCollisionInformation(resetType));
		}

		// Perform broadphase collision detection.
		if  ( reintegrateRecollideMode & hkpWorld::RR_MODE_RECOLLIDE_BROADPHASE )
		{
			collideEntitiesBroadPhaseContinuous(entities, numEntities, world);
		}

		// Warning: islands do not get merged here, therefore the following call is suited to process entities from multiple islands.
		// It activates all inactive islands it touches.

		// Perform narrowphase collision detection.
		if  ( reintegrateRecollideMode & hkpWorld::RR_MODE_RECOLLIDE_NARROWPHASE )
		{
			// recollide all agents, this will invalidate tims
			collideEntitiesNarrowPhaseContinuous(entities, numEntities, *world->m_collisionInput);
		}
		else if ( reintegrateRecollideMode & hkpWorld::RR_MODE_REINTEGRATE )
		{
			// Reintegrate TIMs for all agents involved
			resetCollisionInformationForEntities(entities, numEntities, world, hkpSimulation::RESET_TIM);
		}

	}

	world->unlockAndAttemptToExecutePendingOperations();

	world->m_collisionInput->m_stepInfo = originalStepInfo;
}



void hkpContinuousSimulation::assertThereIsNoCollisionInformationForEntities( hkpEntity** entities, int numEntities, hkpWorld* world )
{
#if defined HK_DEBUG
	HK_ASSERT(0xad44bb3e, numEntities && entities[0]->getWorld() );

	if (m_toiEvents.getSize())
	{
		HK_COMPILE_TIME_ASSERT( sizeof(hkUlong) == sizeof(hkpLinkedCollidable*) );

		hkMap<hkUlong>::Temp entitiesMap(numEntities);

		{
			for (int i = 0; i < numEntities; i++)
			{
				hkpEntity* entity = entities[i];

				// Check that every entity is only passed once
				HK_ON_DEBUG( hkMap<hkUlong>::Iterator it = entitiesMap.findKey(hkUlong(entity)); )
					HK_ASSERT2(0xad45db3, !entitiesMap.isValid(it), "Same entity passed more than once to hkpContinuousSimulation::removeToiEventsOfEntities.");

				entitiesMap.insert(hkUlong(entity), 0);
			}
		}

		{
			for (int i = 0; i < m_toiEvents.getSize(); i++)
			{
				hkpToiEvent& toiEvent = m_toiEvents[i];

				hkMap<hkUlong>::Iterator it0 = entitiesMap.findKey(hkUlong(toiEvent.m_entities[0]));
				hkMap<hkUlong>::Iterator it1 = entitiesMap.findKey(hkUlong(toiEvent.m_entities[1]));

				HK_ASSERT2(0xad67dd8a, !entitiesMap.isValid(it0) && !entitiesMap.isValid(it1), "TOI found for one of the checked entities." );
			}
		}
	}
#endif // if defined HK_DEBUG
}

void hkpContinuousSimulation::removeCollisionInformationForAgent( hkpAgentNnEntry* agent )
{
	HK_ACCESS_CHECK_OBJECT( hkpGetRigidBody(agent->m_collidable[0])->getWorld(), HK_ACCESS_RW );

	for (int i = m_toiEvents.getSize()-1; i >= 0; i--)
	{
		if (m_toiEvents[i].m_contactMgr == agent->m_contactMgr)
		{
			// the TOI-events are unsorted anyways..
			m_toiEvents.removeAt(i);
		}
	}
}

void hkpContinuousSimulation::assertThereIsNoCollisionInformationForAgent( hkpAgentNnEntry* agent )
{
	for (int i = 0; i < m_toiEvents.getSize(); i++)
	{
		HK_ASSERT2(0xad789dda, m_toiEvents[i].m_contactMgr != agent->m_contactMgr, "TOI events relating to the agent were found on the TOI-events list.");
	}
}


void hkpContinuousSimulation::warpTime( hkReal warpDeltaTime )
{
	//
	// Reset time of all TOI events queued in the world
	//
	HK_ASSERT2(0xad33ba11, m_world->m_simulationType >= hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS, "hkpContinuousSimulation::warpTime called incorrectly.");

	hkpContinuousSimulation& sim = static_cast<hkpContinuousSimulation&>(*m_world->m_simulation);
	for (int i = 0; i < sim.m_toiEvents.getSize(); i++)
	{
		sim.m_toiEvents[i].m_time += warpDeltaTime;
	}
}


//////////////////////////////////////////////////////////////////////////
//
//  Continuous Collision Detection
//
//////////////////////////////////////////////////////////////////////////


void hkpContinuousSimulation::collideInternal( const hkStepInfo& stepInfoIn )
{
	HK_TIMER_BEGIN("Collide", HK_NULL);

	if ( m_previousStepResult != HK_STEP_RESULT_SUCCESS )
	{
		// Clear all TOIs that may have been generated
		for ( int i = m_toiEvents.getSize()-1; i>=0; i-- )
		{
			fireToiEventRemoved( m_toiEvents[i] );
		}
		m_toiEvents.clear();
	}

	//
	// Initialize all parameters of the dynamics step into that depend on the stepInfo
	//
	{
		// Step Info
		m_world->m_dynamicsStepInfo.m_stepInfo = stepInfoIn;
		m_world->m_collisionInput->m_stepInfo   = stepInfoIn;

		// Update Solver Info
		hkpSolverInfo& solverInfo  = m_world->m_dynamicsStepInfo.m_solverInfo;
		solverInfo.m_deltaTime	  = stepInfoIn.m_deltaTime    * solverInfo.m_invNumSteps;
		solverInfo.m_invDeltaTime = stepInfoIn.m_invDeltaTime * solverInfo.m_numSteps;
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
			collideEntitiesBroadPhaseContinuous( island->m_entities.begin(), island->m_entities.getSize(), m_world );

			if (hkMemoryStateIsOutOfMemory(21)  )
			{
				m_world->unlockAndAttemptToExecutePendingOperations();
				HK_TIMER_END();
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
			collideIslandNarrowPhaseContinuous( island, *m_world->m_collisionInput );

			if ( hkMemoryStateIsOutOfMemory(22)  )
			{
				m_world->unlockAndAttemptToExecutePendingOperations();
				HK_TIMER_END();
				return;
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

	HK_TIMER_END();
}


void hkpContinuousSimulation::collideEntitiesBroadPhaseContinuousFindPairs( hkpEntity** entities, int numEntities, hkpWorld* world, hkArray<hkpBroadPhaseHandlePair>& newPairs, hkArray<hkpBroadPhaseHandlePair>& delPairs )
{
	if(!numEntities)
	{
		return;
	}

	HK_ASSERT2(0x3454285e, numEntities > 0, "No entities in an island?");
	HK_ASSERT(0xad63ee35, world->areCriticalOperationsLocked());
	HK_ON_DEBUG( assertThereIsNoCollisionInformationForEntities(entities, numEntities, world) );
	HK_ON_DEBUG_MULTI_THREADING( hkpSimulationIsland* island = HK_NULL );

	HK_TIMER_BEGIN_LIST("BroadPhase", "GatherAabbs");

	{
		hkAabbUint32* aabbs32 = hkAllocateStack<hkAabbUint32>( numEntities );
		const hkpBroadPhaseHandle** broadPhaseHandles = hkAllocateStack<const hkpBroadPhaseHandle*>(numEntities);

		int numAabbs = numEntities;
		{
			hkpEntity **e = entities;

			hkAabbUint32* aabb32 = aabbs32;
			const hkpBroadPhaseHandle **handle = broadPhaseHandles;

			for( int i = numAabbs-1; i >=0 ; e++, i-- )
			{
				hkpEntity* entity = *e;
				
				hkCheckDeterminismUtil::checkMt(0xad000300, entity->getUid());
				//hkCheckDeterminismUtil::checkMt(0xad000301, entity->getSimulationIsland()->m_storageIndex); 
				hkCheckDeterminismUtil::checkMt(0xad000302, static_cast<hkpRigidBody*>(entity)->getTransform());

#if !defined(HK_DEBUG_MULTI_THREADING)
				if ( i >= 4 ) { hkMath::forcePrefetch<64>( e[4]->getCollidable() ); }
#endif
				//
				// some debug code
				//
				{
					HK_ASSERT2(0xad33e242, entity->getWorld() == world, "Entities don't belong to the world being processed.");
					hkCheckDeterminismUtil::checkMt(0xad000001, entity->getUid());
#if defined(HK_DEBUG_MULTI_THREADING)
					if (!world->m_multiThreadCheck.isMarkedForWrite() )
					{
						HK_ASSERT( 0xf023ed45, island==HK_NULL || island == entity->getSimulationIsland());
						island = entity->getSimulationIsland();
					}
#endif
				}

				const hkpCollidable* collidable = entity->getCollidable();

				handle[0] = collidable->getBroadPhaseHandle();

				//
				// Calculate AABB if not yet done (e.g. no IntegrateMotion job was processed yet or (on PlayStation(R)3) the shape was not allowed to go onto SPU).
				//
				if ( !collidable->m_boundingVolumeData.isValid() )
				{
					//HK_TIME_CODE_BLOCK("CalcAabbs", HK_NULL);
					hkpEntityAabbUtil::entityBatchRecalcAabb(world->getCollisionInput(), &entity, 1);
				}

				// Copy the already calculated AABB into our local contiguous array.
			#ifdef HK_ARCH_ARM
				HK_ASSERT2(0x46aefcee, (((hkUlong)&collidable->m_boundingVolumeData) & 0x3) == 0, "Unaligned bounding volume data!");
			#else
				HK_ASSERT2(0x46aefcee, (((hkUlong)&collidable->m_boundingVolumeData) & 0xF) == 0, "Unaligned bounding volume data!");
			#endif
				const hkAabbUint32& tmp = reinterpret_cast<const hkAabbUint32&>(collidable->m_boundingVolumeData);
				hkAabbUtil::uncompressExpandedAabbUint32(tmp, *aabb32);

				aabb32++;
				handle++;
			}
		}

		HK_TIMER_SPLIT_LIST("3AxisSweep");

		{
			world->m_broadPhase->lock();

			// we can do the const cast, as we now have the lock to the broadphase
			hkpBroadPhaseHandle** nonConstHandles = const_cast<hkpBroadPhaseHandle**>(broadPhaseHandles);

			world->m_broadPhase->updateAabbsUint32( nonConstHandles, aabbs32, numAabbs, newPairs, delPairs );

			world->m_broadPhase->unlock();
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
	}
	HK_TIMER_END_LIST();
}


void hkpContinuousSimulation::collideEntitiesBroadPhaseContinuous( hkpEntity** entities, int numEntities, hkpWorld* world, hkChar* exportFinished )
{
	if(!numEntities)
	{
		return;
	}

	HK_ASSERT2(0x3454285e, numEntities > 0, "No entities in an island?");
	HK_ASSERT(0xad63ee35, world->areCriticalOperationsLocked());
	HK_ON_DEBUG( assertThereIsNoCollisionInformationForEntities(entities, numEntities, world) );
	HK_ON_DEBUG_MULTI_THREADING( hkpSimulationIsland* island = HK_NULL );

	HK_TIMER_BEGIN_LIST("BroadPhase", "GatherAabbs");

	hkLocalArray<hkpBroadPhaseHandlePair> newPairs( world->m_broadPhaseUpdateSize );
	hkLocalArray<hkpBroadPhaseHandlePair> delPairs( world->m_broadPhaseUpdateSize );
	{
		hkAabbUint32* aabbs32 = hkAllocateStack<hkAabbUint32>( numEntities );
		const hkpBroadPhaseHandle** broadPhaseHandles = hkAllocateStack<const hkpBroadPhaseHandle*>(numEntities);

		int numAabbs = numEntities;
		{
			hkpEntity **e = entities;

			hkAabbUint32* aabb32 = aabbs32;
			const hkpBroadPhaseHandle **handle = broadPhaseHandles;

			for( int i = numAabbs-1; i >=0 ; e++, i-- )
			{
				hkpEntity* entity = *e;

#if !defined(HK_DEBUG_MULTI_THREADING)
				if ( i >= 4 ) { hkMath::forcePrefetch<64>( e[4]->getCollidable() ); }
#endif
				//
				// some debug code
				//
				{
					HK_ASSERT2(0xad33e242, entity->getWorld() == world, "Entities don't belong to the world being processed.");
					hkCheckDeterminismUtil::checkMt( 0xf00001e0, entity->getUid());
#if defined(HK_DEBUG_MULTI_THREADING)
					if (!world->m_multiThreadCheck.isMarkedForWrite() )
					{
						HK_ASSERT( 0xf023ed45, island==HK_NULL || island == entity->getSimulationIsland());
						island = entity->getSimulationIsland();
					}
					// we need read access, so no other people can get write access and change the data
					entity->markForRead();
#endif
				}

				const hkpCollidable* collidable = entity->getCollidable();

				handle[0] = collidable->getBroadPhaseHandle();

				//
				// Calculate AABB if not yet done (e.g. no IntegrateMotion job was processed yet or (on PlayStation(R)3) the shape was not allowed to go onto SPU).
				//
				if ( !collidable->m_boundingVolumeData.isValid() )
				{
					//HK_TIME_CODE_BLOCK("CalcAabbs", HK_NULL);
					hkpEntityAabbUtil::entityBatchRecalcAabb(world->getCollisionInput(), &entity, 1);
				}

					// Copy the already calculated AABB into our local contiguous array.
				#ifdef HK_ARCH_ARM
					HK_ASSERT2(0x46aefcee, (((hkUlong)&collidable->m_boundingVolumeData) & 0x3) == 0, "Unaligned bounding volume data!");
				#else
					HK_ASSERT2(0x46aefcee, (((hkUlong)&collidable->m_boundingVolumeData) & 0xF) == 0, "Unaligned bounding volume data!");
				#endif
				const hkAabbUint32& tmp = reinterpret_cast<const hkAabbUint32&>(collidable->m_boundingVolumeData);
				hkAabbUtil::uncompressExpandedAabbUint32(tmp, *aabb32);

				aabb32++;
				handle++;
			}
		}

		HK_TIMER_SPLIT_LIST("3AxisSweep");

		{
			world->m_broadPhase->lock();

			// we can do the const cast, as we now have the lock to the broadphase
			hkpBroadPhaseHandle** nonConstHandles = const_cast<hkpBroadPhaseHandle**>(broadPhaseHandles);

			world->m_broadPhase->updateAabbsUint32( nonConstHandles, aabbs32, numAabbs, newPairs, delPairs );

			world->m_broadPhase->unlock();
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
	}

	waitForSolverExport(exportFinished);

	removeAndAddPairs(world, entities, delPairs, newPairs);

#if defined(HK_DEBUG_MULTI_THREADING)
	{ for( int e = 0; e < numEntities;	e++ ) { entities[e]->unmarkForRead(); } }
#endif

	HK_TIMER_END_LIST();
}


void hkpContinuousSimulation::collideIslandNarrowPhaseContinuous( hkpSimulationIsland* isle, const hkpProcessCollisionInput& input)
{
	HK_ASSERT(0xad63ee34, isle->getWorld()->areCriticalOperationsLocked() && isle->m_isInActiveIslandsArray);
	HK_TIMER_BEGIN("NarrowPhase", HK_NULL);
	{
		HK_ALIGN16(typedef hkpProcessCollisionOutput hkAlignedCollisionOutput);
		hkAlignedCollisionOutput processOutput(isle);

		hkpAgentNnTrack *const tracks[2] = { &isle->m_narrowphaseAgentTrack, &isle->m_midphaseAgentTrack };
		for ( int j = 0; j < 2; ++j )
		{
			hkpAgentNnTrack& agentTrack = *tracks[j];
			HK_FOR_ALL_AGENT_ENTRIES_BEGIN( agentTrack, entry )
			{
				hkpCollidable* collA = entry->getCollidableA();
				hkpCollidable* collB = entry->getCollidableB();

				// check for at least one of the objects being active. We are disabling this optimization, as objects can be moved using setPosition
				//if ( collA->getMotionState()->m_deactivationCounter | collB->getMotionState()->m_deactivationCounter )
				{
					//hkMath::prefetch128( collA );
					//hkMath::prefetch128( collB );
					hkMath::prefetch128( entry->m_contactMgr );
					
					hkMath::prefetch128( hkAddByteOffset(entry, 128) );

					input.m_collisionQualityInfo = input.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex );
					input.m_createPredictiveAgents = input.m_collisionQualityInfo->m_useContinuousPhysics;

					processOutput.reset();
					hkAgentNnMachine_ProcessAgent( entry, input, processOutput, entry->m_contactMgr );

					{
						if (hkMemoryStateIsOutOfMemory(23) )
						{
							HK_TIMER_END();
							return;
						}
					}

					if ( !processOutput.isEmpty() )
					{
						entry->m_contactMgr->processContact( *collA, *collB, input, processOutput );
					}

					if ( processOutput.hasToi()  )
					{
						HK_ASSERT( 0xf0324354, input.m_stepInfo.m_startTime <= processOutput.m_toi.m_time );
						HK_ASSERT2(0xad8765dd, processOutput.m_toi.m_time >= m_currentTime, "Generating a TOI event before hkpWorld->m_currentTime.");
						addToiEvent(processOutput, *entry );
					}
				}
				//else
				//{
				//	// deactive agent, just touch it to allow it to update its internal time information
				//	hkAgentNnMachine_TouchAgent( entry, input );
				//}
			}
			HK_FOR_ALL_AGENT_ENTRIES_END;
		}
	}
	HK_TIMER_END();
}




// This is a copy-paste of hkpSimulation's version + activation + extra TOI event generation
// <todo: interface inconsistency: discrete takes hkpWorld* and no collisionInput (it assumes that hkpWorld's collison input is valid), and this function does the oposite.... and ignores the input anyways)
// well, you shouldn't need to pass the collison info, unless you do some black magic here...
// the world should keep a consistent time + input.stepInfo information.
void hkpContinuousSimulation::collideEntitiesNarrowPhaseContinuous( hkpEntity** entities, int numEntities, const hkpProcessCollisionInput& input)
{
	HK_ASSERT2(0xadfe825d, numEntities, "Must call the function with a non-zero number of entities.");
	HK_ASSERT2(0xadfe825e, entities[0]->getWorld()->areCriticalOperationsLocked(), "World must be locked when calling hkpContinuousSimulation::collideEntitiesNarrowPhaseContinuous.");

	processAgentsOfEntities( const_cast<hkpEntity**>(entities), numEntities, input, (AgentEntryProcessFunction)(&hkpContinuousSimulation::processAgentCollideContinuous), FIND_CONTACTS_DEFAULT );
}

// void hkpContinuousSimulation::collideEntityPair( hkpEntity* entityA, hkpEntity* entityB )
// {
// 	{
// 		processOutput.reset();
// 		processInput.m_collisionQualityInfo = processInput.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex );
// 		// if both bodies are inactive -- perform discrete collision detection only.
// 		processInput.m_createPredictiveAgents = processInput.m_collisionQualityInfo->m_useContinuousPhysics;
//
// 		hkAgentNnMachine_ProcessAgent( entry, processInput, processOutput, entry->m_contactMgr );
//
// 		if ( !processOutput.isEmpty() )
// 		{
// 			hkpCollidable& collA = *entry->getCollidableA();
// 			hkpCollidable& collB = *entry->getCollidableB();
// 			entry->m_contactMgr->processContact( collA, collB, processInput, processOutput );
// 		}
//
// 		// Adding TOI events to world's queue
// 		if ( processOutput.hasToi() )
// 		{
// 			HK_ASSERT( 0xf0324354, processInput.m_stepInfo.m_startTime <= processOutput.m_toi.m_time );
// 			HK_ON_DEBUG( hkpRigidBody* body = hkpGetRigidBody(entry->getCollidableA()); )
// 				HK_ASSERT2(0xad8765dd, body && body->getWorld() && processOutput.m_toi.m_time >= m_currentTime, "Generating a TOI event before hkpWorld->m_currentTime.");
// 			addToiEvent(processOutput, *entry);
// 		}
// 	}
// }

void hkpContinuousSimulation::processAgentCollideContinuous(hkpAgentNnEntry* entry, const hkpProcessCollisionInput& processInput, hkpProcessCollisionOutput& processOutput)
{
	processOutput.reset();
	processInput.m_collisionQualityInfo = processInput.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex );
	// if both bodies are inactive -- perform discrete collision detection only.
	processInput.m_createPredictiveAgents = processInput.m_collisionQualityInfo->m_useContinuousPhysics;

	hkAgentNnMachine_ProcessAgent( entry, processInput, processOutput, entry->m_contactMgr );

	if (hkMemoryStateIsOutOfMemory(24) )
	{
		return;
	}

	if ( !processOutput.isEmpty() )
	{
		hkpCollidable& collA = *entry->getCollidableA();
		hkpCollidable& collB = *entry->getCollidableB();
		entry->m_contactMgr->processContact( collA, collB, processInput, processOutput );
	}

	// Adding TOI events to world's queue
	if ( processOutput.hasToi() )
	{
		HK_ASSERT( 0xf0324354, processInput.m_stepInfo.m_startTime <= processOutput.m_toi.m_time );
		HK_ON_DEBUG( hkpRigidBody* body = hkpGetRigidBody(entry->getCollidableA()); )
		HK_ASSERT2(0xad8765dd, body && body->getWorld() && processOutput.m_toi.m_time >= m_currentTime, "Generating a TOI event before hkpWorld->m_currentTime.");
		addToiEvent(processOutput, *entry);
	}
}



//////////////////////////////////////////////////////////////////////////
//
//  Adding & removing TOI events
//
//////////////////////////////////////////////////////////////////////////



void hkpContinuousSimulation::addToiEvent(const hkpProcessCollisionOutput& processOutput, const hkpAgentNnEntry& entry)
{
	if ( m_toiEvents.getSize() == m_toiEvents.getCapacity())
	{
		HK_WARN_ALWAYS(0xf0323454, "TOI event queue full, consider using HK_COLLIDABLE_QUALITY_DEBRIS for some objects or increase hkpWorldCinfo::m_sizeOfToiEventQueue" );
		return;
	}

	hkpToiEvent& event = *m_toiEvents.expandByUnchecked(1);

	event.m_time = processOutput.m_toi.m_time;
	event.m_useSimpleHandling = m_world->getCollisionDispatcher()->getCollisionQualityInfo(entry.m_collisionQualityIndex)->m_useSimpleToiHandling;
	event.m_seperatingVelocity = processOutput.m_toi.m_seperatingVelocity;
	event.m_contactPoint = processOutput.m_toi.m_contactPoint;
	event.m_entities[0] = reinterpret_cast<hkpEntity*>(entry.m_collidable[0]->getOwner());
	event.m_entities[1] = reinterpret_cast<hkpEntity*>(entry.m_collidable[1]->getOwner());
	event.m_properties = static_cast<const hkpContactPointProperties&>(processOutput.m_toi.m_properties);
	event.m_contactMgr = static_cast<hkpSimpleConstraintContactMgr*>(entry.m_contactMgr);

	hkString::memCpy4(event.m_extendedUserDatas, processOutput.m_toi.m_properties.m_extendedUserDatas, sizeof(event.m_extendedUserDatas) >> 2);

	if ( hkDebugToi && HK_IS_TRACE_ENABLED( event.m_entities[0], event.m_entities[1] ) )
	{
		hkToiPrintf("add.toi.event", "#    add  TOI     @ %2.7f: %-6s %-6s \n", event.m_time, event.m_entities[0]->getName(), event.m_entities[1]->getName());
	}
}


void hkpContinuousSimulation::removeToiEventsOfEntities( hkpEntity** entities, int numEntities )
{
	HK_ASSERT2(0xad34bb3e, numEntities && entities[0]->getWorld() && entities[0]->getWorld()->areCriticalOperationsLocked(), "The world must be locked when removing TOI events with hkpContinuousSimulation::removeToiEventsOfEntities().");

	HK_ACCESS_CHECK_OBJECT( entities[0]->getWorld(), HK_ACCESS_RW );

	if (m_toiEvents.getSize())
	{
		HK_COMPILE_TIME_ASSERT( sizeof(hkUlong) == sizeof(hkpLinkedCollidable*) );

		hkMap<hkUlong> entitiesMap(numEntities);

		{
			for (int i = 0; i < numEntities; i++)
			{
				hkpEntity* entity = entities[i];

				// Check that every entity is only passed once
				HK_ON_DEBUG( hkMap<hkUlong>::Iterator it = entitiesMap.findKey(hkUlong(entity)); )
				HK_ASSERT2(0xad45db3, !entitiesMap.isValid(it), "Same entity passed more than once to hkpContinuousSimulation::removeToiEventsOfEntities.");

				entitiesMap.insert(hkUlong(entity), 0);
			}
		}

		{
			for (int i = 0; i < m_toiEvents.getSize(); i++)
			{
				hkpToiEvent& toiEvent = m_toiEvents[i];

				hkMap<hkUlong>::Iterator it0 = entitiesMap.findKey(hkUlong(toiEvent.m_entities[0]));
				hkMap<hkUlong>::Iterator it1 = entitiesMap.findKey(hkUlong(toiEvent.m_entities[1]));

				if ( entitiesMap.isValid(it0) || entitiesMap.isValid(it1) )
				{
					// remove this event, and don't worry about the order of remaining m_toiEvents
					fireToiEventRemoved( toiEvent );
					m_toiEvents.removeAt(i--);
				}
			}
		}
	}
}

void hkpContinuousSimulation::removeToiEventsOfEntity( hkpEntity* entity )
{
	HK_ACCESS_CHECK_OBJECT( entity->getWorld(), HK_ACCESS_RW );

	for (int i = 0; i < m_toiEvents.getSize(); i++)
	{
		hkpToiEvent& toiEvent = m_toiEvents[i];
		if (toiEvent.m_entities[0] == entity || toiEvent.m_entities[1] == entity)
		{
			// remove this event, and don't worry about the order of remaining events
			fireToiEventRemoved( m_toiEvents[i] );
			m_toiEvents.removeAt(i--);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//
//  One-island collision detection used in simulateToi
//
//////////////////////////////////////////////////////////////////////////


	// Collides those entities with all other entities which DO NOT necessarily belong to the SAME island.
void hkpContinuousSimulation::collideEntitiesOfOneIslandNarrowPhaseContinuous_toiOnly( hkpEntity** entities, int numEntities, const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection )
{
	// Narrowphase -- !! process each agent once only.
	HK_ASSERT(0xf0ff0044, numEntities > 0);
	HK_ASSERT(0xad63ee33, entities[0]->getWorld()->areCriticalOperationsLocked());

	// Assume all entities belong to the same island
	hkpSimulationIsland* island = entities[0]->getSimulationIsland();

#if defined(HK_DEBUG)
	{ for (int i = 0; i < numEntities; i++){	HK_ASSERT(0xf0ff0045, entities[i]->getSimulationIsland() == island);	} }
#endif

	// use a lookup table to determine whether a given agent should be fired
	int totalNumEntities = island->m_entities.getSize();
	hkLocalArray<hkBool> isProcessed(totalNumEntities);
	isProcessed.setSizeUnchecked(totalNumEntities);
	hkString::memSet(isProcessed.begin(), 0, totalNumEntities);

	HK_ALIGN16(typedef hkpProcessCollisionOutput hkAlignedCollisionOutput);
	hkAlignedCollisionOutput processOutput(island);

	HK_TIMER_BEGIN( "NarrowPhaseTOI", HK_NULL );

	// process agents for entities only, when both agent.entities have already been marked-processed
	// OR when the other entity belongs to a different island
	{
		for (int i = 0; i < numEntities; i++)
		{
			hkpEntity& entity = *entities[i];
			hkpLinkedCollidable& linkedCollidable = *entity.getLinkedCollidable();

			HK_ASSERT(0xad0987fe, entity.getSimulationIsland() == island);

			isProcessed[ entity.m_storageIndex ] = true;

			// cycle through all agents.
			hkArray<struct hkpLinkedCollidable::CollisionEntry> collisionEntriesTmp;
			linkedCollidable.getCollisionEntriesSorted(collisionEntriesTmp); 
			const hkArray<struct hkpLinkedCollidable::CollisionEntry>& collisionEntries = collisionEntriesTmp;

			for (int a = 0; a < collisionEntries.getSize(); a++) 
			{
				const hkpLinkedCollidable::CollisionEntry& centry = collisionEntries[a]; 

				hkpEntity* partner = static_cast<hkpEntity*>(centry.m_partner->getOwner());
				HK_ASSERT2( 0xf0321245, hkpGetRigidBody(centry.m_partner), "Internal error, entity expected, something else found");
				// Note: partner may be in another simulation island -- that is okay
				// as this call is (was meant to be) used by simulateToi only, which locks the world
				// and islands are not merged after the last collide broadphase.
				if ( partner->getSimulationIsland() == island &&	isProcessed[ partner->m_storageIndex ])
				{
					continue;
				}

				hkpAgentNnEntry* entry = centry.m_agentEntry;

				const hkBool32 useContinuousPhysics = m_world->getCollisionDispatcher()->getCollisionQualityInfo( entry->m_collisionQualityIndex )->m_useContinuousPhysics;
				const hkBool useSimplifiedToi = m_world->getCollisionDispatcher()->getCollisionQualityInfo( entry->m_collisionQualityIndex )->m_useSimpleToiHandling;
				if (useContinuousPhysics && !useSimplifiedToi)
				{
					processOutput.reset();
					input.m_collisionQualityInfo = input.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex );
					input.m_createPredictiveAgents = input.m_collisionQualityInfo->m_useContinuousPhysics;

					hkAgentNnMachine_ProcessAgent( entry, input, processOutput, entry->m_contactMgr );

					if (hkMemoryStateIsOutOfMemory(25) )
					{
						goto END;
					}

					if ( !processOutput.isEmpty() )
					{
						hkpCollidable& collA = *entry->getCollidableA();
						hkpCollidable& collB = *entry->getCollidableB();
						entry->m_contactMgr->processContact( collA, collB, input, processOutput );
					}
					if ( processOutput.hasToi() )
					{
						HK_ASSERT( 0xf0324354, input.m_stepInfo.m_startTime <= processOutput.m_toi.m_time );
						HK_ASSERT2(0xad8765dd, processOutput.m_toi.m_time >= m_currentTime, "Generating a TOI event before hkpWorld->m_currentTime.");
						addToiEvent(processOutput, *entry);
					}
				}
				else if (HK_NULL == entitiesNeedingPsiCollisionDetection.getWithDefault(entity.getUid(), HK_NULL))
				{
					entity.addReference();
					entitiesNeedingPsiCollisionDetection.insert(entity.getUid(), &entity);
				}
			}
		}
	}
END:
	HK_TIMER_END();
}

namespace {
	struct EntryAndIsland
	{
		hkpAgentNnEntry* m_entry;
		hkpSimulationIsland* m_island;
	};

}

void hkpContinuousSimulation::collideEntitiesNeedingPsiCollisionDetectionNarrowPhase_toiOnly( const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection )
{
	HK_TIMER_BEGIN("Recollide PSI", HK_NULL);

	m_world->lockCriticalOperations();

	hkPointerMap<hkpAgentNnEntry*, int> entriesToCollideMap;
	hkArray<EntryAndIsland> entriesToCollide;

	// Collect all agentNnEntries to collide.
	// Iterate through all entities, and collect non-continuous collision entries.
	//
	{
		hkPointerMap<hkpEntity*, int>::Iterator it = entitiesNeedingPsiCollisionDetection.getIterator();
		while (entitiesNeedingPsiCollisionDetection.isValid(it))
		{
			const hkpEntity* entity = entitiesNeedingPsiCollisionDetection.getValue(it);

			hkCheckDeterminismUtil::checkMt( 0xad000002, entity->getUid() );
			hkCheckDeterminismUtil::checkMt( 0xad000003, entity->m_storageIndex );

			HK_ASSERT2(0xad763234, !entity->isFixed() || m_world->m_simulationType == hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS, "The entity cannot be fixed if we're using hkpMultithreadedSimulation, or we need to sort the collision entries in the next line.");
			const hkArray<hkpLinkedCollidable::CollisionEntry>& collisionEntries = entity->getLinkedCollidable()->getCollisionEntriesDeterministicUnchecked();
			const int numCollisionEntries = collisionEntries.getSize();
			for (int e = 0; e < numCollisionEntries; e++)
			{
				hkpAgentNnEntry* entry = collisionEntries[e].m_agentEntry;

				if (!entriesToCollideMap.getWithDefault(entry, HK_NULL))
				{
					const hkpCollisionQualityInfo* collisionQualityInfo = m_world->getCollisionDispatcher()->getCollisionQualityInfo(entry->m_collisionQualityIndex);
					hkBool32 isContinuous = collisionQualityInfo->m_useContinuousPhysics;
					hkBool isSimplified = collisionQualityInfo->m_useSimpleToiHandling;
					hkBool32 motionIsFrozen = static_cast<const hkpRigidBody*>(entity)->getRigidMotion()->getMotionState()->getSweptTransform().getInvDeltaTimeSr().isEqualZero();
					// if nont continuouse or if (uses simple TOI and was backstepped/frozen)
					if (!isContinuous || (isSimplified && motionIsFrozen))
					{
						hkpSimulationIsland* island = entity->getSimulationIsland();
						if ( HK_VERY_UNLIKELY(entity->isFixed()) )
						{
							// get other entity connected to the entry
							const hkpEntity* entityA = hkpGetRigidBodyUnchecked( entry->m_collidable[0] );
							const hkpEntity* entityB = hkpGetRigidBodyUnchecked( entry->m_collidable[1] );
							const hkpEntity* otherEntity = hkSelectOther(entity, entityA, entityB);
							island = otherEntity->getSimulationIsland();

							HK_ASSERT2(0xad903062, ( ( entry->m_nnTrackType == HK_AGENT3_NARROWPHASE_TRACK ) ? hkAgentNnMachine_IsEntryOnTrack(island->m_narrowphaseAgentTrack, entry) : hkAgentNnMachine_IsEntryOnTrack(island->m_midphaseAgentTrack, entry) ), "Agent entry not found in the island, as expected.");
						}

						entriesToCollideMap.insert(entry, 1);
						EntryAndIsland& ei = entriesToCollide.expandOne();
						ei.m_entry = entry;
						ei.m_island = island;
					}
				}
			}

			entity->removeReference();
			it = entitiesNeedingPsiCollisionDetection.getNext(it);
		}

		// References are removed. Now clear the entity list.
		entitiesNeedingPsiCollisionDetection.clear();
	}

	// Collide all found entries
	//
	{
		for (int i = 0; i < entriesToCollide.getSize(); i++)
		{
			hkpAgentNnEntry* entry = entriesToCollide[i].m_entry;
			hkpSimulationIsland* island = entriesToCollide[i].m_island;

			HK_ASSERT2(0xad903061, ( ( entry->m_nnTrackType == HK_AGENT3_NARROWPHASE_TRACK ) ? hkAgentNnMachine_IsEntryOnTrack(island->m_narrowphaseAgentTrack, entry) : hkAgentNnMachine_IsEntryOnTrack(island->m_midphaseAgentTrack, entry) ), "Agent entry not found in the island, as expected.");

			HK_ALIGN16(typedef hkpProcessCollisionOutput hkAlignedCollisionOutput);
			hkAlignedCollisionOutput processOutput(island);

			processOutput.reset();

			// Disable Toi's
			input.m_collisionQualityInfo = input.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::COLLISION_QUALITY_PSI );
			input.m_createPredictiveAgents = input.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex )->m_useContinuousPhysics;

			hkAgentNnMachine_ProcessAgent( entry, input, processOutput, entry->m_contactMgr );

			if (hkMemoryStateIsOutOfMemory(26) )
			{
				goto END;
			}

			if ( !processOutput.isEmpty() )
			{
				hkpCollidable& collA = *entry->getCollidableA();
				hkpCollidable& collB = *entry->getCollidableB();
				entry->m_contactMgr->processContact( collA, collB, input, processOutput );
			}

			HK_ASSERT2(0xad810151, !processOutput.hasToi(), "No TOI's allowed in this function." );
		}
	}

END:
	m_world->unlockAndAttemptToExecutePendingOperations();

	HK_TIMER_END();
}


void hkpContinuousSimulation::collideEntitiesOfOneIslandContinuous_toiOnly( hkpEntity** entities, int numEntities, hkpWorld* world, const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection )
{
	HK_ON_DETERMINISM_CHECKS_ENABLED(m_determinismCheckFrameCounter++);

	if (numEntities)
	{
		HK_ASSERT2(0XAD4D4D32, entities[0]->getWorld()->areCriticalOperationsLocked(), "Internal error: world must be locked when calling hkpContinuousSimulation::collideEntitiesOfOneIslandContinuous_toiOnly(...).");

		collideEntitiesBroadPhaseContinuous (entities, numEntities, world);

		if (hkMemoryStateIsOutOfMemory(27) )
		{
			return;
		}
		collideEntitiesOfOneIslandNarrowPhaseContinuous_toiOnly(entities, numEntities, input, entitiesNeedingPsiCollisionDetection );
	}
}

void hkpContinuousSimulation::deleteToiEventsForActiveEntitiesInIsland( const hkpSimulationIsland* island, const hkFixedArray<hkUchar>& entityState, hkArray<hkpToiEvent>& toiEvents )
{
	HK_ASSERT2(0xad3dfb3f, island->m_world && island->m_world->areCriticalOperationsLocked(), "The world must be locked when removing TOI events with hkpContinuousSimulation::deleteToiEventsForActiveEntitiesInIsland.");

	// clear the table entries.
	{
		for (int i = toiEvents.getSize()-1; i >=0; i--)
		{
			hkpToiEvent& event = toiEvents[i];

			// assumption: we know that all existing TOI events are between entities of one island
			if ( ( event.m_entities[0]->getSimulationIsland() == island && ENTITY_STATE(event.m_entities[0]) == HK_TOI_ENTITY_STATE_ACTIVE ) ||
				( event.m_entities[1]->getSimulationIsland() == island && ENTITY_STATE(event.m_entities[1]) == HK_TOI_ENTITY_STATE_ACTIVE ) )
			{
				if ( hkDebugToi && HK_IS_TRACE_ENABLED( event.m_entities[0], event.m_entities[1] ) )
				{
					hkToiPrintf("Toi.del", "#    Del  TOI     @ %2.7f: %-6s %-6s \n", toiEvents[i].m_time, event.m_entities[0]->getName(), event.m_entities[1]->getName());
				}
				fireToiEventRemoved( toiEvents[i] );
				toiEvents.removeAt(i);
			}

		}
	}
}



//////////////////////////////////////////////////////////////////////////
//
//  TOI handling
//
//////////////////////////////////////////////////////////////////////////


// The functions are commented further in the code

		// called from simulateToi:
			// helper
			HK_FORCE_INLINE void hkLs_buildStepInfos( hkpWorld* world, const hkpToiEvent& event, hkReal physicsDeltaTime, hkStepInfo& stepInfo, hkpSolverInfo& solverInfo, hkpConstraintQueryIn& constraintQueryIn);
			// localized solving
			// result tells whether any velocities have been changed, and bodies need to be recollided
			 hkResult hkLs_localizedSolveToi( const hkpToiResources& toiResources, hkpConstraintSolverResources& solverResources, hkpToiEvent& event, hkpToiResourceMgr& toiResourceMgr, hkpWorld* world, hkArray<hkpEntity*>& activeEntities, hkFixedArray<hkUchar>& entityState, hkReal rotateNormal );
				void hkLs_toiCheckValidityOfConstraints(hkpConstraintSolverResources& solverResources, hkpProcessCollisionInput& processInput, hkArray<hkpConstraintSchemaInfo>& constraintStatus, int& firstNonActiveConstraintStatus, hkpConstraintInstance::ConstraintPriority minPriorityToProcess, hkFixedArray<hkUchar>& entityState, const hkArray<hkpEntity*>& touchedEntities, hkArray<hkpEntity*>& toBeActivated);
					int hkLs_areVelocitiesOk( hkReal psiTimeLeft, const hkpConstraintSchemaInfo& constraintStatus, const hkReal* velocities, int numVelocities, const hkpProcessCollisionInput& processInput);
					void hkLs_toiActivateEntitiesAndCheckConstraints(hkpProcessCollisionInput& collisionInput, hkpConstraintInstance::ConstraintPriority minPriorityToProcess, hkTime time, const hkArray<hkpEntity*>& toBeActivated, hkFixedArray<hkUchar>& entityState, hkArray<hkpEntity*>& newTouchedEntities, hkArray<hkpConstraintInstance*>& newTouchedConstraints);
					void hkLs_collectAgentEntriesToProcess( const hkpEntity* entity, hkpConstraintInstance::ConstraintPriority minPriorityToProcess, hkTime time, hkFixedArray<hkUchar>& entityState, hkArray<hkpAgentNnEntry*>& entriesOut);
					HK_FORCE_INLINE hkBool hkLs_isConstraintInteresting(hkpConstraintInstance* constraint, hkpConstraintInstance::ConstraintPriority minPriorityToProcess);

				// helpers:
				HK_FORCE_INLINE void hkLs_updateTransformIfNeeded(hkpEntity* entity, hkTime time, hkFixedArray<hkUchar>& entityState);
				// helper2:
				void hkLs_toiActivateConstraintsLinkingToFixedAndKeyframedEntities(hkArray<hkpConstraintSchemaInfo>& constraintStatus, int& firstNonActiveConstraintStatus, int numOfConstraintsToCheckAtTheEndOfTheList);

				// UNDONE ( ?? )
				void hkLs_toiActivateConstraintsLinkingActivatedEntities(hkArray<hkpConstraintSchemaInfo>& constraintStatus, int& firstNonActiveConstraintStatus, hkFixedArray<hkUchar>& entityState);

			HK_FORCE_INLINE hkBool hkLs_isAgentInteresting(hkpAgentNnEntry* entry,  hkpConstraintInstance::ConstraintPriority minPriorityToProcess);

			// Set all velocityAccumulators to initial velocity for all passed entities.
		void hkLs_toiResetVelocityAccumulatorsForEntities(hkpConstraintSolverResources& solverResources, const hkArray<hkpEntity*>& entities);

			// return: true == areOK; false == are violated;
		hkBool hkLs_toiCheckFinalValidityOfCriticalConstraints(hkpConstraintSolverResources& solverResources, const hkArray<hkpConstraintSchemaInfo>& constraintStatus, hkpProcessCollisionInput& processInput, hkArray<hkpToiResourceMgr::ConstraintViolationInfo>* violatedConstraints);


		void hkLs_backstepAndFreezeEntireIsland(hkTime time, hkpSimulationIsland* island, hkFixedArray<hkUchar>&entityState, hkArray<hkpEntity*>& activeEntities);


// all inline versions of the above hkLs funcs
// must be defined before use for gcc compilers:

// Build hkStepInfo, hksolverInfo and constraintQueryIn structures for the TOI event about to be handled.
HK_FORCE_INLINE void hkLs_buildStepInfos(hkpWorld* world, const hkpToiEvent& event, hkReal physicsDeltaTime, hkStepInfo& stepInfoOut, hkpSolverInfo& solverInfoOut, hkpConstraintQueryIn& constraintQueryInOut)
{
	// setup infos
	stepInfoOut.m_startTime = event.m_time;
	stepInfoOut.m_endTime   = world->getCurrentPsiTime();
	stepInfoOut.m_deltaTime = stepInfoOut.m_endTime - stepInfoOut.m_startTime;
	stepInfoOut.m_invDeltaTime = 1.0f / stepInfoOut.m_deltaTime;

	// Make a temporary copy of world.m_solverInfo and modify it
	solverInfoOut = world->m_dynamicsStepInfo.m_solverInfo;

	solverInfoOut.m_numSteps      = 1;
	solverInfoOut.m_invNumSteps   = 1.0f;
	solverInfoOut.m_numMicroSteps = 1;
	solverInfoOut.m_invNumSteps   = 1.0f;

	solverInfoOut.m_deltaTime	 = physicsDeltaTime;
	solverInfoOut.m_invDeltaTime= 1.0f / physicsDeltaTime;

	solverInfoOut.m_globalAccelerationPerSubStep.setZero(); //setMul4( solverInfoOut.m_deltaTime, world->m_gravity );
	solverInfoOut.m_globalAccelerationPerStep.setZero();    //setMul4( stepInfoOut.m_deltaTime, world->m_gravity );

	solverInfoOut.setTauAndDamping( 0.5f, 1.4f );

	// initialize ConstraintQueryIn structure

	//user solverInfoOut created above. hksolverInfoOut& solverInfoOut = world->m_dynamicsStepInfo.m_solverInfoOut;
	constraintQueryInOut.set( solverInfoOut, stepInfoOut, world->m_violatedConstraintArray );
}

// Updates transform to the specified time (TOI), if it hasn't been done yet.
// Tech info: is based on the entityState array.
HK_FORCE_INLINE void hkLs_updateTransformIfNeeded(hkpEntity* entity, hkTime time, hkFixedArray<hkUchar>& entityState)
{
	HK_ASSERT(0xf0ff0047, !entity->isFixed());
	if (ENTITY_STATE(entity) == HK_TOI_ENTITY_STATE_ZERO)
	{
		ENTITY_STATE(entity) =  HK_TOI_ENTITY_STATE_TRANSFORM_UPDATED;
		hkpRigidBody* body = static_cast<hkpRigidBody*>( entity );
		hkMotionState& ms = body->getRigidMotion()->m_motionState;
		hkSweptTransformUtil::lerp2( ms.getSweptTransform(), time, ms.getTransform() );
	}
}

HK_FORCE_INLINE void hkLs_updateTransform(hkpEntity* entity, hkTime time)
{
	HK_ASSERT(0xf0ff0047, !entity->isFixed());
	hkpRigidBody* body = static_cast<hkpRigidBody*>( entity );
	hkMotionState& ms = body->getRigidMotion()->m_motionState;
	hkSweptTransformUtil::lerp2( ms.getSweptTransform(), time, ms.getTransform() );
}

HK_FORCE_INLINE hkBool hkLs_isConstraintInteresting(hkpConstraintInstance* constraint, hkpConstraintInstance::ConstraintPriority minPriorityToProcess)
{
	if (minPriorityToProcess == hkpConstraintInstance::PRIORITY_PSI)
	{
		HK_WARN_ONCE(0xadbdd88d, "You should not process PRIORITY_PSI constraints in TOI events..");
	}
	HK_ASSERT2(0xadbc7ddd, constraint->getType() != hkpConstraintInstance::TYPE_CHAIN || (constraint->getPriority() == hkpConstraintInstance::PRIORITY_PSI && constraint->getPriority() < minPriorityToProcess), "Only hkpConstraintInstance::TYPE_NORMAL constraints can be processed in TOI events. (Processing constraint chains will very likely cause a crash.)");

	return minPriorityToProcess <= constraint->getPriority();
}

HK_FORCE_INLINE hkBool hkLs_isAgentInteresting(hkpAgentNnEntry* entry, hkpConstraintInstance::ConstraintPriority minPriorityToProcess)
{
	HK_ASSERT(0xad000061, entry->m_collisionQualityIndex >= hkpCollisionDispatcher::COLLISION_QUALITY_PSI && entry->m_collisionQualityIndex <= hkpCollisionDispatcher::COLLISION_QUALITY_CHARACTER);

	HK_ASSERT2(0xad54d4d2, (   ( minPriorityToProcess == hkpConstraintInstance::PRIORITY_TOI         && entry->m_collisionQualityIndex >= hkUchar(hkpCollisionDispatcher::COLLISION_QUALITY_TOI)        )
		|| ( minPriorityToProcess == hkpConstraintInstance::PRIORITY_TOI_HIGHER  && entry->m_collisionQualityIndex >= hkUchar(hkpCollisionDispatcher::COLLISION_QUALITY_TOI_HIGHER) )
		|| ( minPriorityToProcess == hkpConstraintInstance::PRIORITY_TOI_FORCED  && entry->m_collisionQualityIndex >= hkUchar(hkpCollisionDispatcher::COLLISION_QUALITY_TOI_FORCED) )
		||   minPriorityToProcess == hkpConstraintInstance::PRIORITY_PSI
		)
		==
		(minPriorityToProcess <= entry->m_collisionQualityIndex),
		"Enumeration values corrupted.");

	return minPriorityToProcess <= entry->m_collisionQualityIndex;
}






void hkpContinuousSimulation::handleSimpleToi( hkpWorld* world, hkpToiEvent& event, hkReal physicsDeltaTime, hkReal rotateNormal )
{
	HK_TIMER_BEGIN("SimpleTOI", HK_NULL);

	m_world->lockCriticalOperations();

	{
		hkInplaceArray<hkpEntity*,2>          toBeActivated_unused;

		//
		//	Trigger confirmToi method of the contact manager.
		//  This usually fires contact processed callbacks.
		//  This also allows to apply simple TOI-collision response before further localized solving.
		//
		event.m_contactMgr->confirmToi( event, rotateNormal, toBeActivated_unused );

		// 
		// This is simple handling, so we ignore the activation state & always backstep debris objects.
		//

		hkInplaceArray<hkpEntity*, 2> entitiesToRecollide;
		for (int i = 0; i < 2; i++)
		{
			hkpEntity* entity = event.m_entities[i];

			if (HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI == entity->getCollidable()->getQualityType())
			{
				// Backstep body's motion
				hkMotionState& motionState = entity->getMotion()->m_motionState;
				hkSweptTransformUtil::freezeMotionState( event.m_time, motionState );

				// Invalidate TIM's in agents and put entities as PSI entities to collide at the end of the frame.
				hkpWorldAgentUtil::invalidateTim(entity, *world->getCollisionInput());

				// After a debris object is backstepped, all it's consecutive TOIs (with landscape only) should be removed from the TOI event queue.
				for (int t = m_toiEvents.getSize()-1; t >= 0; t--)
				{
					hkpToiEvent& anotherEvent = m_toiEvents[t];

					if (anotherEvent.m_entities[0] == entity || anotherEvent.m_entities[1] == entity)
					{
						//HK_ON_DEBUG( hkpEntity* otherEntity = hkSelectOther( entity, anotherEvent.m_entities[0], anotherEvent.m_entities[1]) );
						//HK_ASSERT2(0xad81016a, otherEntity->getMotionState()->getSweptTransform().getInvDeltaTime() == 0.0f, "This assert may be over-restrictive: Using simple TOI handling for debris against non-fixed objects.");
						fireToiEventRemoved(anotherEvent);
						m_toiEvents.removeAt(t);
					}
				}

				entitiesToRecollide.pushBackUnchecked(entity);
			}
		}

		HK_ASSERT2(0xad810212, entitiesToRecollide.getSize(), "A simple TOI reported, but no entity of quality type HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI found.");

		// Remove the processed TOI
		fireToiEventRemoved(event);

		// We froze the motion, so we can skip broadphase collision detection.
		// Re-collide (continuous && not-simplified) agents.
		// Mark entities that have entries to be recollided at the end of the frame.
		if (entitiesToRecollide.getSize())
		{
			collideEntitiesOfOneIslandNarrowPhaseContinuous_toiOnly(entitiesToRecollide.begin(), entitiesToRecollide.getSize(), *world->getCollisionInput(), m_entitiesNeedingPsiCollisionDetection);
		}
	}

	m_world->unlockAndAttemptToExecutePendingOperations();

	HK_TIMER_END();

}

// \param rotateNormal set by the contactPointCallback for this TOI.
void hkpContinuousSimulation::simulateToi( hkpWorld* world, hkpToiEvent& event, hkReal physicsDeltaTime, hkReal rotateNormal )
{
	createSolverDebugOstream();

#ifdef USE_SOLVER_DEBUG_OSTREAM
	if (debugOstream) { debugOstream->printf("******* Toi %i     ******\n", m_toiCounter); }
#endif

	HK_ASSERT2(0xadf257fe, (event.m_entities[0]->getSimulationIsland()->m_isInActiveIslandsArray || event.m_entities[0]->isFixed())
						&& (event.m_entities[1]->getSimulationIsland()->m_isInActiveIslandsArray || event.m_entities[1]->isFixed()),
						   "This function may cause unpredicted behavior when executed for inactive entities." );

	// Init: call toiListener to determine whether the TOI-Event should be handled
	// and to allocate solverResources
	hkpToiResources toiResources;
	if ( HK_FAILURE == m_toiResourceMgr->beginToiAndSetupResources(event, m_toiEvents, toiResources ) )
	{
		return;
	}

	m_world->lockCriticalOperations();


	// Active entities, which will be back-stepped and reintegrated.
	hkInplaceArray<hkpEntity*,64> activeEntities;
	{
			// Init solver infos and state
		hkStepInfo stepInfo;
		hkpSolverInfo solverInfo;
		hkpConstraintQueryIn constraintQueryIn(HK_ON_CPU(&hkpBeginConstraints));
		hkLs_buildStepInfos(world, event, physicsDeltaTime, stepInfo, solverInfo, constraintQueryIn);

			// Uses toiResource.m_scratchPad for solver memory, otherwise tries to acquire the default scratchpad itself
		hkpConstraintSolverResources solverResources;
		hkpConstraintSolverSetup::initializeSolverState( stepInfo, solverInfo, constraintQueryIn, toiResources.m_scratchpad, toiResources.m_scratchpadSize, toiResources.m_priorityClassMap, toiResources.m_priorityClassRatios, solverResources);


		HK_TIMER_BEGIN_LIST("TOI", "SolveToi");
		hkToiPrintf("Toi.sim", "#  TOI %2.7f: %-6s %-6s \n", event.m_time, RETURN_BODY_NAME(event.m_entities[0]), RETURN_BODY_NAME(event.m_entities[1]) );

		//
		//	TOI
		//
		
		// Get the right island that we'll use through out the scope of this function
		hkpSimulationIsland* island = ( !event.m_entities[0]->isFixed()) ? event.m_entities[0]->getSimulationIsland() : event.m_entities[1]->getSimulationIsland();
		int sizeOfIsland = island->m_entities.getSize();

		// Quick lookup list for activationStatus of bodies.
		hkLocalBuffer<hkUchar> entityState(sizeOfIsland);
		hkString::memSet( entityState.begin(), HK_TOI_ENTITY_STATE_ZERO, sizeOfIsland );


		// Localized Solving -- determines the appropriate set of entities to 'activate', back-step, process collision response for, and reintegrate.
		//             Runs SCR or the constraint solver to compute collision response.
		const hkResult localizedSolveToiResult = hkLs_localizedSolveToi(toiResources, solverResources, event, *m_toiResourceMgr, world, activeEntities, entityState, rotateNormal );

		
		//if (m_world->m_enableForceLimitBreachedSecondaryEventsFromToiSolver)
		//{
		//	// Generate breakage events
		//	for (int i = 0; i < 2; i++)
		//	{
		//		if (solverResources.m_elemTemp[i].m_begin != solverResources.m_elemTemp[i].m_current)
		//		{
		//			hkExportBreachedImpulses(solverInfo, solverResources.m_elemTemp[i].m_begin, solverResources.m_schemas[i].m_begin, solverResources.m_accumulators);

		//			hkpImpulseLimitBreachedHeader* h = (hkpImpulseLimitBreachedHeader*)(solverResources.m_schemas[i].m_begin);
		//			if ( h->m_numBreached )
		//			{
		//				hkpContactImpulseLimitBreachedListenerInfo* bi = (hkpContactImpulseLimitBreachedListenerInfo*)(&h->getElem(0));
		//				hkpWorldCallbackUtil::fireContactImpulseLimitBreached( m_world, bi, h->m_numBreached );
		//			}
		//		}
		//	}
		//}

		// Release solver's memory
		hkpConstraintSolverSetup::shutdownSolver(solverResources);

		// Remove the processed TOI
		fireToiEventRemoved(event);

		if (HK_SUCCESS == localizedSolveToiResult)
		{
			//
			//	Delete old and create new TOI events for active bodies by TOI collisions
			//
			HK_INTERNAL_TIMER_SPLIT_LIST( "EvtCleanup" );
			deleteToiEventsForActiveEntitiesInIsland( island, entityState, m_toiEvents );
			hkReferencedObject::addReferences( activeEntities.begin(), activeEntities.getSize() );	// necessary because of unlockAndAttemptToExecutePendingOperations() could remove the entity from the world
		}
		else
		{
			activeEntities.clear();
		}
	}
	m_toiResourceMgr->endToiAndFreeResources(event, m_toiEvents, toiResources);

	HK_TIMER_SPLIT_LIST( "PendingOperations" );

	HK_TIMER_SPLIT_LIST( "Collide" );
	if ( activeEntities.getSize() )
	{
		HK_ASSERT( 0xf0342332, event.m_time == world->m_collisionInput->m_stepInfo.m_startTime );
		// 	const hkStepInfo physicsStepInfo( event.m_time, m_currentPsiTime );		world->m_collisionInput->m_stepInfo = physicsStepInfo;

		//
		// remove all entities which are no longer part of the world, might have been removed by unlockAndAttemptToExecutePendingOperations()
		//	(typically in break events)
		//
		{
			for (int i = activeEntities.getSize()-1; i>=0; i--)
			{
				hkpEntity* entity = activeEntities[i];
				if ( entity->getWorld() == HK_NULL )
				{
					activeEntities.removeAt(i);
					entity->removeReference();
				}
			}
		}

		collideEntitiesOfOneIslandContinuous_toiOnly(activeEntities.begin(), activeEntities.getSize(), world, *world->getCollisionInput(), m_entitiesNeedingPsiCollisionDetection);
		hkReferencedObject::removeReferences( activeEntities.begin(), activeEntities.getSize() );
	}
	m_world->unlockAndAttemptToExecutePendingOperations();

	HK_TIMER_END_LIST();


	flushSolverDebugOstream();
}

// This method will go through all TOI entries and process them.
void hkpContinuousSimulation::processAgentNnEntries( hkpAgentNnEntry *const * entries, int numEntries, const hkpProcessCollisionInput& collisionInput, hkpSimulationIsland* island, hkpContinuousSimulation::CollisionQualityOverride collisionQualityOverride )
{
	HK_TIMER_BEGIN( "NarrowPhaseTOI", HK_NULL );

	for (int i = 0; i < numEntries; i++)
	{
		hkpAgentNnEntry* entry = entries[i];

		//
		// collide/update manifold
		//
		HK_ALIGN16(typedef hkpProcessCollisionOutput hkAlignedCollisionOutput);
		hkAlignedCollisionOutput processOutput(island);
		processOutput.reset();

		switch(collisionQualityOverride)
		{
		case PROCESS_NORMALLY: 
			collisionInput.m_collisionQualityInfo = collisionInput.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex );
			break;
		case DISABLE_TOIS: 
			collisionInput.m_collisionQualityInfo = collisionInput.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::COLLISION_QUALITY_PSI );
			break;
		case DISABLE_TOIS_AND_EXPAND_MANIFOLD: 
			collisionInput.m_collisionQualityInfo = collisionInput.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::COLLISION_QUALITY_TMP_EXPAND_MANIFOLD );
			if (entry->m_collisionQualityIndex == hkpCollisionDispatcher::COLLISION_QUALITY_CHARACTER )
			{
				collisionInput.m_collisionQualityInfo = collisionInput.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::COLLISION_QUALITY_CHARACTER );
			}
			break;
		default:
			HK_ASSERT2(0xad239412, false, "Undefined override value.");
		}

		hkpCollisionQualityInfo* origInfo = collisionInput.m_dispatcher->getCollisionQualityInfo( entry->m_collisionQualityIndex );
		collisionInput.m_createPredictiveAgents = origInfo->m_useContinuousPhysics;


		//check now
		//	HK_ASSERT2(0xad000180, pair->m_collisionAgent, "Internal Error: The agent was removed from hkCollidablePair");

		hkAgentNnMachine_ProcessAgent( entry, collisionInput, processOutput, entry->m_contactMgr );
		if ( !processOutput.isEmpty() )
		{
			entry->m_contactMgr->processContact( *entry->getCollidableA(), *entry->getCollidableB(), collisionInput, processOutput );
			hkToiPrintf("sub.toi.man", "#      sub.toi.entMan    --manifold present \n");
		}
		else
		{
			hkToiPrintf("sub.toi.man", "#      sub.toi.entMan    --NO manifold  \n");
		}

		if ( processOutput.hasToi()  )
		{
			HK_ASSERT2(0xad234353, collisionQualityOverride == hkpContinuousSimulation::PROCESS_NORMALLY, "No TOIs expected "
				"(collisionQualityOverride set to DISABLE_TOI or DISABLE_TOI_AND_EXPAND_MANIFOLD) when generating manifolds only");
			HK_ASSERT( 0xf032435f, collisionInput.m_stepInfo.m_startTime <= processOutput.m_toi.m_time );
			HK_ASSERT2(0xad8765df, processOutput.m_toi.m_time >= m_currentTime, "Generating a TOI event before hkpWorld->m_currentTime.");
			addToiEvent(processOutput, *entry );
		}
	}

	HK_TIMER_END();
}


// Attempts to generate contact points for all 'interesting' agents. For each agent, the function updates
// the transform of the other body to the TOI-event time and attempts to expand contact manifold there.
void hkLs_collectAgentEntriesToProcess( const hkpEntity* entity, hkpConstraintInstance::ConstraintPriority minPriorityToProcess, hkTime time, hkFixedArray<hkUchar>& entityState, hkLocalArray<hkpAgentNnEntry*>& entriesOut )
{
	hkToiPrintf("sub.toi.entMan", "#      sub.toi.entMan  for %s -- updateEntityManifolds\n", RETURN_BODY_NAME(entity));

	hkArray<struct hkpLinkedCollidable::CollisionEntry> collisionEntriesTmp;
	entity->getLinkedCollidable()->getCollisionEntriesSorted(collisionEntriesTmp);
	const hkArray<struct hkpLinkedCollidable::CollisionEntry>& collisionEntries = collisionEntriesTmp;

	for (int i = 0; i < collisionEntries.getSize(); i++) 
	{
		// force/update contact manifolds for newly added constraints
		const hkpLinkedCollidable::CollisionEntry& ce = collisionEntries[i]; 

		if ( hkLs_isAgentInteresting(ce.m_agentEntry, minPriorityToProcess) )
		{
			hkpLinkedCollidable* otherCollidable = ce.m_partner;
			hkpEntity* otherEntity = static_cast<hkpEntity*>(otherCollidable->getOwner());

			// Only need to check:
			//  - fixed bodies
			//  - not-active bodies (as all the active ones have already updated this agent/manifold).
			if (otherEntity->isFixed() || ENTITY_STATE(otherEntity) < HK_TOI_ENTITY_STATE_MANIFOLD_UPDATED)
			{
				// TODO check whether the manifold has been updated already: store time-of-manifold in agent

				hkToiPrintf("sub.toi.entMan", "#      sub.toi.entMan    to %s \n", RETURN_BODY_NAME( otherEntity));
				if (!otherEntity->isFixed())
				{
					HK_ASSERT(0xf0ff0046, otherEntity->getSimulationIsland() == entity->getSimulationIsland());
					hkLs_updateTransformIfNeeded(otherEntity, time, entityState);
				}

				// We will only collect all TOI entries here and process them in one batch later on.
				entriesOut.pushBack(ce.m_agentEntry);
			}
			else
			{
				hkToiPrintf("sub.toi.entMan", "#      sub.toi.entMan    to %s (manifold already updated)\n", RETURN_BODY_NAME( otherEntity));
			}
		}
		else
		{
#ifdef HK_DEBUG_TOI
			hkpEntity* otherEntity = static_cast<hkpEntity*>(ce.m_partner->getOwner());
			char oname[100]; otherEntity->getName(oname, 100);
			hkToiPrintf("sub.toi.entMan", "#      sub.toi.entMan    to %s (agent not interesting)\n", RETURN_BODY_NAME( otherEntity));
#endif
		}
	}
}




HK_COMPILE_TIME_ASSERT( (int)hkpConstraintInstance::PRIORITY_PSI == (int)hkpCollisionDispatcher::COLLISION_QUALITY_PSI );
HK_COMPILE_TIME_ASSERT( (int)hkpConstraintInstance::PRIORITY_SIMPLIFIED_TOI_UNUSED == (int)hkpCollisionDispatcher::COLLISION_QUALITY_SIMPLIFIED_TOI );
HK_COMPILE_TIME_ASSERT( (int)hkpConstraintInstance::PRIORITY_TOI == (int)hkpCollisionDispatcher::COLLISION_QUALITY_TOI );
HK_COMPILE_TIME_ASSERT( (int)hkpConstraintInstance::PRIORITY_TOI_HIGHER == (int)hkpCollisionDispatcher::COLLISION_QUALITY_TOI_HIGHER );
HK_COMPILE_TIME_ASSERT( (int)hkpConstraintInstance::PRIORITY_TOI_FORCED == (int)hkpCollisionDispatcher::COLLISION_QUALITY_TOI_FORCED );


	// Returns the index of the failing contact point (or the friction component), that breaches the allowed velocities the most.
	// Returns -1 if all contacts are ok.
int hkLs_areVelocitiesOk( hkReal psiTimeLeft, const hkpConstraintSchemaInfo& constraintStatus, const hkReal* velocities, int numVelocities, const hkpProcessCollisionInput& processInput)
{
	//  thought: OTHER CONSTRAINTS CONNECTED TO THE BODY ?? (don't want to connect 10 new force_quality, only to satisfy a single minor_priority constraint )
	//     - pre-calculate an entity's priority & don't touch the 'delicate ones'

	int idx = constraintStatus.m_constraint->getPriority();

	HK_ON_DEBUG
	(
		switch(idx)
		{
			case hkpConstraintInstance::PRIORITY_PSI:
				HK_ASSERT2(0xad5bb332, idx == hkpCollisionDispatcher::COLLISION_QUALITY_PSI, "Internal error: ConstraintPriority and CollisionQuality enumerations must have properly assigned, correlated values.");
				break;
			case hkpConstraintInstance::PRIORITY_TOI:
				HK_ASSERT2(0xad5bb333, idx == hkpCollisionDispatcher::COLLISION_QUALITY_TOI, "Internal error: ConstraintPriority and CollisionQuality enumerations must have properly assigned, correlated values.");
				break;
			case hkpConstraintInstance::PRIORITY_TOI_HIGHER:
				HK_ASSERT2(0xad5bb334, idx == hkpCollisionDispatcher::COLLISION_QUALITY_TOI_HIGHER, "Internal error: ConstraintPriority and CollisionQuality enumerations must have properly assigned, correlated values.");
				break;
			case hkpConstraintInstance::PRIORITY_TOI_FORCED:
				HK_ASSERT2(0xad5bb335, idx == hkpCollisionDispatcher::COLLISION_QUALITY_TOI_FORCED, "Internal error: ConstraintPriority and CollisionQuality enumerations must have properly assigned, correlated values.");
				break;
			default:
				HK_ASSERT2(0xf0ff0048,0, "Unknown or invalid constraint priority");
				break;

		}
	);


	// Determine if there will be an extra TOI caused by this constraint -- add it if so.
	hkpCollisionQualityInfo* info = processInput.m_dispatcher->getCollisionQualityInfo( hkUchar(idx) );

	hkReal maxValue = constraintStatus.m_allowedPenetrationDepth * info->m_maxContraintViolation;
	int maxIdx = -1;

	for (int i = 0; i < numVelocities; i++)
	{
		hkReal value = hkMath::fabs(velocities[i]) * psiTimeLeft;
		if ( value > maxValue )
		{
			maxValue = value;
			maxIdx = i;
		}
	}

	return maxIdx;
}


// Checks whether constraints linking two bodies of which one is 'activated' and one 'touched' only are violated. If one is
// violated, then both: the constraint and the 'touched' body it connects to are marked 'active'. Which means, that the newly
// activated body will be back-stepped, and its motionState will be reintegrated after collision response is properly resolved
// for the whole set of activated bodies + constraints.
//
// Initially only contact constraints were considered to be qualified as 'interesting', however now you can also try to set
// constraint priority to hkpConstraintInstance::PRIORITY_TOI for regular constraints. This generally results in nicer behavior,
// as such constraints are then checked and potentially solved at every related TOI. But it may also be a cause for occasional
// explosions as proper 'isConstraintValid' function has not been implemented for constraints other then simple contact
// constraints.
//
// This uses a magic trick of Oli's where already-processed-constraints are grouped at the beginning of the hkArray<ConstraintSchemaInfo>.
//  - This way we don't redundantly process those constraints
//  - and also it's easy to undo activation of constraints (as they are ordered chronologically by their time of activation)
//
void hkLs_toiCheckValidityOfConstraints(hkpConstraintSolverResources& solverResources, hkpProcessCollisionInput& processInput, hkArray<hkpConstraintSchemaInfo>& constraintStatus, int& firstNonActiveConstraintStatus, hkpConstraintInstance::ConstraintPriority minPriorityToProcess, hkFixedArray<hkUchar>& entityState, const hkArray<hkpEntity*>& touchedEntities, hkArray<hkpEntity*>& toBeActivated)
{
	HK_ASSERT(0xad000015, !toBeActivated  .getSize());
	HK_ASSERT(0xad000016,  touchedEntities.getSize());
	HK_ASSERT(0xad000014, firstNonActiveConstraintStatus < constraintStatus.getSize());

	// Set all touchedEntities' velocityAccumulators to initial velocity (and store their current values first).
	hkVector4* origVelStack = hkAllocateStack<hkVector4>(touchedEntities.getSize()*2);
	{
		hkVector4* origVel = origVelStack;
		for (int i = 0; i < touchedEntities.getSize(); i++)
		{
			hkpEntity* entity = touchedEntities[i];
			hkpVelocityAccumulator* accumulator = hkAddByteOffset(solverResources.m_accumulators, entity->m_solverData);
			hkpMotion* motion = static_cast<hkpMotion*>(entity->getMotion());
			*(origVel++) = accumulator->m_linearVel;
			*(origVel++) = accumulator->m_angularVel;

			//accumulator->m_linearVel  = motion->m_linearVelocity;
			//accumulator->m_angularVel = motion->m_angularVelocity;
			hkSweptTransformUtil::getVelocity( motion->m_motionState, accumulator->m_linearVel, accumulator->m_angularVel);
		}
	}

//#ifdef HK_DEBUG_TOI_DETAIL
//	hkpSimulationIsland* island = touchedEntities[0]->getSimulationIsland();
//	int debugNumObjects = island->getEntities().getSize()+1;
//	hkArray<hkReal> debugVelocities;
//	{
//		debugVelocities.setSize( debugNumObjects * debugNumObjects );
//		for (int i = 0; i < debugVelocities.getSize(); i++)
//		{
//			debugVelocities[i] = 0.0f;
//		}
//	}
//#endif

	{
		for (int i = firstNonActiveConstraintStatus; i < constraintStatus.getSize(); i++)
		{
			hkpConstraintSchemaInfo& cStatus = constraintStatus[i];

			GET_CONSTRAINT_NAME(constraintName, cStatus.m_constraint);

			if ( hkLs_isConstraintInteresting(cStatus.m_constraint, minPriorityToProcess) )
			{
				// filter through inactive_entity--acitve_entity constraints only
				hkpEntity** e = cStatus.m_constraint->getInternal()->m_entities;

				HK_ASSERT(0xf0ff0050, !e[0]->isFixed() && !e[1]->isFixed() &&
									  !e[0]->isFixedOrKeyframed() && !e[1]->isFixedOrKeyframed());

				HK_ASSERT(0xad56bfea, ENTITY_STATE(e[0]) != HK_TOI_ENTITY_STATE_ACTIVE || ENTITY_STATE(e[1]) != HK_TOI_ENTITY_STATE_ACTIVE);

				// Get index of the the not-active entity (which is to be set as 'pending' for activation).
				int tmpIdx = ENTITY_STATE(e[0]) == HK_TOI_ENTITY_STATE_ACTIVE;
				hkpEntity* entity = e[tmpIdx];

				HK_ASSERT(0xf0ff0051, e[1-tmpIdx]->isFixed() || ENTITY_STATE(e[1-tmpIdx]) == HK_TOI_ENTITY_STATE_ACTIVE);
				HK_ASSERT(0xf0ff0052, ENTITY_STATE(entity) != HK_TOI_ENTITY_STATE_ACTIVE);

				if (ENTITY_STATE(entity) & HK_TOI_ENTITY_STATE_PENDING_FLAG )
				{
					hkAlgorithm::swap(constraintStatus[firstNonActiveConstraintStatus++], constraintStatus[i]);
					continue;
				}

				// Are constraint conditions/velocities violated ?
				{
					hkReal velocities[256];
					int numVelocities = hkSolveGetToiViolatingConstraintVelocity(*solverResources.m_solverInfo, cStatus.m_schema, solverResources.m_accumulators, 256, velocities);
					hkBool satisfied = (-1 == hkLs_areVelocitiesOk(solverResources.m_stepInfo->m_deltaTime, cStatus, velocities, numVelocities, processInput));
//#	ifdef HK_DEBUG_TOI_DETAIL
//					if ( numVelocities )
//					{
//						hkReal minVel = HK_REAL_MAX;
//						for ( int v = 0; v < numVelocities; v++){ minVel = hkMath::min2( minVel, velocities[v] );	}
//						if ( minVel > 1000.0f)
//						{
//							minVel *= 1.0f;
//						}
//						int i0 = (e[0]->isFixed())? debugNumObjects-1 : e[0]->getStorageIndex();
//						int i1 = (e[1]->isFixed())? debugNumObjects-1 : e[1]->getStorageIndex();
//						debugVelocities[ i0 * debugNumObjects + i1 ] = minVel;
//						debugVelocities[ i1 * debugNumObjects + i0 ] = minVel;
//					}
//#	endif

					if (!satisfied) //|| F())
					{
						HK_ASSERT(0xf0ff0053, ENTITY_STATE(entity) == HK_TOI_ENTITY_STATE_TOUCHED || ENTITY_STATE(entity) == HK_TOI_ENTITY_STATE_TRANSFORM_UPDATED || ENTITY_STATE(entity) == HK_TOI_ENTITY_STATE_MANIFOLD_UPDATED);	// HK_TOI_ENTITY_STATE_TRANSFORM_UPDATED is set if we ran out of resources
						hkToiPrintf("sub.toi.conVer", "#      sub.toi.conver    constraint violated : %s \n", constraintName);

						ENTITY_STATE(entity) |= HK_TOI_ENTITY_STATE_PENDING_FLAG;
						toBeActivated.pushBack(entity);
						hkAlgorithm::swap(constraintStatus[firstNonActiveConstraintStatus++], constraintStatus[i]);
					}
					else
					{
						hkToiPrintf("sub.toi.conVer", "#      sub.toi.conver    constraint ok       : %s \n", constraintName);
					}
				}
			}
			else
			{
				hkToiPrintf("sub.toi.conVer", "#      sub.toi.conver    constraint not interesting: %s \n", constraintName);
			}
		}
	}

	{
		// Set all touchedEntities' velocityAccumulators back to their current values
		hkVector4* origVel = origVelStack;
		for (int i = 0; i < touchedEntities.getSize(); i++)
		{
			hkpEntity* entity = touchedEntities[i];
			hkpVelocityAccumulator* accumulator = hkAddByteOffset(solverResources.m_accumulators, entity->m_solverData);
			accumulator->m_linearVel  = *(origVel++);
			accumulator->m_angularVel = *(origVel++);
		}
	}

	hkDeallocateStack<hkVector4>(origVelStack, touchedEntities.getSize() * 2);

//#ifdef HK_DEBUG_TOI_DETAIL
//	{
//		hkpSimulationIsland* island = touchedEntities[0]->getSimulationIsland();
//		hkToiPrintf("VEL","=======   ");
//		for ( int y = 0; y < debugNumObjects-1; y++ )
//		{
//			char buffer[256];
//			hkpEntity* entity = island->m_entities[y];
//			entity->getName(buffer, 200);
//			hkToiPrintf("VEL", "%8s", buffer );
//		}
//		hkToiPrintf("VEL","   Fixed\n");
//
//		for ( int x = 0; x < debugNumObjects; x++ )
//		{
//			char s = ' ';
//			char buffer[256];
//			if ( x < debugNumObjects-1)
//			{
//				hkpEntity* entity = island->m_entities[x];
//				switch ( entityState[x] )
//				{
//				case HK_TOI_ENTITY_STATE_ACTIVE: s = 'A'; break;
//				case HK_TOI_ENTITY_STATE_TOUCHED: s = 'T'; break;
//				case HK_TOI_ENTITY_STATE_TRANSFORM_UPDATED: s = 'U'; break;
//				case HK_TOI_ENTITY_STATE_PENDING_AND_TOUCHED: s = 'a'; break; xxx
//				default: break;
//				}
//				entity->getName(buffer, 200);
//			}
//			else
//			{
//				s = '#';
//				hkString::sprintf(buffer, "Fixed\t");
//			}
//			hkToiPrintf("VEL","%c:%-10s", s, buffer );
//
//			for ( int z = 0; z < debugNumObjects; z++ )
//			{
//				hkToiPrintf("VEL","%6.1f  ", debugVelocities[ z * debugNumObjects + x ]);
//			}
//			hkToiPrintf("VEL","\n");
//		}
//	}
//#endif

}

static HK_FORCE_INLINE hkBool less_EntityPtr(const hkpEntity* a, const hkpEntity* b)
{
	return a->getUid() < b->getUid();
}

static HK_FORCE_INLINE hkBool less_ConstraintPtr(const hkpConstraintInstance* a, const hkpConstraintInstance* b)
{
	HK_ASSERT2(0xad567bdc, a->getSlaveEntity() == b->getSlaveEntity(), "Slave entity must be shared between constraints.");

	if ( a->getMasterEntity()->getUid() < b->getMasterEntity()->getUid() )
	{
		return true;
	}

	if ( a->getMasterEntity()->getUid() == b->getMasterEntity()->getUid() )
	{
		return hkUlong(a->m_internal) < hkUlong(b->m_internal); // This compares internal's storageIndex in the master list of the shared master entity
	}

	return false;
}


// Input: toBeActivated entity array
// generateNewTouchedEntitiesAndNewActiveConstraints
void hkLs_toiActivateEntitiesAndCheckConstraints(hkpProcessCollisionInput& collisionInput, hkpConstraintInstance::ConstraintPriority minPriorityToProcess, hkTime time,
																	const hkArray<hkpEntity*>& toBeActivated, hkFixedArray<hkUchar>& entityState, hkArray<hkpEntity*>& newTouchedEntities, hkArray<hkpConstraintInstance*>& newTouchedConstraints)
{
	HK_ASSERT(0xf0ff0054,  toBeActivated.getSize());
	HK_ASSERT(0xf0ff0055, !newTouchedEntities   .getSize());
	HK_ASSERT(0xf0ff0056, !newTouchedConstraints.getSize());

	//
	//	Make touched entities active
	//  Collide those entities with all but active
	//
	//	Find all constraints to the just activate bodies and
	//   - add them to the newTouchedConstraints array
	//   - put new entities into the touchedEntities array
	//
	{
		hkLocalArray<hkpAgentNnEntry*> entries(1000);

		// Collect all entries.
		{
			for (int i = 0; i < toBeActivated.getSize(); i++)
			{
				hkpEntity* entity = toBeActivated[i];

				hkLs_updateTransformIfNeeded(entity, time, entityState);
				HK_ASSERT2(0xad763432, ENTITY_STATE(entity) & HK_TOI_ENTITY_STATE_PENDING_FLAG, "All entities to be activated are expected to be pending.");

				ENTITY_STATE(entity) = HK_TOI_ENTITY_STATE_MANIFOLD_UPDATED;

				hkLs_collectAgentEntriesToProcess( entity, minPriorityToProcess, time, entityState, entries );
			}
		}

#if 1 && defined(HK_DEBUG)
		// Check that there are no duplicated entries in the list.
		{
			for (int a = 0; a < entries.getSize(); a++)
			{
				hkpAgentNnEntry* entryA = entries[a];
				for (int b = a+1; b < entries.getSize(); b++)
				{
					hkpAgentNnEntry* entryB = entries[b];
					HK_ASSERT(0xaf341e32, entryA != entryB);
				}
			}
		}
#endif

		// Process all entries in one huge batch.
		{
			const hkStepInfo oldStepInfo = collisionInput.m_stepInfo;

			collisionInput.m_stepInfo.m_endTime		 = time;
			collisionInput.m_stepInfo.m_deltaTime	 = time - collisionInput.m_stepInfo.m_startTime;
			collisionInput.m_stepInfo.m_invDeltaTime = (collisionInput.m_stepInfo.m_deltaTime > 0) ? (1.0f / collisionInput.m_stepInfo.m_deltaTime) : 0.0f;

			hkpContinuousSimulation* cs = static_cast<hkpContinuousSimulation*>( toBeActivated[0]->getWorld()->m_simulation );
			HK_ON_DETERMINISM_CHECKS_ENABLED(cs->m_determinismCheckFrameCounter++; )
			cs->processAgentNnEntries(entries.begin(), entries.getSize(), collisionInput, toBeActivated[0]->getSimulationIsland(), hkpContinuousSimulation::DISABLE_TOIS_AND_EXPAND_MANIFOLD);

			collisionInput.m_stepInfo = oldStepInfo;
		}
	}

	{
		for (int i = 0; i < toBeActivated.getSize(); i++)
		{
			hkpEntity* entity = toBeActivated[i];

			ENTITY_STATE(entity) = HK_TOI_ENTITY_STATE_ACTIVE;

			// now contact constraints use an internal version of add/remove constraints and we do not need the following line.
			//entity->getSimulationIsland()->getWorld()->m_pendingOperations->executeAllPending();

			{
				const hkSmallArray<struct hkConstraintInternal>& constraintMasters = entity->getConstraintMasters();

				for (int j = 0; j < constraintMasters.getSize(); j++ )
				{
					const hkConstraintInternal* c = &constraintMasters[j];

					GET_CONSTRAINT_NAME(constraintName, c->m_constraint);
					if (! hkLs_isConstraintInteresting(c->m_constraint, minPriorityToProcess))
					{
						hkToiPrintf("sub.toi.newcon", "#      sub.toi.newcon  constr not interesting : %s \n", constraintName);
						continue;
					}

					hkpEntity* otherEntity = c->getOtherEntity(entity);

					if ( otherEntity->isFixed() ) // keyframed bodies are handled in the else-case)
					{
						newTouchedConstraints.pushBack(c->m_constraint);
						HK_ASSERT( 0xf0212345, otherEntity->m_solverData == 0);
						hkToiPrintf("sub.toi.newcon", "#      sub.toi.newcon  new touched constraint : %s \n", constraintName);
					}
					// if its active, the constraint is already there
					else if ( ENTITY_STATE(otherEntity) != HK_TOI_ENTITY_STATE_ACTIVE)
					{
						if ( otherEntity->getCollidable()->getQualityType() == HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI )
						{
							HK_WARN_ONCE(0xad810213, "An entity of quality type HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI gets included in full toi solving (caused by other non-simplified-toi interaction). This is likely to cause penetrations for collisions of simplified-toi quality. Adjust your constraint priorities or change minimum quality of constraints to be included in toi solving.");
						}

						newTouchedConstraints.pushBack(c->m_constraint);
						hkToiPrintf("sub.toi.newcon", "#      sub.toi.newcon  new touched constraint : %s \n", constraintName);
						if ( ENTITY_STATE(otherEntity ) < HK_TOI_ENTITY_STATE_TOUCHED)
						{
							newTouchedEntities.pushBack(otherEntity);
							hkLs_updateTransformIfNeeded(otherEntity, time, entityState);
							hkToiPrintf("sub.toi.newcon", "#      sub.toi.newcon  new touched entity     : %s \n", RETURN_BODY_NAME(otherEntity));
							HK_ASSERT(0x514613b4, ENTITY_STATE(otherEntity) == HK_TOI_ENTITY_STATE_TRANSFORM_UPDATED );
							ENTITY_STATE(otherEntity) = HK_TOI_ENTITY_STATE_TOUCHED;
						}
					}
				}
			}
			{
				const hkArray<class hkpConstraintInstance*>&  constraintSlaves = entity->getConstraintSlaves();
				HK_ASSERT( 0xf0323454, !entity->isFixed() );	// if the enitity is fixed,  constraintSlaves would be non deterministic

				const int startNewTouchedConstraints = newTouchedConstraints.getSize();
				const int startNewTouchedEntities = newTouchedEntities.getSize();

				for (int j = 0; j < constraintSlaves.getSize(); j++ )
				{
					hkConstraintInternal* c = constraintSlaves[j]->getInternal();

					GET_CONSTRAINT_NAME(constraintName, c->m_constraint);
					if (! hkLs_isConstraintInteresting(c->m_constraint, minPriorityToProcess))
					{
						hkToiPrintf("sub.toi.newcon", "#      sub.toi.newcon  constr not interesting : %s \n", constraintName);
						continue;
					}
					//hkToiPrintf("sub.toi.newcon", "#      sub.toi.newcon  constr is  interesting : %s \n", constraintName);

					hkpEntity* otherEntity = c->getOtherEntity(entity);

					if ( otherEntity->isFixed() )//ad_m2 || otherBody->getMotionType() == hkpMotion::MOTION_KEYFRAMED)
					{
						newTouchedConstraints.pushBack(c->m_constraint);
						hkToiPrintf("sub.toi.newcon", "#      sub.toi.newcon  new touched constraint : %s \n", constraintName);
					}
					// if its active, the constraint is already there
					else if ( ENTITY_STATE(otherEntity) != HK_TOI_ENTITY_STATE_ACTIVE)
					{
						newTouchedConstraints.pushBack(c->m_constraint);
						hkToiPrintf("sub.toi.newcon", "#      sub.toi.newcon  new touched constraint : %s \n", constraintName);
						if ( ENTITY_STATE(otherEntity) < HK_TOI_ENTITY_STATE_TOUCHED)
						{
							newTouchedEntities.pushBack(otherEntity);
							hkLs_updateTransformIfNeeded(otherEntity, time, entityState);
							hkToiPrintf("sub.toi.newcon", "#      sub.toi.newcon  new touched entity     : %s \n", RETURN_BODY_NAME(otherEntity));
							HK_ASSERT(0x693ab4aa, ENTITY_STATE(otherEntity) == HK_TOI_ENTITY_STATE_TRANSFORM_UPDATED );
							ENTITY_STATE(otherEntity) = HK_TOI_ENTITY_STATE_TOUCHED;
						}
					}
				}

				// only sort the newTouchedEntities/Constraints added from the constraintSlaves list. This is done for deterministic simulation.
				hkSort(newTouchedConstraints.begin() + startNewTouchedConstraints, newTouchedConstraints.getSize() - startNewTouchedConstraints, less_ConstraintPtr);
				hkSort(newTouchedEntities.begin()    + startNewTouchedEntities   , newTouchedEntities.getSize()    - startNewTouchedEntities,    less_EntityPtr);
			}
		}
	}

}

// Constraints linking to fixed and keyframed bodies should be marked as activated immediately, as those fixec bodies never get 'activated'.
void hkLs_toiActivateConstraintsLinkingToFixedAndKeyframedEntities(hkArray<hkpConstraintSchemaInfo>& constraintStatus, int& firstNonActiveConstraintStatus, int numOfConstraintsToCheckAtTheEndOfTheList)
{
	HK_ASSERT(0xf0ff0056, firstNonActiveConstraintStatus <= constraintStatus.getSize() - numOfConstraintsToCheckAtTheEndOfTheList);

	for (int i = constraintStatus.getSize() - numOfConstraintsToCheckAtTheEndOfTheList; i < constraintStatus.getSize(); i++)
	{
		hkpRigidBody** body = reinterpret_cast<hkpRigidBody**>(constraintStatus[i].m_constraint->getInternal()->m_entities);
		if (body[0]->isFixedOrKeyframed() || body[1]->isFixedOrKeyframed())
		{
			//swap constraints
			hkAlgorithm::swap(constraintStatus[firstNonActiveConstraintStatus++], constraintStatus[i]);
		}
	}

}

// Constraints linking to other already-activated bodies should be marked as activated immediately.
void hkLs_toiActivateConstraintsLinkingActivatedEntities(hkArray<hkpConstraintSchemaInfo>& constraintStatus, int& firstNonActiveConstraintStatus, hkFixedArray<hkUchar>& entityState)
{
	for (int i = firstNonActiveConstraintStatus; i < constraintStatus.getSize(); i++)
	{
		hkpEntity** e = constraintStatus[i].m_constraint->getInternal()->m_entities;
		if (ENTITY_STATE(e[0]) == HK_TOI_ENTITY_STATE_ACTIVE && ENTITY_STATE(e[1]) == HK_TOI_ENTITY_STATE_ACTIVE)
		{
			//swap constraints
			hkAlgorithm::swap(constraintStatus[firstNonActiveConstraintStatus++], constraintStatus[i]);
		}
	}

}

	//
void hkLs_toiResetVelocityAccumulatorsForEntities(hkpConstraintSolverResources& solverResources, const hkArray<hkpEntity*>& entities)
{
	{
		for (int i = 0; i < entities.getSize(); i++)
		{
			hkpEntity* entity = entities[i];
			hkpVelocityAccumulator* accumulator = hkAddByteOffset(solverResources.m_accumulators, entity->m_solverData);
			hkpMotion* motion = static_cast<hkpMotion*>(entity->getMotion());
			accumulator->m_linearVel  = motion->m_linearVelocity;

			// Note: angularVelocity is in local space.

			// Use coreFromWorldMatrix
			hkpConstraintSolverResources::VelocityAccumTransformBackup* accumBackup = &solverResources.m_accumulatorsBackup[entity->m_solverData/sizeof(hkpVelocityAccumulator)];


			// set angular velocity
			accumulator->m_angularVel._setRotatedDir( accumBackup->m_coreTworldRotation, motion->m_angularVelocity );
		}
	}
}


// Sets hkMotionState of backstepped bodies to a 'stationary' transform (trans(0) == trans(1)).
// Sets motionState's start-end times to this-TOI's-time -to- time-of-next-PSI
// Caution: this actually does backstep HK_KEYFRAMED_MOTION'ed objects.
void hkLs_backstepAndFreezeEntireIsland(hkTime time, hkpSimulationIsland* island, hkFixedArray<hkUchar>&entityState, hkArray<hkpEntity*>& activeEntities)
{
	for (int e = 0; e < island->m_entities.getSize(); e++)
	{
		hkpRigidBody& body = *static_cast<hkpRigidBody*>(island->m_entities[e]);
		hkMotionState& motionState = body.getRigidMotion()->m_motionState;

		HK_ASSERT2(0xf0ff009d, !hkMath::equal( time, motionState.getSweptTransform().getBaseTime() + 1.0f / motionState.getSweptTransform().getInvDeltaTime() ), "Internal: Optimization note: Backstepping for this entity might alrady have been performed." );
		{
			// backstep, if the body has not been backstepped yet.
			hkSweptTransformUtil::backStepMotionState( time, motionState );
			if (entityState[e] != HK_TOI_ENTITY_STATE_ACTIVE)
			{
				HK_ASSERT(0xf0ff005a, activeEntities.indexOf(&body) < 0);
				activeEntities.pushBack(&body);
				entityState[e] = HK_TOI_ENTITY_STATE_ACTIVE;
				// BUT MANIFOLDS ARE NOT UPDATED (for bodies that have not been touched; and partly updated for the touched ones)
			}
			HK_ASSERT(0xf0ff005b, activeEntities.indexOf(&body) >= 0);
		}


		// set initial and final positions/orientations in hkSweptTransform to present pos (pos1)
		motionState.getSweptTransform().m_centerOfMass0 = motionState.getSweptTransform().m_centerOfMass1;
		motionState.getSweptTransform().m_rotation0 = motionState.getSweptTransform().m_rotation1;

		motionState.getSweptTransform().m_centerOfMass0(3) = time;
		motionState.getSweptTransform().m_centerOfMass1(3) = 1.0f / (island->m_world->getCurrentPsiTime() - time);
	}
}


// Restores hkpMotion::hkTransform (to SweptTranform()->transform(t1)) for all TRANSFORM_UPDATED and TOUCHED entities in an island
void hkLs_restoreTransformOnBodiesWithUpdatedTransform(hkpSimulationIsland* island, hkFixedArray<hkUchar>& entityState, hkpConstraintSolverResources& solverResources)
{
	if (island->m_entities.getSize()) { hkToiPrintf("sub-upd", "#    updating transforms\r\n" ); }

	for (int i = 0; i < island->m_entities.getSize(); i++)
	{
		int thisEntityState = entityState[i];
		if ( thisEntityState == HK_TOI_ENTITY_STATE_TRANSFORM_UPDATED ||
			thisEntityState == HK_TOI_ENTITY_STATE_TOUCHED           )
		{
			hkpRigidBody* body = static_cast<hkpRigidBody*>(island->m_entities[i]);
			hkMotionState* motionState = &body->getRigidMotion()->m_motionState;
			hkSweptTransformUtil::calcTransAtT1( motionState->getSweptTransform(), motionState->getTransform() );
		}
		if ( thisEntityState == HK_TOI_ENTITY_STATE_MANIFOLD_UPDATED ||
			 thisEntityState == HK_TOI_ENTITY_STATE_TOUCHED           )
		{
			hkpRigidBody* body = static_cast<hkpRigidBody*>(island->m_entities[i]);
			// copy velocities out.
			hkpConstraintSolverResources::VelocityAccumTransformBackup* accumBackup = &solverResources.m_accumulatorsBackup[body->m_solverData/sizeof(hkpVelocityAccumulator)];
			hkpVelocityAccumulator* accumulator = hkAddByteOffset(solverResources.m_accumulators, body->m_solverData);
			// The following line does: copy the first column of the getCoreFromWorldMatrix() from the backup struct.

			hkpMotion* motion = static_cast<hkpMotion*>(body->getMotion());
			motion->m_linearVelocity = accumulator->m_linearVel;
			motion->m_angularVelocity._setRotatedInverseDir( accumBackup->m_coreTworldRotation, accumulator->m_angularVel );

			//consider:  when we have a subsequent TOI, then the previous resulting velocity stored in the motion will be overwritten hence the delayed response (only stored in the motion's velocity, and not in the motion's positions (at t0&t1)) from the previous TOI is lost.
		}
	}
}


hkBool hkLs_toiCheckFinalValidityOfCriticalConstraints(hkpConstraintSolverResources& solverResources, const hkArray<hkpConstraintSchemaInfo>& constraintStatus, hkpProcessCollisionInput& processInput, hkArray<hkpToiResourceMgr::ConstraintViolationInfo>* violatedConstraints)
{
	HK_ASSERT(0xf0ff0057, constraintStatus.getSize());

	hkBool areOk = true;
	for (int i = 0; i < constraintStatus.getSize(); i++)
	{
		const hkpConstraintSchemaInfo& cStatus = constraintStatus[i];

		if (cStatus.m_constraint->getPriority() != hkpConstraintInstance::PRIORITY_TOI_FORCED)
		{
			continue;
		}

		{
			hkReal velocities[256];

			int numVelocities = hkSolveGetToiViolatingConstraintVelocity(*solverResources.m_solverInfo, cStatus.m_schema, solverResources.m_accumulators, 256, velocities);
			int failingIndex = hkLs_areVelocitiesOk(solverResources.m_stepInfo->m_deltaTime, cStatus, velocities, numVelocities, processInput);


			if (failingIndex >= 0)
			{
				if (! violatedConstraints)
				{
					return false;
				}
				areOk = false;

				hkpToiResourceMgr::ConstraintViolationInfo& info =  violatedConstraints->expandOne();
				info.m_constraint = cStatus.m_constraint;

				if (hkpConstraintData::CONSTRAINT_TYPE_CONTACT == cStatus.m_constraint->getData()->getType())
				{
					const hkpSimpleContactConstraintData* contactData = static_cast<const hkpSimpleContactConstraintData*>(cStatus.m_constraint->getData());
					info.m_contactPoint = &contactData->m_atom->getContactPoints()[failingIndex];
					info.m_contactPointProperties = contactData->m_atom->getContactPointPropertiesStream(failingIndex)->asProperties();
					// XXX copy impulse applied
				}
				else
				{
					info.m_contactPoint = HK_NULL;
					info.m_contactPointProperties = HK_NULL;
				}
			}
		}
	}

	return areOk;
}


// This is the main loop of TOI-event handling.
//
// Initially this function attempts to resolve a TOI event only performing collision response computation for the initial
// pair of two interpenetrating objects. If this causes violation of neighboring constraints, then those constraints are
// gradually activated one by one. And therefore the sub-system of rigidBodies and constraints to solve is expanded.
//
// <todo> say: How do we choose agents/constraints to activate
//
// <todo> say: How does the loop look like
//
// <todo> say: How do we handle critical constraints?
// \param rotateNormal this is set by the contactPointCallbacks fired earlier.
//
// /return Returns HK_SUCCESS if localized solving was performed.
//         Returns HK_FAILURE if localized solving was skipped according to contactManager's confirmToi() call results.
//
hkResult hkLs_localizedSolveToi( const hkpToiResources& toiResources, hkpConstraintSolverResources& solverResources, hkpToiEvent& event, hkpToiResourceMgr& toiResourceMgr, hkpWorld* world, hkArray<hkpEntity*>& activeEntities, hkFixedArray<hkUchar>& entityState, hkReal rotateNormal )
{
		//
		// Initialize local variables
		//

		// activeEntities are the ones which get their velocity changed through a TOI
	HK_ASSERT(0xf0ff0058, !activeEntities.getSize());

		// constraints information for all constraints which we have the jacobians build.
		// the first firstNonActiveConstraintStatus are active, which means they are actively pushing objects
		// the rest is just for checking the velocities
	hkInplaceArray<hkpConstraintSchemaInfo, 64> constraintStatus;
	int firstNonActiveConstraintStatus = 0;							// the index of the first non active constraints in the constraintStatus array


	hkpSimulationIsland* island = ( !event.m_entities[0]->isFixed()) ? event.m_entities[0]->getSimulationIsland() : event.m_entities[1]->getSimulationIsland();


	HK_INTERNAL_TIMER_SPLIT_LIST( "2BodyCollide" );

		//
		// Touched entities.
		// These are entities which are connected to the system which is solved during a TOI
		// These touched entities eventually become toBeActivated entities, which become active entities
		//
	hkLocalArray<hkpEntity*> touchedEntities(island->m_entities.getSize());
	{
		if (!event.m_entities[0]->isFixed())
		{
			touchedEntities.pushBackUnchecked(event.m_entities[0]);
			hkLs_updateTransform(event.m_entities[0], event.m_time);
			ENTITY_STATE( event.m_entities[0] ) = HK_TOI_ENTITY_STATE_TOUCHED;
		}
		if (!event.m_entities[1]->isFixed())
		{
			touchedEntities.pushBackUnchecked(event.m_entities[1]);
			hkLs_updateTransform(event.m_entities[1], event.m_time);
			ENTITY_STATE( event.m_entities[1] ) = HK_TOI_ENTITY_STATE_TOUCHED;
		}
	}

		//
		//	toBeActivated are the bodies, which are connected to active constraints and
		//  the function decided to turn the into active entities
		//
	hkInplaceArray<hkpEntity*,64>          toBeActivated;
	{
		//
		//	Trigger confirmToi method of the contact manager.
		//  This usually fires contact processed callbacks.
		//  This also allows to apply simple TOI-collision response before further localized solving.
		//
		event.m_contactMgr->confirmToi( event, rotateNormal, toBeActivated );

		if (!toBeActivated.getSize())
		{
			// Restore transforms in bodies and return.
			if (!event.m_entities[0]->isFixed())
			{
				hkpRigidBody* bodyA = static_cast<hkpRigidBody*>( event.m_entities[0] );
				hkMotionState* motionState = &bodyA->getRigidMotion()->m_motionState;
				hkSweptTransformUtil::calcTransAtT1( motionState->getSweptTransform(), motionState->getTransform() );
			}
			if (!event.m_entities[1]->isFixed())
			{
				hkpRigidBody* bodyB = static_cast<hkpRigidBody*>( event.m_entities[1] );
				hkMotionState* motionState = &bodyB->getRigidMotion()->m_motionState;
				hkSweptTransformUtil::calcTransAtT1( motionState->getSweptTransform(), motionState->getTransform() );
			}
			return HK_FAILURE;
		}

	}

	hkpEntity* initialActivatedEntities[2] = { toBeActivated[0], toBeActivated.getSize() > 1 ? toBeActivated[1] : HK_NULL };

	HK_ON_DEBUG( ENTITY_STATE(initialActivatedEntities[0]) = HK_TOI_ENTITY_STATE_PENDING_AND_TOUCHED; )
	HK_ON_DEBUG( if (initialActivatedEntities[1]) { ENTITY_STATE(initialActivatedEntities[1]) = HK_TOI_ENTITY_STATE_PENDING_AND_TOUCHED; } )

	hkToiPrintf("sub.scr", "#    SCR	%2.7f: %-6s %-6s \n\n", event.m_time, event.m_entities[0]->getName(), event.m_entities[1]->getName() );

	
	// We used simple collidionResponse so we still need to build accumulators for the two TOI bodies
	// info: the bodies are already backstepped
	hkpConstraintSolverSetup::internalAddAccumulators(solverResources, &touchedEntities[0], touchedEntities.getSize());

	hkBool toBeActivatedArrayIsAlreadyPreSet = true;

	HK_ASSERT(0xad000019, toBeActivated.getSize());

	// Initialized out of the internal for-loop
	hkInplaceArray<hkpEntity*,64>              moreTouchedEntities;
	hkInplaceArray<hkpConstraintInstance*,64>  moreConstraints;

		//
		//	Try to expand the active set toiResources.m_numToiSolverIterations times
		//
	for ( int maxNumFullSteps = toiResources.m_numToiSolverIterations; maxNumFullSteps>0; maxNumFullSteps--)
	{
		//
		//	Adds constraints and entities as long as constraint velocities are violated
		//  Note: This whole loop assumes that velocities are not integrated (contact manifolds don't change)
		//
		HK_ASSERT(0xad000004, touchedEntities.getSize() || firstNonActiveConstraintStatus == constraintStatus.getSize() );

		while(1)
		{
			HK_INTERNAL_TIMER_SPLIT_LIST( "ExpandSystem" );
			moreConstraints.clear();
			moreTouchedEntities.clear();

			// Check velocity change in touched-bodies' motions
			// compare their rigidMotions and velocityAccumulators

			//hkToiPrintf("subSolve", "#    loop     ab: %d ib: %da	\r\n", activeEntities.getSize(), touchedEntities.getSize() );

			int updatedFirstNonActiveConstraintStatus = firstNonActiveConstraintStatus;

			if (!toBeActivatedArrayIsAlreadyPreSet)
			{
				if (firstNonActiveConstraintStatus < constraintStatus.getSize()) // !touchedEntities.getSize() ==> firstNonActiveConstraintStatus == constraintStatus.getSize()
				{
					toBeActivated.clear();
					hkLs_toiCheckValidityOfConstraints(solverResources, *world->m_collisionInput, constraintStatus, updatedFirstNonActiveConstraintStatus, toiResources.m_minPriorityToProcess, entityState, touchedEntities, toBeActivated);
					if ( !toBeActivated.getSize() )
					{
						goto RUN_FULL_SOLVER; // no bodies to activate --> run solver.
					}
				}
				else
				{
					// no more constraints which connect to inactive bodies --> run solver.
					goto RUN_FULL_SOLVER;
				}
			}
			HK_ASSERT(0x1234442, !toBeActivatedArrayIsAlreadyPreSet || (toBeActivated.getSize() && touchedEntities.getSize() &&  !activeEntities.getSize()) );

			hkLs_toiActivateEntitiesAndCheckConstraints(*world->getCollisionInputRw(), toiResources.m_minPriorityToProcess, event.m_time, toBeActivated, entityState, moreTouchedEntities, moreConstraints);

			// debug: sets objects to MAGENTA if there is no constraint for the initial TOI pair
			//if (0 && hkDebug && toBeActivatedArrayIsAlreadyPreSet && moreConstraints.getSize() )
			//{
			//	HK_SET_OBJECT_COLOR(hkUlong(touchedEntities[0]->getCollidable()), hkColor::MAGENTA);
			//	if (touchedEntities.getSize() ==2 ) { HK_SET_OBJECT_COLOR(hkUlong(touchedEntities[1]->getCollidable()), hkColor::MAGENTA); }
			//}
			// if ( hkDebugToi )
			// {
			// 	for ( int i = 0; i < moreTouchedEntities.getSize(); i++ ){		hkToiPrintf("sub-add", "#    add ent:%s\n", moreTouchedEntities[i]->getName());			}
			// 	for ( int c = 0; c < moreConstraints.getSize(); c++ )	{		hkToiPrintf("sub-add", "#    add con:%s-%s\n", moreConstraints[c]->getEntityA()->getName(), moreConstraints[c]->getEntityB()->getName());	}
			// }
			toBeActivatedArrayIsAlreadyPreSet = false;

			HK_INTERNAL_TIMER_SPLIT_LIST( "buildAcc+Jac" );

			//
			// Calculate whether new constraints and accumulators will fit into memory
			//
			if ( ! hkpConstraintSolverSetup::internalIsMemoryOkForNewAccumulators(solverResources, moreTouchedEntities.begin(), moreTouchedEntities.getSize())
			  || ! hkpConstraintSolverSetup::internalIsMemoryOkForNewJacobianSchemas(solverResources, moreConstraints.begin(), moreConstraints.getSize())
			  || toiResources.m_maxNumConstraints    < constraintStatus.getSize() + moreConstraints.getSize()
			  || toiResources.m_maxNumEntities       < activeEntities.getSize()   + touchedEntities.getSize() + moreTouchedEntities.getSize()
			  || toiResources.m_maxNumActiveEntities < activeEntities.getSize()   + toBeActivated.getSize())
			{
				// Not enough solverResources to expand localized body set.
				hkToiPrintf("sub-nomem", "#    NO MEMORY\r\n" );

					// unmark all toBeActivated/newTouchedEntities
				{	for (int i = 0; i < toBeActivated.getSize(); i++){	     	ENTITY_STATE(toBeActivated[i])       = HK_TOI_ENTITY_STATE_MANIFOLD_UPDATED;  } }
				{	for (int i = 0; i < moreTouchedEntities.getSize(); i++){	ENTITY_STATE(moreTouchedEntities[i]) = HK_TOI_ENTITY_STATE_TRANSFORM_UPDATED; }	}

				hkpToiResourceMgrResponse response = toiResourceMgr.resourcesDepleted();

				if (response == HK_TOI_RESOURCE_MGR_RESPONSE_BACKSTEP)
				{
					HK_INTERNAL_TIMER_SPLIT_LIST( "Backstep" );
					hkLs_backstepAndFreezeEntireIsland(event.m_time, island, entityState, activeEntities);

					HK_INTERNAL_TIMER_SPLIT_LIST( "InvalidTIMs" );
					for (int i = 0; i < activeEntities.getSize(); i++)	{	hkpWorldAgentUtil::invalidateTim(activeEntities[i], *world->m_collisionInput);	}

					// HACK: clear active entities array, so that none are recollided. At the same time do not reset entityState array, which is used to remove pending TOI events for active entities.
					activeEntities.clear();
					return HK_SUCCESS;
				}

				// in all other cases jump to the solver directly
				goto RUN_FULL_SOLVER;
			}
			else
			{
				// Enough solverResources to expand localized body set.

				//ToiPrintf("subSolve", "    more     ne: %d nc: %da	\n", moreTouchedEntities.getSize(), moreConstraints.getSize() );

				// can never fail -- memory had been checked
				if (hkDebugToi){for ( int i = 0; i < toBeActivated.getSize(); i++){	hkToiPrintf("sub-act", "#    act ent:%s\n", toBeActivated[i]->getName());}}

				hkpConstraintSolverSetup::internalAddAccumulators(    solverResources, moreTouchedEntities.begin(), moreTouchedEntities.getSize());
				hkpConstraintSolverSetup::internalAddJacobianSchemas(solverResources, moreConstraints.begin(),     moreConstraints.getSize(), constraintStatus);

				firstNonActiveConstraintStatus = updatedFirstNonActiveConstraintStatus;

				// Remove pending==activated entities from touched list.
				{
					for (int i = 0; i < touchedEntities.getSize(); i++)
					{
						HK_ASSERT(0xf0ff0509, !(ENTITY_STATE(touchedEntities[i]) & HK_TOI_ENTITY_STATE_PENDING_FLAG));
						if (ENTITY_STATE(touchedEntities[i]) == HK_TOI_ENTITY_STATE_ACTIVE)
						{
							HK_TOI_COLORING( HK_SET_OBJECT_COLOR( hkUlong(touchedEntities[i]->getCollidable()), 0xffff7700); )
							touchedEntities.removeAt(i--);
						}
					}
				}

				activeEntities.insertAt (activeEntities.getSize(),  toBeActivated.begin(), toBeActivated.getSize() );

				// putting touched dynamic bodies on active entities list
				{
					touchedEntities.reserve( touchedEntities.getSize() + moreTouchedEntities.getSize() );
					for (int i = 0; i < moreTouchedEntities.getSize(); i++)
					{
						if (static_cast<hkpRigidBody*>(moreTouchedEntities[i])->getMotionType() != hkpMotion::MOTION_KEYFRAMED)
						{
							HK_TOI_COLORING( HK_SET_OBJECT_COLOR( hkUlong(moreTouchedEntities[i]->getCollidable()), 0xff7777ff); )
							touchedEntities.pushBackUnchecked( moreTouchedEntities[i]) ;
						}
					}
				}

				//
				// Go through all newly added constraints, and check if they link to fixed bodies.
				//   If so -- mark them as active, so that they are skipped in toiCheckValidityOfConstraints()
				//   which returns a list of entities to be activated (and does not have support for handlind fixed entities).
				//
					// (only mark active, so we never attempt to activate fixed bodies)
					// TODO do the same for keyframed bodies
				hkLs_toiActivateConstraintsLinkingToFixedAndKeyframedEntities(constraintStatus, firstNonActiveConstraintStatus, moreConstraints.getSize() );
				hkLs_toiActivateConstraintsLinkingActivatedEntities(          constraintStatus, firstNonActiveConstraintStatus, entityState );

				//
				// Perform incremental subSolving for the newly added constraints
				//
				hkpConstraintSolverSetup::subSolve(solverResources, hkpConstraintSolverSetup::SOLVER_MODE_INCREMENTAL_CONTINUE);
				
				//hkToiPrintf("sub.toi.inc", "#      sub.toi.inc   %d\n", solverResources.m_elemTemp[0].m_current - solverResources.m_elemTemp[0].m_begin );
				//hkToiPrintf("sub.toi.inc", "#      sub.toi.inc   %d\n", solverResources.m_elemTemp[1].m_current - solverResources.m_elemTemp[1].m_begin );
				
				hkToiPrintf("sub.toi.inc", "#      sub.toi.inc   %d\n", solverResources.m_elemTempCurrent - solverResources.m_elemTemp );

				// <TODO> optionally reapply the collision response which caused the TOI event being processed
				//      if no constraint was created. Use hkpSimpleCollisionResponse

			}

			// break the loop if no new constraints/entities was added
			if ( !moreConstraints.getSize() && !moreTouchedEntities.getSize() )
			{
				goto RUN_FULL_SOLVER;
			}
		} // while(1)

RUN_FULL_SOLVER:

		HK_INTERNAL_TIMER_SPLIT_LIST( "Solver" );
		{
			// full solve (1 step only)
			hkToiPrintf("sub.toi.fulsol",   "#      sub.toi.fulsol--- c:%-2d ae:%-2d te:%-2d \n", constraintStatus.getSize(), activeEntities.getSize(), touchedEntities.getSize() );
			hkpConstraintSolverSetup::subSolve(solverResources, hkpConstraintSolverSetup::SOLVER_MODE_PROCESS_ALL);
		}
	}

	HK_INTERNAL_TIMER_SPLIT_LIST( "ForcedConstr" );

		//
		// perform final check and iterations for forcedQuality-Constraints
		//
	const int highestPrioritySchemas = hkpConstraintSolverResources::NUM_PRIORITY_CLASSES - 1;
	if (solverResources.m_schemas[highestPrioritySchemas].m_begin != solverResources.m_schemas[highestPrioritySchemas].m_current)
	{
		hkLs_toiResetVelocityAccumulatorsForEntities(solverResources, touchedEntities);
		hkBool forcedConstraintsAreSatisfied = hkLs_toiCheckFinalValidityOfCriticalConstraints(solverResources, constraintStatus, *world->m_collisionInput, HK_NULL);
		if (!forcedConstraintsAreSatisfied)
		{
			int triesLeft = toiResources.m_numForcedToiFinalSolverIterations;
			while (!forcedConstraintsAreSatisfied && ( --triesLeft >= 0) )
			{
				// solve
				
				//hkSolveStepJacobians(*solverResources.m_solverInfo, solverResources.m_schemas[1].m_begin, solverResources.m_accumulators, solverResources.m_elemTemp[1].m_begin);
				
				hkSolveStepJacobians(*solverResources.m_solverInfo, solverResources.m_schemas[highestPrioritySchemas].m_begin, solverResources.m_accumulators, solverResources.m_elemTemp);

				// check validity of forced-quality constraints, assuming touched entities to be not affected/reintegrated.
				hkLs_toiResetVelocityAccumulatorsForEntities(solverResources, touchedEntities);
				forcedConstraintsAreSatisfied = hkLs_toiCheckFinalValidityOfCriticalConstraints(solverResources, constraintStatus, *world->m_collisionInput, HK_NULL);
			}
		}

		if ( !forcedConstraintsAreSatisfied )
		{
			hkInplaceArray<hkpToiResourceMgr::ConstraintViolationInfo, 32> violatedConstraints;
			// Get all failing constraints
			hkLs_toiCheckFinalValidityOfCriticalConstraints(solverResources, constraintStatus, *world->m_collisionInput, &violatedConstraints);

			hkpToiResourceMgrResponse response = toiResourceMgr.cannotSolve(violatedConstraints);
			if (response == HK_TOI_RESOURCE_MGR_RESPONSE_BACKSTEP)
			{
				HK_INTERNAL_TIMER_SPLIT_LIST( "Backstep" );
				hkLs_backstepAndFreezeEntireIsland(event.m_time, island, entityState, activeEntities);

				HK_INTERNAL_TIMER_SPLIT_LIST( "InvalidTims" );
				{ for (int i = 0; i < activeEntities.getSize(); i++){		hkpWorldAgentUtil::invalidateTim(activeEntities[i], *world->m_collisionInput);	} }

				// HACK: clear active entities array, so that none recollided. At the same time do not reset entityState array, which is used to remove pending TOI events for active entities.
				activeEntities.clear();
				return HK_SUCCESS;
			}
			HK_ASSERT2(0xf0325478, response == HK_TOI_RESOURCE_MGR_RESPONSE_CONTINUE, "hkpToiResourceMgr::cannotSolve should only return HK_TOI_LISTENER_RESPONSE_CONTINUE or HK_TOI_LISTENER_RESPONSE_BACKSTEP ");
		}
	}

	if (hkDebugToi)
	{
		hkStringBuf str1; {	for (int e = 0; e < activeEntities.getSize(); e++){		str1 += BODY_NAME(activeEntities[e]); str1 += " ";	}		}
		hkStringBuf str2; {	for (int e = 0; e < touchedEntities.getSize(); e++){	str2 += BODY_NAME(touchedEntities[e]); str2 += " "; }		}

		hkStringBuf str3;
		{	for (int c = 0; c < firstNonActiveConstraintStatus; c++)
			{
				hkpConstraintInstance* c2 = constraintStatus[c].m_constraint;
				str3 += BODY_NAME(c2->getInternal()->m_entities[0]);
				str3 += "-";
				str3 += BODY_NAME(c2->getInternal()->m_entities[1]);
				str3 += (c2->getPriority() == hkpConstraintInstance::PRIORITY_TOI_FORCED ? "F" : hkpConstraintInstance::PRIORITY_TOI_HIGHER ? "h" : c2->getPriority() == hkpConstraintInstance::PRIORITY_TOI ? "t" : c2->getPriority() == hkpConstraintInstance::PRIORITY_PSI ? "p" : "!");
				str3 += " ";
		}	}

		hkStringBuf str4;
		{	for (int c = firstNonActiveConstraintStatus; c < constraintStatus.getSize(); c++)
			{
				hkpConstraintInstance* c2 = constraintStatus[c].m_constraint;
				str4 += BODY_NAME(c2->getInternal()->m_entities[0]);
				str4 += "-";
				str4 += BODY_NAME(c2->getInternal()->m_entities[1]);
				str4 += (c2->getPriority() == hkpConstraintInstance::PRIORITY_TOI_FORCED ? "F" : hkpConstraintInstance::PRIORITY_TOI_HIGHER ? "h" : c2->getPriority() == hkpConstraintInstance::PRIORITY_TOI ? "t" : c2->getPriority() == hkpConstraintInstance::PRIORITY_PSI ? "p" : "!");
				str4 += " ";
		}	}
		hkToiPrintf("sub.toi.summary", "#      sub.toi.summary  active  : %s \n", str1.cString());
		hkToiPrintf("sub.toi.summary", "#      sub.toi.summary  touched : %s \n", str2.cString());
		hkToiPrintf("sub.toi.summary", "#      sub.toi.summary  actConst: %s \n", str3.cString());
		hkToiPrintf("sub.toi.summary", "#      sub.toi.summary  touConst: %s \n", str4.cString());
	}

	//
	// Activate initial TOI pair if they have not been activated yet [happens if you run out of resources]
	//
	if (!activeEntities.getSize())
	{
		activeEntities.insertAt(0, initialActivatedEntities[0]);
		ENTITY_STATE(activeEntities[0]) = HK_TOI_ENTITY_STATE_ACTIVE;
		if (initialActivatedEntities[1] != HK_NULL)
		{
			activeEntities.insertAt(1, initialActivatedEntities[1]);
			ENTITY_STATE( activeEntities[1] ) = HK_TOI_ENTITY_STATE_ACTIVE;
		}
	}

	// Color initial TOI pair
	{
		HK_TOI_COLORING
		(
			HK_SET_OBJECT_COLOR( hkUlong(initialActivatedEntities[0]->getCollidable()), 0xffff0000);
			if (initialActivatedEntities[1] != HK_NULL)
			{
				HK_SET_OBJECT_COLOR( hkUlong(initialActivatedEntities[1]->getCollidable()), 0xffff0000);
			}
			else
			{
				hkpEntity* otherEntity = hkSelectOther( initialActivatedEntities[0], event.m_entities[0], event.m_entities[1] );
				if (!otherEntity->isFixedOrKeyframed())
				{
					switch ( ENTITY_STATE(otherEntity) )
					{
					case HK_TOI_ENTITY_STATE_TOUCHED:	HK_SET_OBJECT_COLOR( hkUlong(otherEntity->getCollidable()), 0xffff00aa); 			break;
					case HK_TOI_ENTITY_STATE_ACTIVE:	HK_SET_OBJECT_COLOR( hkUlong(otherEntity->getCollidable()), 0xffffaa00); 			break;
					default:							HK_ASSERT(0xad6b433d, false);
					}
				}
			}
		)
	}


	//
	// Finally actually backstep all activated entities to the precise TOI time
	//
	// TODO optimize : we only need to copy from motionState.getTransform to motionState.getSweptTransform().pos/rot(t1)
	HK_INTERNAL_TIMER_SPLIT_LIST( "IntegMotions" );
	{
		for (int e = 0; e < activeEntities.getSize(); e++)
		{
			hkpEntity& body = *activeEntities[e];
			hkMotionState& motionState = body.getMotion()->m_motionState;
			hkSweptTransformUtil::backStepMotionState( event.m_time, motionState );
		}
	}


		//
		// Integrate bodies
		//
	hkpConstraintSolverSetup::oneStepIntegrate( *solverResources.m_solverInfo, *solverResources.m_stepInfo, solverResources.m_accumulators, activeEntities.begin(), activeEntities.getSize());

		// Reset the transform on touched but not activated entities
		// hkpMotion::hkTransform on TRANSFORM_UPDATED and TOUCHED entities
	hkLs_restoreTransformOnBodiesWithUpdatedTransform(island, entityState, solverResources);

		//
		//	invalid the tims for all collision pairs of active entities
		//
	HK_INTERNAL_TIMER_SPLIT_LIST( "InvalidTims" );
	{
		for (int i = 0; i < activeEntities.getSize(); i++)
		{
			hkpWorldAgentUtil::invalidateTim(activeEntities[i], *world->m_collisionInput);
		}
	}
	hkToiPrintf("sub.end", "#    ---- \n");

	return HK_SUCCESS;
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
