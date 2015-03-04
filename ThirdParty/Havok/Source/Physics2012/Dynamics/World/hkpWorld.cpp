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
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>

#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseHandle.h>

#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Dispatch/ContactMgr/hkpNullContactMgrFactory.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>

#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/Filter/Null/hkpNullCollisionFilter.h>
#include <Physics2012/Collide/Filter/DefaultConvexList/hkpDefaultConvexListFilter.h>

#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>
#include <Physics2012/Collide/Agent/hkpCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionData.h>
#include <Physics2012/Collide/Agent/Query/hkpLinearCastCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Agent/Query/hkpCdBodyPairCollector.h>

#include <Physics2012/Collide/Query/CastUtil/hkpSimpleWorldRayCaster.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCaster.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldLinearCaster.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobQueueUtils.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobQueueUtils.h>

#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>
#include <Physics2012/Dynamics/Phantom/hkpShapePhantom.h>

#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpNullAction.h>
#include <Physics2012/Dynamics/Phantom/hkpPhantom.h>
#include <Physics2012/Dynamics/Phantom/hkpPhantomBroadPhaseListener.h>
#include <Physics2012/Dynamics/World/Util/BroadPhase/hkpEntityEntityBroadPhaseListener.h>
#include <Physics2012/Dynamics/World/Util/BroadPhase/hkpBroadPhaseBorderListener.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldMemoryUtil.h>
#include <Physics2012/Dynamics/World/BroadPhaseBorder/hkpBroadPhaseBorder.h>

#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>
#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstance.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstanceAction.h>

#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>

#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>
#include <Physics2012/Dynamics/Collide/Deprecated/hkpReportContactMgr.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>

#include <Physics2012/Dynamics/World/Simulation/hkpSimulation.h>

#include <Physics2012/Dynamics/World/Simulation/Continuous/hkpContinuousSimulation.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>

#if defined(HK_PLATFORM_MULTI_THREAD) && (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)
#	include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobQueueUtils.h>
#	include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#	include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpMultithreadedSimulation.h>
#endif

#include <Physics2012/Dynamics/World/Simulation/Backstep/hkpBackstepSimulation.h>

#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>
#include <Physics2012/Dynamics/World/Maintenance/Default/hkpDefaultWorldMaintenanceMgr.h>
#include <Physics2012/Dynamics/World/Memory/hkpWorldMemoryAvailableWatchDog.h>

#include <Physics/ConstraintSolver/Jacobian/hkpJacobianSchema.h>

#include <Physics2012/Collide/Util/Welding/hkpWeldingUtility.h>

#include <Physics2012/Collide/Agent3/CollectionCollection3/hkpCollectionCollectionAgent3.h>

#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpCollisionCallbackUtil.h>

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpFixedBufferCdPointCollector.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuWorldGetClosestPointsJob.h>

#include <Physics2012/Dynamics/World/Extensions/hkpWorldExtension.h>
#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/ToiResourceMgr/hkpToiResourceMgr.h>

#if defined HK_DEBUG
	// Only used in hkpWorld::constrainedDynamicBodiesCanCollide() below.
#	include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#endif // #if defined HK_DEBUG

#if defined HK_ENABLE_DETERMINISM_CHECKS
#	include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>
#endif


hkBool hkpWorld::m_forceMultithreadedSimulation = false;

hkpMultithreadConfig::hkpMultithreadConfig()
{
#if defined(HK_PLATFORM_HAS_SPU)
	m_maxNumConstraintsSolvedSingleThreaded       = 4;
#else
	m_maxNumConstraintsSolvedSingleThreaded       = 70;
#endif
}

hkpWorld::hkpWorld( hkFinishLoadedObjectFlag flag ) :
m_phantoms(flag)
{
}

void hkpWorld::removeAll()
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	//
	// Cleanup notes: Actions and constraints should be removed by deleting the entities,
	// so we don't explicitly remove them here.
	//
	//

	// NOTE: if collide has been called and integrate has not been called yet
	// by the hkHalfSteppingUtility for example then the
	// m_simulationState will be CAN_NOT_REMOVE_ENTITIES_AND_CONSTRAINTS...
	// This will prevent the world from being destroyed as the loops below will continuously
	// fail to remove any entities.  Here we will set the m_simulationState to
	// CAN_REMOVE_ENTITIES_AND_CONSTRAINTS.  This should be safe as the world is about to
	// be deleted so this state variable should no longer be critical.

	const hkArray<hkpSimulationIsland*>& activeIslands = getActiveSimulationIslands();
	{
		removePhantomBatch( getPhantoms().begin(), getPhantoms().getSize() );

		hkArray< hkpEntity* >::Temp islandEntities;

		// Avoid non-LIFO frees by reserving once for all upcoming append's.
		{
			int numEntities = m_fixedIsland->getEntities().getSize();
			for ( int i = 0; i < activeIslands.getSize(); ++i )
			{
				numEntities += activeIslands[i]->getEntities().getSize();
			}
			for ( int i = 0; i < m_inactiveSimulationIslands.getSize(); ++i )
			{
				numEntities += m_inactiveSimulationIslands[i]->getEntities().getSize();
			}
			islandEntities.reserve( numEntities );
		}

		for ( int i = 0; i < activeIslands.getSize(); ++i )
		{
			islandEntities.append( activeIslands[i]->getEntities().begin(), activeIslands[i]->getEntities().getSize() );
		}

		for ( int i = 0; i < m_inactiveSimulationIslands.getSize(); ++i )
		{
			islandEntities.append( m_inactiveSimulationIslands[i]->getEntities().begin(), m_inactiveSimulationIslands[i]->getEntities().getSize() );
		}

		int worldFixedRigidBodyIndex = m_fixedIsland->getEntities().indexOf( m_fixedRigidBody );
		HK_ASSERT2( 0x2db9267f, worldFixedRigidBodyIndex != -1, "The world's fixed rigid body cannot be located" );
		worldFixedRigidBodyIndex += islandEntities.getSize();

		islandEntities.append( m_fixedIsland->getEntities().begin(), m_fixedIsland->getEntities().getSize() );

		// Do not remove the world's fixed rigid body as it is removed later
		islandEntities.removeAt( worldFixedRigidBodyIndex );

		removeEntityBatch( islandEntities.begin(), islandEntities.getSize() );
	}

	if (!m_wantSimulationIslands)
	{
		// Remove the only active simulation island
		HK_ASSERT(0x2d89367f, activeIslands.getSize() == 1 && activeIslands[0]->getEntities().getSize() == 0);
		hkpWorldOperationUtil::removeIslandFromDirtyList(this, activeIslands.back());
		delete m_activeSimulationIslands.back();
		m_activeSimulationIslands.popBack();
	}

	hkpWorldCallbackUtil::fireWorldRemoveAll( this );
}

hkpWorld::~hkpWorld()
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	HK_ASSERT2( 0xf0df2321, m_isLocked == 0, "You cannot delete the world when its locked, use markForWrite instead" );

	// avoid any spurious asserts, if the user has just added some entities
	HK_ASSERT(0xad000086, !areCriticalOperationsLocked() && !m_pendingOperationsCount);
	//unlockAndAttemptToExecutePendingOperations();

#if defined(HK_PLATFORM_MULTI_THREAD) && (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)
	delete m_multithreadedSimulationJobData;
#endif

	m_maintenanceMgr->removeReference();
	m_maintenanceMgr = HK_NULL;
	if ( m_modifyConstraintCriticalSection )
	{
		delete m_modifyConstraintCriticalSection;
		m_modifyConstraintCriticalSection = HK_NULL;
	}

	if ( m_propertyMasterLock )
	{
		delete m_propertyMasterLock;
		m_propertyMasterLock = HK_NULL;
	}

	m_memoryWatchDog = HK_NULL;

	removeAll();

	{
		// remove fixed stuff

		removeEntity( m_fixedRigidBody );
		m_fixedRigidBody = HK_NULL;

		hkpWorldOperationUtil::removeIslandFromDirtyList(this, m_fixedIsland);
		delete m_fixedIsland;
		m_fixedIsland = HK_NULL;
	}

	// Deleting the fixed island may need to access this critical section so we delete it after.
	if ( m_islandDirtyListCriticalSection )
	{
		delete m_islandDirtyListCriticalSection;
		m_islandDirtyListCriticalSection = HK_NULL;
	}

	// Release references to extensions.
	{
		const int numExtensions = m_worldExtensions.getSize();
		for ( int i = 0; i < numExtensions; ++i )
		{
			m_worldExtensions[i]->removedFromWorld( this );
		}
	}


	hkpWorldCallbackUtil::fireWorldDeleted( this );

	m_broadPhase->removeReference();
	m_broadPhase = HK_NULL;

	m_collisionDispatcher->removeReference();
	m_collisionDispatcher = HK_NULL;

	delete m_broadPhaseDispatcher;
	if ( m_broadPhaseBorder )
	{
		m_broadPhaseBorder->removeReference();
	}

	delete m_phantomBroadPhaseListener;
	delete m_entityEntityBroadPhaseListener;
	delete m_broadPhaseBorderListener;

	m_collisionFilter->removeReference();
	m_convexListFilter->removeReference();

	delete m_collisionInput->m_config;
	delete m_collisionInput;

	m_simulation->removeReference();

	HK_ON_DEBUG( hkpDebugInfoOnPendingOperationQueues::cleanup(this); );
	delete m_pendingOperations;

	delete m_violatedConstraintArray;
}

static void hkWorld_setupContactMgrFactories( hkpWorld* world, hkpCollisionDispatcher* dis )
{
	hkpContactMgrFactory* simple = new hkpSimpleConstraintContactMgr::Factory( world );
	hkpContactMgrFactory* rep    = new hkpReportContactMgr::Factory( world );
	hkpContactMgrFactory* none   = new hkpNullContactMgrFactory();



	// simple
	dis->registerContactMgrFactoryWithAll( simple, hkpMaterial::RESPONSE_SIMPLE_CONTACT );
	// Response reporting is deprecated.
	dis->registerContactMgrFactoryWithAll( rep, hkpMaterial::RESPONSE_REPORTING );
	dis->registerContactMgrFactoryWithAll( none, hkpMaterial::RESPONSE_NONE );

	simple->removeReference();
	rep->removeReference();
	none->removeReference();
}

// This method should be called if you have changed the collision filter for the world.
void hkpWorld::updateCollisionFilterOnWorld( hkpUpdateCollisionFilterOnWorldMode updateMode, hkpUpdateCollectionFilterMode updateShapeCollectionFilter )
{
	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::UpdateFilterOnWorld op;
		op.m_collisionFilterUpdateMode = updateMode;
		op.m_updateShapeCollections = updateShapeCollectionFilter;
		queueOperation( op );
		return;
	}
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	//
	// Proceed whit the proper operation
	//

	//HK_ASSERT2(0x623c210c,  m_filterUpdateState == CAN_UPDATE_FILTERS, "You are trying to update collision filters during a collide() call. This can lead to the system destroying a collision agent with a function in the current call stack, and so is not allowed.");

	blockExecutingPendingOperations(true);

	HK_TIMER_BEGIN( "UpdateFilterOnWorld", HK_NULL);
	if ( updateMode == HK_UPDATE_FILTER_ON_WORLD_FULL_CHECK )
	{


		// This method should be called if you have changed the world collision filter
		{
			// Active Islands
			const hkArray<hkpSimulationIsland*>& activeIslands = getActiveSimulationIslands();
			for ( int i = 0; i < activeIslands.getSize(); ++i )
			{
				const hkArray<hkpEntity*>& entities = activeIslands[i]->getEntities();

				for( int j = 0; j < entities.getSize(); j++ )
				{
					hkpEntity* entity = entities[j];
					updateCollisionFilterOnEntity( entity, HK_UPDATE_FILTER_ON_ENTITY_FULL_CHECK, updateShapeCollectionFilter );
				}
			}
		}

		{
			// Inactive Islands
			for ( int i = 0; i < m_inactiveSimulationIslands.getSize(); ++i )
			{
				const hkArray<hkpEntity*>& entities = m_inactiveSimulationIslands[i]->getEntities();

				for( int j = 0; j < entities.getSize(); j++ )
				{
					hkpEntity* entity = entities[j];
					updateCollisionFilterOnEntity( entity, HK_UPDATE_FILTER_ON_ENTITY_FULL_CHECK, updateShapeCollectionFilter);
				}
			}
		}

	}
	else
	{
		// do not requery broadphase -- only check each agent once
		lockCriticalOperations();

		const hkArray<hkpSimulationIsland*>* arrays[2] = { &getActiveSimulationIslands(), &m_inactiveSimulationIslands };

		for ( int a = 0; a < 2; a++)
		{
			hkInplaceArray<hkpAgentNnEntry*, 32> agentsToRemove;
			// Active + Inactive Islands
			for ( int i = 0; i < arrays[a]->getSize(); ++i )
			{
				hkpSimulationIsland* island = (*arrays[a])[i];
				hkpAgentNnTrack *const tracks[2] = { &island->m_narrowphaseAgentTrack, &island->m_midphaseAgentTrack };

				for ( int j = 0; j < 2; ++j )
				{
					hkpAgentNnTrack& track = *tracks[j];

					agentsToRemove.clear();
					HK_FOR_ALL_AGENT_ENTRIES_BEGIN(track, agentEntry)
					{
						// Verify existing collision agents

						if (!getCollisionFilter()->isCollisionEnabled( *agentEntry->m_collidable[0],  *agentEntry->m_collidable[1] ))
						{
							goto removeAgentLabel;
						}

						// check for disabled collisions, especially landscape = landscape ones
						{
							hkpCollidableQualityType qt0 = agentEntry->m_collidable[0]->getQualityType();
							hkpCollidableQualityType qt1 = agentEntry->m_collidable[1]->getQualityType();
							int collisionQuality = getCollisionDispatcher()->getCollisionQualityIndex( qt0, qt1 );
							if ( collisionQuality == hkpCollisionDispatcher::COLLISION_QUALITY_INVALID )
							{
								goto removeAgentLabel;
							}
							else if(collisionQuality != agentEntry->m_collisionQualityIndex)
							{
								HK_WARN(0xad381441, "Collision quality between two entities has changed. "
									"The agent is removed and collision between the objects is effectively disabled. "
									"Run updateCollisionFilterOnWorld() with HK_UPDATE_FILTER_ON_WORLD_FULL_CHECK "
									"option to recreate the agent.");
								goto removeAgentLabel;
							}
						}

						// check collections
						if ( updateShapeCollectionFilter == HK_UPDATE_COLLECTION_FILTER_PROCESS_SHAPE_COLLECTIONS )
						{
							hkAgentNnMachine_UpdateShapeCollectionFilter( agentEntry, *getCollisionInput(), *island );
						}
						continue;

						{
removeAgentLabel:

							agentsToRemove.pushBack(agentEntry);
					
							// Request split check.
							island->m_splitCheckRequested = true;
						}
					}
					HK_FOR_ALL_AGENT_ENTRIES_END;
				}

				while(agentsToRemove.getSize())
				{
					hkpAgentNnEntry* agent = agentsToRemove.back();
					agentsToRemove.popBack();
					hkpWorldAgentUtil::removeAgentAndItsToiEvents(agent);
				}
			}
		}
		unlockCriticalOperations();
	}

	if (updateMode == HK_UPDATE_FILTER_ON_WORLD_FULL_CHECK )
	{
		for ( int i = 0; i < m_phantoms.getSize(); i++ )
		{
			hkpPhantom* phantom = m_phantoms[i];
			updateCollisionFilterOnPhantom( phantom, updateShapeCollectionFilter );
		}
	}

	blockExecutingPendingOperations(false);
	attemptToExecutePendingOperations();


	HK_TIMER_END();
}

static void HK_CALL hkWorld_updateFilterOnSinglePhantom( hkpPhantom* phantom, hkpCollidable* collidable, hkpCollisionFilter* filter  )
{
	hkBool oldOverlapping = phantom->isOverlappingCollidableAdded( collidable );
	hkpCollidable* phantomCollidable = phantom->getCollidableRw();

	if( filter->isCollisionEnabled( *phantomCollidable, *collidable ) )
	{
		if ( !oldOverlapping )
		{
			phantom->addOverlappingCollidable( collidable );
		}

		if( collidable->getType() == hkpWorldObject::BROAD_PHASE_PHANTOM )
		{
			hkBool otherOldOverlapping = static_cast<hkpPhantom*>(collidable->getOwner())->isOverlappingCollidableAdded( phantomCollidable );
			if( !otherOldOverlapping )
			{
				static_cast<hkpPhantom*>(collidable->getOwner())->addOverlappingCollidable( phantomCollidable );
			}
		}
	}
	else
	{
		if ( oldOverlapping )
		{
			phantom->removeOverlappingCollidable( collidable );
		}

		if( collidable->getType() == hkpWorldObject::BROAD_PHASE_PHANTOM )
		{
			hkBool otherOldOverlapping = static_cast<hkpPhantom*>(collidable->getOwner())->isOverlappingCollidableAdded( phantomCollidable );
			if( otherOldOverlapping )
			{
				static_cast<hkpPhantom*>(collidable->getOwner())->removeOverlappingCollidable( phantomCollidable );
			}
		}
	}
}

void hkpWorld::updateCollisionFilterOnPhantom( hkpPhantom* phantom, hkpUpdateCollectionFilterMode updateShapeCollectionFilter )
{
	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::UpdateFilterOnPhantom op;
		op.m_phantom = phantom;
		op.m_updateShapeCollections = updateShapeCollectionFilter;
		queueOperation( op );
		return;
	}

	//
	// Proceed with the proper operation
	//
	HK_ASSERT2(0x6c6fbb7e,  phantom->getWorld() == this, "Trying to update a phantom that has not been added to the world");
	HK_ACCESS_CHECK_WITH_PARENT( this, HK_ACCESS_RO, phantom, HK_ACCESS_RW );

	lockCriticalOperations();
	HK_TIMER_BEGIN_LIST("UpdateFilterOnPhantom", "broadphase" );


	// Get the list of overlapping pairs and see which ones are to be removed and which are to be added
	hkpCollidable* phantomCollidable = phantom->getCollidableRw();
	hkLocalArray<hkpBroadPhaseHandlePair> pairsOut( m_broadPhaseQuerySize );
	m_broadPhase->reQuerySingleObject( phantomCollidable->getBroadPhaseHandle(), pairsOut );

	HK_TIMER_SPLIT_LIST("UpdateOverlaps");

	// Sort the pairsOut list
	for( int i = 0; i<pairsOut.getSize(); i++ )
	{
		// check for not having self overlaps
		HK_ASSERT2( 0xf043defd, pairsOut[i].m_b != phantomCollidable->getBroadPhaseHandle(), "Error in Broadphase: query object returned in query result" );
		if( pairsOut[i].m_b == phantomCollidable->getBroadPhaseHandle() )
		{
			// Ignore self overlaps
			continue;
		}
		hkpCollidable* collidable = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pairsOut[i].m_b)->getOwner() );
		hkWorld_updateFilterOnSinglePhantom( phantom, collidable, m_collisionFilter );
	}

	if ( updateShapeCollectionFilter )
	{
		HK_TIMER_SPLIT_LIST("collectionFilter");
		phantom->updateShapeCollectionFilter();
	}

	unlockAndAttemptToExecutePendingOperations();
	HK_TIMER_END_LIST();
}

void hkpWorld::reenableCollisionBetweenEntityPair( hkpEntity* entityA, hkpEntity* entityB )
{
	
	HK_ASSERT2( 0x3af8c80f, ( entityA->getCollidable()->getShape() ) && ( entityB->getCollidable()->getShape() ), "You should not enable collisions where one of the entities does not have a shape." );

	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::UpdateFilterOnEntityPair op;
		op.m_entityA = entityA;
		op.m_entityB = entityB;
		queueOperation( op );
		return;
	}

	// check the broadphase
	bool isOverlapping = m_broadPhase->areAabbsOverlapping( entityA->getCollidable()->getBroadPhaseHandle(), entityB->getCollidable()->getBroadPhaseHandle());
	if (!isOverlapping )
	{
		return;
	}

	// check the filter
	if ( !m_collisionFilter->isCollisionEnabled( *entityA->getCollidable(), *entityB->getCollidable() ) )
	{
		return;
	}

	hkpAgentNnEntry* entry = hkAgentNnMachine_FindAgent( entityA->getLinkedCollidable(), entityB->getLinkedCollidable() );
	if ( entry )
	{
		return;
	}


	hkpTypedBroadPhaseHandlePair newPair;
	newPair.m_a = const_cast<hkpTypedBroadPhaseHandle*>(entityA->getCollidable()->getBroadPhaseHandle());
	newPair.m_b = const_cast<hkpTypedBroadPhaseHandle*>(entityB->getCollidable()->getBroadPhaseHandle());

	m_broadPhaseDispatcher->addPairs( &newPair, 1, m_collisionFilter );
}


// This method should be called if you have altered the collision filtering information for this entity.
void hkpWorld::updateCollisionFilterOnEntity( hkpEntity* entity, hkpUpdateCollisionFilterOnEntityMode updateMode, hkpUpdateCollectionFilterMode updateShapeCollectionFilter )
{
	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::UpdateFilterOnEntity op;
		op.m_entity = entity;
		op.m_collisionFilterUpdateMode = updateMode;
		op.m_updateShapeCollections = updateShapeCollectionFilter;
		queueOperation( op );
		return;
	}

	//
	// Proceed with the proper operation
	//
	HK_ASSERT2(0XAD000103, entity->getWorld() == this, "Error: updatingCollisionFilter on a body not inserted into this world.");

	HK_ACCESS_CHECK_WITH_PARENT( this, HK_ACCESS_RW, entity, HK_ACCESS_RW );

	// If you don't want to lock the world than get an extra entity->addReference()
	HK_TIMER_BEGIN_LIST( "UpdateFilterOnEntity", "init");
	{
		lockCriticalOperations();

		// We force entry sorting for fixed objects.
		if (entity->isFixed())
		{
			entity->getLinkedCollidable()->sortEntries();
		}

		// Recreate a list of present broad-phase pairs
		hkInplaceArray<hkpBroadPhaseHandlePair,128> updatedPairs;

		if (updateMode == HK_UPDATE_FILTER_ON_ENTITY_FULL_CHECK)
		{
			// Get the list of overlapping pairs and see which ones are to be removed and which are to be added
			HK_TIMER_SPLIT_LIST("broadphase");

			m_broadPhase->reQuerySingleObject( entity->getCollidable()->getBroadPhaseHandle(), updatedPairs );

			//
			//	Do phantoms
			//
			{
				HK_TIMER_SPLIT_LIST("phantom");

				for (int i = 0; i < updatedPairs.getSize(); i++ )
				{
					hkpCollidable* collidable = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(updatedPairs[i].m_b)->getOwner() );
					hkpPhantom* phantom = hkpGetPhantom(collidable);

					if (phantom)
					{
						hkWorld_updateFilterOnSinglePhantom( phantom, entity->getCollidableRw(), m_collisionFilter );
						if ( updateShapeCollectionFilter )
						{
							phantom->updateShapeCollectionFilter();
						}

						// Remove entry from the list
						updatedPairs.removeAt(i--);
					}
				}
			}

			//
			// Do entities
			//

			const hkArray<struct hkpLinkedCollidable::CollisionEntry>& collisionEntries = entity->getLinkedCollidable()->getCollisionEntriesDeterministicUnchecked();

			hkInplaceArray<hkpBroadPhaseHandlePair,128> presentPairs;
			{
				for (int i = 0; i < collisionEntries.getSize(); i++)
				{
					hkpBroadPhaseHandlePair& pair = presentPairs.expandOne();
					pair.m_a = collisionEntries[i].m_agentEntry->getCollidableA()->getBroadPhaseHandle(); 
					pair.m_b = collisionEntries[i].m_agentEntry->getCollidableB()->getBroadPhaseHandle(); 
				}
			}

			// Create a to-remove and to-add lists
			hkpTypedBroadPhaseDispatcher::removeDuplicates(presentPairs, updatedPairs);

			// Add pairs executed after verifying existing agents
			HK_ASSERT(0xf0764312, presentPairs.getSize() == 0);
		}

		//
		// Verify existing collision agents
		//
		{
			HK_TIMER_SPLIT_LIST("checkAgts");
			hkpCollidableQualityType qt0 = entity->getLinkedCollidable()->getQualityType();

			hkArray<struct hkpLinkedCollidable::CollisionEntry>& collisionEntries = entity->getLinkedCollidable()->getCollisionEntriesDeterministicUnchecked();

			//info: cannot store hkpAgentNnEntries for later, cos their location changes every time an entry is removed.

			for (int i = 0; i < collisionEntries.getSize(); i++)
			{
				const hkpLinkedCollidable::CollisionEntry& entry = collisionEntries[i];

				if (!getCollisionFilter()->isCollisionEnabled( *entity->getCollidable(),  *entry.m_partner ))
				{
					goto removeAgentLabel;
				}

				// check for disabled collisions, especially landscape = landscape ones
				{
					hkpCollidableQualityType qt1 = entry.m_partner->getQualityType();
					int collisionQuality = getCollisionDispatcher()->getCollisionQualityIndex( qt0, qt1 );
					if ( collisionQuality == hkpCollisionDispatcher::COLLISION_QUALITY_INVALID )
					{
						goto removeAgentLabel;
					}
					else if(collisionQuality != entry.m_agentEntry->m_collisionQualityIndex)
					{
						hkpBroadPhaseHandlePair& pair = updatedPairs.expandOne();
						pair.m_a = collisionEntries[i].m_agentEntry->getCollidableA()->getBroadPhaseHandle();
						pair.m_b = collisionEntries[i].m_agentEntry->getCollidableB()->getBroadPhaseHandle();
						goto removeAgentLabel;
					}
				}

				// check collections
				if ( updateShapeCollectionFilter == HK_UPDATE_COLLECTION_FILTER_PROCESS_SHAPE_COLLECTIONS )
				{
					hkpEntity* entityA = entity;
					hkpEntity* entityB = static_cast<hkpEntity*>(entry.m_partner->getOwner());
					hkpSimulationIsland* island = (entityA->isFixed() )? entityB->getSimulationIsland(): entityA->getSimulationIsland();

					hkAgentNnMachine_UpdateShapeCollectionFilter( entry.m_agentEntry, *getCollisionInput(), *island );
				}
				continue;

				{
removeAgentLabel:
					//remove agent
					HK_ON_DEBUG(int oldSize = collisionEntries.getSize());
					hkpWorldAgentUtil::removeAgentAndItsToiEvents(entry.m_agentEntry);
					HK_ASSERT(0xf0ff002a, oldSize - 1 == collisionEntries.getSize());
					// the collision entries list just shrinked, so set the index to the first unchecked entry
					i--;

					// Request split check.
					entity->getSimulationIsland()->m_splitCheckRequested = true;
				}
			}
		}

		//
		// (Continuation of broadphase check) add new agents:
		//

		// only performed at FULL_BROAD_PHASE_CHECK
		if (updatedPairs.getSize() > 0)
		{
			HK_TIMER_SPLIT_LIST("addAgts");
			// filter and add pairsOut
			// this list includes entity-phantom pairs as well
			m_broadPhaseDispatcher->addPairs( static_cast<hkpTypedBroadPhaseHandlePair*>(&updatedPairs[0]), updatedPairs.getSize(), getCollisionFilter() );
		}



#		ifdef HK_ENABLE_EXTENSIVE_WORLD_CHECKING
		{
			hkpSimulationIsland* island = entity->getSimulationIsland();
			if ( !island->isFixed())
			{
				island->isValid();
			}
		}
#		endif

		unlockAndAttemptToExecutePendingOperations();
	}
	HK_TIMER_END_LIST();
}

void hkpWorld::reintegrateAndRecollideEntities( hkpEntity** entityBatch, int numEntities, ReintegrationRecollideMode mode )
{
	hkpWorld* world = this;
	if (world->areCriticalOperationsLocked())
	{
		hkWorldOperation::ReintegrateAndRecollideEntityBatch op;
		op.m_entities = const_cast<hkpEntity**>(entityBatch);
		op.m_numEntities = hkObjectIndex(numEntities);
		op.m_mode = hkUint8(mode);
		world->queueOperation( op );
		return;
	}

	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	//markForWrite();	// not allowed as we cannot ensure single threaded access here
	m_simulation->reintegrateAndRecollideEntities( entityBatch, numEntities, this, mode );
	//unmarkForWrite();
}

hkpEntity* hkpWorld::addEntity( hkpEntity* entity, enum hkpEntityActivation initialActivationState)
{
	HK_ASSERT2( 0x7f090345, entity, "You can not add a null entity to a world.");

	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::AddEntity op;
		op.m_entity = entity;
		op.m_activation = initialActivationState;
		queueOperation( op );
		return HK_NULL;
	}
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	HK_INTERNAL_TIMER_BEGIN_LIST("AddEntity", "Island");

	//
	// Proceed with the proper operation
	//

	// history: m_simulationState = CAN_NOT_REMOVE_ENTITIES_AND_CONSTRAINTS;

	HK_ASSERT2(0x3f9eb209,  entity->getWorld() == HK_NULL, "You are trying to add an entity to a world, which has already been added to a world" );
	HK_ASSERT2(0xf0ff0030, entity->m_actions.isEmpty(), "Entity to add has already actions attached, this is wrong");
	

	const hkpShape* shape = entity->m_collidable.getShape();

	entity->setCachedShapeData(this, shape);

	// check if the collidable back ptr to the motion state is set
	// as it may not be due to packedfile serialization
	if (!entity->m_collidable.getMotionState())
	{
		hkMotionState*	motionState = entity->getMotionState();
		entity->m_collidable.setMotionState( motionState );
	}
	// The motions state as added may be 4 dimensional (have time, which is probably
	// nothing in relation to this world time)
	// so we make sure to set the invDeltaTime to 0 on it which
	// makes it just a 3D placement (it will set itself up with a
	// time quantum upon next step).
	hkMotionState* ms = static_cast<hkpRigidBody*>( entity )->getRigidMotion()->getMotionState();
	hkSimdReal zero; zero.setZero();
	hkSweptTransformUtil::setTimeInformation(zero, zero, *ms);
	entity->m_motion.m_deactivationNumInactiveFrames[0] = 0;
	entity->m_motion.m_deactivationNumInactiveFrames[1] = 0;

	// Simulation Island
	allowCriticalOperations(false);
	{
		// Assign world-unique id
		entity->m_uid = ++m_lastEntityUid;
		// Add a reference to the entity
		entity->addReference();
		// add island
		hkpWorldOperationUtil::addEntitySI( this, entity, initialActivationState );

		hkUint8* deactFlags = m_dynamicsStepInfo.m_solverInfo.m_deactivationNumInactiveFramesSelectFlag;
		entity->getMotion()->setWorldSelectFlagsNeg(deactFlags[0], deactFlags[1], m_dynamicsStepInfo.m_solverInfo.m_deactivationIntegrateCounter);
	}
	allowCriticalOperations(true);

#if defined(HK_PLATFORM_HAS_SPU)
	entity->getCollidableRw()->setShapeSizeForSpu();  // needs to be called after world is set on entity
#endif

	lockCriticalOperations();
	HK_INTERNAL_TIMER_SPLIT_LIST("Broadphase");
	{
		// Add the entity to BroadPhase
		hkpWorldOperationUtil::addEntityBP( this, entity );
	}

	// Run callbacks before other pending operations as callbacks fire internal operations which ensure proper state of the world
	//if ( DO_FIRE_CALLBACKS == fireCallbacks )
	HK_INTERNAL_TIMER_SPLIT_LIST("Callbacks");
	{
		// Fire the callbacks
		// notice: order
		hkpWorldCallbackUtil::fireEntityAdded( this, entity );
		hkpEntityCallbackUtil::fireEntityAdded( entity );
	}

	unlockCriticalOperations();

	{
		attemptToExecutePendingOperations();
	}

#ifdef HK_ENABLE_EXTENSIVE_WORLD_CHECKING
	HK_INTERNAL_TIMER_SPLIT_LIST("Validate");
	{
		hkpSimulationIsland* island = entity->getSimulationIsland();

		{
			if ( island ) { island->isValid(); }
		}

	}
#endif

	HK_INTERNAL_TIMER_END_LIST();

	if (HK_NUM_EXTENDED_USER_DATAS_IN_TOI_EVENT / 2 < entity->m_numShapeKeysInContactPointProperties)
	{
		HK_WARN(0xac8755aa, "A body request more extended user datas, than half of what can be stored for ToiEvents for both bodies." );
	}

	return entity;
}


void hkpWorld::addEntityBatch( hkpEntity*const* entityBatch, int numEntities, hkpEntityActivation initialActivationState )
{
	if( numEntities <= 0 )
	{
		return;
	}

	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::AddEntityBatch op;
		op.m_entities = const_cast<hkpEntity**>(entityBatch);
		HK_ASSERT(0xf0ff0040, numEntities < HK_INVALID_OBJECT_INDEX );
		op.m_numEntities = static_cast<hkObjectIndex>(numEntities);
		op.m_activation = initialActivationState;
		queueOperation( op );
		return;
	}

	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	HK_TIMER_BEGIN_LIST("AddEntities", "init")

		//SimulationState savedState = static_cast<SimulationState>(m_simulationState);
		//m_simulationState = CAN_NOT_REMOVE_ENTITIES_AND_CONSTRAINTS;

		lockCriticalOperations();

	hkArray< hkpBroadPhaseHandle* >::Temp collList;
	collList.reserveExactly( numEntities );
	hkArray< hkAabb > aabbList;
	aabbList.reserveExactly( numEntities );

	hkpSimulationIsland* gabriolaIsland;

	// create an island
	bool usedGabriola = false;
	if (m_wantSimulationIslands)
	{
		gabriolaIsland = new hkpSimulationIsland(this);
		gabriolaIsland->m_splitCheckRequested = true;

		if (initialActivationState == HK_ENTITY_ACTIVATION_DO_ACTIVATE)
		{
			gabriolaIsland->m_storageIndex = (hkObjectIndex)getActiveSimulationIslands().getSize();
			gabriolaIsland->m_isInActiveIslandsArray = true;
			gabriolaIsland->m_activeMark = true;
			// will be added to active list if used.
		}
		else
		{
			gabriolaIsland->m_storageIndex = (hkObjectIndex)m_inactiveSimulationIslands.getSize();
			gabriolaIsland->m_isInActiveIslandsArray = false;
			gabriolaIsland->m_activeMark = false;
			// will be added to inactive list if used.
		}
	}
	else
	{
		gabriolaIsland = getActiveSimulationIslands()[0];
		gabriolaIsland->m_entities.reserve( gabriolaIsland->m_entities.getSize() + numEntities );
	}

	{
		hkReal extraRadius = getCollisionInput()->getTolerance() * .5f;
		hkSimdReal zero; zero.setZero();
		for( int i = 0; i < numEntities; i++ )
		{
			hkpEntity* entity = entityBatch[i];

			HK_ASSERT2( 0xad5fbd63, entity, "You can not batch with a null entity to a world.");
			HK_ASSERT2( 0xad5fbd64, entity->getWorld() == HK_NULL, "You are trying to add an entity, which already belongs to an hkpWorld.");

			hkpCollidable* collidable = entity->getCollidableRw();
			const hkpShape* shape = collidable->getShape();

			// Assign world-unique id
			entity->m_uid = ++m_lastEntityUid;

			entity->addReference();
			if (!entity->getCollidable()->getMotionState()) // may be null due to packfile serialize
			{
				entity->m_collidable.setMotionState( entity->getMotionState() );
			}
			hkMotionState* ms = static_cast<hkpRigidBody*>( entity )->getRigidMotion()->getMotionState();
			hkSweptTransformUtil::setTimeInformation(zero, zero, *ms); // set time to 0 with no invdelta (so not swept)
			entity->m_motion.m_deactivationNumInactiveFrames[0] = 0;
			entity->m_motion.m_deactivationNumInactiveFrames[1] = 0;

			entity->setWorld( this );

			hkUint8* deactFlags = m_dynamicsStepInfo.m_solverInfo.m_deactivationNumInactiveFramesSelectFlag;
			entity->getMotion()->setWorldSelectFlagsNeg(deactFlags[0], deactFlags[1], m_dynamicsStepInfo.m_solverInfo.m_deactivationIntegrateCounter);

			if ( entity->isFixed() )
			{
				m_fixedIsland->internalAddEntity( entity );
			}
			else
			{
				usedGabriola = true;
				gabriolaIsland->internalAddEntity(entity);
			}			

			collidable->m_boundingVolumeData.invalidate();

			if (shape)
			{
				// add the shape to the broadphase and merge islands as necessary
				hkAabb& aabb = *aabbList.expandByUnchecked(1);
				collidable->getShape()->getAabb( collidable->getTransform(), extraRadius, aabb );
				collList.pushBackUnchecked( collidable->getBroadPhaseHandle() );

				entity->setCachedShapeData(this, shape);

#if defined(HK_PLATFORM_HAS_SPU)
				entity->getCollidableRw()->setShapeSizeForSpu();
#endif
			}
		}
	}

	if (m_wantSimulationIslands)
	{
		if (usedGabriola)
		{
			HK_ON_DETERMINISM_CHECKS_ENABLED( gabriolaIsland->m_uTag = gabriolaIsland->m_entities[0]->m_uid );
			gabriolaIsland->m_splitCheckFrameCounter = hkUchar(gabriolaIsland->m_entities[0]->m_uid);

			hkArray<hkpSimulationIsland*>& islandArray = initialActivationState == HK_ENTITY_ACTIVATION_DO_ACTIVATE
				? const_cast<hkArray<hkpSimulationIsland*>&>(getActiveSimulationIslands())
				: m_inactiveSimulationIslands;
			islandArray.pushBack(gabriolaIsland);
		}
		else
		{
			delete gabriolaIsland;
		}
	}

	hkLocalArray< hkpBroadPhaseHandlePair > pairsOut( m_broadPhaseQuerySize );

	HK_TIMER_SPLIT_LIST("Broadphase");

	m_broadPhase->addObjectBatch( collList, aabbList, pairsOut );

	HK_TIMER_SPLIT_LIST("CreateAgents");

	m_broadPhaseDispatcher->addPairs( static_cast<hkpTypedBroadPhaseHandlePair*>(pairsOut.begin()), pairsOut.getSize(), getCollisionFilter() );

	HK_TIMER_SPLIT_LIST("AddedCb");

	{
		for( int i = 0; i < numEntities; i++ )
		{
			hkpEntity* entity = entityBatch[i];
			hkpWorldCallbackUtil::fireEntityAdded( this, entity );
			hkpEntityCallbackUtil::fireEntityAdded( entity );
		}
	}

	unlockAndAttemptToExecutePendingOperations();

	HK_TIMER_END_LIST();
}


hkBool hkpWorld::removeEntity( hkpEntity* entity )
{
	HK_ASSERT(0x72576e5f, entity);

	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::RemoveEntity op;
		op.m_entity = entity;
		queueOperation( op );
		return false;
	}

	HK_ASSERT2(0x72576e5e,  entity->getWorld() == this, "You are trying to remove an entity from a world to which it is not added");

	//
	// Proceed with the proper operation
	//

	lockCriticalOperations();

	// Update the BroadPhase
	HK_INTERNAL_TIMER_BEGIN_LIST("RemEntity", "Broadphase");
	hkpWorldOperationUtil::removeEntityBP( this, entity );
	HK_ASSERT(0xad000095, 0 == entity->getLinkedCollidable()->getCollisionEntriesNonDeterministic().getSize());

	// Fire the callbacks
	HK_INTERNAL_TIMER_SPLIT_LIST("Callbacks");
	// World callbacks are called first (to allow access to the entity's constraints and actions (as they're removed in the entity callback)
	hkpWorldCallbackUtil::fireEntityRemoved( this, entity );
	hkpEntityCallbackUtil::fireEntityRemoved( entity );

	// when should callbacks be called ? with all agents + constraints + etc. in place ?

	HK_ASSERT(0xad000210, entity->m_actions.isEmpty());
	HK_ASSERT(0xad000211, entity->m_constraintsMaster.isEmpty());
	HK_ASSERT(0xad000212, entity->m_constraintsSlave.isEmpty());

#	if defined HK_ENABLE_EXTENSIVE_WORLD_CHECKING
	hkpSimulationIsland* island = (entity->getSimulationIsland() && entity->getSimulationIsland()->m_entities.getSize() == 1) ? HK_NULL : entity->getSimulationIsland();
#	endif

	// do it here as you also need to allow for removal of constraints by the callbacks
	allowCriticalOperations(false);
	{
		HK_INTERNAL_TIMER_SPLIT_LIST("Island");
		HK_ASSERT(0xf0ff0041, entity->getWorld());

		hkpWorldOperationUtil::removeEntitySI( this, entity );

		if ( entity->m_collidable.m_boundingVolumeData.hasAllocations() )
		{
			entity->m_collidable.m_boundingVolumeData.deallocate();
		}

		// If the entity has been loaded from a packfile try and deallocate any internal zero size arrays in the entity.
		// If the arrays have a non-zero size the user will be warned.
		if (entity->m_memSizeAndFlags == 0)
		{
			entity->deallocateInternalArrays();
		}
		entity->removeReference();
	}
	allowCriticalOperations(true);

#	ifdef HK_ENABLE_EXTENSIVE_WORLD_CHECKING
	{
		HK_INTERNAL_TIMER_SPLIT_LIST("Validate");
		if ( island ) { island->isValid(); }
	}
#	endif

	unlockAndAttemptToExecutePendingOperations();

	HK_INTERNAL_TIMER_END_LIST();
	return true;
}




void hkpWorld::removeEntityBatch( hkpEntity*const* entityBatch, int numEntities )
{
	if( numEntities <= 0 )
	{
		return;
	}

	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::RemoveEntityBatch op;
		op.m_entities = const_cast<hkpEntity**>(entityBatch);
		HK_ASSERT(0xf0ff0043, numEntities < HK_INVALID_OBJECT_INDEX);
		op.m_numEntities = (hkObjectIndex)numEntities;
		queueOperation( op );
		return;
	}

	lockCriticalOperations();

	HK_TIMER_BEGIN_LIST("RemEntities", "Init+CallBck");

	// Remove all TOI contact points before calling entity-removed callbacks
	m_simulation->resetCollisionInformationForEntities(const_cast<hkpEntity**>(entityBatch), numEntities, this, hkpSimulation::RESET_TOI);

	// Remove collision agents via broadphase
	{
		hkArray<hkpBroadPhaseHandle*>::Temp collList;
		collList.reserveExactly(numEntities);
		{
			hkpEntity*const* entity = entityBatch;
			hkpEntity*const* entityEnd = entityBatch + numEntities;

			while( entity != entityEnd )
			{
				HK_ASSERT2(0xadb7d62a, *entity, "An HK_NULL found in the entity list for hkpWorld::removeEntityBatch");
				HK_ASSERT2(0xadb7d62b, (*entity)->getWorld() == this, "Trying to remove an entity which does not belong to this hkpWorld.");

				hkpCollidable* c = (*entity)->getCollidableRw();
				if ( c->getShape() != HK_NULL )
				{
					collList.pushBackUnchecked( c->getBroadPhaseHandle() );
				}
				entity++;
			}
		}
		if ( collList.getSize() )
		{
			HK_TIMER_SPLIT_LIST("BroadPhase");
			hkLocalArray< hkpBroadPhaseHandlePair > pairsOut( m_broadPhaseQuerySize );

			m_broadPhase->removeObjectBatch( collList, pairsOut );

			HK_TIMER_SPLIT_LIST("DelAgents");

			m_broadPhaseDispatcher->removePairs( static_cast<hkpTypedBroadPhaseHandlePair*>(pairsOut.begin()), pairsOut.getSize() );
		}
	}

	HK_TIMER_SPLIT_LIST("RemoveCb");
	{
		hkpEntity*const* entity = entityBatch;
		hkpEntity*const* entityEnd = entityBatch + numEntities;

		while( entity != entityEnd )
		{
			// World callbacks are called first (to allow access to the entity's constraints and actions (as they're removed in the entity callback)
			hkpWorldCallbackUtil::fireEntityRemoved( this, *entity );
			hkpEntityCallbackUtil::fireEntityRemoved( *entity );

			hkpWorldOperationUtil::removeEntitySI(this, *entity);

			if ( (*entity)->m_collidable.m_boundingVolumeData.hasAllocations() )
			{
				(*entity)->m_collidable.m_boundingVolumeData.deallocate();
			}

			// If the entity has been loaded from a packfile try and deallocate any internal zero size arrays in the entity.
			// If the arrays have a non-zero size the user will be warned.
			if ( (*entity)->m_memSizeAndFlags == 0 )
			{
				(*entity)->deallocateInternalArrays();
			}
			(*entity)->removeReference();
			entity++;
		}
	}
	HK_TIMER_END_LIST();

	unlockAndAttemptToExecutePendingOperations();
}


void hkpWorld::activateRegion( const hkAabb& aabb )
{
	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::ActivateRegion op;
		op.m_aabb = hkAllocateChunk<hkAabb>(1, HK_MEMORY_CLASS_DYNAMICS);
		hkString::memCpy(op.m_aabb, &aabb, sizeof(hkAabb));
		queueOperation( op );
		return;
	}
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	hkArray<hkpBroadPhaseHandlePair> pairs;
	m_broadPhase->querySingleAabb( aabb, pairs );
	for (int i = 0; i < pairs.getSize(); i++)
	{
		HK_ASSERT2(0xf0ff0098, pairs[i].m_a == HK_NULL, "Internal check.");
		hkpCollidable* coll = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pairs[i].m_b)->getOwner() );
		hkpRigidBody*  body = hkpGetRigidBody(coll);
		if (body)
		{
			body->activate();
		}
	}
}

#if defined HK_DEBUG
static void HK_CALL warnIfConstrainedDynamicBodiesCanCollide( const hkpConstraintInstance* constraint,
															 hkBool bodiesCollisionEnabledBeforeConstraintAdded,
															 hkBool bodiesCollisionEnabledAfterConstraintAdded )
{
	const hkpRigidBody* rigidBodyA = constraint->getRigidBodyA();
	const hkpRigidBody* rigidBodyB = constraint->getRigidBodyB();

	const hkpCollidable* collidableA = rigidBodyA->getCollidable();
	const hkpCollidable* collidableB = rigidBodyB->getCollidable();

	// Some filters (e.g. hkpConstraintCollisionFilter) can remove any agents that were already in place.
	// If that's the case, we don't need to warn.
	bool agentExists = (hkAgentNnMachine_FindAgent( rigidBodyA->getLinkedCollidable(), rigidBodyB->getLinkedCollidable() ) != HK_NULL);

	if ( bodiesCollisionEnabledBeforeConstraintAdded && agentExists )
	{
		// Current HK_WARN max string length is 512 characters. Keep messages a little shorter to be sure they fit.
		char warnInfoString[510];

		// Print body and constraint names if they are available.
		if ( rigidBodyA->getName() && rigidBodyB->getName() && constraint->getName() )
		{
			hkString::snprintf( warnInfoString,
				509,
				"Colliding body and constraint info; hkpRigidBody A name:'%s', pointer:0x%p, filter info:%d. hkpRigidBody B name:'%s', pointer:0x%p, filter info:%d. Constraint name:'%s', pointer:0x%p.",
				rigidBodyA->getName(),
				rigidBodyA,
				collidableA->getCollisionFilterInfo(),
				rigidBodyB->getName(),
				rigidBodyB,
				collidableB->getCollisionFilterInfo(),
				constraint->getName(),
				constraint );
		}
		else
		{
			hkString::snprintf( warnInfoString,
				509,
				"Colliding body and constraint info; hkpRigidBody A pointer:0x%p, filter info:%d. hkpRigidBody B pointer:0x%p, filter info:%d. Constraint pointer:0x%p.",
				rigidBodyA,
				collidableA->getCollisionFilterInfo(),
				rigidBodyB,
				collidableB->getCollisionFilterInfo(),
				constraint );
		}

		//
		// Warnings ID pairs match to allow users to disable irrelevant warnings only.
		// Have 2 warning outputs as there is no way to know how long user hkpRigidBody and hkConstraint names will be...
		//
		if ( bodiesCollisionEnabledAfterConstraintAdded )
		{
			HK_WARN( 0x2a1db936, "Constraint added between two *colliding* dynamic rigid bodies. Check your collision filter logic and setup. Collision between constrained bodies typically leads to unintended artifacts e.g. adjacent, constrained ragdoll limbs colliding leading to 'ragdoll jitter'." );
			HK_WARN( 0x2a1db936, warnInfoString );
		}
		else
		{
			HK_WARN( 0x68c4e1dc, "Constraint added between two *colliding* dynamic rigid bodies. The bodies will collide with one another unless one of the functions hkpWorld::updateCollisionFilter...() is called. Collision between constrained bodies typically leads to unintended artifacts e.g. adjacent, constrained ragdoll limbs colliding leading to 'ragdoll jitter'." );
			HK_WARN( 0x68c4e1dc, warnInfoString );
		}
	}
}

static bool HK_CALL constrainedDynamicBodiesCanCollide( const hkpWorld* world, const hkpConstraintInstance* constraint )
{
	const hkpRigidBody* rigidBodyA = constraint->getRigidBodyA();
	const hkpRigidBody* rigidBodyB = constraint->getRigidBodyB();

	//
	// Only check for collisions between dynamic bodies.
	//
	if ( rigidBodyA && rigidBodyB && (! rigidBodyA->isFixedOrKeyframed() ) && (! rigidBodyB->isFixedOrKeyframed() ) && rigidBodyA->getSimulationIsland() == rigidBodyB->getSimulationIsland() )
	{
		const hkpLinkedCollidable* collidableA = rigidBodyA->getLinkedCollidable();
		const hkpLinkedCollidable* collidableB = rigidBodyB->getLinkedCollidable();

		//
		// Check if the hkpWorld already has an agent for those bodies
		//
		if ( hkAgentNnMachine_FindAgent(collidableA, collidableB) )
		{
			// Use a hkpClosestCdPointCollector class to gather the results of our query.
			hkpClosestCdPointCollector collector;

			// Get the shape type of each shape (this is used to figure out the most appropriate
			// getClosestPoints(...) method to use).
			const hkpShapeType shapeTypeA = collidableA->getShape()->getType();
			const hkpShapeType shapeTypeB = collidableB->getShape()->getType();

			hkpCollisionInput input = *world->getCollisionInput();

			// Ask the collision dispatcher to locate a suitable getClosestPoints(...) method.
			hkpCollisionDispatcher::GetClosestPointsFunc getClosestPointsFunc = world->getCollisionDispatcher()->getGetClosestPointsFunc( shapeTypeA, shapeTypeB );
			getClosestPointsFunc( *collidableA, *collidableB, input, collector );

			return collector.hasHit();		
		}
	}

	return false;
}


#endif

hkpConstraintInstance* hkpWorld::addConstraint( hkpConstraintInstance* constraint )
{
	HK_ASSERT3(0x3c77f996, constraint->getData()->isValid(), "ConstraintInstance " << constraint <<  " with name: " << constraint->getName() << " is invalid!" );
	HK_ASSERT2(0xad675544, (constraint->getData()->getType() < hkpConstraintData::BEGIN_CONSTRAINT_CHAIN_TYPES) ^ (constraint->getType() == hkpConstraintInstance::TYPE_CHAIN), "You're adding an inconsistent constraint which uses hkpConstraintChainData+hkpConstraintInstance or vice versa.");

	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::AddConstraint op;
		op.m_constraint = constraint;
		queueOperation( op );
		return HK_NULL;
	}
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	// Proceed whit the proper operation
	HK_ASSERT2(0xad000103, constraint->getOwner() == HK_NULL, "Error: you are trying to add a constraint, that has already been added to some world");

	// This is to allow loading of pre-3.3 assets.
	//
	// In 3.3 we have two problematic members of the hkpBreakableConstraintData: m_childRuntimeSize and m_childNumSolverResults.
	//
	// Initialization of those members depends on a call to a virtual method of a different object,
	// and we cannot do that safely in our current serialization framework neither at the time
	// of converting assets nor in the finish-up constructor.
	//
	// Therefore we're doing that here.
	//
	if (constraint->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE)
	{
		hkpBreakableConstraintData* data = static_cast<hkpBreakableConstraintData*>(constraint->getDataRw());

		if (data->m_childRuntimeSize == 0)
		{
			hkpConstraintData::RuntimeInfo info;
			data->m_constraintData->getRuntimeInfo( true, info );
			data->m_childRuntimeSize      = hkUint16(info.m_sizeOfExternalRuntime);
			data->m_childNumSolverResults = hkUint16(info.m_numSolverResults);
		}

	}

	// Check that TOI-priority constraints don't connect to simplified-TOI objects
	// Note: this is actually allowed
#if 0 && defined(HK_DEBUG)	
	if (constraint->m_priority >= hkpConstraintInstance::PRIORITY_TOI)
	{
		if (constraint->getEntityA())
		{
			HK_ASSERT2(0xad810214, constraint->getEntityA()->getCollidable()->getQualityType() != HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI, "An entity of quality type HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI is linked to a constraint with priority type PRIOITY_TOI or higher. This is likely to cause penetrations and tunnelling for collisions of simplified-toi quality. This is illegal. Adjust your constraint priorities or entity priorities.");
		}
		if (constraint->getEntityB())
		{
			HK_ASSERT2(0xad810214, constraint->getEntityB()->getCollidable()->getQualityType() != HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI, "An entity of quality type HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI is linked to a constraint with priority type PRIOITY_TOI or higher. This is likely to cause penetrations and tunnelling for collisions of simplified-toi quality. This is illegal. Adjust your constraint priorities or entity priorities.");
		}
	}
#	endif // HK_DEBUG


	hkpConstraintInstance* result;

#if defined HK_DEBUG
	// We have to lock critical operations now in the unlikely event that the collision filter checks
	//	from constrainedDynamicBodiesCanCollide try to remove one of the constraint bodies
	lockCriticalOperations();

	// Check if bodies attached to constraint might collide (arising in unwanted artifacts later).
	hkBool bodiesCollisionEnabledBeforeConstraintAdded = constrainedDynamicBodiesCanCollide( this, constraint );

	// Unlock now, but don't process them events until after the constraint is added.
	unlockCriticalOperations();

#endif // #if defined HK_DEBUG

	blockExecutingPendingOperations(true);
	{
		// info: locking done in the hkpWorldOperationUtil function
		constraint->pointNullsToFixedRigidBody();
		constraint->m_uid = ++m_lastConstraintUid;
		result = hkpWorldOperationUtil::addConstraintImmediately(this, constraint);

		if ( constraint->getType() == hkpConstraintInstance::TYPE_CHAIN )
		{
			// Adding constraint chain's action
			hkpConstraintChainInstance* chain = static_cast<hkpConstraintChainInstance*>(constraint);
			// if the constraint chain instance is part of a physics system, than the action
			// might also be part of that system and already be added to the world. So
			// we have to check whether the action is added already
			if (chain->m_action->getWorld() == HK_NULL)
			{
				addAction(chain->m_action);
			}


			HK_ASSERT2(0xad7877dd, chain->m_chainedEntities.getSize() - 1 <= chain->getData()->getNumConstraintInfos(), "hkpConstraintChainInstance requires more constraintInfos than it has in its hkpConstraintChainData (it has too many hkEntities).");
			HK_ASSERT2(0xad7877de, chain->m_chainedEntities.getSize() >= 2, "hkpConstraintChainInstance has less than 2 chained bodies.");
			if (chain->m_chainedEntities.getSize() - 1 < chain->getData()->getNumConstraintInfos())
			{
				HK_WARN(0xad7877de, "hkpConstraintChainInstance does not use all ConstraintInfos supplied in its hkConstralintChainData.");
			}
		}
	}
	blockExecutingPendingOperations(false);
	attemptToExecutePendingOperations( false );	// no body operations fired here

#if defined HK_DEBUG
	// See notes on bodiesCollisionEnabledBeforeConstraintAdded above
	lockCriticalOperations();

	// Check if bodies attached to constraint will collide (arising in unwanted artifacts later).
	hkBool bodiesCollisionEnabledAfterConstraintAdded = constrainedDynamicBodiesCanCollide( this, constraint );
	warnIfConstrainedDynamicBodiesCanCollide( constraint, bodiesCollisionEnabledBeforeConstraintAdded, bodiesCollisionEnabledAfterConstraintAdded );

	unlockAndAttemptToExecutePendingOperations();
#endif // #if defined HK_DEBUG

	return result;
}


hkpConstraintInstance* hkpWorld::createAndAddConstraintInstance( hkpRigidBody* bodyA, hkpRigidBody* bodyB, hkpConstraintData* constraintData)
{
	hkpConstraintInstance* constraint;

	constraint = new hkpConstraintInstance( bodyA, bodyB, constraintData, hkpConstraintInstance::PRIORITY_PSI );

	constraint->setUserData( constraintData->m_userData );
	this->addConstraint( constraint );
	return constraint;
}


hkBool hkpWorld::removeConstraint( hkpConstraintInstance* constraint)
{
	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::RemoveConstraint op;
		op.m_constraint = constraint;
		queueOperation( op );
		return false;
	}

	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	// Proceed with the proper operation
	HK_ASSERT2(0x6c6f226b, constraint->getOwner() && static_cast<hkpSimulationIsland*>(constraint->getOwner())->getWorld() == this, "Trying to remove a constraint, that has not been added to the world or was added to a different world.");
	HK_ASSERT2(0Xad000114, constraint->getData()->getType() != hkpConstraintData::CONSTRAINT_TYPE_CONTACT, "Error: trying to remove a constactConstraint which is owned and managed by agents only");

	lockCriticalOperations();
	{
		if ( constraint->getType() == hkpConstraintInstance::TYPE_CHAIN )
		{
			// Adding constraint chain's action
			hkpConstraintChainInstance* chain = static_cast<hkpConstraintChainInstance*>(constraint);
			if (chain->m_action->getWorld() == this)
			{
				removeActionImmediately(chain->m_action);
			}
		}

		constraint->addReference();
		// info: locking done in the hkpWorldOperationUtil function
		hkpWorldOperationUtil::removeConstraintImmediately(this, constraint);

		// If the constraint is to the special "fixed" rigid body, we need to take care of it.
		constraint->setFixedRigidBodyPointersToZero( this );

		constraint->removeReference();
	}
	unlockAndAttemptToExecutePendingOperations();

	return true;
}

hkpAction* hkpWorld::addAction( hkpAction* action )
{
	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::AddAction op;
		op.m_action = action;
		queueOperation( op );
		return HK_NULL;
	}
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	HK_ASSERT2(0XAD000108, action->getWorld() == HK_NULL, "Error: trying to add an action, that already has been added to some world");

	// Proceed whit the proper operation
	action->addReference();

	lockCriticalOperations();

	hkInplaceArray< hkpEntity*, 4 > entities;
	action->getEntities( entities );
	action->setWorld( this );

	hkpEntity* firstMovableEntity = HK_NULL;

	for (int i = 0; i < entities.getSize(); ++i)
	{
		HK_ASSERT2(0x3a26883f,  entities[i]->m_world == this, "Error: You tried to add an action which depends on Entities which are not added to the physics" );

		entities[i]->m_actions.pushBack( action );

		hkpSimulationIsland* island = entities[i]->getSimulationIsland();
		if ( !island->isFixed() )
		{
			if ( firstMovableEntity == HK_NULL )
			{
				firstMovableEntity = entities[i];
				island->addAction(action);
			}
			else
			{
				// check to see if islands need to be merged
				if ( firstMovableEntity->getSimulationIsland() != entities[i]->getSimulationIsland() )
				{
					hkpWorldOperationUtil::mergeIslands(this, firstMovableEntity, entities[i]);
				}
			}
		}
	}

	// When all entities are fixed, add the action to the fixed island
	if (firstMovableEntity == HK_NULL)
	{
		HK_ASSERT2(0xad34fe33, entities.getSize(), "You tried to add an action which has no entities specified.");
		entities[0]->getSimulationIsland()->addAction(action);
	}

	// Run pending operations before firing callbacks to make sure that all merge-island-requests are processed before other operations.
	unlockAndAttemptToExecutePendingOperations();

	hkpWorldCallbackUtil::fireActionAdded( this, action );

	// This action might have already been removed in the above callback. Still though we assume the user has keeps a reference to
	// it outside of this call, so we can safely return the pointer.

	return action;
}



void hkpWorld::removeAction( hkpAction* action )
{
	// Check if operation may be performed now
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::RemoveAction op;
		op.m_action = action;
		queueOperation( op );
		return;
	}
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	HK_ASSERT2(0xad000107, action->getWorld() == this, "Error: removing an action that already has been removed from the world (note: it may still be hanging on the actionList of some entity, as it is only removed form it in actionRemovedCallbacks. And those callbacks are not ordered.)");

	// Proceed with the proper operation
	removeActionImmediately(action);
}

void hkpWorld::removeActionImmediately( hkpAction* action )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0xad000600, this == action->getWorld(), "Removing an action which is not in this world.");
	// Add temporary reference
	action->addReference();

	lockCriticalOperations();
	// Fire callback while the action is still in the this
	hkpWorldCallbackUtil::fireActionRemoved( this, action );

	//  TODO clear comments:
	//  Not needed with locking:
	//	The action might already have been removed from withing the callbacks, therefore proceed only if it still is in the this.
	//	if (action->getWorld() == this)
	{
		hkInplaceArray< hkpEntity*, 4 > entities;
		action->getEntities( entities );
		for (int i = 0; i < entities.getSize(); ++i)
		{
			HK_ASSERT2(0xad000220, entities[i]->getWorld() == this, "Error: action being removed is attached to an entity not insterted into its world.");
			//detachActionFromEntity(action, entities[i]);
			int idx = entities[i]->m_actions.indexOf(action);
			HK_ASSERT2(0xad000240, idx >= 0, "You tried to remove an action that was never added while removing an action" );
			entities[i]->m_actions.removeAt(idx);
		}

		hkpSimulationIsland* island = action->getSimulationIsland();
		island->removeAction( action );
		action->setWorld( HK_NULL );
		action->removeReference();

		hkpWorldOperationUtil::putIslandOnDirtyList(island->getWorld(), island);
	}

	unlockAndAttemptToExecutePendingOperations();

	// Remove temporary reference
	action->removeReference();
}

void hkpWorld::attachActionToEntity(hkpAction* action, hkpEntity* entity)
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0xad000230, action->getWorld(), "Error: this function is only meant to be called from action, and only when they are inserted into the world");
	//if (action->getWorld())
	{
		HK_ASSERT2(0xad000221, entity->getWorld() == this, "Error: attaching an entity not inserted into the world");
		HK_ASSERT2(0xad000222, entity->m_actions.indexOf(action) < 0 , "Error: You tried to add the same action twice");
		entity->m_actions.pushBack( action );

		if (action->getSimulationIsland()->isFixed() && !entity->isFixed())
		{
			action->getSimulationIsland()->removeAction(action);
			entity->getSimulationIsland()->addAction(action);
		}
		else if ( entity->getSimulationIsland() != action->getSimulationIsland() && !entity->isFixed() )
		{
			// HACK: taking an arbitrary entity form action's island -- if it get's deleted before the merge is performed, the
			//       islands may not get merged.
			hkpWorldOperationUtil::mergeIslands(this, entity, action->getSimulationIsland()->m_entities[0]);
		}

	}

}

void hkpWorld::detachActionFromEntity(hkpAction* action, hkpEntity* entity)
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0xad000230, action->getWorld(), "Error: this function is only meant to be called from action, and only when they are inserted into the world");
	//if (action->getWorld())
	{
		HK_ASSERT2(0xad000223, entity->getWorld() == this, "Error: detaching from entity not inserted to the world");

		const int idx = entity->m_actions.indexOf( action );
		HK_ASSERT2(0x3ef53a57, idx >= 0, "You tried to remove an action that was never added" );
		entity->m_actions.removeAt(idx);

		entity->getSimulationIsland()->m_splitCheckRequested = true;

		//
		// And now find a valid island for the action
		//
		hkInplaceArray< hkpEntity*, 4 > entities;
		action->getEntities( entities );
		HK_ASSERT2( 0x46fe7c1a, entities.indexOf( entity ) >= 0, "Error: detaching from entity not attached to the action" );
		HK_ASSERT2( 0x69e11690, entities.getSize() > 1, "Cannot have an action with no entities in the world" );

		hkpSimulationIsland* newIsland = HK_NULL;
		for (int i = 0; i < entities.getSize(); ++i)
		{
			if (entities[i] != entity)
			{
				newIsland = entities[i]->getSimulationIsland();
				if (!newIsland->isFixed())
				{
					break;
				}
			}
		}

		if (newIsland != action->getSimulationIsland())
		{
			action->getSimulationIsland()->removeAction(action);
			newIsland->addAction(action);
		}
	} 
}

hkpPhantom* hkpWorld::addPhantom( hkpPhantom* phantom )
{
	HK_ASSERT2(0x13c74a8e,  phantom, "Cannot pass an HK_NULL as parameter to hkpWorld::addPhantom");

	// Check if operation may be performed now
	if (areCriticalOperationsLockedForPhantoms())
	{
		hkWorldOperation::AddPhantom op;
		op.m_phantom = phantom;
		queueOperation( op );
		return HK_NULL;
	}
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	//
	// Proceed whit the proper operation
	//

	HK_ASSERT2(0x13c74a8e,  phantom->getWorld() == HK_NULL, "Trying to add a phantom to a hkpWorld twice");

	lockCriticalOperations();

	// check if the collidable back ptr to the motion state is set
	// as it may not be due to packedfile serialization
	if (!phantom->m_collidable.getMotionState())
	{
		phantom->m_collidable.setMotionState( phantom->getMotionState() );
	}

	phantom->setWorld( this );

#if defined(HK_PLATFORM_HAS_SPU)
	phantom->getCollidableRw()->setShapeSizeForSpu();
#endif
	phantom->addReference();
	m_phantoms.pushBack( phantom );

	hkpWorldOperationUtil::addPhantomBP(this, phantom);

	//disable + execute here ?

	hkpWorldCallbackUtil::firePhantomAdded( this, phantom );
	phantom->firePhantomAdded();

	unlockAndAttemptToExecutePendingOperations();

	return phantom;
}



void hkpWorld::addPhantomBatch( hkpPhantom*const* phantomBatch, int numPhantoms )
{
	if( numPhantoms <= 0 )
	{
		return;
	}
	// Check if operation may be performed now
	if (areCriticalOperationsLockedForPhantoms())
	{
		hkWorldOperation::AddPhantomBatch op;
		op.m_phantoms = const_cast<hkpPhantom**>(phantomBatch);
		op.m_numPhantoms = hkObjectIndex(numPhantoms);
		queueOperation( op );
		return;
	}
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	lockCriticalOperations();

	hkLocalArray< hkpBroadPhaseHandle* > collList(numPhantoms);
	hkLocalArray< hkAabb > aabbList(numPhantoms);

	aabbList.setSizeUnchecked( numPhantoms );
	collList.setSizeUnchecked( numPhantoms );

	for( int i = 0; i < numPhantoms; i++ )
	{
		hkpPhantom* phantom = phantomBatch[i];

		HK_ASSERT2(0xad87bc8a, phantom, "An HK_NULL found in a phantom batch.");
		HK_ASSERT2(0xad87bc89, phantom->getWorld() == HK_NULL, "Trying to add a phantom which already belongs to an hkpWorld.");

		if (!phantom->getCollidable()->getMotionState()) // may be null due to packfile serialize
		{
			phantom->m_collidable.setMotionState( phantom->getMotionState() );
		}

		phantom->setWorld( this );

#if defined(HK_PLATFORM_HAS_SPU)
		phantom->getCollidableRw()->setShapeSizeForSpu();
#endif
		collList[i] = ( phantom->getCollidableRw()->getBroadPhaseHandle() );
		phantom->calcAabb( aabbList[i] );

		phantom->setBoundingVolumeData(aabbList[i]);
		//kd tree manager is notified by the callback

		phantom->addReference();

		m_phantoms.pushBack( phantom );
		hkpWorldCallbackUtil::firePhantomAdded( this, phantom );
		phantom->firePhantomAdded();
	}

	hkLocalArray< hkpBroadPhaseHandlePair > newPairs( m_broadPhaseQuerySize );


	m_broadPhase->addObjectBatch( collList, aabbList, newPairs );

	// check for changes
	m_broadPhaseDispatcher->addPairs( static_cast<hkpTypedBroadPhaseHandlePair*>(newPairs.begin()), newPairs.getSize(), getCollisionFilter() );

	unlockAndAttemptToExecutePendingOperations();

}



void hkpWorld::removePhantom( hkpPhantom* phantom )
{
	HK_ASSERT2(0x13c74a8f,  phantom, "Cannot pass an HK_NULL as parameter to hkpWorld::removePhantom");

	// Check if operation may be performed now
	if (areCriticalOperationsLockedForPhantoms())
	{
		hkWorldOperation::RemovePhantom op;
		op.m_phantom = phantom;
		queueOperation( op );
		return;
	}

	//
	// Proceed whit the proper operation
	//

	HK_ASSERT2(0x627789e0,  phantom->getWorld() == this, "Trying to remove a phantom from a hkpWorld to which it was not added");
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	lockCriticalOperations();

	//
	//	fire callbacks
	//
	hkpWorldCallbackUtil::firePhantomRemoved( this, phantom );
	phantom->firePhantomRemoved();

	hkpWorldOperationUtil::removePhantomBP(this, phantom);

	//
	//	remove phantom from list
	//
	m_phantoms.removeAt(m_phantoms.indexOf( phantom ) );
	phantom->setWorld( HK_NULL );

	// If the entity has been loaded from a packfile try and deallocate any internal zero size arrays in the phantom.
	// If the arrays have a non-zero size the user will be warned.
	if ( phantom->m_memSizeAndFlags == 0 )
	{
		phantom->deallocateInternalArrays();
	}
	phantom->removeReference();

	unlockAndAttemptToExecutePendingOperations();
}



void hkpWorld::removePhantomBatch( hkpPhantom*const* phantomBatch, int numPhantoms )
{
	if( numPhantoms <= 0 )
	{
		return;
	}
	//	HK_ASSERT2(0xf0ff009c, !areCriticalOperationsLocked(), "Error: removing phantoms is not allowed when the world is locked (it might be safe, but also might cause problems if you change BroadPhase from an add/removeEntity, for example.)");

	if (areCriticalOperationsLockedForPhantoms())
	{
		hkWorldOperation::RemovePhantomBatch op;
		op.m_phantoms = const_cast<hkpPhantom**>(phantomBatch);
		op.m_numPhantoms = hkObjectIndex(numPhantoms);
		queueOperation( op );
		return;
	}
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	lockCriticalOperations();

	//
	//	fire callbacks
	//
	hkLocalArray< hkpBroadPhaseHandle* > collList(numPhantoms);
	{
		for( int i = 0; i < numPhantoms; i++ )
		{
			hkpPhantom* phantom = phantomBatch[i];
			HK_ASSERT2(0xad87bc87, phantom, "An HK_NULL pointer found in phantom batch.");
			HK_ASSERT2(0xad87bc88, phantom->getWorld() == this, "Trying to remove a phantom which does not belong to this hkpWorld.");
			collList.pushBackUnchecked( phantom->getCollidableRw()->getBroadPhaseHandle() );
			hkpWorldCallbackUtil::firePhantomRemoved( this, phantom );
			phantom->firePhantomRemoved();
		}
	}

	//
	//	remove pairs
	//
	if ( collList.getSize() )
	{
		hkLocalArray< hkpBroadPhaseHandlePair > removedPairs( m_broadPhaseQuerySize );
		m_broadPhase->removeObjectBatch( collList, removedPairs );
		m_broadPhaseDispatcher->removePairs( static_cast<hkpTypedBroadPhaseHandlePair*>(removedPairs.begin()), removedPairs.getSize() );
	}

	//
	// remove phantom from phantom list
	//
	{
		for( int i = 0; i < numPhantoms; i++ )
		{
			hkpPhantom* phantom = phantomBatch[i];
			phantom->setWorld( HK_NULL );
			m_phantoms.removeAt(m_phantoms.indexOf( phantom ) );
			// If the entity has been loaded from a packfile try and deallocate any internal zero size arrays in the phantom
			// If the arrays have a non-zero size the user will be warned.
			if ( phantom->m_memSizeAndFlags == 0 )
			{
				phantom->deallocateInternalArrays();
			}
			phantom->removeReference();
		}
	}

	unlockAndAttemptToExecutePendingOperations();
}

void hkpWorld::addPhysicsSystem( const hkpPhysicsSystem* sys )
{
	// Rigid bodies
	if(sys->getRigidBodies().getSize() > 0)
	{
		// Check for any HK_NULL entries in rigid body array.  If we have a HK_NULL entry
		// we cannot add the rigid bodies as a batch and must add them individually.
		if(sys->getRigidBodies().indexOf(HK_NULL) == -1)
		{
			addEntityBatch( (hkpEntity*const*)( sys->getRigidBodies().begin() ), sys->getRigidBodies().getSize(),
				sys->isActive() ? HK_ENTITY_ACTIVATION_DO_ACTIVATE : HK_ENTITY_ACTIVATION_DO_NOT_ACTIVATE );
		}
		else
		{
			HK_WARN(0x31a2b8a3, "hkPhysicsSystem contains a HK_NULL rigid body.  Cannot add rigid bodies as batch.  Adding individually.");

			for(hkInt32 i = 0; i < sys->getRigidBodies().getSize(); ++i)
			{
				if(sys->getRigidBodies()[i] != HK_NULL)
				{
					addEntity(sys->getRigidBodies()[i], sys->isActive() ? HK_ENTITY_ACTIVATION_DO_ACTIVATE : HK_ENTITY_ACTIVATION_DO_NOT_ACTIVATE);
				}
			}
		}
	}

	// Phantoms
	if(sys->getPhantoms().getSize() > 0)
	{
		// Check for any HK_NULL entries in phantom array.  If we have a HK_NULL entry
		// we cannot add the phantoms as a batch and must add them individually.
		if(sys->getPhantoms().indexOf(HK_NULL) == -1)
		{
			addPhantomBatch(sys->getPhantoms().begin(), sys->getPhantoms().getSize());
		}
		else
		{
			HK_WARN(0x31a2b8a3, "hkPhysicsSystem contains a HK_NULL phantom.  Cannot add phantoms as batch.  Adding individually.");

			for(hkInt32 i = 0; i < sys->getPhantoms().getSize(); ++i)
			{
				if(sys->getPhantoms()[i] != HK_NULL)
				{
					addPhantom(sys->getPhantoms()[i]);
				}
			}
		}
	}

	// actions & constraints -- in this order, as the constraint chains also add their 'instance actions' independently.
	for (int a=0; a < sys->getActions().getSize(); ++a)
	{
		// allow null actions for now
		if (sys->getActions()[a])
		{
			addAction(sys->getActions()[a]);
		}
	}
	for (int c=0; c < sys->getConstraints().getSize(); ++c)
	{
		// Check for HK_NULL constraints
		if(sys->getConstraints()[c] != HK_NULL)
		{
			// Make sure rigid bodies of the constraint are in the world.  They may not be because they could
			// be HK_NULL and therefore not added to the world by the checks above.
			if ( ( sys->getConstraints()[c]->getEntityA() == HK_NULL) && ( sys->getConstraints()[c]->getEntityB() == HK_NULL ) )
			{
				HK_WARN(0x615d642c, "hkPhysicsSystem contains a constraint with both entities set to HK_NULL.  Constraint not added.");
			}
			else
			{
				addConstraint(sys->getConstraints()[c]);
			}
		}
		else
		{
			HK_WARN(0x615d642c, "hkPhysicsSystem contains a HK_NULL constraint.  HK_NULL constraint not added.");
		}
	}
}

void hkpWorld::removePhysicsSystem( const hkpPhysicsSystem* sys )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	// contraints & actions
	for (int c=0; c < sys->getConstraints().getSize(); ++c)
	{
		// Only remove constraints that have an owner.
		if(sys->getConstraints()[c] && sys->getConstraints()[c]->getOwner() != HK_NULL)
		{
			removeConstraint(sys->getConstraints()[c]);
		}
	}
	for (int a=0; a < sys->getActions().getSize(); ++a)
	{
		if(sys->getActions()[a])
		{
			removeAction(sys->getActions()[a]);
		}
	}

	// Rigid bodies
	{
		// Check for any HK_NULL entries in rigid body array.  If we have a HK_NULL entry
		// we cannot remove the rigid bodies as a batch and must remove them individually.
		if(sys->getRigidBodies().indexOf(HK_NULL) == -1)
		{
			removeEntityBatch( (hkpEntity*const*)( sys->getRigidBodies().begin() ), sys->getRigidBodies().getSize() );
		}
		else
		{
			for(hkInt32 i = 0; i < sys->getRigidBodies().getSize(); ++i)
			{
				if(sys->getRigidBodies()[i] != HK_NULL)
				{
					removeEntity(sys->getRigidBodies()[i]);
				}
			}
		}
	}

	// Phantoms
	{
		// Check for any HK_NULL entries in phantom array.  If we have a HK_NULL entry
		// we cannot remove the phantoms as a batch and must remove them individually.
		if(sys->getPhantoms().indexOf(HK_NULL) == -1)
		{
			removePhantomBatch(sys->getPhantoms().begin(), sys->getPhantoms().getSize());
		}
		else
		{
			HK_WARN(0x31a2b8a3, "hkPhysicsSystem contains a HK_NULL phantom.  Cannot add phantoms as batch.  Adding individually.");

			for(hkInt32 i = 0; i < sys->getPhantoms().getSize(); ++i)
			{
				if(sys->getPhantoms()[i] != HK_NULL)
				{
					removePhantom(sys->getPhantoms()[i]);
				}
			}
		}
	}
}

//
// Gravity (convenience)
//

void hkpWorld::setGravity( const hkVector4& gravity )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	// We are not waking up objects
	m_gravity = gravity;
}


//
// Listener registration
//

void hkpWorld::addActionListener( hkpActionListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x7d1e9387, m_actionListeners.indexOf(worldListener) < 0, "You tried to add a world action listener twice" );

	m_actionListeners.pushBack( worldListener );
}

void hkpWorld::removeActionListener( hkpActionListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_actionListeners.indexOf(worldListener);
	HK_ASSERT2(0x52e10e50, i >= 0, "You tried to remove a world action listener, which was never added" );
	m_actionListeners[i] = HK_NULL;
}


void hkpWorld::addConstraintListener( hkpConstraintListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x1a5b73b6, m_constraintListeners.indexOf( worldListener ) < 0, "You tried to add a world constraint listener twice" );

	m_constraintListeners.pushBack( worldListener );
}

void hkpWorld::removeConstraintListener( hkpConstraintListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_constraintListeners.indexOf( worldListener );
	HK_ASSERT2(0x14e7d731, i >= 0, "You tried to remove a world constraint listener, which was never added" );
	m_constraintListeners[i] = HK_NULL;
}

void hkpWorld::addEntityListener( hkpEntityListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x41fecf63, m_entityListeners.indexOf(worldListener) < 0, "You tried to add a world entity listener twice" );
	m_entityListeners.pushBack( worldListener );
}

void hkpWorld::removeEntityListener( hkpEntityListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_entityListeners.indexOf(worldListener);
	HK_ASSERT2(0x7e5dcf64, i >= 0, "You tried to remove a world entity listener, which was never added" );
	m_entityListeners[i] = HK_NULL;
}

void hkpWorld::addPhantomListener( hkpPhantomListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x4aa03aaf, m_phantomListeners.indexOf(worldListener) < 0, "You tried to add a world entity listener twice" );
	m_phantomListeners.pushBack( worldListener );
}

void hkpWorld::removePhantomListener( hkpPhantomListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_phantomListeners.indexOf(worldListener);
	HK_ASSERT2(0x25ce777c, i >= 0, "You tried to remove a hkpPhantomListener, which was never added" );
	m_phantomListeners[i] = HK_NULL;
}

void hkpWorld::addIslandActivationListener( hkpIslandActivationListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x1a14bf93, m_islandActivationListeners.indexOf(worldListener) < 0, "You tried to add a world activation listener twice" );
	m_islandActivationListeners.pushBack( worldListener );
}

void hkpWorld::removeIslandActivationListener( hkpIslandActivationListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_islandActivationListeners.indexOf( worldListener );
	HK_ASSERT2(0x137408f4, i >= 0, "You tried to remove a world activation listener, which was never added" );
	m_islandActivationListeners[i] = HK_NULL;
}

void hkpWorld::addWorldPostCollideListener( hkpWorldPostCollideListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x63a352b1, m_worldPostCollideListeners.indexOf(worldListener) < 0, "You tried to add a world post detection listener twice" );
	m_worldPostCollideListeners.pushBack( worldListener );
}

void hkpWorld::removeWorldPostCollideListener( hkpWorldPostCollideListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_worldPostCollideListeners.indexOf(worldListener);
	HK_ASSERT2(0x67c333b0, i >= 0, "You tried to remove a world post detection listener, which was never added" );
	m_worldPostCollideListeners[i] = HK_NULL;
}


void hkpWorld::addWorldPostSimulationListener( hkpWorldPostSimulationListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x619dae7f, m_worldPostSimulationListeners.indexOf(worldListener) < 0, "You tried to add a world post simulation listener twice" );
	m_worldPostSimulationListeners.pushBack( worldListener );
}

void hkpWorld::removeWorldPostSimulationListener( hkpWorldPostSimulationListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_worldPostSimulationListeners.indexOf(worldListener);
	HK_ASSERT2(0x5eb3cb29, i >= 0, "You tried to remove a world post simulation listener, which was never added" );
	m_worldPostSimulationListeners[i] = HK_NULL;
}

void hkpWorld::addWorldPostIntegrateListener( hkpWorldPostIntegrateListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x619dae7f, m_worldPostIntegrateListeners.indexOf(worldListener) < 0, "You tried to add a world post simulation listener twice" );
	m_worldPostIntegrateListeners.pushBack( worldListener );
}

void hkpWorld::removeWorldPostIntegrateListener( hkpWorldPostIntegrateListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_worldPostIntegrateListeners.indexOf(worldListener);
	HK_ASSERT2(0x5eb3cb29, i >= 0, "You tried to remove a world post simulation listener, which was never added" );
	m_worldPostIntegrateListeners[i] = HK_NULL;
}


void hkpWorld::addIslandPostCollideListener( hkpIslandPostCollideListener* islandListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x5c983e76, m_islandPostCollideListeners.indexOf(islandListener) < 0, "You tried to add a island post detection listener twice" );
	m_islandPostCollideListeners.pushBack( islandListener );
}

void hkpWorld::removeIslandPostCollideListener( hkpIslandPostCollideListener* islandListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_islandPostCollideListeners.indexOf(islandListener);
	HK_ASSERT2(0x60701a8c, i >= 0, "You tried to remove a island post detection listener, which was never added" );
	m_islandPostCollideListeners[i] = HK_NULL;
}


void hkpWorld::addIslandPostIntegrateListener( hkpIslandPostIntegrateListener* islandListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x4654251a, m_islandPostIntegrateListeners.indexOf(islandListener) < 0, "You tried to add a island post simulation listener twice" );
	m_islandPostIntegrateListeners.pushBack( islandListener );
}

void hkpWorld::removeIslandPostIntegrateListener( hkpIslandPostIntegrateListener* islandListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_islandPostIntegrateListeners.indexOf(islandListener);
	HK_ASSERT2(0x143fdef2, i >= 0, "You tried to remove a island post simulation listener, which was never added" );
	m_islandPostIntegrateListeners[i] = HK_NULL;
}


void hkpWorld::addContactListener( hkpContactListener* collisionListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x1debcc37, m_contactListeners.indexOf(collisionListener) < 0, "You tried to add a world collision listener twice" );
	m_contactListeners.pushBack( collisionListener );
}


void hkpWorld::removeContactListener( hkpContactListener* collisionListener)
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_contactListeners.indexOf(collisionListener);
	HK_ASSERT2(0x6c3fe017, i >= 0, "You tried to remove a world collision listener, which was never added" );
	m_contactListeners[i] = HK_NULL;
}


void hkpWorld::addWorldDeletionListener( hkpWorldDeletionListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x2ad1617f, m_worldDeletionListeners.indexOf( worldListener ) < 0, "You tried to add a world deletion listener twice" );
	m_worldDeletionListeners.pushBack( worldListener );
}

void hkpWorld::removeWorldDeletionListener( hkpWorldDeletionListener* worldListener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_worldDeletionListeners.indexOf( worldListener );
	HK_ASSERT2(0x12f005e2, i >= 0, "You tried to remove a world deletion listener, which was never added" );
	m_worldDeletionListeners[i] = HK_NULL;
}

void hkpWorld::addContactImpulseLimitBreachedListener( hkpContactImpulseLimitBreachedListener* listener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2(0x2ad1617f, m_contactImpulseLimitBreachedListeners.indexOf( listener ) < 0, "You tried to add a hkpContactImpulseLimitBreachedListener twice" );
	m_contactImpulseLimitBreachedListeners.pushBack( listener );
}

void hkpWorld::removeContactImpulseLimitBreachedListener( hkpContactImpulseLimitBreachedListener* listener )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int i = m_contactImpulseLimitBreachedListeners.indexOf( listener );
	HK_ASSERT2(0x12f005e2, i >= 0, "You tried to remove a hkpContactImpulseLimitBreachedListener, which was never added" );
	m_contactImpulseLimitBreachedListeners[i] = HK_NULL;
}

void hkpWorld::addWorldExtension( hkpWorldExtension* extension )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	HK_ASSERT2( 0x34a7e8b5, m_worldExtensions.indexOf( extension ) == -1, "Cannot add an extension which is already in the world." );

	m_worldExtensions.pushBack( extension );
	extension->addedToWorld( this );
}

void hkpWorld::removeWorldExtension( hkpWorldExtension* extension )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	const int index = m_worldExtensions.indexOf( extension );
	HK_ASSERT2( 0x34a7e8b2, index != -1, "Cannot remove an extension which isn't in the world." );
	extension->removedFromWorld( this );
	m_worldExtensions.removeAt( index );
}

hkpWorldExtension* hkpWorld::findWorldExtension( int id ) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );

	HK_ASSERT2( 0x3a89e12d, id != -1, "You should not search for an extension with id -1." );
	const int numExtension = m_worldExtensions.getSize();
	for ( int i = 0; i < numExtension; ++i )
	{
		hkpWorldExtension *const extension = m_worldExtensions[i];
		if ( extension->getId() == id )
		{
			return extension;
		}
	}
	return HK_NULL;
}

hkTime hkpWorld::getCurrentTime() const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	return m_simulation->getCurrentTime();
}

hkTime hkpWorld::getCurrentPsiTime() const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	return m_simulation->getCurrentPsiTime();
}


hkpStepResult hkpWorld::stepDeltaTime( hkReal physicsDeltaTime )
{
	HK_CHECK_FLUSH_DENORMALS();

#if defined(HK_PLATFORM_HAS_SPU)
	if ( m_collisionDispatcher->m_agent3Registered && !hkpCollectionCollectionAgent3::g_agentRegistered)
	{
		HK_WARN_ONCE(0xad234123, "The hkpCollectionCollectionAgent3 must be registered on PS3 if you use process hkpShapeCollections on SPU");
	}
#endif

	lock();
	hkpWorldMemoryUtil::watchHeapMemory( this );
	HK_ON_ENABLE_INTERNAL_DATA_RANDOMIZATION( randomizeInternalState() );
	hkpStepResult result = m_simulation->stepDeltaTime( physicsDeltaTime );

	// Update broad phase structures
	if( m_broadPhase->getType() == hkpBroadPhase::BROADPHASE_TREE_16BIT )
	{
		HK_ASSERT2( 0x3a6e4912, hkpBroadPhase::s_updateTreeBroadPhaseFunction, "hkpTreeBroadPhase is not registered" );
		hkpBroadPhase::s_updateTreeBroadPhaseFunction( m_broadPhase, physicsDeltaTime );
	}
	else if ( m_broadPhase->getType() == hkpBroadPhase::BROADPHASE_TREE_32BIT )
	{
		HK_ASSERT2( 0x3a6e4912, hkpBroadPhase::s_updateTreeBroadPhaseFunction32, "hkpTreeBroadPhase32 is not registered" );
		hkpBroadPhase::s_updateTreeBroadPhaseFunction32( m_broadPhase, physicsDeltaTime );
	}

	unlock();
	return result;
}


hkpStepResult hkpWorld::integrate( hkReal physicsDeltaTime )
{
	lock();
	hkpStepResult result = m_simulation->integrate( physicsDeltaTime );
	unlock();
	return result;
}

hkpStepResult hkpWorld::collide()
{
	lock();
	hkpStepResult result = m_simulation->collide();
	unlock();
	return result;
}

hkpStepResult hkpWorld::advanceTime()
{
	lock();
	hkpStepResult result = m_simulation->advanceTime();
	unlock();
	return result;
}


void hkpWorld::setFrameTimeMarker( hkReal frameDeltaTime )
{
	m_simulation->setFrameTimeMarker( frameDeltaTime );
}

bool hkpWorld::isSimulationAtMarker() const
{
	return m_simulation->isSimulationAtMarker();
}

bool hkpWorld::isSimulationAtPsi() const
{
	return m_simulation->isSimulationAtPsi( );
}

#if defined(HK_PLATFORM_MULTI_THREAD) && (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)


hkpStepResult hkpWorld::stepMultithreaded( hkJobQueue* jobQueue, hkJobThreadPool* threadPool, hkReal physicsDeltaTime )
{
	HK_CHECK_FLUSH_DENORMALS();

#if defined(HK_PLATFORM_HAS_SPU)
	if ( m_collisionDispatcher->m_agent3Registered && !hkpCollectionCollectionAgent3::g_agentRegistered)
	{
		HK_WARN_ONCE(0xad234123, "The hkpCollecitonCollecitonAgent3 must be registered on PS3 if you use process hkpShapeCollections on SPU");
	}
#endif

	hkpStepResult result = initMtStep( jobQueue, physicsDeltaTime );
	if ( result != HK_STEP_RESULT_SUCCESS )
	{
		return result;
	}
	threadPool->processAllJobs( jobQueue );
	jobQueue->processAllJobs( );
	threadPool->waitForCompletion();

	result = finishMtStep( jobQueue, threadPool );
	return result;
}

hkpStepResult hkpWorld::initMtStep( hkJobQueue* jobQueue, hkReal physicsDeltaTime )
{
	HK_ASSERT2( 0x1298af35, m_simulationType == hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED, "You cannot call this function on the world if you are not using a multithreaded simulation.");
#if defined HK_ENABLE_INTERNAL_DATA_RANDOMIZATION
	lock();
	randomizeInternalState();
	unlock();
#endif

	if ( m_memoryWatchDog ) 
	{
		lock();
		hkpWorldMemoryUtil::watchHeapMemory( this );
		unlock();
	}

	return m_simulation->stepBeginSt( jobQueue, physicsDeltaTime );
}


hkpStepResult hkpWorld::finishMtStep( hkJobQueue* jobQueue, hkJobThreadPool* threadPool )
{
	HK_ASSERT2( 0x1298af35, m_simulationType == hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED, "You cannot call this function on the world if you are not using a multithreaded simulation.");
	hkpStepResult result = m_simulation->finishMtStep( jobQueue, threadPool );
	if ( result != HK_STEP_RESULT_SUCCESS )
	{
		return result;
	}

	// Update broad phase structures
	if ( m_broadPhase->getType() == hkpBroadPhase::BROADPHASE_TREE_16BIT )
	{
		HK_ASSERT2( 0x3a6e4912, hkpBroadPhase::s_updateTreeBroadPhaseFunction, "hkpTreeBroadPhase is not registered" );
		markForRead();
		hkpBroadPhase::s_updateTreeBroadPhaseFunction( m_broadPhase, m_simulation->getPhysicsDeltaTime() );
		unmarkForRead();
	}
	else if ( m_broadPhase->getType() == hkpBroadPhase::BROADPHASE_TREE_32BIT )
	{
		HK_ASSERT2( 0x3a6e4912, hkpBroadPhase::s_updateTreeBroadPhaseFunction32, "hkpTreeBroadPhase32 is not registered" );
		markForRead();
		hkpBroadPhase::s_updateTreeBroadPhaseFunction32( m_broadPhase, m_simulation->getPhysicsDeltaTime() );
		unmarkForRead();
	}

	return HK_STEP_RESULT_SUCCESS;
}


void hkpWorld::getMultithreadConfig( hkpMultithreadConfig& config )
{
	HK_ASSERT2(0x192fa846, m_simulationType == hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED, "This function can only be called for a multithreaded simulation");
	m_simulation->getMultithreadConfig( config );
}

void hkpWorld::setMultithreadConfig( const hkpMultithreadConfig& config, hkJobQueue* queue )
{
	HK_ASSERT2(0x192fa846, m_simulationType == hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED, "This function can only be called for a multithreaded simulation");
	m_simulation->setMultithreadConfig( config, queue );
}
#else
hkpStepResult hkpWorld::initMtStep( hkJobQueue* jobQueue, hkReal frameDeltaTime ){			HK_ASSERT2( 0xf032ed45, false, "Your platform does not support multithreaded simulation");	return HK_STEP_RESULT_SUCCESS; }
hkpStepResult hkpWorld::finishMtStep( hkJobQueue* jobQueue, hkJobThreadPool* threadPool ){									HK_ASSERT2( 0xf032ed47, false, "Your platform does not support multithreaded simulation"); return HK_STEP_RESULT_SUCCESS;	}
void hkpWorld::getMultithreadConfig( hkpMultithreadConfig& config ){ HK_ASSERT2( 0xf032ed4b, false, "Your platform does not support multithreaded simulation");	}
void hkpWorld::setMultithreadConfig( const hkpMultithreadConfig& config, hkJobQueue* queue ){ HK_ASSERT2( 0xf032ed4c, false, "Your platform does not support multithreaded simulation");	}
#endif

#if defined (HK_ENABLE_DETERMINISM_CHECKS)
static HK_FORCE_INLINE hkBool less_hkSimulationIslandPtr( const hkpSimulationIsland* a, const hkpSimulationIsland* b )
{
	// the below works because m_isInActiveIslandsArray is a boolean value
	return ( a->m_isInActiveIslandsArray < b->m_isInActiveIslandsArray || a->m_uTag < b->m_uTag );
}
#endif

#if defined (HK_ENABLE_DETERMINISM_CHECKS)
namespace {
	struct UidAndIdx
	{
		hkUint32 m_entityUid;
		hkObjectIndex m_endPointIdx;
	};

	bool cmpLessUidAndIdx(const UidAndIdx& x0, const UidAndIdx& x1)
	{
		return x0.m_endPointIdx < x1.m_endPointIdx;
	}

}
#endif

void hkpWorld::checkDeterminismOfIslandBroadPhase(const hkpSimulationIsland* island)
{
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	HK_TIME_CODE_BLOCK("hkpWorld::checkDeterminismOfIslandBroadPhase", HK_NULL);
	island->m_world->m_broadPhase->lock();
	const hkArray<hkpEntity*>& entities = island->m_entities;

	hkCheckDeterminismUtil::checkMt( 0xf0000180, entities.getSize());

	//const hkp3AxisSweep* bp = static_cast<const hkp3AxisSweep*>(((const hkpWorld*)(this))->getBroadPhase());
	
	for (int ei = 0; ei < entities.getSize(); ei++)
	{
		hkpEntity* e = entities[ei];
		hkUint32 nodeId = e->m_collidable.m_broadPhaseHandle.m_id;
		hkCheckDeterminismUtil::checkMt(0xf0000181, e->getUid());
		hkCheckDeterminismUtil::checkMt(0xf0000182, nodeId);
// 		hkp3AxisSweep::hkpBpNode node = bp->m_nodes[nodeId];
// 		node.m_handle = HK_NULL;
// 		hkCheckDeterminismUtil::checkMt( 0xf0000183, node );
	}
	island->m_world->m_broadPhase->unlock();
#endif
}

void hkpWorld::checkDeterminism()
{
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	HK_TIME_CODE_BLOCK("hkpWorld::checkDeterminism", HK_NULL );
	hkArray<const hkpSimulationIsland*> islands;
	{
		// for all motions: check with loaded data
		{
			islands.pushBack(this->getFixedIsland());
			islands.insertAt(islands.getSize(), getActiveSimulationIslands().begin(), getActiveSimulationIslands().getSize());
			islands.insertAt(islands.getSize(), getInactiveSimulationIslands().begin(), getInactiveSimulationIslands().getSize());
		}
	}

	hkAlgorithm::quickSort(islands.begin(), islands.getSize(), less_hkSimulationIslandPtr);

	//
	//	Check entity specific data
	//
	{
		hkCheckDeterminismUtil::checkMt( 0xf0000184, getActiveSimulationIslands().getSize() );
		hkCheckDeterminismUtil::checkMt( 0xf0000185, m_inactiveSimulationIslands.getSize() );

		for (int i = 0; i < islands.getSize(); i++)
		{
			const hkpSimulationIsland& island = *islands[i];
			HK_ASSERT2(0xad8655d4, island.m_uTag != hkUint32(-1), "invalid island uid");
			hkCheckDeterminismUtil::checkMt( 0xf0000186, island.m_uTag );
			hkCheckDeterminismUtil::checkMt( 0xf0000187, island.getEntities().getSize() );
			hkCheckDeterminismUtil::checkMt( 0xf0000188, island.m_storageIndex );
			if ( i )
			{
				hkCheckDeterminismUtil::checkMt( 0xf00000c2, island.m_splitCheckFrameCounter);
			}


			for (int e = 0; e < island.m_entities.getSize(); e++)
			{
				hkpRigidBody& body = static_cast<hkpRigidBody&>(*island.m_entities[e]);
				{
					hkCheckDeterminismUtil::checkMt( 0xf0000189, body.getUid() );
					hkCheckDeterminismUtil::checkMt( 0xf000018a, body.getCollidable()->getBroadPhaseHandle()->m_id );
				}

				//
				//	Motions
				//
				{
					hkpMotion* motion = body.getRigidMotion();
					hkReal position = motion->getPosition().lengthSquared<3>().getReal();		hkCheckDeterminismUtil::checkMt(0xf000018b, position);
					hkReal rotation = motion->getRotation().m_vec.lengthSquared<4>().getReal();	hkCheckDeterminismUtil::checkMt(0xf000018c, rotation);
					hkReal linVel = motion->getLinearVelocity().lengthSquared<3>().getReal();	hkCheckDeterminismUtil::checkMt(0xf000018d, linVel);
					hkReal angVel = motion->getAngularVelocity().lengthSquared<3>().getReal();	hkCheckDeterminismUtil::checkMt(0xf000018e, angVel);
				}

				for (int c = 0; c < body.m_constraintsMaster.getSize(); c++)
				{
					const hkpConstraintInstance* constraint = body.m_constraintsMaster[c].m_constraint;
					hkCheckDeterminismUtil::checkMt( 0xf0000190, constraint->m_uid ^ constraint->getOtherEntity(&body)->getUid());
					//hkCheckDeterminismUtil::checkMt( 0xf0000190, constraint->m_uid );
					//hkCheckDeterminismUtil::checkMt( 0xf0000191, constraint->getOtherEntity(&body)->getUid() );
				}
			}
		}
	}

	//
	// Check determinism of entire broadphase
	//
	getBroadPhase()->checkDeterminism();


	//
	//	Check collision information
	//
	if(1)
	{
		for (int i = 0; i < islands.getSize(); i++)
		{
			const hkpSimulationIsland* island = islands[i];
			const hkpAgentNnTrack* tracks[2] = { &island->m_narrowphaseAgentTrack, &island->m_midphaseAgentTrack };
			for ( int j = 0; j < 2; ++j )
			{
				const hkpAgentNnTrack* track = tracks[j];
				hkCheckDeterminismUtil::checkMt(0xf00001a0, track->m_sectors.getSize());
				HK_ACCESS_CHECK_WITH_PARENT( island->m_world, HK_ACCESS_RO, island, HK_ACCESS_RW );
				HK_FOR_ALL_AGENT_ENTRIES_BEGIN((*track), entry)
				{
					int uidA = hkpGetRigidBody(entry->getCollidableA())->getUid();
					int uidB = hkpGetRigidBody(entry->getCollidableB())->getUid();
					hkCheckDeterminismUtil::checkMt( 0xf00001a1, (uidA + uidB) ^ entry->m_agentType);

					// not working .. hkCheckDeterminismUtil::checkMt( &entry->m_agentIndexOnCollidable[0],2 );
					hkpSimpleConstraintContactMgr* mgr = (hkpSimpleConstraintContactMgr*)entry->m_contactMgr;
					if ( mgr->getConstraintInstance() )
					{
						hkCheckDeterminismUtil::checkMt( 0xf00001a4, mgr->m_contactConstraintData.m_atom->m_numContactPoints);
						hkpSimpleContactConstraintAtom* atoms = mgr->m_contactConstraintData.m_atom;
						const hkContactPoint* contactPoints = atoms->getContactPoints();
						hkCheckDeterminismUtil::checkMtCrc( 0xf00001a5, contactPoints, atoms->m_numContactPoints );
						// 	hkpContactPointPropertiesStream* props = atoms->getContactPointPropertiesStream();		// this does not work because the m_user member could hold a pointer
						// 	int propSize = atoms->getContactPointPropertiesStriding();
						// 	hkCheckDeterminismUtil::checkMtCrc( 0xf00001a5, (char*)props, propSize * atoms->m_numContactPoints );
					}
				}
				HK_FOR_ALL_AGENT_ENTRIES_END;
			}
		}
	}
#endif // if defined (HK_ENABLE_DETERMINISM_CHECKS)

}

void hkpWorld::getCinfo(hkpWorldCinfo& info) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	// NOTE: The order of these variables matches the order in the world Cinfo. Please keep them in sync
	// if you are making changes.

	// Basic setup
	info.m_gravity = m_gravity;
	info.m_broadPhaseQuerySize = m_broadPhaseQuerySize;
	info.m_broadPhaseWorldAabb.m_min  = m_broadPhaseExtents[0];
	info.m_broadPhaseWorldAabb.m_max  = m_broadPhaseExtents[1];
	info.m_collisionTolerance = m_collisionInput->getTolerance();
	info.m_collisionFilter = m_collisionFilter;
	info.m_convexListFilter = m_convexListFilter;
	
	info.m_broadPhaseType = m_broadPhaseType;
	info.m_broadPhaseBorderBehaviour = (m_broadPhaseBorder)? m_broadPhaseBorder->m_type : hkpWorldCinfo::BROADPHASE_BORDER_DO_NOTHING;
	info.m_mtPostponeAndSortBroadPhaseBorderCallbacks = (m_broadPhaseBorder)? m_broadPhaseBorder->m_postponeAndSortCallbacks : hkBool(false);

	info.m_expectedMaxLinearVelocity = m_collisionDispatcher->m_expectedMaxLinearVelocity;
	info.m_expectedMinPsiDeltaTime   = m_collisionDispatcher->m_expectedMinPsiDeltaTime;

	info.m_snapCollisionToConvexEdgeThreshold = m_snapCollisionToConvexEdgeThreshold;
	info.m_snapCollisionToConvexEdgeThreshold = m_snapCollisionToConcaveEdgeThreshold;
	info.m_enableToiWeldRejection = m_enableToiWeldRejection;

	info.m_memoryWatchDog = m_memoryWatchDog;

	// Optimizations
	info.m_broadPhaseNumMarkers = m_broadPhaseNumMarkers;
	info.m_sizeOfToiEventQueue  = m_sizeOfToiEventQueue;
	info.m_contactPointGeneration = m_contactPointGeneration;
	info.m_allowToSkipConfirmedCallbacks = m_collisionInput->m_allowToSkipConfirmedCallbacks;
	info.m_contactRestingVelocity = m_dynamicsStepInfo.m_solverInfo.m_contactRestingVelocity;

	// Solver Settings
	info.m_solverTau = m_dynamicsStepInfo.m_solverInfo.m_tau;
	info.m_solverDamp = m_dynamicsStepInfo.m_solverInfo.m_damping;
	info.m_solverIterations = m_dynamicsStepInfo.m_solverInfo.m_numSteps;
	info.m_solverMicrosteps = m_dynamicsStepInfo.m_solverInfo.m_numMicroSteps;
	info.m_maxConstraintViolation = hkMath::sqrt(m_dynamicsStepInfo.m_solverInfo.m_maxConstraintViolationSqrd);
	info.m_forceCoherentConstraintOrderingInSolver = m_dynamicsStepInfo.m_solverInfo.m_forceCoherentConstraintOrderingInSolver;

	// Solver's Deactivation Settings
	info.m_deactivationNumInactiveFramesSelectFlag0 = m_dynamicsStepInfo.m_solverInfo.m_deactivationNumInactiveFramesSelectFlag[0];
	info.m_deactivationNumInactiveFramesSelectFlag1 = m_dynamicsStepInfo.m_solverInfo.m_deactivationNumInactiveFramesSelectFlag[1];
	info.m_deactivationIntegrateCounter = m_dynamicsStepInfo.m_solverInfo.m_deactivationIntegrateCounter;

	// Internal algorithm settings
	info.m_iterativeLinearCastEarlyOutDistance = m_collisionInput->m_config->m_iterativeLinearCastEarlyOutDistance;
	info.m_iterativeLinearCastMaxIterations = m_collisionInput->m_config->m_iterativeLinearCastMaxIterations;
	info.m_enableDeprecatedWelding = m_collisionInput->m_enableDeprecatedWelding;
	info.m_shouldActivateOnRigidBodyTransformChange = m_shouldActivateOnRigidBodyTransformChange;
	info.m_toiCollisionResponseRotateNormal    = m_toiCollisionResponseRotateNormal;
	
	//info.m_enableForceLimitBreachedSecondaryEventsFromToiSolver = m_enableForceLimitBreachedSecondaryEventsFromToiSolver;
	info.m_useCompoundSpuElf = m_useCompoundSpuElf;
	info.m_maxSectorsPerMidphaseCollideTask    = m_maxSectorsPerMidphaseCollideTask;
	info.m_maxSectorsPerNarrowphaseCollideTask = m_maxSectorsPerNarrowphaseCollideTask;
	info.m_processToisMultithreaded			   = m_processToisMultithreaded;	
	info.m_maxEntriesPerToiMidphaseCollideTask         = m_maxEntriesPerToiMidphaseCollideTask;
	info.m_maxEntriesPerToiNarrowphaseCollideTask         = m_maxEntriesPerToiNarrowphaseCollideTask;
	info.m_maxNumToiCollisionPairsSinglethreaded = m_maxNumToiCollisionPairsSinglethreaded;
	info.m_deactivationReferenceDistance       = m_deactivationReferenceDistance;
	info.m_numToisTillAllowedPenetrationSimplifiedToi = m_numToisTillAllowedPenetrationSimplifiedToi;
	info.m_numToisTillAllowedPenetrationToi           = m_numToisTillAllowedPenetrationToi;
	info.m_numToisTillAllowedPenetrationToiHigher     = m_numToisTillAllowedPenetrationToiHigher;
	info.m_numToisTillAllowedPenetrationToiForced     = m_numToisTillAllowedPenetrationToiForced;


	// Debugging flags
	info.m_enableDeactivation = m_wantDeactivation;
	info.m_simulationType = m_simulationType;
	info.m_frameMarkerPsiSnap = m_simulation->m_frameMarkerPsiSnap;

	info.m_enableSimulationIslands = m_wantSimulationIslands;
	info.m_processActionsInSingleThread = m_processActionsInSingleThread;
	info.m_allowIntegrationOfIslandsWithoutConstraintsInASeparateJob = m_allowIntegrationOfIslandsWithoutConstraintsInASeparateJob;
	info.m_minDesiredIslandSize = m_minDesiredIslandSize;

	// CollisionCallbackUtil
	info.m_fireCollisionCallbacks = ( hkpCollisionCallbackUtil::findCollisionCallbackUtil( this ) != HK_NULL );
}


hkWorldMemoryAvailableWatchDog* hkpWorld::getMemoryWatchDog( ) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	return m_memoryWatchDog;
}

void hkpWorld::setMemoryWatchDog( hkWorldMemoryAvailableWatchDog* watchDog )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	m_memoryWatchDog = watchDog;
}


hkpWorld::hkpWorld( const hkpWorldCinfo& infoBase, unsigned int sdkversion )
{
	hkpWorldCinfo	info = infoBase;
	m_lastEntityUid = hkUint32(-1);
	HK_ON_DETERMINISM_CHECKS_ENABLED( m_lastIslandUid = hkUint32(-1) );
	m_lastConstraintUid = hkUint32(-1);
	m_isLocked = false;
	m_assertOnRunningOutOfSolverMemory = true;

	m_violatedConstraintArray	=	new hkpViolatedConstraintArray();

#	if defined (HK_ENABLE_DETERMINISM_CHECKS)
	HK_WARN_ONCE(0xf0233212, "The engine is compiled with special determinism checks, see hkCheckDeterminismUtil.h");
#	endif

	//
	//	Check consistency of stepSizes in info HVK-929
	//
#ifdef HK_DEBUG
	{
		hkReal estimatedDt = 0.016f; // or shall we use info.m_expectedMinPsiDeltaTime
		hkReal gravLen = info.m_gravity.length<3>().getReal();
		if ( gravLen > 0.0f )
		{
			hkReal distanceTravelledInOneFrameDueToGravitationalAcceleration = 0.5f * gravLen * estimatedDt * estimatedDt;
			if ( distanceTravelledInOneFrameDueToGravitationalAcceleration > info.m_collisionTolerance * 2.0f )
			{
				HK_WARN( 0xf0de4354, "Your m_collisionTolerance seems to be very small" );
			}
			hkReal distanceTravelledInTenFramesDueToGravitationalAcceleration = 0.5f * gravLen * 10.0f * estimatedDt * 10.0f * estimatedDt;
			if ( distanceTravelledInTenFramesDueToGravitationalAcceleration < info.m_collisionTolerance  )
			{
				HK_WARN( 0xf0de4355, "Your m_collisionTolerance seems to be very big" );
			}

			hkReal velocityInOneFrame = 1.0f * estimatedDt * gravLen;
			if ( velocityInOneFrame > info.m_contactRestingVelocity )
			{
				HK_WARN( 0xf0de4356, "Your m_contactRestingVelocity seems to be too small" );
			}
		}
	}
#endif


	//
	// Operation delaying manager
	//
	m_pendingOperations = new hkpWorldOperationQueue(this);
	m_pendingOperationQueues = HK_NULL;
	HK_ON_DEBUG( hkpDebugInfoOnPendingOperationQueues::init(this);	);
	m_pendingOperationsCount = 0;
	m_pendingBodyOperationsCount = 0;
	m_criticalOperationsLockCount = 0;
	m_criticalOperationsLockCountForPhantoms = 0;
	m_blockExecutingPendingOperations = false;
	m_criticalOperationsAllowed = true;
	m_modifyConstraintCriticalSection = HK_NULL;
	m_propertyMasterLock = HK_NULL;
	m_islandDirtyListCriticalSection = HK_NULL;
	m_pendingOperationQueueCount = 1;
	m_destructionWorld = HK_NULL;

	//
	// Broadphase
	//
	{

		// Extents
		m_broadPhaseExtents[0] = info.m_broadPhaseWorldAabb.m_min;
		m_broadPhaseExtents[1] = info.m_broadPhaseWorldAabb.m_max;
		HK_ASSERT2(0x1570b067,  m_broadPhaseExtents[0].allLess<3>(m_broadPhaseExtents[1]), "Each axis of world size MUST be > 0.0 !");

		// Sweep and prune markers
		HK_ASSERT2(0x465fe452, info.m_broadPhaseNumMarkers == 0, "There is currently an issue with markers that can cause a crash. You should disable markers until this is fixed");
		m_broadPhaseNumMarkers = info.m_broadPhaseNumMarkers;

		m_broadPhaseType = info.m_broadPhaseType;
		m_broadPhase = HK_NULL;

		// Check if we need a sweep and prune broad phase. Usually we do.
		if( m_broadPhaseType != hkpWorldCinfo::BROADPHASE_TYPE_TREE )
		{
			HK_ASSERT2(0x7372ab86, hkpBroadPhase::s_createSweepAndPruneBroadPhaseFunction, "hkp3AxisSweepBroadPhase is not registered" );
			m_broadPhase = hkpBroadPhase::s_createSweepAndPruneBroadPhaseFunction( m_broadPhaseExtents[0], m_broadPhaseExtents[1], info.m_broadPhaseNumMarkers );
		}

		// Check if we need a tree broad phase. In hybrid mode this wraps the sweep and prune broad phase.
		if( m_broadPhaseType == hkpWorldCinfo::BROADPHASE_TYPE_TREE || m_broadPhaseType == hkpWorldCinfo::BROADPHASE_TYPE_HYBRID )
		{
			HK_ASSERT2(0x3a6e4912, hkpBroadPhase::s_createTreeBroadPhaseFunction, "hkpTreeBroadPhase is not registered" );
			hkpBroadPhase* childBroadPhase = m_broadPhase;
			m_broadPhase = hkpBroadPhase::s_createTreeBroadPhaseFunction( childBroadPhase );
			if( childBroadPhase )
			{
				childBroadPhase->removeReference();
			}
		}
		
		// this is used as a guess for how many overlapping objects will be found in the broadphase on the addition of an entity or phantom
		m_broadPhaseQuerySize = info.m_broadPhaseQuerySize; // 1024
		m_broadPhaseUpdateSize = m_broadPhaseQuerySize / 2;
	}

	m_sizeOfToiEventQueue = info.m_sizeOfToiEventQueue;

	markForWrite();


	if (sdkversion != HAVOK_SDK_VERSION_NUMBER)
	{
		HK_ERROR(0x53c94b42, "** Havok libs built with version [" << HAVOK_SDK_VERSION_NUMBER << "], used with code built with [" << sdkversion << "]. **");
	}


	m_gravity = info.m_gravity;


	//
	// Used to be in hkpWorld::updateFromCinfo
	//



	// Initialize welding information
	hkpWeldingUtility::initWeldingTable( info.m_snapCollisionToConvexEdgeThreshold, info.m_snapCollisionToConcaveEdgeThreshold );
	// Set the values here just for getCinfo
	m_snapCollisionToConvexEdgeThreshold = info.m_snapCollisionToConvexEdgeThreshold;
	m_snapCollisionToConcaveEdgeThreshold = info.m_snapCollisionToConvexEdgeThreshold;
	m_enableToiWeldRejection = info.m_enableToiWeldRejection;



	// activation upon hkpRigidBody::set Position/Rotation/Transform
	m_shouldActivateOnRigidBodyTransformChange = info.m_shouldActivateOnRigidBodyTransformChange;

	m_toiCollisionResponseRotateNormal = info.m_toiCollisionResponseRotateNormal;
	
	//m_enableForceLimitBreachedSecondaryEventsFromToiSolver = info.m_enableForceLimitBreachedSecondaryEventsFromToiSolver;
	
	m_useCompoundSpuElf = info.m_useCompoundSpuElf;

	HK_ASSERT2( 0x34febc22, info.m_maxSectorsPerMidphaseCollideTask > 0, "maxSectorsPerMidphaseCollideTask must be greater than 0." );
	m_maxSectorsPerMidphaseCollideTask = info.m_maxSectorsPerMidphaseCollideTask;
	HK_ASSERT2( 0x34febc22, info.m_maxSectorsPerNarrowphaseCollideTask > 0, "maxSectorsPerNarrowphaseCollideTask must be greater than 0." );
	m_maxSectorsPerNarrowphaseCollideTask = info.m_maxSectorsPerNarrowphaseCollideTask;
	m_processToisMultithreaded		   = info.m_processToisMultithreaded;	
	HK_ASSERT2( 0x34febc22, info.m_maxEntriesPerToiMidphaseCollideTask > 0, "maxSectorsPerMidphaseToiCollideTask must be greater than 0." );
	m_maxEntriesPerToiMidphaseCollideTask      = info.m_maxEntriesPerToiMidphaseCollideTask;
	HK_ASSERT2( 0x34febc22, info.m_maxEntriesPerToiNarrowphaseCollideTask > 0, "maxSectorsPerNarrowphaseCollideTask must be greater than 0." );
	m_maxEntriesPerToiNarrowphaseCollideTask      = info.m_maxEntriesPerToiNarrowphaseCollideTask;
	m_maxNumToiCollisionPairsSinglethreaded = info.m_maxNumToiCollisionPairsSinglethreaded;

	m_deactivationReferenceDistance = info.m_deactivationReferenceDistance;

	// Solver info initialization
	{
		hkpSolverInfo& si = m_dynamicsStepInfo.m_solverInfo;
		si.setTauAndDamping( info.m_solverTau, info.m_solverDamp );

		// new values
		si.m_contactRestingVelocity = info.m_contactRestingVelocity;
		si.m_numSteps    = info.m_solverIterations;
		si.m_invNumSteps = 1.0f / info.m_solverIterations;

		si.m_numMicroSteps    = info.m_solverMicrosteps;
		si.m_maxConstraintViolationSqrd	  = info.m_maxConstraintViolation * info.m_maxConstraintViolation;
		si.m_invNumMicroSteps = 1.0f / info.m_solverMicrosteps;
		si.m_forceCoherentConstraintOrderingInSolver = info.m_forceCoherentConstraintOrderingInSolver;
		si.m_deactivationNumInactiveFramesSelectFlag[0] = info.m_deactivationNumInactiveFramesSelectFlag0;
		si.m_deactivationNumInactiveFramesSelectFlag[1] = info.m_deactivationNumInactiveFramesSelectFlag1;
		si.m_deactivationIntegrateCounter = info.m_deactivationIntegrateCounter;

		si.m_deltaTime = 0.0f;
		si.m_invDeltaTime = 0.0f;


		const hkReal expectedDeltaTime = 0.016f;
		hkReal gravity = info.m_gravity.length<3>().getReal();
		if ( gravity == 0.0f )
		{
			gravity = 9.81f;
		}

		const hkReal averageObjectSize = gravity * 0.1f;

		for ( int i = 0; i < hkpSolverInfo::DEACTIVATION_CLASSES_END; i++)
		{
			hkReal relVelocityThres;  // relative to gravity*1sec
			hkReal relDeceleration;	  // factor of the gravity at relVelocityThres
			switch (i)
			{
			case hkpSolverInfo::DEACTIVATION_CLASS_INVALID:
			case hkpSolverInfo::DEACTIVATION_CLASS_OFF:
				relVelocityThres   = HK_REAL_EPSILON;
				relDeceleration    = 0.0f;
				break;
			case hkpSolverInfo::DEACTIVATION_CLASS_LOW:
				relVelocityThres   = 0.01f;   // = 10cm/sec
				relDeceleration    = 0.08f;
				break;
			case hkpSolverInfo::DEACTIVATION_CLASS_MEDIUM:
				relVelocityThres   = 0.017f;   // = 17cm/sec
				relDeceleration    = 0.2f;
				break;
			case hkpSolverInfo::DEACTIVATION_CLASS_HIGH:
				relVelocityThres   = 0.02f;   // = 20cm/sec
				relDeceleration    = 0.3f;
				break;
			default:
			case hkpSolverInfo::DEACTIVATION_CLASS_AGGRESSIVE:
				relVelocityThres   = 0.025f;   // = 25cm/sec
				relDeceleration    = 0.4f;
				break;
			}
			hkpSolverInfo::DeactivationInfo& di = si.m_deactivationInfo[i];

			const hkReal velocityThres = gravity * relVelocityThres;

			hkReal deceleration = gravity * relDeceleration / velocityThres;

			di.m_slowObjectVelocityMultiplier = 1.0f - expectedDeltaTime * si.m_invNumSteps * deceleration;

			di.m_linearVelocityThresholdInv = 1.0f / velocityThres;
			di.m_angularVelocityThresholdInv = 1.0f / (averageObjectSize * velocityThres);

			if (relDeceleration > 0)
			{
				di.m_relativeSleepVelocityThreshold = expectedDeltaTime * si.m_invNumSteps / relDeceleration;
			}
			else
			{
				di.m_relativeSleepVelocityThreshold = HK_REAL_MAX / 16.0f;
			}
			hkReal q = info.m_deactivationReferenceDistance;
			di.m_maxDistSqrd[0] =  q      *  q;
			di.m_maxDistSqrd[1] = (q * 4) * (q * 4);
			di.m_maxRotSqrd[0]  = (q * 2) * (q * 2);
			di.m_maxRotSqrd[1]  = (q * 8) * (q * 8);
		}
	}

	//
	// End of code that was in updateFromCinfo
	//


	m_memoryWatchDog = info.m_memoryWatchDog;


	// Simulation islands and deactivation
	m_wantSimulationIslands = info.m_enableSimulationIslands;
	m_wantDeactivation = info.m_enableDeactivation;
	if (!m_wantSimulationIslands && m_wantDeactivation)
	{
		m_wantDeactivation = false;
		HK_WARN(0xad678954, "Cannot use deactivation when not using simulation islands. Deactivation disabled.");
	}
	m_processActionsInSingleThread = info.m_processActionsInSingleThread;
	m_allowIntegrationOfIslandsWithoutConstraintsInASeparateJob = info.m_allowIntegrationOfIslandsWithoutConstraintsInASeparateJob;


	//
	// Collision detection bridge
	//
	{
		m_broadPhaseDispatcher    = new hkpTypedBroadPhaseDispatcher();
		m_phantomBroadPhaseListener = new hkpPhantomBroadPhaseListener();
		m_entityEntityBroadPhaseListener = new hkpEntityEntityBroadPhaseListener(this);
		m_broadPhaseBorderListener = new hkpBroadPhaseBorderListener();

		m_broadPhaseDispatcher->setBroadPhaseListener(m_phantomBroadPhaseListener, hkpWorldObject::BROAD_PHASE_ENTITY,  hkpWorldObject::BROAD_PHASE_PHANTOM);
		m_broadPhaseDispatcher->setBroadPhaseListener(m_phantomBroadPhaseListener, hkpWorldObject::BROAD_PHASE_PHANTOM, hkpWorldObject::BROAD_PHASE_ENTITY);
		m_broadPhaseDispatcher->setBroadPhaseListener(m_phantomBroadPhaseListener, hkpWorldObject::BROAD_PHASE_PHANTOM, hkpWorldObject::BROAD_PHASE_PHANTOM);

		m_broadPhaseDispatcher->setBroadPhaseListener(m_entityEntityBroadPhaseListener, hkpWorldObject::BROAD_PHASE_ENTITY, hkpWorldObject::BROAD_PHASE_ENTITY);

		// Extra five records for the broadphase borders
		m_broadPhaseDispatcher->setBroadPhaseListener(m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_ENTITY, hkpWorldObject::BROAD_PHASE_BORDER);
		m_broadPhaseDispatcher->setBroadPhaseListener(m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_BORDER, hkpWorldObject::BROAD_PHASE_ENTITY);

		// Use border listeners to handle border-phantom overlaps.
		m_broadPhaseDispatcher->setBroadPhaseListener(m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_PHANTOM, hkpWorldObject::BROAD_PHASE_BORDER);
		m_broadPhaseDispatcher->setBroadPhaseListener(m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_BORDER, hkpWorldObject::BROAD_PHASE_PHANTOM);
		m_broadPhaseDispatcher->setBroadPhaseListener(m_broadPhaseBorderListener, hkpWorldObject::BROAD_PHASE_BORDER, hkpWorldObject::BROAD_PHASE_BORDER);

		hkpContactMgrFactory* defaultCmFactory = new hkpSimpleConstraintContactMgr::Factory( this );

		// add a default collision filter (returns always true)
		m_collisionDispatcher = new hkpCollisionDispatcher( hkpNullAgent::createNullAgent, defaultCmFactory );

		defaultCmFactory->removeReference();

		if ( info.m_collisionFilter == HK_NULL )
		{
			m_collisionFilter = new hkpNullCollisionFilter();
		} 
		else
		{
			m_collisionFilter = info.m_collisionFilter;
			m_collisionFilter->addReference();

			// Filter may have to be initialized. e.g. constraintCollisionFilter should be added to world as constraintListener
			m_collisionFilter->init( this );
		}

		if ( info.m_convexListFilter == HK_NULL )
		{
			m_convexListFilter = new hkpDefaultConvexListFilter();
		}
		else
		{
			m_convexListFilter = info.m_convexListFilter;
			m_convexListFilter->addReference();
		}


		m_collisionInput = new hkpProcessCollisionInput;
		hkpProcessCollisionInput& pci = * m_collisionInput;

		pci.m_weldClosestPoints = false;

		pci.m_dispatcher = m_collisionDispatcher;
		pci.m_tolerance = info.m_collisionTolerance;
		pci.m_filter = m_collisionFilter;
		pci.m_convexListFilter = m_convexListFilter;

		m_contactPointGeneration = info.m_contactPointGeneration;
		pci.m_allowToSkipConfirmedCallbacks = info.m_allowToSkipConfirmedCallbacks;

		pci.m_config = new hkpCollisionAgentConfig();
		pci.m_config->m_iterativeLinearCastEarlyOutDistance = info.m_iterativeLinearCastEarlyOutDistance;
		pci.m_config->m_iterativeLinearCastMaxIterations = info.m_iterativeLinearCastMaxIterations;
		pci.m_enableDeprecatedWelding= info.m_enableDeprecatedWelding;

		pci.m_createPredictiveAgents = false;
		pci.m_collisionQualityInfo = pci.m_dispatcher->getCollisionQualityInfo( hkpCollisionDispatcher::COLLISION_QUALITY_PSI );

		pci.m_stepInfo.set( 0,0 );
		//
		// Calculate some basic values needed for 32bit AABBs
		//
		{
			hkVector4 span;		span.setSub( info.m_broadPhaseWorldAabb.m_max, info.m_broadPhaseWorldAabb.m_min);
			hkVector4 spanInv;	spanInv.setReciprocal(span); spanInv.setComponent<3>(hkSimdReal_1);

			pci.m_aabb32Info.m_bitScale.setMul( hkSimdReal::fromFloat(hkAabbUtil::AABB_UINT32_MAX_FVALUE), spanInv );
			pci.m_aabb32Info.m_bitOffsetLow.setNeg<4>( info.m_broadPhaseWorldAabb.m_min );
			hkVector4 rounding; rounding.setMul( hkSimdReal::fromFloat(1.0f/hkAabbUtil::AABB_UINT32_MAX_FVALUE), span);
			pci.m_aabb32Info.m_bitOffsetHigh.setAdd(pci.m_aabb32Info.m_bitOffsetLow, rounding);

			pci.m_aabb32Info.m_bitScale.zeroComponent<3>();
			pci.m_aabb32Info.m_bitOffsetLow.zeroComponent<3>();
			pci.m_aabb32Info.m_bitOffsetHigh.zeroComponent<3>();

			m_broadPhase->set32BitOffsetAndScale(pci.m_aabb32Info.m_bitOffsetLow, pci.m_aabb32Info.m_bitOffsetHigh, pci.m_aabb32Info.m_bitScale);
		}

		hkWorld_setupContactMgrFactories( this, getCollisionDispatcher() );
	}

#if defined(HK_PLATFORM_MULTI_THREAD) && (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)
	// override simulation type
	if ( m_forceMultithreadedSimulation )
	{
		hkpWorldCinfo* worldInfo = const_cast<hkpWorldCinfo*>( &info );
		worldInfo->m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED;
	}
#endif

	m_minDesiredIslandSize = 0;
	{
		m_simulationType = info.m_simulationType;
		switch(m_simulationType)
		{
		case hkpWorldCinfo::SIMULATION_TYPE_DISCRETE:
			{
				HK_ASSERT2( 0x3965f694, hkpSimulation::createDiscrete, "hkpSimulation is not registered" );
				m_simulation = hkpSimulation::createDiscrete( this );
				break;
			}
		case hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS:
			{
				HK_ASSERT2( 0x9d83a22, hkpSimulation::createContinuous, "hkpContinuousSimulation is not registered" );
				m_simulation = hkpSimulation::createContinuous( this );
				break;
			}
		case hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED:
#if defined(HK_PLATFORM_MULTI_THREAD) && (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)
			{
				m_modifyConstraintCriticalSection = new hkCriticalSection( 4000 );
				m_propertyMasterLock = new hkCriticalSection( 4000 );
				HK_ASSERT2( 0x2727d0e1, hkpSimulation::createMultithreaded, "hkpMultithreadedSimulation is not registered" );
				m_simulation = hkpSimulation::createMultithreaded( this );
				m_minDesiredIslandSize = info.m_minDesiredIslandSize;
				break;
			}
#else
			{
				HK_WARN(0xf032ed54, "Multithreaded simulation not supported on this platform. Defaulting to continuous simulation.");
				HK_ASSERT2( 0x9d83a22, hkpSimulation::createContinuous, "hkpContinuousSimulation is not registered" );
				m_simulation = hkpSimulation::createContinuous( this );
				break;
			}

#endif
		default:
			{
				HK_ASSERT2( 0xf032ed54, 0, "Invalid simulation type. Please select a valid type." );
				break;
			}

		}
		m_islandDirtyListCriticalSection = new hkCriticalSection( 4000 );
		m_simulation->m_frameMarkerPsiSnap = info.m_frameMarkerPsiSnap;
	}

	//
	//	Surface qualities
	//
	{
		hkReal gravLen = m_gravity.length<3>().getReal();
		if ( gravLen == 0.0f )
		{
			gravLen = 9.81f;
		}

		HK_ASSERT2( 0xaf2141e0, info.m_numToisTillAllowedPenetrationSimplifiedToi > 1.0f, "m_numToisTillAllowedPenetrationSimplifiedToi has to be > 1.0" );
		HK_ASSERT2( 0xaf2141e1, info.m_numToisTillAllowedPenetrationToi           > 1.0f, "m_numToisTillAllowedPenetrationToi has to be > 1.0" );
		HK_ASSERT2( 0xaf2141e2, info.m_numToisTillAllowedPenetrationToiHigher     > 1.0f, "m_numToisTillAllowedPenetrationToiHigher has to be > 1.0" );
		HK_ASSERT2( 0xaf2141e3, info.m_numToisTillAllowedPenetrationToiForced     > 1.0f, "m_numToisTillAllowedPenetrationToiForced has to be > 1.0" );

		m_numToisTillAllowedPenetrationSimplifiedToi = info.m_numToisTillAllowedPenetrationSimplifiedToi;
		m_numToisTillAllowedPenetrationToi           = info.m_numToisTillAllowedPenetrationToi;
		m_numToisTillAllowedPenetrationToiHigher     = info.m_numToisTillAllowedPenetrationToiHigher;
		m_numToisTillAllowedPenetrationToiForced     = info.m_numToisTillAllowedPenetrationToiForced;

		hkpCollisionDispatcher::InitCollisionQualityInfo input;
		input.m_gravityLength = gravLen;
		input.m_collisionTolerance = m_collisionInput->m_tolerance;
		input.m_minDeltaTime = info.m_expectedMinPsiDeltaTime; // 50 Hz
		input.m_maxLinearVelocity = info.m_expectedMaxLinearVelocity;
		input.m_numToisTillAllowedPenetrationSimplifiedToi = info.m_numToisTillAllowedPenetrationSimplifiedToi;
		input.m_numToisTillAllowedPenetrationToi           = info.m_numToisTillAllowedPenetrationToi;
		input.m_numToisTillAllowedPenetrationToiHigher     = info.m_numToisTillAllowedPenetrationToiHigher;
		input.m_numToisTillAllowedPenetrationToiForced     = info.m_numToisTillAllowedPenetrationToiForced;
		input.m_wantContinuousCollisionDetection = info.m_simulationType >= hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS ;
		input.m_enableNegativeManifoldTims = info.m_contactPointGeneration == hkpWorldCinfo::CONTACT_POINT_REJECT_MANY;
		input.m_enableNegativeToleranceToCreateNon4dContacts = info.m_contactPointGeneration >= hkpWorldCinfo::CONTACT_POINT_REJECT_DUBIOUS;
		input.m_defaultConstraintPriority = hkpConstraintInstance::PRIORITY_PSI;
		input.m_toiConstraintPriority = hkpConstraintInstance::PRIORITY_TOI;
		input.m_toiHigherConstraintPriority = hkpConstraintInstance::PRIORITY_TOI_HIGHER;
		input.m_toiForcedConstraintPriority = hkpConstraintInstance::PRIORITY_TOI_FORCED;
		input.m_enableToiWeldRejection = info.m_enableToiWeldRejection;

		m_collisionDispatcher->initCollisionQualityInfo( input );

		hkpCollisionDispatcher* dis = m_collisionDispatcher;
		m_collisionInput->m_collisionQualityInfo = dis->getCollisionQualityInfo( dis->COLLISION_QUALITY_PSI );

	}


	//
	// Simulation Islands
	//
	{
		m_fixedIsland = new hkpSimulationIsland(this);
		m_fixedIsland->m_storageIndex = HK_INVALID_OBJECT_INDEX;
		m_fixedIsland->m_activeMark = false;
		m_fixedIsland->m_isInActiveIslandsArray = false;
#ifdef HK_DEBUG_MULTI_THREADING
		// we disable this flag for fixed islands
		m_fixedIsland->m_allowIslandLocking = true;
#endif

		if (!m_wantSimulationIslands)
		{
			hkpSimulationIsland* activeIsland = new hkpSimulationIsland(this);
			m_activeSimulationIslands.pushBack( activeIsland );
			activeIsland->m_storageIndex = 0;
		}
	}


	//
	// Add the fixed rigid body
	// NOTE: This rigid body has no shape, and we do not need it to,
	// so we temporarily disable the associated rigid body construction warning
	//
	{
		hkpRigidBodyCinfo rbci;
		rbci.m_motionType = hkpMotion::MOTION_FIXED;
		rbci.m_mass = 0;
		m_fixedRigidBody = new hkpRigidBody( rbci );
		m_fixedRigidBody->m_npData = 0; 
		addEntity( m_fixedRigidBody );
		HK_ON_DETERMINISM_CHECKS_ENABLED( m_fixedIsland->m_uTag = m_fixedRigidBody->m_uid );
		m_fixedRigidBody->removeReference();
	}

	{
		m_dynamicsStepInfo.m_stepInfo.set(0.0f, 1.0f/60.0f);
		m_collisionInput->m_dynamicsInfo = &m_dynamicsStepInfo;
	}

	// note: do not manually unmark the broadphase as this is done in hkpWorld::unmarkForWrite()!
	m_broadPhase->markForWrite();

	//
	//	Broadphase border
	//
	if ( info.m_broadPhaseBorderBehaviour != info.BROADPHASE_BORDER_DO_NOTHING )
	{
		m_broadPhaseBorder = new hkpBroadPhaseBorder( this, info.m_broadPhaseBorderBehaviour, info.m_mtPostponeAndSortBroadPhaseBorderCallbacks );
	}
	else
	{
		m_broadPhaseBorder = HK_NULL;
	}

	m_maintenanceMgr = new hkpDefaultWorldMaintenanceMgr();
	m_maintenanceMgr->init(this);

	// Add a collisionCallbackUtil if necessary.
	if ( info.m_fireCollisionCallbacks )
	{
		hkpCollisionCallbackUtil::requireCollisionCallbackUtil( this );
	}

	unmarkForWrite();
	if ( info.m_simulationType != hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED)
	{
		m_multiThreadCheck.disableChecks();
	}

#if defined(HK_PLATFORM_MULTI_THREAD) && (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)
	m_multithreadedSimulationJobData = new hkpMtThreadStructure();
#endif

	// Verification of initialization of internal solver arrays
	HK_ON_DEBUG( hkpJacobianSchema::verifySchemaInfoArrays() );
}


void hkpWorld::shiftBroadPhase( const hkVector4& shiftDistance, hkVector4& effectiveShiftDistanceOut, CachedAabbUpdate updateAabbs )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_ASSERT2( 0xf0df2dfe, !areCriticalOperationsLocked(), "You cannot call this function from a callback");

	HK_TIMER_BEGIN("Broadphase Shift", HK_NULL);

	hkArray<hkpBroadPhaseHandlePair> newPairs;
	m_broadPhase->shiftBroadPhase(shiftDistance, effectiveShiftDistanceOut, newPairs);
	m_broadPhase->getOffsetLowHigh32bit(m_collisionInput->m_aabb32Info.m_bitOffsetLow, m_collisionInput->m_aabb32Info.m_bitOffsetHigh);

	m_broadPhaseExtents[0].add( effectiveShiftDistanceOut );
	m_broadPhaseExtents[1].add( effectiveShiftDistanceOut );


	lockCriticalOperations();
	m_broadPhaseDispatcher->addPairs( static_cast<hkpTypedBroadPhaseHandlePair*>(newPairs.begin()), newPairs.getSize(), getCollisionFilter() );


	if (m_broadPhaseBorder)
	{
		// upate AABB of broadphase border phantoms
		hkpPhantom** phantoms =  m_broadPhaseBorder->m_phantoms;
		for ( int i = 0; i < 6; i++ )
		{
			hkpPhantom* phantom = phantoms[ i ];
			switch( phantom->getType() )
			{
			case HK_PHANTOM_AABB:
				{
					hkpAabbPhantom* aabbPhantom = static_cast<hkpAabbPhantom*>( phantom );
					// Now do a bad trick to move the AABB.
					hkAabb& aabb = const_cast<hkAabb&>( aabbPhantom->getAabb() );
					aabb.m_min.add( effectiveShiftDistanceOut );
					aabb.m_max.add( effectiveShiftDistanceOut );
					break;
				}
			case HK_PHANTOM_SIMPLE_SHAPE:
			case HK_PHANTOM_CACHING_SHAPE:
				{
					hkpShapePhantom* shapePhantom = static_cast<hkpShapePhantom*>( phantom );
					hkTransform& transform = const_cast<hkTransform&>( shapePhantom->getTransform());
					transform.getTranslation().add( effectiveShiftDistanceOut );
					break;
				}
			default:
				HK_ASSERT2(0xf041604,0,"Unknown Phantom Type" );
			}
		}
	}
	else
	{
		HK_WARN(0xad906071, "No broad phase border exists. You need to handle out-of-broadphase objects manaully");
	}

	if ( updateAabbs == SHIFT_BROADPHASE_UPDATE_ENTITY_AABBS )
	{
		hkArray<hkpEntity*> bodies;

		const hkpSimulationIsland* fixed = getFixedIsland();
		const hkArray<hkpSimulationIsland*>& active = getActiveSimulationIslands();
		const hkArray<hkpSimulationIsland*>& inactive = getInactiveSimulationIslands();

		// Merge all islands into one array
		hkArray<const hkpSimulationIsland*> allIslands(active.getSize() + inactive.getSize() + 1);

		int numFixed = fixed != HK_NULL ? 1 : 0;
		hkArray<const hkpSimulationIsland*>::copy(allIslands.begin(), fixed != HK_NULL ? &fixed : HK_NULL, numFixed);
		hkArray<const hkpSimulationIsland*>::copy(allIslands.begin() + numFixed, active.begin(), active.getSize());
		hkArray<const hkpSimulationIsland*>::copy(allIslands.begin() + numFixed + active.getSize(), inactive.begin(), inactive.getSize());

		// Collect all entities in islands with a valid shape
		for (int i = 0; i < allIslands.getSize(); i++ )
		{
			const hkpSimulationIsland* island = allIslands[i];
			for (int b = 0; b < island->getEntities().getSize(); b++ )
			{
				if( island->getEntities()[b]->m_collidable.m_shape)
				{
					bodies.pushBack(island->getEntities()[b]);
				}
			}
		}

		hkpEntityAabbUtil::entityBatchRecalcAabb(m_collisionInput, bodies.begin(), bodies.getSize());
	}

	unlockAndAttemptToExecutePendingOperations();

	HK_TIMER_END();
}


void hkpWorld::setCollisionFilter( hkpCollisionFilter* filter,
								  hkBool             runUpdateCollisionFilterOnWorld,
								  hkpUpdateCollisionFilterOnWorldMode          updateMode,
								  hkpUpdateCollectionFilterMode updateShapeCollectionFilter )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	if ( filter == HK_NULL )
	{
		filter = new hkpNullCollisionFilter();
	}
	else
	{
		filter->addReference();
	}

	if ( m_collisionFilter )
	{
		m_collisionFilter->removeReference();
	}

	m_collisionFilter = filter;
	m_collisionInput->m_filter = m_collisionFilter;

	m_collisionFilter->init( this );

	{
		if (runUpdateCollisionFilterOnWorld)
		{
			updateCollisionFilterOnWorld(updateMode, updateShapeCollectionFilter);
		}
		else
		{
#if defined(HK_DEBUG)
			if ( ( getActiveSimulationIslands().getSize() != 0) || ( m_inactiveSimulationIslands.getSize() != 0 ) )
			{
				HK_WARN(0x4a5454cb, "You are setting the collision filter after adding entities. Collisions between these entities will not have been filtered correctly."\
					" You can use hkpWorld::updateCollisionFilter() to make sure the entities are filtered according to the new filter");
			}
#endif
		}
	}
}

void hkpWorld::checkAccessGetActiveSimulationIslands() const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	if ( !(this->m_multiThreadCheck.isMarkedForWrite() || this->m_multiThreadCheck.isMarkedForReadRecursive()))
	{
		// if we are in multi threaded mode, we need the job queue to be locked
		HK_ASSERT( 0xf0232344, m_simulationType == hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED);
		//HK_ON_DEBUG(const hkpMultiThreadedSimulation* mts = static_cast<const hkpMultiThreadedSimulation*>(m_simulation));
		//HK_ASSERT2( 0xf0232345, mts->m_jobQueue.m_criticalSection.isEntered(), "In the multithreaded section of hkpWorld, you cannot use getActiveSimulationIslands()"  );
	}
}

void hkpWorld::setBroadPhaseBorder( hkpBroadPhaseBorder* b )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	if ( m_broadPhaseBorder )
	{
		m_broadPhaseBorder->deactivate();
		m_broadPhaseBorder->removeReference();
	}

	m_broadPhaseBorder = b;
	if ( b )
	{
		b->addReference();
	}
}

hkpBroadPhaseBorder* hkpWorld::getBroadPhaseBorder() const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	return m_broadPhaseBorder;
}


void hkpWorld::castRay( const hkpWorldRayCastInput& input, hkpRayHitCollector& collector ) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	HK_TIME_CODE_BLOCK("worldCastRayCollector",HK_NULL);
	hkpWorldRayCaster rayCaster;
	rayCaster.castRay( *getBroadPhase(), input, getCollisionFilter(), HK_NULL, collector );
}

void hkpWorld::castRay( const hkpWorldRayCastInput& input, hkpWorldRayCastOutput& output ) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	HK_TIME_CODE_BLOCK("worldCastRay",HK_NULL);
	hkpSimpleWorldRayCaster rayCaster;
	rayCaster.castRay( *getBroadPhase(), input, getCollisionFilter(), HK_NULL, output );
}

void hkpWorld::castRayMt( hkpWorldRayCastCommand* commandArray, int numCommands, hkJobQueue* jobQueue, hkJobThreadPool* jobThreadPool, hkSemaphoreBusyWait*	semaphore, int numCommandsPerJob ) const
{
	hkpCollisionQueryJobHeader* jobHeader = hkAllocateChunk<hkpCollisionQueryJobHeader>( 1, HK_MEMORY_CLASS_DEMO );
	hkpWorldRayCastJob worldRayCastJobBase( getCollisionInput(), jobHeader, commandArray, numCommands, m_broadPhase, semaphore );
	worldRayCastJobBase.setRunsOnSpuOrPpu();

	while( worldRayCastJobBase.m_numCommands > 0 )
	{
		hkpWorldRayCastJob worldRaycastJob = worldRayCastJobBase;
		worldRaycastJob.m_numCommands = hkMath::min2( worldRayCastJobBase.m_numCommands, numCommandsPerJob );
		worldRayCastJobBase.m_numCommands -= numCommandsPerJob;
		worldRayCastJobBase.m_commandArray += numCommandsPerJob;

		jobQueue->addJob( worldRaycastJob, hkJobQueue::JOB_LOW_PRIORITY );
	}

	jobThreadPool->processAllJobs( jobQueue );
	jobQueue->processAllJobs();
	jobThreadPool->waitForCompletion();
	semaphore->acquire();

	hkDeallocateChunk( jobHeader, 1, HK_MEMORY_CLASS_DEMO );
}

void hkpWorld::getClosestPoints( const hkpCollidable* collA, const hkpCollisionInput& input, hkpCdPointCollector& collector) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );

	hkAabb aabb;
	{
		// the AABBs in the broadphase are already expanded by getCollisionInput()->getTolerance() * 0.5f, so we only have to
		// increase our AABB by the restTolerance
		hkReal restTolerance = input.getTolerance() - getCollisionInput()->getTolerance() * 0.5f;
		collA->getShape()->getAabb( collA->getTransform(), restTolerance, aabb );
	}

	HK_TIMER_BEGIN_LIST("hkpWorld::getClosestPoints", "BroadPhase");
	hkInplaceArray<hkpBroadPhaseHandlePair,128> hits;
	m_broadPhase->querySingleAabb( aabb, hits );

	HK_TIMER_SPLIT_LIST("NarrowPhase");
	const hkpShapeType typeA = collA->getShape()->getType();
	hkpBroadPhaseHandlePair* p = hits.begin();
	for (int i = hits.getSize() -1; i>=0; p++, i--)
	{
		const hkpTypedBroadPhaseHandle* tp = static_cast<const hkpTypedBroadPhaseHandle*>( p->m_b );
		const hkpCollidable* collB = static_cast<hkpCollidable*>(tp->getOwner());
		if ( collA == collB )
		{
			continue;
		}

		if ( !getCollisionFilter()->isCollisionEnabled( *collA, *collB ))
		{
			continue;
		}

		const hkpShape* shapeB = collB->getShape();
		if ( !shapeB )
		{
			continue;
		}

		hkpShapeType typeB = shapeB->getType();

		hkpCollisionDispatcher::GetClosestPointsFunc getClosestPointFunc = input.m_dispatcher->getGetClosestPointsFunc( typeA, typeB );
		getClosestPointFunc( *collA, *collB, input, collector );
	}

	HK_TIMER_END_LIST();
}

void hkpWorld::getClosestPointsMt(hkpWorldGetClosestPointsCommand* commandArray, int numCommands, hkJobQueue* jobQueue, hkJobThreadPool* jobThreadPool,
								  hkSemaphoreBusyWait* semaphore, int numCommandsPerJob) const
{
	hkpCollisionQueryJobHeader* jobHeader = hkAllocateChunk<hkpCollisionQueryJobHeader>(1, HK_MEMORY_CLASS_DEMO);
	hkpWorldGetClosestPointsJob baseJob(getCollisionInput(), jobHeader, commandArray, numCommands, m_broadPhase, getCollisionInput()->getTolerance(), semaphore);
	baseJob.setRunsOnSpuOrPpu();

	while( baseJob.m_numCommands > 0 )
	{
		hkpWorldGetClosestPointsJob job = baseJob;
		job.m_numCommands = hkMath::min2( baseJob.m_numCommands, numCommandsPerJob );
		baseJob.m_numCommands -= numCommandsPerJob;
		baseJob.m_commandArray += numCommandsPerJob;

		jobQueue->addJob( job, hkJobQueue::JOB_LOW_PRIORITY );
	}

	jobThreadPool->processAllJobs( jobQueue );
	jobQueue->processAllJobs();
	jobThreadPool->waitForCompletion();
	semaphore->acquire();

	hkDeallocateChunk( jobHeader, 1, HK_MEMORY_CLASS_DEMO );
}

void hkpWorld::getPenetrations( const hkpCollidable* collA, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector ) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );

	hkAabb aabb;
	{
		hkReal restTolerance = hkMath::max2( hkReal(0.0f), input.getTolerance() - getCollisionInput()->getTolerance() * 0.5f );
		collA->getShape()->getAabb( collA->getTransform(), restTolerance, aabb );
	}

	HK_TIMER_BEGIN_LIST("hkpWorld::getPenetrations", "BroadPhase" );
	hkInplaceArray<hkpBroadPhaseHandlePair,128> hits;
	m_broadPhase->querySingleAabb( aabb, hits );

	HK_TIMER_SPLIT_LIST("NarrowPhase")
	const hkpShapeType typeA = collA->getShape()->getType();
	hkpBroadPhaseHandlePair* p = hits.begin();
	for (int i = hits.getSize() -1; i>=0; p++, i--)
	{
		const hkpTypedBroadPhaseHandle* tp = static_cast<const hkpTypedBroadPhaseHandle*>( p->m_b );
		const hkpCollidable* collB = static_cast<hkpCollidable*>(tp->getOwner());

		if ( collA == collB )
		{
			continue;
		}

		if ( !getCollisionFilter()->isCollisionEnabled( *collA, *collB ))
		{
			continue;
		}

		const hkpShape* shapeB = collB->getShape();
		if ( !shapeB )
		{
			continue;
		}

		hkpShapeType typeB = shapeB->getType();

		hkpCollisionDispatcher::GetPenetrationsFunc getPenetrationsFunc = input.m_dispatcher->getGetPenetrationsFunc( typeA, typeB );
		getPenetrationsFunc( *collA, *collB, input, collector );
		if ( collector.getEarlyOut() )
		{
			break;
		}
	}

	HK_TIMER_END_LIST();
}

void hkpWorld::linearCast( const hkpCollidable* collA, const hkpLinearCastInput& input, hkpCdPointCollector& castCollector, hkpCdPointCollector* startPointCollector ) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	HK_TIME_CODE_BLOCK("worldLinCast",HK_NULL);
	hkpWorldLinearCaster linearCaster;
	hkpBroadPhaseAabbCache* broadPhaseCache = HK_NULL;
	linearCaster.linearCast( *getBroadPhase(), collA, input, getCollisionFilter(), *getCollisionInput(), getCollisionInput()->m_config, broadPhaseCache, castCollector, startPointCollector );
}

void hkpWorld::linearCastMt( hkpWorldLinearCastCommand* commandArray, int numCommands, hkJobQueue* jobQueue, hkJobThreadPool* jobThreadPool,
							hkSemaphoreBusyWait* semaphore, int numCommandsPerJob) const
{
	hkpCollisionQueryJobHeader* jobHeader = hkAllocateChunk<hkpCollisionQueryJobHeader>(1, HK_MEMORY_CLASS_DEMO);
	hkpWorldLinearCastJob worldLinearCastJobBase(getCollisionInput(), jobHeader, commandArray, numCommands, m_broadPhase, semaphore);
	worldLinearCastJobBase.setRunsOnSpuOrPpu();
	
	while ( worldLinearCastJobBase.m_numCommands > 0 )
	{
		hkpWorldLinearCastJob  worldLinearCastJob = worldLinearCastJobBase;
		worldLinearCastJob.m_numCommands = hkMath::min2( worldLinearCastJobBase.m_numCommands, numCommandsPerJob);
		worldLinearCastJobBase.m_numCommands -= numCommandsPerJob;
		worldLinearCastJobBase.m_commandArray += numCommandsPerJob;

		jobQueue->addJob( worldLinearCastJob, hkJobQueue::JOB_LOW_PRIORITY);
	}

	jobThreadPool->processAllJobs( jobQueue );
	jobQueue->processAllJobs();
	jobThreadPool->waitForCompletion();
	semaphore->acquire();

	hkDeallocateChunk(jobHeader, 1, HK_MEMORY_CLASS_DEMO);
}

/////////////////////////////////////////////////////////////
//
// Serialization / systems support.
//
/////////////////////////////////////////////////////////////

static hkBool HK_CALL enumerateAllInactiveEntitiesInWorld(const hkpWorld* world, hkpPhysicsSystem* sys)
{
	hkBool hasFixedEntities = false;
	if (world->getFixedIsland())
	{
		const hkArray<hkpEntity*>& e = world->getFixedIsland()->getEntities();
		for (int ei=0; ei < e.getSize(); ++ei)
		{
			// Sometimes we have a dummy rigid body in the island, so we ignore it
			if ( ei == 0 && e[ei]->getCollidable()->getShape() == HK_NULL )
			{
				continue;
			}
			sys->addRigidBody( static_cast<hkpRigidBody*>( e[ei] ) );
		}
		hasFixedEntities = e.getSize() > 0;
	}

	hkArray<hkpSimulationIsland*>::const_iterator si_it;
	const hkArray<hkpSimulationIsland*>& inactive_sim_islands = world->getInactiveSimulationIslands();
	for ( si_it = inactive_sim_islands.begin(); si_it != inactive_sim_islands.end(); si_it++)
	{
		const hkArray<hkpEntity*>& e = (*si_it)->getEntities();
		for (int ei=0; ei < e.getSize(); ++ei)
		{
			sys->addRigidBody( static_cast<hkpRigidBody*>( e[ei] ) );
		}
	}
	return inactive_sim_islands.getSize() > 0 || hasFixedEntities;
}

static hkBool HK_CALL enumerateAllActiveEntitiesInWorld(const hkpWorld* world, hkpPhysicsSystem* sys)
{
	hkArray<hkpSimulationIsland*>::const_iterator si_it;
	const hkArray<hkpSimulationIsland*>& active_sim_islands = world->getActiveSimulationIslands();
	// set active state of the physics system if active rigid bodies were found
	for ( si_it = active_sim_islands.begin(); si_it != active_sim_islands.end(); si_it++)
	{
		const hkArray<hkpEntity*>& e = (*si_it)->getEntities();
		for (int ei=0; ei < e.getSize(); ++ei)
		{
			sys->addRigidBody( static_cast<hkpRigidBody*>( e[ei] ) );
		}
	}
	return active_sim_islands.getSize() > 0;
}

static void HK_CALL enumerateAllEntitiesInWorld(const hkpWorld* world, hkpPhysicsSystem* sys)
{
	enumerateAllInactiveEntitiesInWorld( world, sys );
	sys->setActive(enumerateAllActiveEntitiesInWorld( world, sys ));
}

static void enumerateAllConstraintsInIsland( hkpSimulationIsland* island, hkpPhysicsSystem* sys)
{
	for (int e = 0; e < island->getEntities().getSize(); ++e)
	{
		hkpEntity* entity = island->getEntities()[e];

		const hkSmallArray<struct hkConstraintInternal>&  constraintMasters = entity->getConstraintMasters();

		for ( int c = 0; c < constraintMasters.getSize(); c++)
		{
			const hkConstraintInternal* ci = &constraintMasters[c];
			hkpConstraintAtom* atom = hkpWorldConstraintUtil::getTerminalAtom(ci);
			hkpConstraintAtom::AtomType type = atom->getType();
			if (type != hkpConstraintAtom::TYPE_CONTACT )
			{
				sys->addConstraint( ci->m_constraint );
			}
		}
	}
}

static void enumerateAllConstraintsInWorld(const hkpWorld* world, hkpPhysicsSystem* sys)
{
	// Get the list from each island.
	// A constraint can not exist in two islands so we can just add them all inti one big list without checking for duplicates.
	hkArray<hkpSimulationIsland*>::const_iterator si_it;
	const hkArray<hkpSimulationIsland*>& active_sim_islands = world->getActiveSimulationIslands();
	for ( si_it = active_sim_islands.begin(); si_it != active_sim_islands.end(); si_it++)
	{
		enumerateAllConstraintsInIsland( (*si_it), sys );
	}

	const hkArray<hkpSimulationIsland*>& inactive_sim_islands = world->getInactiveSimulationIslands();
	for ( si_it = inactive_sim_islands.begin(); si_it != inactive_sim_islands.end(); si_it++)
	{
		enumerateAllConstraintsInIsland( (*si_it), sys );
	}
}

static void enumerateAllActionsInWorld(const hkpWorld* world, hkpPhysicsSystem* sys)
{
	// Get the list from each island.
	// A constraint can not exist in two islands so we can just add them all inti one big list withut checking for duplicates.
	hkArray<hkpSimulationIsland*>::const_iterator si_it;
	const hkArray<hkpSimulationIsland*>& active_sim_islands = world->getActiveSimulationIslands();
	for ( si_it = active_sim_islands.begin(); si_it != active_sim_islands.end(); si_it++)
	{
		const hkArray<hkpAction*>& a = (*si_it)->getActions();
		for (int ai=0; ai < a.getSize(); ++ai)
		{
			sys->addAction( a[ai] );
		}
	}

	const hkArray<hkpSimulationIsland*>& inactive_sim_islands = world->getInactiveSimulationIslands();
	for ( si_it = inactive_sim_islands.begin(); si_it != inactive_sim_islands.end(); si_it++)
	{
		const hkArray<hkpAction*>& a = (*si_it)->getActions();
		for (int ai=0; ai < a.getSize(); ++ai)
		{
			sys->addAction( a[ai] );
		}
	}
}

static void enumerateAllPhantomsInWorld(const hkpWorld* world, hkpPhysicsSystem* sys)
{
	hkpBroadPhaseBorder* border = world->getBroadPhaseBorder();
	const hkArray<hkpPhantom*>& phantoms = world->getPhantoms();
	if ( border )
	{
		for (int pi=0; pi < phantoms.getSize(); ++pi)
		{
			hkpPhantom* p = phantoms[pi];
			if ( p == border->m_phantoms[0] ) continue;
			if ( p == border->m_phantoms[1] ) continue;
			if ( p == border->m_phantoms[2] ) continue;
			if ( p == border->m_phantoms[3] ) continue;
			if ( p == border->m_phantoms[4] ) continue;
			if ( p == border->m_phantoms[5] ) continue;
			sys->addPhantom(p);
		}
	}
	else
	{
		for (int pi=0; pi < phantoms.getSize(); ++pi)
		{
			sys->addPhantom(phantoms[pi]);
		}
	}
}



hkpPhysicsSystem* hkpWorld::getWorldAsOneSystem() const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	hkpPhysicsSystem* sys = new hkpPhysicsSystem();

	//
	// rigid bodies
	//
	enumerateAllEntitiesInWorld(this, sys);
	enumerateAllPhantomsInWorld(this, sys);

	// Constraints and Actions
	enumerateAllConstraintsInWorld( this, sys);
	enumerateAllActionsInWorld( this, sys);

	return sys;
}

void hkpWorld::getWorldAsSystems(hkArray<hkpPhysicsSystem*>& systemsInOut) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	//
	// fixed/inactive rigid bodies
	//
	hkpPhysicsSystem* inactive = new hkpPhysicsSystem();
	if (enumerateAllInactiveEntitiesInWorld( this, inactive ))
	{
		inactive->setActive(false);
		systemsInOut.pushBack(inactive);
		inactive = HK_NULL;
	}

	// reuse inactive if it was not used above
	hkpPhysicsSystem* active = inactive ? inactive : new hkpPhysicsSystem();
	systemsInOut.pushBack(active);

	//
	// active rigid bodies
	//
	enumerateAllActiveEntitiesInWorld( this, active );
	enumerateAllPhantomsInWorld( this, active );

	// Constraints and Actions
	enumerateAllConstraintsInWorld( this, active );
	enumerateAllActionsInWorld( this, active );
}

/////////////////////////////////////////////////////////////
//
//  Locking the world, and delaying worldOperations
//
/////////////////////////////////////////////////////////////

void hkpWorld::internal_executePendingOperations()
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_INTERNAL_TIMER_BEGIN("WorldOps", this);
	m_pendingOperationsCount = 0;
	m_pendingOperations->executeAllPending();
	HK_INTERNAL_TIMER_END();
}

void hkpWorld::internal_executePendingBodyOperations()
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	HK_INTERNAL_TIMER_BEGIN("BodyOps", this);
	m_pendingBodyOperationsCount = 0;
	m_pendingOperations->executeAllPendingBodyOperations();
	HK_INTERNAL_TIMER_END();
}

void hkpWorld::checkConstraintsViolated()
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	int numViolatedConstraints = m_violatedConstraintArray->m_nextFreeElement;
	if ( numViolatedConstraints >= hkpViolatedConstraintArray::CAPACITY)
	{
		numViolatedConstraints = hkpViolatedConstraintArray::CAPACITY;
		HK_WARN_ONCE(0x34fe3455, "Running out of violated constraint buffer, this will lead to nondeterministic behavior");
	}
	for(int i=0; i<numViolatedConstraints; ++i)
	{
		hkpWorldCallbackUtil::fireConstraintViolated(this,m_violatedConstraintArray->m_constraints[i]);
	}
	m_violatedConstraintArray->reset();
}

void hkpWorld::queueOperation(const hkWorldOperation::BaseOperation& operation)
{
	m_pendingOperations->queueOperation(operation);
}

void hkpWorld::addBodyOperation(hkpRigidBody* breakingBody, hkpBodyOperation* operation, int priority, /*hkpBodyOperation::ExecutionState*/int hint)
{
	m_pendingOperations->queueBodyOperation(breakingBody, operation, priority, (hkpBodyOperation::ExecutionState)hint);
}

hkWorldOperation::UserCallback* hkpWorld::queueCallback(hkWorldOperation::UserCallback* callback, hkUlong userData)
{
	if (areCriticalOperationsLocked())
	{
		hkWorldOperation::UserCallbackOperation operation;
		operation.m_userCallback = callback;
		operation.m_userData = userData;
		queueOperation(operation);
		return callback;
	}

	callback->worldOperationUserCallback(userData);
	return callback;
}


void hkpWorld::findInitialContactPoints( hkpEntity** entities, int numEntities )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	HK_ON_DEBUG( m_simulation->assertThereIsNoCollisionInformationForEntities(entities, numEntities, this) );

	hkStepInfo stepInfo( m_simulation->getCurrentPsiTime(), m_simulation->getCurrentPsiTime() );
	m_simulation->collideEntitiesDiscrete( entities, numEntities, this, stepInfo, hkpSimulation::FIND_CONTACTS_EXTRA );
}

void hkpWorld::findInitialContactPointsOfAllEntities()
{
	const hkArray<hkpSimulationIsland*>& activeIslands = getActiveSimulationIslands();
	for (int i = 0; i < activeIslands.getSize(); i++)
	{
		findInitialContactPoints( activeIslands[i]->m_entities.begin(), activeIslands[i]->m_entities.getSize());
	}
	for (int i = 0; i < m_inactiveSimulationIslands.getSize(); i++)
	{
		findInitialContactPoints( m_inactiveSimulationIslands[i]->m_entities.begin(), m_inactiveSimulationIslands[i]->m_entities.getSize());
	}
}


void hkpWorld::calcRequiredSolverBufferSize( hkWorldMemoryAvailableWatchDog::MemUsageInfo& infoOut )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );

	// clean up the islands before computing the memory usage
	hkpWorldOperationUtil::cleanupDirtyIslands( this ); 

	// get the sum and the max of the memory usages per each island
	infoOut.m_maxRuntimeBlockSize = 0;
	infoOut.m_sumRuntimeBlockSize = 0;
	infoOut.m_largestSimulationIsland = HK_NULL;

	const hkArray<hkpSimulationIsland*>& activeIslands = getActiveSimulationIslands();

	for (int i = activeIslands.getSize()-1; i>=0; i--)
	{
		hkpSimulationIsland* activeIsland = activeIslands[i];
		HK_ASSERT(0x367e587f,  activeIsland->m_storageIndex == i );

		int memUsage = activeIsland->getMemUsageForIntegration();
		infoOut.m_sumRuntimeBlockSize += memUsage;

		if ( memUsage > infoOut.m_maxRuntimeBlockSize )
		{
			infoOut.m_maxRuntimeBlockSize = memUsage;
			infoOut.m_largestSimulationIsland = activeIsland;
		}
	}

	
	// include the memory needed by the TOIs to the total buffer size
	if ( m_simulationType == hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS || m_simulationType == hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED )
	{
		int scratchpadSize = static_cast<hkpContinuousSimulation*>( m_simulation )->m_toiResourceMgr->getScratchpadCapacity();
		// we're taking the maximum since nothing else exists in the buffer when TOIs are solved 
		infoOut.m_sumRuntimeBlockSize = hkMath::max2( infoOut.m_sumRuntimeBlockSize, scratchpadSize );
	}
}


void hkpWorld::lock()
{
	//m_worldLock->enter();
	hkReferencedObject::lockAll();
	markForWrite();
	m_isLocked++;
}

void hkpWorld::unlock()
{
	m_isLocked--;
	unmarkForWrite();
	//m_worldLock->leave();
	hkReferencedObject::unlockAll();
}

void hkpWorld::lockReadOnly()
{
	markForRead();
}

void hkpWorld::unlockReadOnly()
{
	unmarkForRead();
}

bool hkpWorld::checkUnmarked()
{
#if defined(HK_DEBUG)
	if ( m_multiThreadCheck.isCheckingEnabled() )
	{
		if ( m_multiThreadCheck.isMarkedForWrite() || m_multiThreadCheck.isMarkedForReadRecursive() )
		{
			HK_ASSERT2( 0xf032e323, false, "You cannot have a lock on the world now");
		}
	}
#endif
	return true;
}



void hkpWorld::lockIslandForConstraintUpdate( hkpSimulationIsland* island )
{
	if ( !m_modifyConstraintCriticalSection )
	{
		return;
	}
	if (!m_multiThreadCheck.isMarkedForWrite() )
	{
		HK_ASSERT2( 0xf02134ed, island->m_allowIslandLocking, "You can only call this function during collision callbacks or when the world is locked");
	}

	m_modifyConstraintCriticalSection->enter();
	HK_ON_DEBUG_MULTI_THREADING(island->markForWrite());
	HK_ON_DEBUG_MULTI_THREADING(m_fixedIsland->markForWrite());
}

void hkpWorld::lockForIslandSplit( hkpSimulationIsland* island )
{
	if ( !m_modifyConstraintCriticalSection )
	{
		return;
	}

	m_modifyConstraintCriticalSection->enter();
	HK_ON_DEBUG_MULTI_THREADING(island->markForWrite());
	HK_ON_DEBUG_MULTI_THREADING(m_fixedIsland->markForWrite());
}

void hkpWorld::unlockIslandForConstraintUpdate( hkpSimulationIsland* island )
{
	if ( !m_modifyConstraintCriticalSection )
	{
		return;
	}
	if (!m_multiThreadCheck.isMarkedForWrite() )
	{
		HK_ASSERT2( 0xf02134fd, island->m_allowIslandLocking, "You can only call this function during collision callbacks or when the world is locked");
	}
	island->unmarkForWrite();
	m_fixedIsland->unmarkForWrite();
	m_modifyConstraintCriticalSection->leave();
}

void hkpWorld::unlockForIslandSplit( hkpSimulationIsland* island )
{
	if ( !m_modifyConstraintCriticalSection )
	{
		return;
	}

	island->unmarkForWrite();
	m_fixedIsland->unmarkForWrite();
	m_modifyConstraintCriticalSection->leave();
}

void hkpWorld::setMultithreadedAccessChecking( MtAccessChecking accessCheckState )
{
	if ( MT_ACCESS_CHECKING_ENABLED == accessCheckState )
	{
		m_multiThreadCheck.enableChecks();
		m_broadPhase->getMultiThreadCheck().enableChecks();
	}
	else
	{
		m_multiThreadCheck.disableChecks();
		m_broadPhase->getMultiThreadCheck().disableChecks();
	}
}

hkpWorld::MtAccessChecking hkpWorld::getMultithreadedAccessChecking() const
{
	HK_ASSERT2(0xad906241, m_multiThreadCheck.isCheckingEnabled() == m_broadPhase->getMultiThreadCheck().isCheckingEnabled(), "hkpWorld's and hkpBroadPhase's m_multiThreadedChecks enable/disable state is not the same.");
	if ( m_multiThreadCheck.isCheckingEnabled() )
	{
		return MT_ACCESS_CHECKING_ENABLED;
	}
	else
	{
		return MT_ACCESS_CHECKING_DISABLED;
	}
}


#ifdef HK_DEBUG_MULTI_THREADING
void hkpWorld::markForRead( ) const
{
	m_multiThreadCheck.markForRead();
	m_broadPhase->markForRead();
}

void hkpWorld::markForWrite( )
{
	m_multiThreadCheck.markForWrite();
	m_broadPhase->markForWrite();
}

void hkpWorld::unmarkForRead( ) const
{
	m_multiThreadCheck.unmarkForRead();
	m_broadPhase->unmarkForRead();
}

void hkpWorld::unmarkForWrite()
{
	m_broadPhase->unmarkForWrite();
	m_multiThreadCheck.unmarkForWrite();
}
#endif

#ifdef HK_ENABLE_DETERMINISM_CHECKS
void hkpWorld::checkDeterminismInAgentNnTracks(const hkpWorld* world)
{
	hkArray<hkpSimulationIsland*> allIslands;
	allIslands.insertAt(0, world->getActiveSimulationIslands().begin(), world->getActiveSimulationIslands().getSize());
	allIslands.insertAt(allIslands.getSize(), world->getInactiveSimulationIslands().begin(), world->getInactiveSimulationIslands().getSize());
	for (int i = 0; i < allIslands.getSize(); i++)
	{
		checkDeterminismInAgentNnTracks(allIslands[i]);
	}
}

void hkpWorld::checkDeterminismInAgentNnTracks(const hkpSimulationIsland* island)
{
	const hkpAgentNnTrack* tracks[2] = { &island->m_narrowphaseAgentTrack, &island->m_midphaseAgentTrack };
	
	for ( int j = 0; j < 2; ++j )
	{
		const hkpAgentNnTrack *const track = tracks[j];
		//const hkpAgentNnTrack& track = island->m_narrowphaseAgentTrack;
		for (int i = 0; i < track->m_sectors.getSize(); i++)
		{
			hkpAgentNnSector* sector = track->m_sectors[i];
			hkpAgentNnEntry* sectorEnd = hkAddByteOffset( sector->getBegin(), track->getSectorSize( i ) );
			for (hkpAgentNnEntry* entry = sector->getBegin(); entry < sectorEnd; entry = hkAddByteOffset(entry, track->getSectorSize(0)))
			{
				hkpRigidBody* rba = hkpGetRigidBody(entry->m_collidable[0]);
				hkpRigidBody* rbb = hkpGetRigidBody(entry->m_collidable[1]);

				hkCheckDeterminismUtil::checkMt( 0xf00001a8, rba->m_uid);
				hkCheckDeterminismUtil::checkMt( 0xf00001a9, rbb->m_uid);
			}
		}
	}
}
#endif 

#if defined HK_ENABLE_INTERNAL_DATA_RANDOMIZATION

static hkPseudoRandomGenerator prng(1);

void hkpWorld::randomizeInternalState()
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	for (int fi = 0; fi < getFixedIsland()->m_entities.getSize(); fi++)
	{
		hkpEntity* e = getFixedIsland()->m_entities[fi];

		// Randomize constraint slaves list of fixed bodies.

		for (int si = 0; si < e->m_constraintsSlave.getSize(); si++)
		{
			int newIndex = (int)prng.getRandRange(0.0f, (hkReal)e->m_constraintsSlave.getSize()-0.01f);

			hkAlgorithm::swap(e->m_constraintsSlave[si], e->m_constraintsSlave[newIndex]);
			// link masters to new slaves
			e->m_constraintsSlave[si]->m_internal->m_slaveIndex = (hkObjectIndex)si;
			e->m_constraintsSlave[newIndex]->m_internal->m_slaveIndex = (hkObjectIndex)newIndex;
		}

		// Randomize collision entries lists of fixed bodies.
		hkArray<hkpLinkedCollidable::CollisionEntry>& collisionEntries = e->getLinkedCollidable()->getCollisionEntriesNonDeterministic();
		for (int ei = 0; ei < collisionEntries.getSize(); ei++)
		{
			int newIndex = (int)prng.getRandRange(0.0f, (hkReal)collisionEntries.getSize()-0.01f);

			if (ei != newIndex)
			{
				hkAlgorithm::swap(collisionEntries[ei], collisionEntries[newIndex]);
				int collidableIndex;
				collidableIndex = e->getLinkedCollidable() == collisionEntries[ei].m_agentEntry->getCollidableB();
				collisionEntries[ei].m_agentEntry->m_agentIndexOnCollidable[collidableIndex] = (hkObjectIndex)ei;
				collidableIndex = e->getLinkedCollidable() == collisionEntries[newIndex].m_agentEntry->getCollidableB();
				collisionEntries[newIndex].m_agentEntry->m_agentIndexOnCollidable[collidableIndex] = (hkObjectIndex)newIndex;
			}
		}
	}
}
#endif

//////////////////////////////////////////////////////////////////////
//
//  Extra large-block comments moved from hkpWorld.h
//
//////////////////////////////////////////////////////////////////////

// "concept: blocking of execution of pending operations"

// Concept: Suppreses attempts of execution of pending operations when hkWordl::attemptToExecutePendingOperations()
//          is called. Allows you to execute a series of critical operations (without locking the hkpWorld) and only execute
//          pending operations at the end (as each critical operation locks the world itself, therefore potentially putting
//          operations on the pending queue).
// Info:    This is only a boolean flag, not a blockCount.
// Usage:   Block the world before executing the first operation, unblock after executing the last one.
//          Call attemptToExecutePendingOperations explicitly then.
// Example: see hkpWorld::updateCollisionFilterOnWorld


// "concept: allowing critical operations"

// Debugging utility: monitoring of critical operations executions.
// When 'critical' operations are NOT allowed, areCriticalOperationsLocked() fires an assert whenever called.
// We assume that hkpWorld::isLocked is called by every 'critical' operation at its beginning, to check whether
// the operation should be performed immediately or put on a pending list.

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
