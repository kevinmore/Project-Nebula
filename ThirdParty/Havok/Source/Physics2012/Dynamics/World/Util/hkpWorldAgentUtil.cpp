/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Monitor/hkMonitorStream.h>


#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Dispatch/ContactMgr/hkpContactMgrFactory.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpCollisionAgent.h>
#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnTrack.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>

#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>
#include <Physics2012/Dynamics/World/Simulation/hkpSimulation.h>


// If both pointers are the same & point to a fixed island -- return it.
static HK_FORCE_INLINE hkpSimulationIsland* getAnyNonFixedIsland( hkpSimulationIsland* islandA, hkpSimulationIsland* islandB)
{
	if (!islandA->isFixed())
	{
		return islandA;
	}
	if (!islandB->isFixed())
	{
		return islandB;
	}
	HK_ASSERT2(0x48af8302, islandA == islandB, "Internal error: two different fixed islands.");
	return islandA;
}




hkpAgentNnEntry* hkpWorldAgentUtil::addAgent( hkpLinkedCollidable* collA, hkpLinkedCollidable* collB, const hkpProcessCollisionInput& input )
{
	HK_ASSERT2(0Xad000710, !hkAgentNnMachine_FindAgent(collA, collB), "An agent already exists between the two collidables specified.");

	hkpEntity* entityA = static_cast<hkpEntity*>( collA->getOwner() );
	hkpEntity* entityB = static_cast<hkpEntity*>( collB->getOwner() );

	// Request island merge
	hkpWorldOperationUtil::mergeIslandsIfNeeded( entityA, entityB );

	//   Choose the island to add new agent to 
	//   merge might have been delayed
	hkpSimulationIsland* theIsland = getAnyNonFixedIsland(entityA->getSimulationIsland(), entityB->getSimulationIsland());
	HK_ACCESS_CHECK_WITH_PARENT( theIsland->m_world, HK_ACCESS_IGNORE, theIsland, HK_ACCESS_RW );

	hkpCdBody modBodyA[4];
	hkpCdBody modBodyB[4];
	hkMotionState modMotionA[4];
	hkMotionState modMotionB[4];

	hkPadSpu<hkUchar> cdBodyHasTransformFlag = 0;

	const hkpCdBody* firstNonTransformBodyA = collA;
	const hkpCdBody* firstNonTransformBodyB = collB;
	if (collA->m_shape->getType() == hkcdShapeType::TRANSFORM)
	{
		firstNonTransformBodyA = hkAgentMachine_processTransformedShapes(collA, modBodyA, modMotionA, 4, cdBodyHasTransformFlag);
	}
	if (collB->m_shape->getType() == hkcdShapeType::TRANSFORM)
	{
		firstNonTransformBodyB = hkAgentMachine_processTransformedShapes(collB, modBodyB, modMotionB, 4, cdBodyHasTransformFlag);
	}

	//
	//	Get the agent type and flip information
	//
	int agentType;
	int isFlipped;
	hkAgentNnMachine_GetAgentType( firstNonTransformBodyA, firstNonTransformBodyB, input, agentType, isFlipped );
	if ( isFlipped )
	{
		hkAlgorithm::swap( collA, collB );
		hkAlgorithm::swap( firstNonTransformBodyA, firstNonTransformBodyB );
	}

	//
	// Attempt to create the mgr
	//
	hkpContactMgr* mgr;
	{
		hkpContactMgrFactory* factory = input.m_dispatcher->getContactMgrFactory( entityA->getMaterial().getResponseType(), entityB->getMaterial().getResponseType() );
		mgr = factory->createContactMgr( *collA, *collB, input );
	}

	//
	//	Create the final agent
	//
	
	hkpAgentNnEntry* newAgent = hkAgentNnMachine_CreateAgent( theIsland->m_narrowphaseAgentTrack, collA, firstNonTransformBodyA, collB, firstNonTransformBodyB, cdBodyHasTransformFlag, agentType, input, mgr );

#	ifdef HK_ENABLE_EXTENSIVE_WORLD_CHECKING
	theIsland->isValid();
#	endif


	return newAgent;



//	// suspend agent
//	if (createSuspended)
//	{
//		// info: if entityA and entityB belong to to active/inactive islands, than whether the agent should/shouln't be created
//		//       only depends on which island we initially assign it to.
//		hkpWorldAgentUtil::suspendAgent(pair);
//	}
//
//	return pair;


}

void hkpWorldAgentUtil::removeAgent( hkpAgentNnEntry* agent )
{
	HK_ON_DEBUG( hkpSimulation* simulation = static_cast<hkpEntity*>( agent->m_collidable[0]->getOwner() )->getSimulationIsland()->getWorld()->m_simulation );
	HK_ON_DEBUG( simulation->assertThereIsNoCollisionInformationForAgent(agent) );

	// Remove hkCollisionPair / agent from hkpSimulationIsland
	hkpSimulationIsland* theIsland;
	hkpEntity* entityA = static_cast<hkpEntity*>( agent->m_collidable[0]->getOwner() );
	hkpEntity* entityB = static_cast<hkpEntity*>( agent->m_collidable[1]->getOwner() );
	hkpSimulationIsland* islandA = entityA->getSimulationIsland();
	hkpSimulationIsland* islandB = entityB->getSimulationIsland();

	if (islandA == islandB)
	{
		theIsland = islandA;
		theIsland->m_splitCheckRequested = true;
	}
	else if (entityA->isFixed())
	{
		// don't check whether the island is fixed, cause you'll get a cache miss on the fixed island :-/
		theIsland = islandB;
	}
	else if (entityB->isFixed())
	{
		theIsland = islandA;
	}
	else
	{
		// This should happen only when you add and remove an agent between entities moving one after another (and belonging to two different islands)
		// in a way that their AABBs overlap in-between collision detection run for each of the islands.

		theIsland = getIslandFromAgentEntry(agent, islandA, islandB);

		// we have those, because we may still have a merge request for those entities in the pendingOperation queue
		//  and this is faster than going through the pendingOperations list. And we are too lazy.
		entityA->getSimulationIsland()->m_splitCheckRequested = true;
		entityB->getSimulationIsland()->m_splitCheckRequested = true;
	}
	HK_ACCESS_CHECK_WITH_PARENT( theIsland->m_world, HK_ACCESS_IGNORE, theIsland, HK_ACCESS_RW );


	hkpAgentNnTrack* track = theIsland->getAgentNnTrack( agent->m_nnTrackType );
	hkpCollisionDispatcher* dispatch = theIsland->getWorld()->getCollisionDispatcher();

	hkpContactMgr* mgr = agent->m_contactMgr;
	hkAgentNnMachine_DestroyAgent( *track, agent, dispatch, *theIsland );
	mgr->cleanup();

#	ifdef HK_ENABLE_EXTENSIVE_WORLD_CHECKING
	theIsland->isValid();
#	endif

	//HK_INTERNAL_TIMER_END_LIST();

}

void hkpWorldAgentUtil::removeAgentAndItsToiEvents ( hkpAgentNnEntry* agent )
{
	hkpSimulation* simulation = static_cast<hkpEntity*>( agent->m_collidable[0]->getOwner() )->getSimulationIsland()->getWorld()->m_simulation;
	simulation->removeCollisionInformationForAgent( agent );

	hkpWorldAgentUtil::removeAgent( agent );
}

hkpSimulationIsland* hkpWorldAgentUtil::getIslandFromAgentEntry( hkpAgentNnEntry* entry, hkpSimulationIsland* candidateA, hkpSimulationIsland* candidateB)
{
	// just iterate over sectors of the shorter track
	hkpAgentNnTrack* trackA;
	hkpAgentNnTrack* trackB;
	if ( entry->m_nnTrackType == HK_AGENT3_NARROWPHASE_TRACK )
	{
		trackA = &candidateA->m_narrowphaseAgentTrack;
		trackB = &candidateB->m_narrowphaseAgentTrack;
	}
	else
	{
		trackA = &candidateA->m_midphaseAgentTrack;
		trackB = &candidateB->m_midphaseAgentTrack;
	}
	hkBool searchIsleA = trackA->m_sectors.getSize() <= trackB->m_sectors.getSize();
	hkpAgentNnTrack* trackToSearch = searchIsleA ? trackA : trackB;
	hkBool sectorFound = false;
	hkArray<hkpAgentNnSector*>& sectors = trackToSearch->m_sectors;
	for (int i = 0; i < sectors.getSize(); i++)
	{
		hkpAgentNnSector* sector = sectors[i];
		if (sector->getBegin() <= entry && entry < sector->getEnd() )
		{
			sectorFound = true;
			break;
		}
	}

	// if the agent is not there, then it's in the other track -- just remove it with hkAgentNnMachine_
	return (searchIsleA ^ sectorFound) ? candidateB : candidateA;
}



HK_FORCE_INLINE static hkpAgentData* getAgentData( hkpAgentNnEntry* entry)
{
	hkAgent3::StreamCommand command = hkAgent3::StreamCommand(entry->m_streamCommand);
	if ( command == hkAgent3::STREAM_CALL_WITH_TIM)
	{
		return hkAddByteOffset<hkpAgentData>( entry, hkSizeOf( hkpAgentNnMachineTimEntry ) );
	}
	else
	{
		return hkAddByteOffset<hkpAgentData>( entry, hkSizeOf( hkpAgentNnMachinePaddedEntry ) );
	}
}

void hkpWorldAgentUtil::updateEntityShapeCollectionFilter( hkpEntity* entity, hkpCollisionInput& collisionInput )
{
	HK_ACCESS_CHECK_OBJECT( entity->getWorld(), HK_ACCESS_RW );
	hkpLinkedCollidable* collidable = entity->getLinkedCollidable();

	hkArray<struct hkpLinkedCollidable::CollisionEntry> collisionEntriesTmp;
	collidable->getCollisionEntriesSorted(collisionEntriesTmp);
	const hkArray<struct hkpLinkedCollidable::CollisionEntry>& collisionEntries = collisionEntriesTmp;

	for (int i = 0; i < collisionEntries.getSize(); i++)
	{
		hkpAgentNnEntry* entry = collisionEntries[i].m_agentEntry; 

		hkAgent3::UpdateFilterFunc func = collisionInput.m_dispatcher->getAgent3UpdateFilterFunc(entry->m_agentType);
		if (func)
		{
				// this cast is allowed, as the nn-machine only works between entities
			hkpEntity* entityA = static_cast<hkpEntity*>(entry->getCollidableA()->getOwner());
			hkpEntity* entityB = static_cast<hkpEntity*>(entry->getCollidableB()->getOwner());
			hkpSimulationIsland* island = (entityA->isFixed() )? entityB->getSimulationIsland(): entityA->getSimulationIsland();

			hkpAgentData* agentData = getAgentData(entry);
			func(entry, agentData, *entry->getCollidableA(), *entry->getCollidableB(), collisionInput, entry->m_contactMgr, *island);
		}
	}
}

void hkpWorldAgentUtil::invalidateTim( hkpEntity* entity, const hkpCollisionInput& collisionInput )
{
	hkpLinkedCollidable* collidable = entity->getLinkedCollidable();
	const hkArray<struct hkpLinkedCollidable::CollisionEntry>& collisionEntries = collidable->getCollisionEntriesNonDeterministic(); // order doesn't matter

	for (int i = 0; i < collisionEntries.getSize(); i++)
	{
		hkpAgentNnEntry* entry = collisionEntries[i].m_agentEntry; 
		hkAgentNnMachine_InvalidateTimInAgent( entry, collisionInput );
	}
}

void hkpWorldAgentUtil::warpTime( hkpSimulationIsland* island, hkTime oldTime, hkTime newTime, const hkpCollisionInput& collisionInput )
{
	HK_ACCESS_CHECK_WITH_PARENT( island->m_world, HK_ACCESS_RO, island, HK_ACCESS_RW );
	HK_FOR_ALL_AGENT_ENTRIES_BEGIN(island->m_narrowphaseAgentTrack, entry)
	{
		hkAgentNnMachine_WarpTimeInAgent(entry, oldTime, newTime, collisionInput );
	}
	HK_FOR_ALL_AGENT_ENTRIES_END;
	HK_FOR_ALL_AGENT_ENTRIES_BEGIN(island->m_midphaseAgentTrack, entry)
	{
		hkAgentNnMachine_WarpTimeInAgent(entry, oldTime, newTime, collisionInput );
	}
	HK_FOR_ALL_AGENT_ENTRIES_END;
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
