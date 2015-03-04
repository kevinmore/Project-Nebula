/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Dynamics/SuspendInactiveAgents/hkpSuspendInactiveAgentsUtil.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent3/hkpAgent3.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>

#include <Physics2012/Collide/Agent/Collidable/hkpCdBody.h>

hkpSuspendInactiveAgentsUtil::hkpSuspendInactiveAgentsUtil(hkpWorld* world, OperationMode mode, InitContactsMode initContactsMode )
:	m_world(world), m_mode(mode), m_initContactsMode(initContactsMode)
{
	addReference();
	world->addWorldDeletionListener( this );
	world->addIslandActivationListener( this );
}

hkpSuspendInactiveAgentsUtil::~hkpSuspendInactiveAgentsUtil()
{
	if ( m_world )
	{
		m_world->removeWorldDeletionListener( this );
		m_world = HK_NULL;
	}
}
		
namespace {

	class NeverCollideFilter : public hkpCollisionFilter
	{
		virtual hkBool isCollisionEnabled( const hkpCollidable& a, const hkpCollidable& b ) const { return false; }
		virtual	hkBool isCollisionEnabled( const hkpCollisionInput& input, const hkpCdBody& a, const hkpCdBody& b, const hkpShapeContainer& bContainer, hkpShapeKey bKey  ) const { return false; }
		virtual hkBool isCollisionEnabled( const hkpCollisionInput& input, const hkpCdBody& collectionBodyA, const hkpCdBody& collectionBodyB, const HK_SHAPE_CONTAINER& containerShapeA, const HK_SHAPE_CONTAINER& containerShapeB, hkpShapeKey keyA, hkpShapeKey keyB ) const { return false; }
		virtual hkBool isCollisionEnabled( const hkpShapeRayCastInput& aInput, const hkpShapeContainer& bContainer, hkpShapeKey bKey ) const { return false; }
		virtual hkBool isCollisionEnabled( const hkpWorldRayCastInput& a, const hkpCollidable& collidableB ) const { return false; }
	};

	class Clear1nTracksFilter : public hkpCollisionFilter
	{ 
	public:
		Clear1nTracksFilter( const hkpCollisionFilter* filter ) : m_originalFilter(filter) { HK_ASSERT2(0xad7865dd, m_originalFilter, "Original filter must be specified.");  }

		~Clear1nTracksFilter() {  }

		virtual hkBool isCollisionEnabled( const hkpCollidable& a, const hkpCollidable& b ) const 
		{
			HK_ASSERT2(0xad78d6a0, false, "This function should be never called."); return true;
		}

		virtual hkBool isCollisionEnabled( const hkpCollisionInput& input, const hkpCdBody& a, const hkpCdBody& b, const hkpShapeContainer& bCollection, hkpShapeKey bKey  ) const  
		{	
			if ( b.getShape()->getType() == hkcdShapeType::MOPP 
			  || b.getShape()->getType() == hkcdShapeType::BV_TREE
			  || b.getShape()->getType() == hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE )
			{
				return false;
			}
			return m_originalFilter->isCollisionEnabled (input, a, b, bCollection, bKey);
		}

		virtual hkBool isCollisionEnabled( const hkpCollisionInput& input, const hkpCdBody& a, const hkpCdBody& b, const HK_SHAPE_CONTAINER& containerShapeA, const HK_SHAPE_CONTAINER& containerShapeB, hkpShapeKey keyA, hkpShapeKey keyB ) const
		{
			if (   a.getShape()->getType() == hkcdShapeType::MOPP 
				|| a.getShape()->getType() == hkcdShapeType::BV_TREE
				|| a.getShape()->getType() == hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE)
			{
				return false;
			}
			if (   b.getShape()->getType() == hkcdShapeType::MOPP 
				|| b.getShape()->getType() == hkcdShapeType::BV_TREE
				|| b.getShape()->getType() == hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE)
			{
				return false;
			}
			return m_originalFilter->isCollisionEnabled (input, a, b, containerShapeA, containerShapeB, keyA, keyB );
		}

		virtual hkBool isCollisionEnabled( const hkpShapeRayCastInput& aInput, const hkpShapeContainer& bContainer, hkpShapeKey bKey ) const  
		{	
			HK_ASSERT2(0xad78d6a0, false, "This function should be never called."); return true;
		}

		virtual hkBool isCollisionEnabled( const hkpWorldRayCastInput& a, const hkpCollidable& collidableB ) const  
		{	
			HK_ASSERT2(0xad78d6a0, false, "This function should be never called."); return true;
		}

	protected:
		const hkpCollisionFilter* m_originalFilter;
	};
}

static void HK_CALL removeEmptyAgent( hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
	if ( entry->m_numContactPoints == 0 )
	{
		entry->m_size = 0;
	}
}

void hkpSuspendInactiveAgentsUtil::islandDeactivatedCallback( hkpSimulationIsland* island )
{
	// This is only called from hkpWorldOperationUtil::cleanupDirtyIslands.
	HK_ACCESS_CHECK_OBJECT( island->getWorld(), HK_ACCESS_RW );
	HK_ASSERT2( 0xad7899de, island->getWorld()->areCriticalOperationsLocked(), "Critical operations are expected to be locked.");

	NeverCollideFilter neverCollideFilter;
	Clear1nTracksFilter clear1nTracksFilter(m_world->getCollisionFilter());

	hkpCollisionInput input = *m_world->getCollisionInput();
	switch(m_mode)
	{
		case SUSPEND_ALL_COLLECTION_AGENTS: input.m_filter = &neverCollideFilter; break;
		case SUSPEND_1N_AGENT_TRACKS:       input.m_filter = &clear1nTracksFilter; break;
		case SUSPEND_UNUSED_CACHES:	        break;
	}

	int gskAgent3Type		= input.m_dispatcher->getAgent3Type( hkcdShapeType::CONVEX, hkcdShapeType::CONVEX, false );
	int gskAgent3TypePred	= input.m_dispatcher->getAgent3Type( hkcdShapeType::CONVEX, hkcdShapeType::CONVEX, true );

	hkAgent3::UpdateFilterFunc oldUpdate		= input.m_dispatcher->m_agent3Func[ gskAgent3Type ].m_updateFilterFunc;
	hkAgent3::UpdateFilterFunc oldUpdatePred	= input.m_dispatcher->m_agent3Func[ gskAgent3TypePred ].m_updateFilterFunc;
	
	input.m_dispatcher->m_agent3Func[ gskAgent3Type ].m_updateFilterFunc	 = removeEmptyAgent;
	input.m_dispatcher->m_agent3Func[ gskAgent3TypePred ].m_updateFilterFunc = removeEmptyAgent;

	hkpAgentNnTrack *const tracks[2] = { &island->m_narrowphaseAgentTrack, &island->m_midphaseAgentTrack };
	for ( int i = 0; i < 2; ++i )
	{
		hkpAgentNnTrack& track = *tracks[i];
		HK_FOR_ALL_AGENT_ENTRIES_BEGIN(track, entry)
		{
			hkUchar oldSize = entry->m_size;
			hkAgentNnMachine_UpdateShapeCollectionFilter( entry, input, *island );
			// Currently this trick only works (setting size to 0 callback above) with 1-n machines. This will reset
			// the size if it is called on a convex - convex n-n agent.
			entry->m_size = oldSize;
		}
		HK_FOR_ALL_AGENT_ENTRIES_END;
	}

	input.m_dispatcher->m_agent3Func[ gskAgent3Type ].m_updateFilterFunc = oldUpdate;
	input.m_dispatcher->m_agent3Func[ gskAgent3TypePred ].m_updateFilterFunc = oldUpdatePred;

}

void hkpSuspendInactiveAgentsUtil::islandActivatedCallback( hkpSimulationIsland* island )
{
    if ( m_mode == SUSPEND_UNUSED_CACHES )
    {
	    return;
    }

	// This is only called from hkpWorldOperationUtil::cleanupDirtyIslands and from the engine, e.g. during island merges.
	// This is not safe is the updateShapeCollectioFilter would remove any agents.

	HK_ACCESS_CHECK_OBJECT( island->getWorld(), HK_ACCESS_RW );
	HK_ASSERT2( 0xad7899df, island->getWorld()->areCriticalOperationsLocked(), "Critical operations are expected to be locked.");

	hkpCollisionInput input = *m_world->getCollisionInput();

	if (m_mode == SUSPEND_ALL_COLLECTION_AGENTS)
	{
		hkpAgentNnTrack *const tracks[2] = { &island->m_narrowphaseAgentTrack, &island->m_midphaseAgentTrack };
		for ( int i = 0; i < 2; ++i )
		{
			hkpAgentNnTrack& track = *tracks[i];
			HK_FOR_ALL_AGENT_ENTRIES_BEGIN(track, entry)
			{
				hkAgentNnMachine_UpdateShapeCollectionFilter( entry, input, *island );
			}
			HK_FOR_ALL_AGENT_ENTRIES_END;
		}
	}

	if (m_initContactsMode == INIT_CONTACTS_FIND )
	{
		m_world->findInitialContactPoints( island->m_entities.begin(), island->m_entities.getSize() );
	}
}

void hkpSuspendInactiveAgentsUtil::worldDeletedCallback( hkpWorld* world )
{
	world->removeWorldDeletionListener( this );
	m_world = HK_NULL;
	removeReference();
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
