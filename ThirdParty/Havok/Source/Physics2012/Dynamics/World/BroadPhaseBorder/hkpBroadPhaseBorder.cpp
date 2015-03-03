/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Physics2012/Dynamics/World/BroadPhaseBorder/hkpBroadPhaseBorder.h>
#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>


//
// Broadphase border implementation
//

hkpBroadPhaseBorder::hkpBroadPhaseBorder( hkpWorld* world, hkpWorldCinfo::BroadPhaseBorderBehaviour type, hkBool postponeAndSortCallbacks )
{
	m_world = world;
	m_type = type;

	m_postponeAndSortCallbacks = postponeAndSortCallbacks;

	addReference();
	m_world->addWorldDeletionListener( this );
	m_world->addWorldPostSimulationListener( this );


	hkAabb infoAABB;

	const hkVector4 minExtents = m_world->m_broadPhaseExtents[0];
	const hkVector4 maxExtents = m_world->m_broadPhaseExtents[1];
	{
		infoAABB.m_min.setSelect<hkVector4ComparisonMask::MASK_X>( maxExtents, minExtents );
		infoAABB.m_max = maxExtents;
		hkpAabbPhantom* ph = new hkpAabbPhantom( infoAABB );
		ph->getCollidableRw()->getBroadPhaseHandle()->setType(hkpWorldObject::BROAD_PHASE_BORDER);
		ph->addPhantomOverlapListener( this );
		m_phantoms[0] = ph;
		world->addPhantom( ph );
	}
	{
		infoAABB.m_min = minExtents;
		infoAABB.m_max.setSelect<hkVector4ComparisonMask::MASK_X>( minExtents, maxExtents );
		hkpAabbPhantom* ph = new hkpAabbPhantom( infoAABB );
		ph->getCollidableRw()->getBroadPhaseHandle()->setType(hkpWorldObject::BROAD_PHASE_BORDER);
		ph->addPhantomOverlapListener( this );
		m_phantoms[1] = ph;
		world->addPhantom( ph );
	}

	{
		infoAABB.m_min.setSelect<hkVector4ComparisonMask::MASK_XZ>( minExtents, maxExtents );
		infoAABB.m_max = maxExtents;
		hkpAabbPhantom* ph = new hkpAabbPhantom( infoAABB );
		ph->getCollidableRw()->getBroadPhaseHandle()->setType(hkpWorldObject::BROAD_PHASE_BORDER);
		ph->addPhantomOverlapListener( this );
		m_phantoms[2] = ph;
		world->addPhantom( ph );
	}
	{
		infoAABB.m_min = minExtents;
		infoAABB.m_max.setSelect<hkVector4ComparisonMask::MASK_XZ>( maxExtents, minExtents );
		hkpAabbPhantom* ph = new hkpAabbPhantom( infoAABB );
		ph->getCollidableRw()->getBroadPhaseHandle()->setType(hkpWorldObject::BROAD_PHASE_BORDER);
		ph->addPhantomOverlapListener( this );
		m_phantoms[3] = ph;
		world->addPhantom( ph );
	}

	{
		infoAABB.m_min.setSelect<hkVector4ComparisonMask::MASK_XY>( minExtents, maxExtents );
		infoAABB.m_max = maxExtents;
		hkpAabbPhantom* ph = new hkpAabbPhantom( infoAABB );
		ph->getCollidableRw()->getBroadPhaseHandle()->setType(hkpWorldObject::BROAD_PHASE_BORDER);
		ph->addPhantomOverlapListener( this );
		m_phantoms[4] = ph;
		world->addPhantom( ph );
	}

	{
		infoAABB.m_min = minExtents;
		infoAABB.m_max.setSelect<hkVector4ComparisonMask::MASK_XY>( maxExtents, minExtents );
		hkpAabbPhantom* ph = new hkpAabbPhantom( infoAABB );
		ph->getCollidableRw()->getBroadPhaseHandle()->setType(hkpWorldObject::BROAD_PHASE_BORDER);
		ph->addPhantomOverlapListener( this );
		m_phantoms[5] = ph;
		world->addPhantom( ph );
	}
}

hkpBroadPhaseBorder::~hkpBroadPhaseBorder()
{
	for (int i = 0; i < 6; i++ )
	{
		if (m_phantoms[i] != HK_NULL)
		{
			m_phantoms[i]->removeReference();
			m_phantoms[i] = HK_NULL;
		}
	}
}


void hkpBroadPhaseBorder::collidableAddedCallback( const hkpCollidableAddedEvent& event )
{
	// The island is locked with a world-scope critical section. We're mt safe.

	hkpRigidBody* body = hkpGetRigidBody( event.m_collidable );
	if ( body )
	{
		if (m_postponeAndSortCallbacks)
		{
			// put on a queue, which is processed in post simulation callback.
			m_entitiesExitingBroadPhase.pushBack(body);
			body->addReference();
		}
		else
		{
			// execute callback immediately 
			maxPositionExceededCallback( body );
		}
	}
}


// Callback implementation 
void hkpBroadPhaseBorder::collidableRemovedCallback( const hkpCollidableRemovedEvent& event )
{
}

static bool hkpBroadPhaseBorder_EntityCmpLess(const hkpEntity* a, const hkpEntity* b)
{
	return a->getUid() < b->getUid();
}

void hkpBroadPhaseBorder::postSimulationCallback( hkpWorld* world )
{
	if (m_entitiesExitingBroadPhase.getSize())
	{
		hkSort(m_entitiesExitingBroadPhase.begin(), m_entitiesExitingBroadPhase.getSize(), hkpBroadPhaseBorder_EntityCmpLess);

		for (int i = 0; i < m_entitiesExitingBroadPhase.getSize(); i++)
		{
			maxPositionExceededCallback( m_entitiesExitingBroadPhase[i] );
		}

		hkReferencedObject::removeReferences(m_entitiesExitingBroadPhase.begin(), m_entitiesExitingBroadPhase.getSize());
		m_entitiesExitingBroadPhase.clear();
	}
}


void hkpBroadPhaseBorder::maxPositionExceededCallback( hkpEntity* entity )
{
	hkpRigidBody* body = static_cast<hkpRigidBody*>(entity);
	switch ( m_type )
	{
		case hkpWorldCinfo::BROADPHASE_BORDER_ASSERT:
			HK_ASSERT2( 0xf013323d, 0, "Entity left the broadphase. See hkpWorldCinfo::BroadPhaseBorderBehaviour for details." );
			body->setMotionType( hkpMotion::MOTION_FIXED );
			break;

		case hkpWorldCinfo::BROADPHASE_BORDER_REMOVE_ENTITY:
			if (entity->getWorld() == m_world)
			{
				m_world->removeEntity( entity );
				HK_WARN( 0x65567363, "Entity left the broadphase and has been removed from the hkpWorld. See hkpWorldCinfo::BroadPhaseBorderBehaviour for details." );
			}
			break;

		case hkpWorldCinfo::BROADPHASE_BORDER_FIX_ENTITY:
			body->setMotionType( hkpMotion::MOTION_FIXED );
			HK_WARN( 0x267bc474, "Entity left the broadphase and has been changed to fixed motion type. See hkpWorldCinfo::BroadPhaseBorderBehaviour for details." );
			break;

		default:
			break;
	}
}

void hkpBroadPhaseBorder::worldDeletedCallback( hkpWorld* world )
{
	m_world->removeWorldDeletionListener( this );
	m_world->removeWorldPostSimulationListener( this );
	removeReference();
}

void hkpBroadPhaseBorder::deactivate()
{
	if ( m_world )
	{
		m_world->removePhantomBatch( reinterpret_cast<hkpPhantom**>(&m_phantoms[0]), 6 );
		for (int i =0; i < 6; i++ )
		{
			m_phantoms[i]->removePhantomOverlapListener( this );
			m_phantoms[i]->removeReference();
			m_phantoms[i] = HK_NULL;
		}
		worldDeletedCallback( m_world );
		m_world = HK_NULL;
	}
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
