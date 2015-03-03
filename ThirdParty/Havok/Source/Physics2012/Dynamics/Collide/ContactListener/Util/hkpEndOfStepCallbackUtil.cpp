/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpEndOfStepCallbackUtil.h>
#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactPointEvent.h>

#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>

hkpEndOfStepCallbackUtil::hkpEndOfStepCallbackUtil()
: m_sequenceNumber( 0 ), m_deterministicOrder( false )
{
}

void hkpEndOfStepCallbackUtil::registerCollision( hkpSimpleConstraintContactMgr* mgr, hkpContactListener* listener, hkpCollisionEvent::CallbackSource source )
{
	HK_ASSERT(0x4d1740b4, mgr != HK_NULL );
	NewCollision& collision = m_newCollisions.expandOne();
	collision.m_mgr = mgr;
	collision.m_listener = listener;
	collision.m_source = source;
	collision.m_sequenceNumber = m_sequenceNumber;
	++m_sequenceNumber;
}

void hkpEndOfStepCallbackUtil::unregisterCollision( hkpSimpleConstraintContactMgr* mgr, hkpContactListener* listener, hkpCollisionEvent::CallbackSource source )
{
	Collision& collision = m_removedCollisions.expandOne();
	collision.m_mgr = mgr;
	collision.m_listener = listener;
	collision.m_source = source;
}

hkBool hkpEndOfStepCallbackUtil::Collision::operator == ( const Collision& other ) const
{
	return ( other.m_mgr == m_mgr ) && ( other.m_listener == m_listener ) && ( other.m_source == m_source ); 
}

inline hkUint64 hkpEndOfStepCallbackUtil::Collision::getUid() const
{
	return ( hkUint64( m_mgr->m_constraint.getEntityA()->m_uid ) << 32 ) | m_mgr->m_constraint.getEntityB()->m_uid;
}

inline hkBool hkpEndOfStepCallbackUtil::Collision::operator < ( const Collision& other ) const
{
	const hkUint64 uid = getUid();
	const hkUint64 otherUid = other.getUid();
	return uid < otherUid;
}

inline hkBool hkpEndOfStepCallbackUtil::NewCollision::operator < ( const NewCollision& other ) const
{
	const hkUint64 uid = getUid();
	const hkUint64 otherUid = other.getUid();
	return ( uid < otherUid ) || ( ( uid == otherUid ) && ( m_sequenceNumber < other.m_sequenceNumber ) );
}

static void mergeArrays( hkArray<hkpEndOfStepCallbackUtil::NewCollision>& source, hkArray<hkpEndOfStepCallbackUtil::Collision>& target )
{
	const hkpEndOfStepCallbackUtil::NewCollision* s = source.begin();
	const hkpEndOfStepCallbackUtil::NewCollision* sEnd = source.end();
	
	// Early out to avoid copying the target.
	if ( s == sEnd )
	{
		return;
	}

	hkArray<hkpEndOfStepCallbackUtil::Collision> other( source.getSize() + target.getSize() );

	hkpEndOfStepCallbackUtil::Collision* t = target.begin();
	hkpEndOfStepCallbackUtil::Collision* tEnd = target.end();
	hkpEndOfStepCallbackUtil::Collision* o = other.begin();

	while ( ( s < sEnd ) && ( t < tEnd ) )
	{
		if ( *static_cast<const hkpEndOfStepCallbackUtil::Collision*>( s ) < *t )
		{
			*o = *s;
			++s;
			++o;
		}
		else
		{
			*o = *t;
			++t;
			++o;
		}
	}
	while ( s < sEnd )
	{
		*o = *s;
		++s;
		++o;
	}
	while ( t < tEnd )
	{
		*o = *t;
		++t;
		++o;
	}
	target.swap( other );
	source.clear();
}

template<typename T>
static void stripArray( hkArray<hkpEndOfStepCallbackUtil::Collision>& removals, hkArray<T>& target )
{
	const int numRemovals = removals.getSize();
	const int numTarget = target.getSize();

	// Early out to avoid copying the target.
	if ( numRemovals == 0 )
	{
		return;
	}

	hkArray<T> other( numTarget );

	
	int o = 0;
	for ( int i = 0; i < numTarget; ++i )
	{
		const int index = removals.indexOf( target[i] );
		if ( index == -1 )
		{
			other[o] = target[i];
			++o;
		}
		else
		{
			removals.removeAt( index );
		}		
	}
	other.setSize( numTarget - ( numRemovals - removals.getSize() ) );
	target.swap( other );
}


void hkpEndOfStepCallbackUtil::postSimulationCallback( hkpWorld* world )
{
	HK_TIMER_BEGIN("EndOfStepCbs", HK_NULL);
	stripArray( m_removedCollisions, m_newCollisions );
	stripArray( m_removedCollisions, m_collisions );

#if defined(HK_DEBUG)
	// Sanity check removals.
	HK_ASSERT2( 0x34d1459e, m_removedCollisions.getSize() == 0, "A collision was unregistered which was never registered." );

	// Sanity check additions.
	const int numNewCollisions = m_newCollisions.getSize();
	for ( int i = 0; i < numNewCollisions; ++i )
	{
		const NewCollision& collision = m_newCollisions[i];
		HK_ASSERT2( 0x8fe12edb, m_newCollisions.indexOf( collision, i + 1 ) == -1, "This collision was registered more than once in this frame." );
		HK_ASSERT2( 0x8fe12eda, m_collisions.indexOf( collision ) == -1, "This collision is already registered collision with the util." );
	}
#endif

	if ( m_deterministicOrder )
	{
		// Sort up to collision UIDs and sequence number, which is sufficient for determinism.
		hkAlgorithm::quickSort( m_newCollisions.begin(), m_newCollisions.getSize() );
		mergeArrays( m_newCollisions, m_collisions );
	}
	else
	{
		//m_collisions.append( m_newCollisions.begin(), m_newCollisions.getSize() ); //todo decide on future of append
		Collision* c = m_collisions.expandBy( m_newCollisions.getSize() );
		for( int i = 0; i < m_newCollisions.getSize(); ++i )
		{
			c[i] = m_newCollisions[i];
		}
		m_newCollisions.clear();
	}

	// HVK-6133 : we try to check if the listener that registered the collision is still
	// listening to the collision events of the body ; if it is not, that means it will 
	// never tell us to unregister a collision, so we remove it.

	const int numCollisions = m_collisions.getSize();
	hkBool anyInvalidCollision = false;
	for ( int i = 0; i < numCollisions; ++i )
	{
		Collision& collision = m_collisions[i];
		hkpSimpleConstraintContactMgr *const mgr = collision.m_mgr;
		hkpConstraintInstance *const instance = mgr->getConstraintInstance();
		
		hkpRigidBody* body = HK_NULL;
		switch ( collision.m_source ) 
		{
			case hkpCollisionEvent::SOURCE_A:
				body = instance->getRigidBodyA();
				break;
			case hkpCollisionEvent::SOURCE_B:
				body = instance->getRigidBodyB();
				break;
			default:
				break;
		}		
		
		if (body && (body->getContactListeners().indexOf(collision.m_listener) == -1))
		{
			anyInvalidCollision = true;
			collision.m_listener = HK_NULL;
		}
		
		// We're only interested in active constraints.
		else if ( instance->getSimulationIsland()->m_activeMark )
		{
			fireContactPointEventsForCollision( mgr, collision.m_listener, collision.m_source );
		}
	}

	if ( anyInvalidCollision )
	{
		int nbColl = m_collisions.getSize();
		hkArray< Collision > validCollisions( nbColl );
		int numValidCollisions = 0;
		for ( int i = 0; i < nbColl; ++i )
		{
			Collision& collision = m_collisions[i];
			if ( collision.m_listener )
			{
				validCollisions[numValidCollisions++] = collision;
			}
		}

		validCollisions.setSize( numValidCollisions );
		m_collisions.swap( validCollisions );
	}

	m_sequenceNumber = 0;
	HK_TIMER_END();
}


void hkpEndOfStepCallbackUtil::fireContactPointEventsForCollision( hkpSimpleConstraintContactMgr* mgr, hkpContactListener* listener, hkpCollisionEvent::CallbackSource source )
{
	hkpRigidBody* bodyA = mgr->getConstraintInstance()->getRigidBodyA();
	hkpRigidBody* bodyB = mgr->getConstraintInstance()->getRigidBodyB();
	hkpSimpleContactConstraintAtom *atom = mgr->getAtom();
	hkpContactPointPropertiesStream* cpp = atom->getContactPointPropertiesStream();
	int cppStriding = atom->getContactPointPropertiesStriding();
	hkContactPoint* cp = atom->getContactPoints();
	int nA = atom->m_numContactPoints;
	const bool callbacksForFullManifold = 0 < ( mgr->getConstraintInstance()->m_internal->m_callbackRequest & hkpConstraintAtom::CALLBACK_REQUEST_CONTACT_POINT_CALLBACK );
	for (int cindex = nA-1; cindex >=0 ; cp++, cpp = hkAddByteOffset(cpp, cppStriding), cindex-- )
	{
		hkpContactPointProperties *const properties = cpp->asProperties();
		if ( ( properties->m_flags & hkContactPointMaterial::CONTACT_IS_NEW ) || callbacksForFullManifold )
		{
			hkpShapeKey *const shapeKeys = reinterpret_cast< hkpShapeKey* >( properties->getStartOfExtendedUserData( atom ) );
			hkpContactPointEvent event( source, bodyA, bodyB, mgr, 
				hkpContactPointEvent::TYPE_MANIFOLD_AT_END_OF_STEP,
				cp, cpp->asProperties(), 
				HK_NULL, HK_NULL, 
				callbacksForFullManifold, ( cindex == nA - 1 ), ( cindex == 0 ),
				shapeKeys,
				HK_NULL, HK_NULL );

			// Fire the callback.
			listener->contactPointCallback( event );
		}
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
