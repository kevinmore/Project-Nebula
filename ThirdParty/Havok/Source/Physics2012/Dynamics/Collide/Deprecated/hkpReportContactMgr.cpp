/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Physics2012/Dynamics/Collide/Deprecated/hkpReportContactMgr.h>
#include <Physics2012/Dynamics/Collide/Deprecated/Dispatch/hkpCollideCallbackDispatcher.h>

#include <Physics2012/Dynamics/World/Simulation/Continuous/hkpContinuousSimulation.h>

#if !defined(HK_PLATFORM_SPU)
#	include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>
#	include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>
#endif

#if !defined(HK_PLATFORM_SPU)
	HK_COMPILE_TIME_ASSERT( hkpContactMgr::TOI_ACCEPT == hkpContactMgr::ToiAccept(HK_CONTACT_POINT_ACCEPT) );
	HK_COMPILE_TIME_ASSERT( hkpContactMgr::TOI_REJECT == hkpContactMgr::ToiAccept(HK_CONTACT_POINT_REJECT) );

hkpReportContactMgr::hkpReportContactMgr( hkpWorld *sm, hkpRigidBody *bodyA, hkpRigidBody *bodyB ): hkpDynamicsContactMgr( hkpContactMgr::TYPE_REPORT_CONTACT_MGR )
{
	m_skipNextNprocessCallbacks = hkMath::min2( bodyA->getContactPointCallbackDelay(), bodyB->getContactPointCallbackDelay() );
	m_world = sm;
	m_bodyA = bodyA;
	m_bodyB = bodyB;
}

hkpReportContactMgr::~hkpReportContactMgr()
{
}
#endif

hkContactPointId hkpReportContactMgr::addContactPointImpl( const hkpCdBody& a, const hkpCdBody& b, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output, const hkpGskCache* contactCache, hkContactPoint& pcp )
{
	hkContactPointId contactPointIdOut = 0; 

	hkpRigidBody* rba = m_bodyA;
	hkpRigidBody* rbb = m_bodyB;

	hkReal projectedVel;
	{
		hkVector4 velA;		rba->getPointVelocity( pcp.getPosition(), velA );
		hkVector4 velB;		rbb->getPointVelocity( pcp.getPosition(), velB );

		hkVector4 deltaVel; deltaVel.setSub( velB, velA );
		projectedVel = deltaVel.dot<3>( pcp.getNormal() ).getReal();
	}

		//
		// fire all events
		//
	hkpManifoldPointAddedEvent event( contactPointIdOut, this, &input, &output, &a,&b, &pcp, contactCache, HK_NULL, projectedVel);
	hkFireContactPointAddedCallback( m_world, rba, rbb, event );

	if ( event.m_status == HK_CONTACT_POINT_REJECT )
	{
		// Note: This will fire the removal event, so all listeners will be correctly informed of the state change.
		removeContactPointImpl( contactPointIdOut, *output.m_constraintOwner.val() );
		return HK_INVALID_CONTACT_POINT;
	}
	else
	{
		m_skipNextNprocessCallbacks = event.m_nextProcessCallbackDelay;
		return contactPointIdOut;
	}
}


hkpReportContactMgr::ToiAccept hkpReportContactMgr::addToiImpl( const hkpCdBody& a, const hkpCdBody& b, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output, hkTime toi, 
														 hkContactPoint& cp, const hkpGskCache* gskCache, hkReal& projectedVelocity, hkpContactPointProperties& properties )
{
	properties.reset();
	hkpToiPointAddedEvent event( this, &input, &output, &a,&b, &cp, gskCache, &properties, toi, projectedVelocity);
	hkFireContactPointAddedCallback( m_world, m_bodyA, m_bodyB, event );

	if (event.m_status == HK_CONTACT_POINT_REJECT)
	{
		removeToiImpl( *output.m_constraintOwner.val(), properties );
	}
	else
	{
		projectedVelocity = event.m_projectedVelocity;
	}

	return hkpContactMgr::ToiAccept( event.m_status);
}

void hkpReportContactMgr::removeToiImpl( class hkCollisionConstraintOwner& constraintOwner, hkpContactPointProperties& properties )
{
	hkpWorld* world = m_bodyA->getWorld();

	// Fire TOI-point removed
	hkpContactPointRemovedEvent removedEvent( HK_INVALID_CONTACT_POINT, this, &constraintOwner, &properties, m_bodyA, m_bodyB );
	hkFireContactPointRemovedCallback( world, m_bodyA, m_bodyB, removedEvent );
}



void hkpReportContactMgr::removeContactPointImpl( hkContactPointId cpId, hkCollisionConstraintOwner& info )
{
		//
		// fire all events
		//
	hkpContactPointRemovedEvent event( cpId, this, &info, HK_NULL, m_bodyA, m_bodyB);
	hkFireContactPointRemovedCallback( m_world, m_bodyA, m_bodyB, event );
}

void hkpReportContactMgr::processContactImpl( const hkpCollidable& a, const hkpCollidable& b, const hkpProcessCollisionInput& input, hkpProcessCollisionData& collisionData )
{
	//
	//	fire all events
	//
	{
		if ( m_skipNextNprocessCallbacks-- != 0) 
		{
			return;
		}

		hkpRigidBody* rba = static_cast<hkpRigidBody*>(a.getOwner());
		hkpRigidBody* rbb = static_cast<hkpRigidBody*>(b.getOwner());

		m_skipNextNprocessCallbacks = hkMath::min2( rba->getContactPointCallbackDelay(), rbb->getContactPointCallbackDelay() );

		//
		// fire all events using frequency information
		//
		{
			hkpContactProcessEvent event( this, &a,&b, &collisionData );
#if !defined(HK_PLATFORM_SPU)
			{
				for ( int i = collisionData.getNumContactPoints()-1; i>=0; i-- )
				{
					event.m_contactPointProperties[i] = HK_NULL;
				}
			}
#endif
			hkFireContactProcessCallback( m_world, rba, rbb, event );
		}
	}
}

#if !defined(HK_PLATFORM_SPU)
void hkpReportContactMgr::confirmToi( struct hkpToiEvent& event, hkReal rotateNormal, class hkArray<class hkpEntity*>& outToBeActivated )
{
	// <TODO> This won't work but it never gets called anyway.
	hkpSimpleConstraintContactMgr* mgr = static_cast< hkpSimpleConstraintContactMgr* >( event.m_contactMgr );
	hkpContactPointEvent cpEvent( hkpCollisionEvent::SOURCE_WORLD, static_cast<hkpRigidBody*>( event.m_entities[0] ), static_cast<hkpRigidBody*>( event.m_entities[1] ), mgr,
		hkpContactPointEvent::TYPE_TOI,
		&event.m_contactPoint, &event.m_properties,
		&event.m_seperatingVelocity, &rotateNormal, 
		false, false, false,
		reinterpret_cast< hkpShapeKey* >( &event.m_extendedUserDatas ),
		HK_NULL, HK_NULL );
	
	hkpWorld* world = event.m_entities[0]->getWorld();
	hkpWorldCallbackUtil::fireContactPointCallback( world, cpEvent );

	cpEvent.m_source = hkpCollisionEvent::SOURCE_A;
	hkpEntityCallbackUtil::fireContactPointCallback( event.m_entities[0], cpEvent );

	cpEvent.m_source = hkpCollisionEvent::SOURCE_B;
	hkpEntityCallbackUtil::fireContactPointCallback( event.m_entities[1], cpEvent );
}

hkpReportContactMgr::Factory::Factory(hkpWorld *mgr)
{
	m_world = mgr;
}

hkpContactMgr*	hkpReportContactMgr::Factory::createContactMgr( const hkpCollidable& a, const hkpCollidable& b, const hkpCollisionInput& env )
{
	hkpRigidBody* bodyA = reinterpret_cast<hkpRigidBody*>(a.getOwner() );
	hkpRigidBody* bodyB = reinterpret_cast<hkpRigidBody*>(b.getOwner() );

	hkpReportContactMgr *mgr = new hkpReportContactMgr( m_world, bodyA, bodyB);
	return mgr;
}
#endif

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
