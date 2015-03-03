/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Collide/ContactModifiers/SoftContact/hkpSoftContactUtil.h>
#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBodyCinfo.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpAddModifierUtil.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

hkpSoftContactUtil::hkpSoftContactUtil( hkpRigidBody* bodyA, hkpRigidBody* optionalBodyB, hkReal forceScale, hkReal maxAccel )
{
	m_bodyA = bodyA;
	m_bodyB = optionalBodyB;

	m_bodyA->m_responseModifierFlags |= hkpResponseModifier::IMPULSE_SCALING;

	m_bodyA->addContactListener( this );
	m_bodyA->addEntityListener( this );

	m_forceScale = forceScale;
	m_maxAcceleration = maxAccel;

	this->addReference();
}

hkpSoftContactUtil::~hkpSoftContactUtil()
{
	if(m_bodyA)
	{
		m_bodyA->removeContactListener( this );
		m_bodyA->removeEntityListener( this );
	}
}


void hkpSoftContactUtil::contactPointCallback( const hkpContactPointEvent& event )
{
	hkpRigidBody* bodyA = event.m_bodies[0];
	hkpRigidBody* bodyB = event.m_bodies[1];

	hkpRigidBody* b = (m_bodyB)? m_bodyB : hkSelectOther( m_bodyA, bodyA, bodyB );

	// The bodies could be in either order so we have to check both cases
	if ( ( (bodyA == m_bodyA) && (bodyB == b) ) || ( (bodyB == m_bodyA) && (bodyA == b) ) )
	{
		hkpAddModifierUtil::setImpulseScalingForContact( event, bodyA, bodyB, m_forceScale, m_maxAcceleration );
	}
}


void hkpSoftContactUtil::entityDeletedCallback( hkpEntity* entity )
{
	HK_ASSERT2(0x76abe9fb, entity == m_bodyA, "hkpSoftContactUtil received an unexpected entity deleted callback");
	entity->removeContactListener( this );
	entity->removeEntityListener( this );
	m_bodyA = HK_NULL;
	m_bodyB = HK_NULL;

	this->removeReference();
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
