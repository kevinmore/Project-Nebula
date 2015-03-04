/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Utilities/Collide/ContactModifiers/CenterOfMassChanger/hkpCenterOfMassChangerUtil.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpAddModifierUtil.h>

#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBodyCinfo.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>


hkpCenterOfMassChangerUtil::hkpCenterOfMassChangerUtil( hkpRigidBody* bodyA, hkpRigidBody* bodyB, const hkVector4& displacementA, const hkVector4& displacementB )
{
	m_bodyA = bodyA;
	m_bodyB = bodyB;

	// We only make space in bodyA's collisions.
	m_bodyA->m_responseModifierFlags |= hkpResponseModifier::CENTER_OF_MASS_DISPLACEMENT;

	m_displacementA = displacementA;
	m_displacementB = displacementB;

	m_bodyA->addContactListener( this );
	m_bodyA->addEntityListener( this );

	this->addReference();
}

hkpCenterOfMassChangerUtil::~hkpCenterOfMassChangerUtil()
{
	if(m_bodyA)
	{
		m_bodyA->removeContactListener( this );
		m_bodyA->removeEntityListener( this );
	}
}


void hkpCenterOfMassChangerUtil::contactPointCallback( const hkpContactPointEvent& event )
{
	if ( event.m_bodies[0]->getMaterial().getResponseType() != hkpMaterial::RESPONSE_SIMPLE_CONTACT )
	{
		return;
	}
	if ( event.m_bodies[1]->getMaterial().getResponseType() != hkpMaterial::RESPONSE_SIMPLE_CONTACT )
	{
		return;
	}

	hkpRigidBody* bodyA = event.m_bodies[0];
	hkpRigidBody* bodyB = event.m_bodies[1];

	// The bodies could be in either order so we have to check both cases
	if ( ( ( bodyA == m_bodyA ) && (bodyB == m_bodyB ) ) ||	( ( bodyB == m_bodyA) && (bodyA == m_bodyB ) ) )
	{
		hkpAddModifierUtil::setCenterOfMassDisplacementForContact( event, m_bodyA, m_bodyB, m_displacementA, m_displacementB );
	}
}


void hkpCenterOfMassChangerUtil::entityDeletedCallback( hkpEntity* entity )
{
	HK_ASSERT2(0x76abe9fb, entity == m_bodyA, "hkpCenterOfMassChangerUtil received an unexpected entity deleted callback");
	entity->removeContactListener( this );
	entity->removeEntityListener( this );
	m_bodyA = HK_NULL;
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
