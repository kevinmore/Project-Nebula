/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpAddModifierUtil.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactListener.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactPointEvent.h>

#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>

#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>


void HK_CALL hkpAddModifierUtil::setInvMassScalingForContact( const hkpContactPointEvent& event, hkpRigidBody* bodyA, hkpRigidBody* bodyB, const hkVector4& factorA, const hkVector4& factorB )
{
	hkpSimulationIsland& island = *event.getSimulationIsland();
	if ( event.m_type != hkpContactPointEvent::TYPE_MANIFOLD )
	{
		hkpResponseModifier::setInvMassScalingForContact( event.m_contactMgr, bodyA, bodyB, island, factorA, factorB );
	}
	else
	{
		HK_ASSERT2( 0x3f4a19f8, hkpResponseModifier::MASS_SCALING & ( bodyA->m_responseModifierFlags | bodyB->m_responseModifierFlags ), "You must set the appropriate response modifier flag in one of the entities to add this modifier in a contact callback." );
		// The island is locked here to primarily prevent us changing the size of the constraint after buffers have been allocated.
		// Since we know we have made enough space for the response modifier, we can temporarily ignore the lock.
		hkMultiThreadCheck checkBackup = island.m_multiThreadCheck;
		island.m_multiThreadCheck.disableChecks();
		hkpResponseModifier::setInvMassScalingForContact( event.m_contactMgr, bodyA, bodyB, island, factorA, factorB );
		island.m_multiThreadCheck = checkBackup;
	}
}


void HK_CALL hkpAddModifierUtil::setInvMassScalingForContact( const hkpContactPointEvent& event, hkpRigidBody* body, const hkVector4& factor )
{
	hkpSimulationIsland& island = *event.getSimulationIsland();
	if ( event.m_type != hkpContactPointEvent::TYPE_MANIFOLD )
	{
		hkpResponseModifier::setInvMassScalingForContact( event.m_contactMgr, body, island, factor );
	}
	else
	{
		HK_ASSERT2( 0x3f4a19f8, hkpResponseModifier::MASS_SCALING & ( event.m_bodies[0]->m_responseModifierFlags | event.m_bodies[1]->m_responseModifierFlags ), "You must set the appropriate response modifier flag in one of the entities to add this modifier in a contact callback." );
		// The island is locked here to primarily prevent us changing the size of the constraint after buffers have been allocated.
		// Since we know we have made enough space for the response modifier, we can temporarily ignore the lock.
		hkMultiThreadCheck checkBackup = island.m_multiThreadCheck;
		island.m_multiThreadCheck.disableChecks();
		hkpResponseModifier::setInvMassScalingForContact( event.m_contactMgr, body, island, factor );
		island.m_multiThreadCheck = checkBackup;
	}
}


void HK_CALL hkpAddModifierUtil::setCenterOfMassDisplacementForContact( const hkpContactPointEvent& event, hkpRigidBody* bodyA, hkpRigidBody* bodyB, const hkVector4& displacementA, const hkVector4& displacementB )
{
	hkpSimulationIsland& island = *event.getSimulationIsland();
	if ( event.m_type != hkpContactPointEvent::TYPE_MANIFOLD )
	{
		hkpResponseModifier::setCenterOfMassDisplacementForContact( event.m_contactMgr, bodyA, bodyB, island, displacementA, displacementB );
	}
	else
	{
		HK_ASSERT2( 0x3f4a19f8, hkpResponseModifier::CENTER_OF_MASS_DISPLACEMENT & ( bodyA->m_responseModifierFlags | bodyB->m_responseModifierFlags ), "You must set the appropriate response modifier flag in one of the entities to add this modifier in a contact callback." );
		// The island is locked here to primarily prevent us changing the size of the constraint after buffers have been allocated.
		// Since we know we have made enough space for the response modifier, we can temporarily ignore the lock.
		hkMultiThreadCheck checkBackup = island.m_multiThreadCheck;
		island.m_multiThreadCheck.disableChecks();
		hkpResponseModifier::setCenterOfMassDisplacementForContact( event.m_contactMgr, bodyA, bodyB, island, displacementA, displacementB );
		island.m_multiThreadCheck = checkBackup;
	}
}


void HK_CALL hkpAddModifierUtil::setImpulseScalingForContact( const hkpContactPointEvent& event, hkpRigidBody* bodyA, hkpRigidBody* bodyB, hkReal usedImpulseFraction, hkReal maxAcceleration )
{
	hkpSimulationIsland& island = *event.getSimulationIsland();
	if ( event.m_type != hkpContactPointEvent::TYPE_MANIFOLD )
	{
		hkpResponseModifier::setImpulseScalingForContact( event.m_contactMgr, bodyA, bodyB, island, usedImpulseFraction, maxAcceleration );
	}
	else
	{
		HK_ASSERT2( 0x3f4a19f8, hkpResponseModifier::IMPULSE_SCALING & ( bodyA->m_responseModifierFlags | bodyB->m_responseModifierFlags ), "You must set the appropriate response modifier flag in one of the entities to add this modifier in a contact callback." );
		// The island is locked here to primarily prevent us changing the size of the constraint after buffers have been allocated.
		// Since we know we have made enough space for the response modifier, we can temporarily ignore the lock.
		hkMultiThreadCheck checkBackup = island.m_multiThreadCheck;
		island.m_multiThreadCheck.disableChecks();
		hkpResponseModifier::setImpulseScalingForContact( event.m_contactMgr, bodyA, bodyB, island, usedImpulseFraction, maxAcceleration );
		island.m_multiThreadCheck = checkBackup;
	}
}

void HK_CALL hkpAddModifierUtil::setSurfaceVelocity( const hkpContactPointEvent& event, hkpRigidBody* body, const hkVector4& velWorld )
{
	hkpSimulationIsland& island = *event.getSimulationIsland();
	if ( event.m_type != hkpContactPointEvent::TYPE_MANIFOLD )
	{
		hkpResponseModifier::setSurfaceVelocity( event.m_contactMgr, body, island, velWorld );
	}
	else
	{
		HK_ASSERT2( 0x3f4a19f8, hkpResponseModifier::SURFACE_VELOCITY & ( event.m_bodies[0]->m_responseModifierFlags | event.m_bodies[1]->m_responseModifierFlags ), "You must set the appropriate response modifier flag in one of the entities to add this modifier in a contact callback." );
		// The island is locked here to primarily prevent us changing the size of the constraint after buffers have been allocated.
		// Since we know we have made enough space for the response modifier, we can temporarily ignore the lock.
		hkMultiThreadCheck checkBackup = island.m_multiThreadCheck;
		island.m_multiThreadCheck.disableChecks();
		hkpResponseModifier::setSurfaceVelocity( event.m_contactMgr, body, island, velWorld );
		island.m_multiThreadCheck = checkBackup;
	}
}


void HK_CALL hkpAddModifierUtil::clearSurfaceVelocity( const hkpContactPointEvent& event, hkpRigidBody* body )
{
	hkpSimulationIsland& island = *event.getSimulationIsland();
	if ( event.m_type != hkpContactPointEvent::TYPE_MANIFOLD )
	{
		hkpResponseModifier::clearSurfaceVelocity( event.m_contactMgr, island, body );
	}
	else
	{
		HK_ASSERT2( 0x3f4a19f8, hkpResponseModifier::SURFACE_VELOCITY & ( event.m_bodies[0]->m_responseModifierFlags | event.m_bodies[1]->m_responseModifierFlags ), "You must set the appropriate response modifier flag in one of the entities to add this modifier in a contact callback." );
		// The island is locked here to primarily prevent us changing the size of the constraint after buffers have been allocated.
		// Since we know we have made enough space for the response modifier, we can temporarily ignore the lock.
		hkMultiThreadCheck checkBackup = island.m_multiThreadCheck;
		island.m_multiThreadCheck.disableChecks();
		hkpResponseModifier::clearSurfaceVelocity( event.m_contactMgr, island, body );
		island.m_multiThreadCheck = checkBackup;
	}
}


void HK_CALL hkpAddModifierUtil::setLowSurfaceViscosity( const hkpContactPointEvent& event )
{
	hkpSimulationIsland& island = *event.getSimulationIsland();
	if ( event.m_type != hkpContactPointEvent::TYPE_MANIFOLD )
	{
		hkpResponseModifier::setLowSurfaceViscosity( event.m_contactMgr, island );
	}
	else
	{
		HK_ASSERT2( 0x3f4a19f8, hkpResponseModifier::VISCOUS_SURFACE & ( event.m_bodies[0]->m_responseModifierFlags | event.m_bodies[1]->m_responseModifierFlags ), "You must set the appropriate response modifier flag in one of the entities to add this modifier in a contact callback." );
		// The island is locked here to primarily prevent us changing the size of the constraint after buffers have been allocated.
		// Since we know we have made enough space for the response modifier, we can temporarily ignore the lock.
		hkMultiThreadCheck checkBackup = island.m_multiThreadCheck;
		island.m_multiThreadCheck.disableChecks();
		hkpResponseModifier::setLowSurfaceViscosity( event.m_contactMgr, island );
		island.m_multiThreadCheck = checkBackup;
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
