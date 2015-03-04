/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>
#include <Physics2012/Dynamics/World/Util/BroadPhase/hkpEntityEntityBroadPhaseListener.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>


hkpEntityEntityBroadPhaseListener::hkpEntityEntityBroadPhaseListener( hkpWorld* world)
{
	m_world = world;
}

void hkpEntityEntityBroadPhaseListener::addCollisionPair( hkpTypedBroadPhaseHandlePair& pair )
{
	hkpLinkedCollidable* collA = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
	hkpLinkedCollidable* collB = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );

	// check for disabled collisions, especially landscape = landscape ones
	const hkpProcessCollisionInput* input = m_world->getCollisionInput();
	{
		hkpCollidableQualityType qt0 = collA->getQualityType();
		hkpCollidableQualityType qt1 = collB->getQualityType();
		hkChar collisionQuality = input->m_dispatcher->getCollisionQualityIndex( qt0, qt1 );
		if ( collisionQuality == hkpCollisionDispatcher::COLLISION_QUALITY_INVALID )
		{
			return;
		}
		hkpCollisionQualityInfo* origInfo = input->m_dispatcher->getCollisionQualityInfo( collisionQuality );
		input->m_createPredictiveAgents = origInfo->m_useContinuousPhysics;
	}

#if defined HK_DEBUG
	// check the motion type
	hkpRigidBody* bodyA = hkpGetRigidBody(collA);
	hkpRigidBody* bodyB = hkpGetRigidBody(collB);
	hkpCollidableQualityType qualityA = bodyA->getCollidable()->getQualityType();
	hkpCollidableQualityType qualityB = bodyB->getCollidable()->getQualityType();
	if( bodyA && bodyB && ( qualityA == HK_COLLIDABLE_QUALITY_FIXED || qualityA == HK_COLLIDABLE_QUALITY_KEYFRAMED ) && ( qualityB == HK_COLLIDABLE_QUALITY_FIXED || qualityB == HK_COLLIDABLE_QUALITY_KEYFRAMED ) )
	{
		HK_WARN( 0xad16c0e6, "Creating an agent between two fixed or keyframed objects. Check the quality types." );
	}
#endif

	hkpWorldAgentUtil::addAgent(collA, collB, *input);
}


void hkpEntityEntityBroadPhaseListener::removeCollisionPair( hkpTypedBroadPhaseHandlePair& pair )
{
	hkpLinkedCollidable* collA = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_a)->getOwner() );
	hkpLinkedCollidable* collB = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(pair.m_b)->getOwner() );

	hkpAgentNnEntry* entry = hkAgentNnMachine_FindAgent(collA, collB);

	if (entry)
	{
		hkpWorldAgentUtil::removeAgent(entry);
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
