/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Utilities/Collide/ContactModifiers/SurfaceVelocity/hkpSurfaceVelocityUtil.h>
#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpAddModifierUtil.h>

hkpSurfaceVelocityUtil::hkpSurfaceVelocityUtil(hkpRigidBody* body, const hkVector4& surfaceVelocityWorld)
{
	this->addReference();

	m_rigidBody = body;
	body->m_responseModifierFlags |= hkpResponseModifier::SURFACE_VELOCITY;
	m_surfaceVelocity = surfaceVelocityWorld;

	m_rigidBody->addContactListener( this );
	m_rigidBody->addEntityListener( this );
}


void hkpSurfaceVelocityUtil::contactPointCallback( const hkpContactPointEvent& event )
{
	hkpAddModifierUtil::setSurfaceVelocity( event, m_rigidBody, m_surfaceVelocity );
}


hkpSurfaceVelocityUtil::~hkpSurfaceVelocityUtil()
{
	if( m_rigidBody )
	{
		m_rigidBody->removeContactListener( this );
		m_rigidBody->removeEntityListener( this );
	}
}

void hkpSurfaceVelocityUtil::entityDeletedCallback( hkpEntity* entity )
{
	HK_ASSERT(0x24a5384b, entity == m_rigidBody);
	entity->removeContactListener( this );
	entity->removeEntityListener( this );
	m_rigidBody = HK_NULL;
	removeReference();
}

void hkpSurfaceVelocityUtil::setSurfaceVelocity( const hkVector4& velWorld )
{
	// performance abort if new velocity equals old velocity
	if ( m_surfaceVelocity.allExactlyEqual<3>(velWorld) )
	{
		return;
	}

	m_surfaceVelocity = velWorld;

	// iterate over all contact managers and update the modifiers' surface velocity value
	{
		hkpLinkedCollidable& collidableEx = *m_rigidBody->getLinkedCollidable();
		const hkArray<struct hkpLinkedCollidable::CollisionEntry>& collisionEntries = collidableEx.getCollisionEntriesNonDeterministic();
		for (int i = 0; i < collisionEntries.getSize(); i++)
		{
			hkpAgentNnEntry* entry = collisionEntries[i].m_agentEntry;
			HK_ASSERT(0xafff008e, entry->m_contactMgr != HK_NULL);

			hkpDynamicsContactMgr* contactManager = static_cast<hkpDynamicsContactMgr*>(entry->m_contactMgr);

			hkpConstraintInstance* instance = contactManager->getConstraintInstance();
			if ( instance && instance->m_internal )
			{
				hkpSimulationIsland* island = instance->getSimulationIsland();
				hkpResponseModifier::setSurfaceVelocity(contactManager, m_rigidBody, *island, m_surfaceVelocity);
			}
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
