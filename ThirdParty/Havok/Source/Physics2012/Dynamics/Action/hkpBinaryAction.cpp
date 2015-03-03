/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Action/hkpBinaryAction.h>
#include <Physics2012/Dynamics/Entity/hkpEntity.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

hkpBinaryAction::hkpBinaryAction(hkpEntity* entityA, hkpEntity* entityB, hkUlong userData)
:	hkpAction( userData ),
	m_entityA(entityA),
	m_entityB(entityB)
{
	if (entityB || entityA)  // if both are null probably due to reg of vtable in serialization.
		_referenceBodies();
}


void hkpBinaryAction::_referenceBodies()
{
	HK_ASSERT2(0xf0ff0089, !getWorld(), "This function is only to be used in an action constructor");
	if (m_entityA == HK_NULL)
	{
		HK_WARN(0x7d2cd135, "hkpBinaryAction: bodyA is a NULL pointer, you can use hkpWorld::getFixedRigidBody()");	
	}
	else
	{
		m_entityA->addReference();
	}

	if (m_entityB == HK_NULL)
	{
		HK_WARN(0x4a40a3fb, "hkpBinaryAction: bodyB is a NULL pointer, you can use hkpWorld::getFixedRigidBody()");	
	}
	else
	{
		m_entityB->addReference();
	}
}


void hkpBinaryAction::entityRemovedCallback(hkpEntity* entity) 
{
	if ( getWorld() != HK_NULL )
	{
		HK_ASSERT(0xad000225, m_entityA->getWorld() && m_entityB->getWorld());
		getWorld()->removeActionImmediately( this );
	}
}

void hkpBinaryAction::getEntities( hkArray<hkpEntity*>& entitiesOut )
{
	entitiesOut.pushBack( m_entityA );
	entitiesOut.pushBack( m_entityB );
}

void hkpBinaryAction::setEntityA(hkpEntity* entityA)
{
	//HK_ASSERT2(0x5ef81388, getWorld() == HK_NULL, "hkpBinaryAction is already added to an hkpWorld. m_entityA cannot be changed.");	
	HK_ASSERT2(0x3deafe13, entityA != HK_NULL, "entityA is a NULL pointer. You can use hkpWorld::getFixedRigidBody().");	

	entityA->addReference();

	//
	// If m_entityA is being changed, remove the old hkpEntity reference and listener.
	//
	if (m_entityA != HK_NULL)
	{
		//HK_WARN(0x2d7b4a9a, "m_entityA is not NULL. This hkpBinaryAction already has an hkpEntity in m_entityA.");

		if (getWorld())
		{
			getWorld()->detachActionFromEntity(this, m_entityA);
		}
		m_entityA->removeReference();
		m_entityA = HK_NULL;
	}

	//
	// Add reference and listener for the new hkpEntity.
	//
	m_entityA = entityA;
	if (getWorld())
	{
		getWorld()->attachActionToEntity(this, m_entityA);
	}
}

void hkpBinaryAction::setEntityB(hkpEntity* entityB)
{
	//HK_ASSERT2(0x4fbad054, getWorld() == HK_NULL, "hkpBinaryAction is already added to an hkpWorld. m_entityB cannot be changed.");	
	HK_ASSERT2(0x571ab0a2, entityB != HK_NULL, "entityB is a NULL pointer. You can use hkpWorld::getFixedRigidBody().");	

	//
	// If m_entityB is being changed, remove the old hkpEntity reference and listener.
	//
	entityB->addReference();

	if (m_entityB != HK_NULL)
	{
		//HK_WARN(0x61c277a8, "m_entityB is not NULL. This hkpBinaryAction already has an hkpEntity in m_entityB.");

		if (getWorld())
		{
			getWorld()->detachActionFromEntity(this, m_entityB);
		}
		m_entityB->removeReference();
		m_entityB = HK_NULL;
	}

	//
	// Add reference and listener for the new hkpEntity.
	//
	m_entityB = entityB;
	if (getWorld())
	{
		getWorld()->attachActionToEntity(this, m_entityB);
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
