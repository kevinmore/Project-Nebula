/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Action/hkpUnaryAction.h>
#include <Physics2012/Dynamics/Entity/hkpEntity.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

hkpUnaryAction::hkpUnaryAction(hkpEntity* entity, hkUlong userData )
: hkpAction( userData ), m_entity(entity)
{
	if (m_entity != HK_NULL)
	{
		m_entity->addReference();
	}
	else
	{
		//	HK_WARN(0x4621e7da, "hkpUnaryAction: body is a NULL pointer");
	}
}

void hkpUnaryAction::entityRemovedCallback(hkpEntity* entity) 
{
	// Remove self from physics.
	if ( getWorld() != HK_NULL )
	{
		getWorld()->removeActionImmediately(this);
	}
}

hkpUnaryAction::~hkpUnaryAction()
{
	if (m_entity != HK_NULL)
	{
		m_entity->removeReference();
		m_entity = HK_NULL;
	}
}

// NB: Only intended to be called pre-simulation i.e. before the hkpUnaryAction is 
// added to an hkpWorld.
void hkpUnaryAction::setEntity(hkpEntity* entity)
{
	//HK_ASSERT2(0x76017ab2, getWorld() == HK_NULL, "This hkpUnaryAction is already added to an hkpWorld.");	
	HK_ASSERT2(0x5163bcc3, entity != HK_NULL, "entity is a NULL pointer. You can use hkpWorld::getFixedRigidBody().");	
	if(m_entity != HK_NULL)
	{
		HK_WARN(0x17aaa816, "m_entity is not NULL. This hkpUnaryAction already had an hkpEntity.");

		if (getWorld())
		{
			getWorld()->detachActionFromEntity(this, m_entity);
		}
		m_entity->removeReference();
		m_entity = HK_NULL;
	}

	m_entity = entity;
	m_entity->addReference();
	if (getWorld())
	{
		getWorld()->attachActionToEntity(this, m_entity);
	}
}

void hkpUnaryAction::getEntities( hkArray<hkpEntity*>& entitiesOut )
{
	entitiesOut.pushBack( m_entity );
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
