/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Collide/ContactModifiers/ViscoseSurface/hkpViscoseSurfaceUtil.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpAddModifierUtil.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

//
// Please do not change this file
//

hkpViscoseSurfaceUtil::hkpViscoseSurfaceUtil( hkpRigidBody* entity )
{
	HK_ASSERT2(0x6740aa92,  entity->getWorld() == HK_NULL, "You can only create a hkpViscoseSurfaceUtil BEFORE you add an entity to the world");

	entity->m_responseModifierFlags |= hkpResponseModifier::VISCOUS_SURFACE;
	m_entity = entity;

	HK_ASSERT2(0x42166b07, entity->getMaterial().getResponseType() == hkpMaterial::RESPONSE_SIMPLE_CONTACT, "The response type of the entity must be hkpMaterial::RESPONSE_SIMPLE_CONTACT" );

	entity->setContactPointCallbackDelay(0);
	entity->addContactListener( this );
	entity->addEntityListener( this );
}


void hkpViscoseSurfaceUtil::contactPointCallback( const hkpContactPointEvent& event )
{
	hkpAddModifierUtil::setLowSurfaceViscosity( event );
}


void hkpViscoseSurfaceUtil::entityDeletedCallback( hkpEntity* entity )
{		
	entity->removeContactListener( this );
	entity->removeEntityListener( this );
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
