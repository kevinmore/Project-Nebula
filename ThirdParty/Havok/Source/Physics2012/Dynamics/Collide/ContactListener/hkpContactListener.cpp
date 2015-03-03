/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactListener.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpCollisionCallbackUtil.h>
#include <Physics2012/Dynamics/Collide/ContactListener/Util/hkpEndOfStepCallbackUtil.h>

void hkpContactListener::registerForEndOfStepContactPointCallbacks( const hkpCollisionEvent& event )
{
	const hkpWorld* world = event.m_bodies[0]->getWorld();
	hkpCollisionCallbackUtil* collisionCallbackUtil = hkpCollisionCallbackUtil::findCollisionCallbackUtil( world );
	HK_ASSERT2( 0x47feba2, collisionCallbackUtil, "You must set hkpWorldCinfo::m_fireCollisionCallbacks to register a collision for end of step callbacks" );
	collisionCallbackUtil->m_endOfStepCallbackUtil.registerCollision( event.m_contactMgr, this, event.m_source );
}


void hkpContactListener::unregisterForEndOfStepContactPointCallbacks( const hkpCollisionEvent& event )
{
	const hkpWorld* world = event.m_bodies[0]->getWorld();
	hkpCollisionCallbackUtil* collisionCallbackUtil = hkpCollisionCallbackUtil::findCollisionCallbackUtil( world );
	HK_ASSERT2( 0x47feba2, collisionCallbackUtil, "You must set hkpWorldCinfo::m_fireCollisionCallbacks to unregister a collision for end of step callbacks" );
	collisionCallbackUtil->m_endOfStepCallbackUtil.unregisterCollision( event.m_contactMgr, this, event.m_source );
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
