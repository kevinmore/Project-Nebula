/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Collide/Deprecated/hkpCollisionListener.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#if ! defined( HK_PLATFORM_SPU )

void hkpCollisionListener::contactPointCallback( const hkpContactPointEvent& event )
{
	if ( ( event.m_contactPointProperties->m_flags & hkContactPointMaterial::CONTACT_IS_NEW ) && ( event.m_type != hkpContactPointEvent::TYPE_MANIFOLD_AT_END_OF_STEP ) )
	{
		// Build a confirmed event from this contact point event.
		hkpContactPointAddedEvent::Type type;
		hkReal rotateNormal;
		if ( event.isToi() )
		{
			type = hkpContactPointAddedEvent::TYPE_TOI;
			rotateNormal = event.getRotateNormal();
		}
		else
		{
			type = hkpContactPointAddedEvent::TYPE_MANIFOLD;
			rotateNormal = 0.0f;
		}
		const hkReal separatingVelocity = event.getSeparatingVelocity();
		hkpContactPointConfirmedEvent confirmedEvent( type, event.m_bodies[0]->getCollidable(), event.m_bodies[1]->getCollidable(), 
			&event.m_contactMgr->m_contactConstraintData, event.m_contactPoint, event.m_contactPointProperties, rotateNormal, separatingVelocity );
		switch( event.m_source )
		{
			case hkpCollisionEvent::SOURCE_A:
				confirmedEvent.m_callbackFiredFrom = event.m_bodies[0];
				break;
			case hkpCollisionEvent::SOURCE_B:
				confirmedEvent.m_callbackFiredFrom = event.m_bodies[1];
				break;
			default:
				confirmedEvent.m_callbackFiredFrom = HK_NULL;
				break;
		}
		// Fire the new event.
		contactPointConfirmedCallback( confirmedEvent );
		
		// Write back changes to these parameters (if applicable).
		event.setSeparatingVelocity( confirmedEvent.m_projectedVelocity );
		if ( event.isToi() )
		{
			event.setRotateNormal( confirmedEvent.m_rotateNormal );
		}
	}
}

#endif // !defined( HK_PLATFORM_SPU )

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
