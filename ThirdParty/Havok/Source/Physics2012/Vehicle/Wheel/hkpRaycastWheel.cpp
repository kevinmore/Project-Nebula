/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Wheel/hkpRaycastWheel.h>
#include <Physics2012/Collide/Query/Collector/RayCollector/hkpClosestRayHitCollector.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastInput.h>
#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>

hkpRaycastWheel::hkpRaycastWheel( hkpClosestRayHitCollector* collector )
{
	if ( collector == HK_NULL )
	{
		m_collector = new hkpClosestRayHitCollector();
		m_ownCollector = true;
	}
	else
	{
		m_collector = collector;
		m_ownCollector = false;
	}
}

hkpRaycastWheel::~hkpRaycastWheel()
{
	if (m_ownCollector)
	{
		delete m_collector;
	}
}

void hkpRaycastWheel::collide()
{
	hkpWorldRayCastInput input;
	input.m_from = m_hardPointWs;
	input.m_to = m_rayEndPointWs;
	input.m_enableShapeCollectionFilter = true;
	input.m_filterInfo = m_collisionFilterInfo;
	m_collector->reset();
	m_phantom->castRay(input, *m_collector);

	if (m_collector->hasHit())
	{
		const hkpWorldRayCastOutput& rayCastOutput = m_collector->getHit();
		HK_ASSERT2( 0x1ee3f67b, rayCastOutput.hasHit(), "Cannot convert to collision output when there is no hit." );

		m_contactPoint.setNormalOnly( rayCastOutput.m_normal );
		int numKeys = hkMath::min2<int>( (int) hkpShapeRayCastOutput::MAX_HIERARCHY_DEPTH, (int) hkpWheel::MAX_NUM_SHAPE_KEYS );
		for( int i = 0; i < numKeys; ++i )
		{
			m_contactShapeKey[i] = rayCastOutput.m_shapeKeys[i];
		}

		hkpRigidBody* groundRigidBody = hkpGetRigidBody( rayCastOutput.m_rootCollidable );
		HK_ASSERT2(0x1f3b75d4,  groundRigidBody, 
			"Your car raycast hit a phantom object. If you don't want this to happen, disable collisions between the wheel raycast phantom and phantoms.\
			\nTo do this, change hkpWheel::m_collisionFilterInfo or hkpCollisionFilter::isCollisionEnabled( const hkpCollidable& a, const hkpCollidable& b )");

		m_contactBody = groundRigidBody; 

		const hkReal wheelRadius = m_radius;
		hkReal hitDistance = rayCastOutput.m_hitFraction * ( m_suspensionLength + wheelRadius );
		m_currentSuspensionLength = hitDistance - wheelRadius;

		hkVector4 contactPointWsPosition; contactPointWsPosition.setAddMul( m_hardPointWs, m_suspensionDirectionWs, hkSimdReal::fromFloat( hitDistance ) );
		m_contactPoint.setPosition( contactPointWsPosition );
		m_contactPoint.setDistance( m_currentSuspensionLength );
		m_contactFriction = m_contactBody->getMaterial().getFriction();

		// Let theta be the angle between the contact normal and the suspension direction.
		hkSimdReal cosTheta = m_contactPoint.getNormal().dot<3>( m_suspensionDirectionWs );
		HK_ASSERT( 0x66b55978,  cosTheta.isLessZero() );

		if ( cosTheta < -hkSimdReal::fromFloat( m_normalClippingAngleCos ) )
		{
			//
			// calculate the suspension velocity
			// 
			hkVector4 chassisVelocityAtContactPoint;
			m_chassis->getPointVelocity( m_contactPoint.getPosition(), chassisVelocityAtContactPoint );

			hkVector4 groundVelocityAtContactPoint;
			groundRigidBody->getPointVelocity( m_contactPoint.getPosition(), groundVelocityAtContactPoint );

			hkVector4 chassisRelativeVelocity; chassisRelativeVelocity.setSub( chassisVelocityAtContactPoint, groundVelocityAtContactPoint );

			hkSimdReal projVel = m_contactPoint.getNormal().dot<3>( chassisRelativeVelocity );

			hkSimdReal inv; inv.setReciprocal(cosTheta); inv = -inv;
			m_suspensionClosingSpeed = (projVel * inv).getReal();
			m_suspensionScalingFactor = inv.getReal();
		}
		else
		{
			m_suspensionClosingSpeed = 0.0f;
			m_suspensionScalingFactor = 1.0f / m_normalClippingAngleCos;
		}
	}
	else
	{
		// No hit
		m_contactBody = HK_NULL; 
		m_currentSuspensionLength = m_suspensionLength;
		m_contactPoint.setPosition( m_rayEndPointWs );

		hkVector4 contactPointWsNormal; contactPointWsNormal.setNeg<4>( m_suspensionDirectionWs );
		m_contactPoint.setNormalOnly( contactPointWsNormal );
		m_contactFriction = 0.0f;
		m_contactPoint.setDistance( m_suspensionLength );
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
