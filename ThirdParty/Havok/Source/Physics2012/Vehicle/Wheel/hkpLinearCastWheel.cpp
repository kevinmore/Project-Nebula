/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Wheel/hkpLinearCastWheel.h>
#include <Physics2012/Collide/Query/CastUtil/hkpLinearCastInput.h>
#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>

hkpLinearCastWheel::hkpLinearCastWheel( hkpShape* shape, hkpClosestCdPointCollector* collector ) :
	m_shape(shape),
	m_maxExtraPenetration(HK_REAL_EPSILON),
	m_startPointTolerance(HK_REAL_EPSILON)
{
	if ( collector == HK_NULL )
	{
		m_collector = new hkpClosestCdPointCollector();
		m_ownCollector = true;
	}
	else
	{
		m_collector = collector;
		m_ownCollector = false;
	}
}

hkpLinearCastWheel::~hkpLinearCastWheel()
{
	if (m_ownCollector)
	{
		delete m_collector;
	}
}

void hkpLinearCastWheel::collide()
{
	// Get the wheel transform and create the collidable
	hkQuaternion wheelRotation;
	wheelRotation.setMul( m_chassis->getRotation(), m_steeringOrientationCs );

	hkTransform wheelTransform;
	wheelTransform.set( wheelRotation, m_hardPointWs );

	hkpCollidable wheel( m_shape, &wheelTransform );
	wheel.setCollisionFilterInfo( m_collisionFilterInfo );

	// Do the linear cast
	m_collector->reset();
	hkpLinearCastInput input;
	input.m_to.setAddMul( m_hardPointWs, m_suspensionDirectionWs, hkSimdReal::fromFloat( m_suspensionLength ) );
	input.m_maxExtraPenetration = m_maxExtraPenetration;
	input.m_startPointTolerance = m_startPointTolerance;
	m_phantom->linearCast( &wheel, input, *m_collector, HK_NULL );

	if ( m_collector->hasHit() )
	{
		hkpRootCdPoint linearCastOutput = m_collector->getHit();

		m_contactPoint = linearCastOutput.m_contact;
		m_contactBody = hkpGetRigidBody( linearCastOutput.m_rootCollidableB );
		HK_ASSERT2( 0x1f3b75d6, m_contactBody, "The wheel hit a phantom object." );
		HK_ASSERT2( 0x1f3b75d5, m_contactBody != m_chassis, "The wheel hit the chassis." );
		m_contactFriction = m_contactBody->getMaterial().getFriction();
		// The full shape key hierarchy is not available.
		m_contactShapeKey[0] = linearCastOutput.m_shapeKeyB;
		m_contactShapeKey[1] = HK_INVALID_SHAPE_KEY;
		m_currentSuspensionLength = m_suspensionLength * linearCastOutput.m_contact.getDistance();

		// Let theta be the angle between the contact normal and the suspension direction.
		hkSimdReal cosTheta = m_contactPoint.getNormal().dot<3>( m_suspensionDirectionWs );

		if ( cosTheta < -hkSimdReal::fromFloat( m_normalClippingAngleCos ) )
		{
			//
			// calculate the suspension velocity
			// 
			hkVector4 chassisVelocityAtContactPoint;
			m_chassis->getPointVelocity( m_contactPoint.getPosition(), chassisVelocityAtContactPoint );

			hkVector4 groundVelocityAtContactPoint;
			m_contactBody->getPointVelocity( m_contactPoint.getPosition(), groundVelocityAtContactPoint );

			hkVector4 chassisRelativeVelocity; chassisRelativeVelocity.setSub( chassisVelocityAtContactPoint, groundVelocityAtContactPoint);

			hkSimdReal projVel = m_contactPoint.getNormal().dot<3>( chassisRelativeVelocity );

			hkSimdReal inv; inv.setReciprocal( cosTheta ); inv = -inv;
			m_suspensionClosingSpeed = ( projVel * inv ).getReal();
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
		m_contactBody = HK_NULL; 
		m_currentSuspensionLength = m_suspensionLength;
		m_suspensionClosingSpeed = 0.0f;
		m_contactPoint.setPosition( m_rayEndPointWs );

		hkVector4 contactPointWsNormal; contactPointWsNormal.setNeg<4>( m_suspensionDirectionWs );
		m_contactPoint.setNormalOnly( contactPointWsNormal );
		m_contactFriction = 0.0f;
		m_suspensionScalingFactor = 1.0f;
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
