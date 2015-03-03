/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/WheelCollide/RayCast/hknpVehicleRayCastWheelCollide.h>
#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>

// A little helper
static HK_FORCE_INLINE const hknpMotion& _getMotion( const hknpWorld* world, hknpBodyId bodyId )
{
	return world->getMotion( world->getBody(bodyId).m_motionId );
}

static bool enableTodoErrors = true;

hknpVehicleRayCastWheelCollide::hknpVehicleRayCastWheelCollide()
:	m_wheelCollisionFilterInfo( 0 )
{
	m_alreadyUsed = false;
	m_type = RAY_CAST_WHEEL_COLLIDE;
}

void hknpVehicleRayCastWheelCollide::init( const hknpVehicleInstance* vehicle )
{

}

hknpVehicleRayCastWheelCollide::~hknpVehicleRayCastWheelCollide()
{

}

void hknpVehicleRayCastWheelCollide::calcWheelsAABB( const hknpVehicleInstance* vehicle, hkAabb& aabbOut ) const
{
	aabbOut.m_min.setMin( vehicle->m_wheelsInfo[0].m_hardPointWs, vehicle->m_wheelsInfo[0].m_rayEndPointWs );
	aabbOut.m_max.setMax( vehicle->m_wheelsInfo[0].m_hardPointWs, vehicle->m_wheelsInfo[0].m_rayEndPointWs );

	for (int w_it=1; w_it<vehicle->m_data->m_numWheels; w_it ++)
	{
		const hknpVehicleInstance::WheelInfo &wheel_info = vehicle->m_wheelsInfo[w_it];
		aabbOut.m_min.setMin( aabbOut.m_min, wheel_info.m_rayEndPointWs );
		aabbOut.m_min.setMin( aabbOut.m_min, wheel_info.m_hardPointWs );
		aabbOut.m_max.setMax( aabbOut.m_max, wheel_info.m_rayEndPointWs );
		aabbOut.m_max.setMax( aabbOut.m_max, wheel_info.m_hardPointWs );
	}
}

void hknpVehicleRayCastWheelCollide::updateBeforeCollisionDetection( const hknpVehicleInstance* vehicle )
{
}

void hknpVehicleRayCastWheelCollide::castSingleWheel( const hknpVehicleInstance::WheelInfo& wheelInfo, hknpWorld* const world, hknpCollisionQueryCollector* collector ) const
{
	hknpRayCastQuery input(wheelInfo.m_hardPointWs, wheelInfo.m_rayEndPointWs);
	world->castRay( input, collector );
}

void hknpVehicleRayCastWheelCollide::collideWheels( const hkReal deltaTime, const hknpVehicleInstance* vehicle, CollisionDetectionWheelOutput* cdInfoOut, hknpWorld* world )
{
	const hkUint8 numWheels = vehicle->m_data->m_numWheels;
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		CollisionDetectionWheelOutput& cd_wheelInfo = cdInfoOut[i];
		const hknpVehicleInstance::WheelInfo& wheel_info = vehicle->m_wheelsInfo[i];

		hknpClosestHitCollector rayCastOutput;
		castSingleWheel( wheel_info, world, &rayCastOutput );

		if ( rayCastOutput.hasHit() )
		{
			getCollisionOutputFromCastResult( vehicle, i, rayCastOutput, cd_wheelInfo, world );
		}
		else
		{
			getCollisionOutputWithoutHit( vehicle, i, cd_wheelInfo );
		}
		wheelCollideCallback( vehicle, i, cd_wheelInfo );
	}
}

void hknpVehicleRayCastWheelCollide::getCollisionOutputFromCastResult( const hknpVehicleInstance* vehicle, hkUint8 wheelInfoNum, const hknpClosestHitCollector& rayCastOutput, CollisionDetectionWheelOutput& cdInfo, hknpWorld* world ) const
{
	HK_ASSERT2( 0x1ee3f67b, rayCastOutput.hasHit(), "Cannot convert to collision output when there is no hit." );

	const hkReal suspensionLength = vehicle->m_suspension->m_wheelParams[wheelInfoNum].m_length;
	const hknpVehicleInstance::WheelInfo& wheel_info = vehicle->m_wheelsInfo[wheelInfoNum];

	cdInfo.m_contactPoint.setNormalOnly( rayCastOutput.getHits()[0].m_normal );
	cdInfo.m_contactShapeKey = rayCastOutput.getHits()[0].m_hitBodyInfo.m_shapeKey;
	cdInfo.m_contactBodyId = rayCastOutput.getHits()[0].m_hitBodyInfo.m_bodyId;

	const hkReal wheelRadius = vehicle->m_data->m_wheelParams[wheelInfoNum].m_radius;
	hkReal hitDistance = rayCastOutput.getHits()[0].m_fraction * ( suspensionLength + wheelRadius );
	cdInfo.m_currentSuspensionLength = hitDistance - wheelRadius;

	hkVector4 contactPointWsPosition; contactPointWsPosition.setAddMul( wheel_info.m_hardPointWs, wheel_info.m_suspensionDirectionWs, hkSimdReal::fromFloat(hitDistance) );
	cdInfo.m_contactPoint.setPosition( contactPointWsPosition );
	cdInfo.m_contactPoint.setDistance( cdInfo.m_currentSuspensionLength );
	cdInfo.m_contactFriction = world->getMaterialLibrary()->getEntry( world->getBody( cdInfo.m_contactBodyId ).m_materialId ).m_dynamicFriction;

	// Let theta be the angle between the contact normal and the suspension direction.
	hkSimdReal cosTheta = cdInfo.m_contactPoint.getNormal().dot<3>( wheel_info.m_suspensionDirectionWs );
	HK_ASSERT( 0x66b55978,  cosTheta.isLessZero() );

	if ( cosTheta < -hkSimdReal::fromFloat(vehicle->m_data->m_normalClippingAngleCos) )
	{
		//
		// calculate the suspension velocity
		//
		hkVector4 chassis_velocity_at_contactPoint;
		vehicle->getChassisMotion().getPointVelocity( cdInfo.m_contactPoint.getPosition(), chassis_velocity_at_contactPoint );

		hkVector4 groundVelocityAtContactPoint;
		_getMotion(world, cdInfo.m_contactBodyId).getPointVelocity( cdInfo.m_contactPoint.getPosition(), groundVelocityAtContactPoint );

		hkVector4 chassisRelativeVelocity; chassisRelativeVelocity.setSub( chassis_velocity_at_contactPoint, groundVelocityAtContactPoint );

		hkSimdReal projVel = cdInfo.m_contactPoint.getNormal().dot<3>( chassisRelativeVelocity );

		hkSimdReal inv; inv.setReciprocal(cosTheta); inv = -inv;
		cdInfo.m_suspensionClosingSpeed = (projVel * inv).getReal();
		cdInfo.m_suspensionScalingFactor = inv.getReal();
	}
	else
	{
		cdInfo.m_suspensionClosingSpeed = 0.0f;
		cdInfo.m_suspensionScalingFactor = 1.0f / vehicle->m_data->m_normalClippingAngleCos;
	}
}

void hknpVehicleRayCastWheelCollide::getCollisionOutputWithoutHit( const hknpVehicleInstance* vehicle, hkUint8 wheelNum, CollisionDetectionWheelOutput& cdInfo ) const
{
	const hkReal suspensionLength = vehicle->m_suspension->m_wheelParams[wheelNum].m_length;
	const hknpVehicleInstance::WheelInfo& wheel_info = vehicle->m_wheelsInfo[wheelNum];

	cdInfo.m_contactBodyId = hknpBodyId::invalid();
	cdInfo.m_currentSuspensionLength = suspensionLength;
	cdInfo.m_suspensionClosingSpeed = 0.0f;
	cdInfo.m_contactPoint.setPosition( wheel_info.m_rayEndPointWs );

	hkVector4 contactPointWsNormal; contactPointWsNormal.setNeg<4>( wheel_info.m_suspensionDirectionWs );
	cdInfo.m_contactPoint.setNormalOnly( contactPointWsNormal );
	cdInfo.m_contactFriction = 0.0f;
	cdInfo.m_suspensionScalingFactor = 1.0f;
	cdInfo.m_contactPoint.setDistance( suspensionLength );
}

void hknpVehicleRayCastWheelCollide::addToWorld( hknpWorld* world )
{
	HK_ASSERT2(0x39bd4c1f,enableTodoErrors,"Not implemented");
}

void hknpVehicleRayCastWheelCollide::removeFromWorld()
{
	HK_ASSERT2(0x465a86e4,enableTodoErrors,"Not implemented");
}

void hknpVehicleRayCastWheelCollide::setCollisionFilterInfo( hkUint32 filterInfo )
{
	HK_ASSERT2(0x279e4ec3,enableTodoErrors,"Not implemented");
	m_wheelCollisionFilterInfo = filterInfo;
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
