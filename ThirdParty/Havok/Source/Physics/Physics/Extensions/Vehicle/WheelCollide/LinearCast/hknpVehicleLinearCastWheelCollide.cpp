/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/WheelCollide/LinearCast/hknpVehicleLinearCastWheelCollide.h>
#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Common/Base/hkBase.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>
#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>

// Debug and vizualization:

#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/Shape/hkDisplayBox.h>
#define HK_WHEEL_START_COLOR hkColor::GREEN
#define HK_WHEEL_END_COLOR hkColor::RED
#define HK_WHEEL_CONTACT_COLOR hkColor::WHITE
// Visualize the wheel casts in start and end state
//#define HK_DEBUG_LINEAR_CAST_WHEELS
// Visualize the wheel contact points.
//#define HK_DEBUG_WHEEL_CONTACT_POINTS


hknpVehicleLinearCastWheelCollide::hknpVehicleLinearCastWheelCollide()
:	m_maxExtraPenetration( HK_REAL_EPSILON ),
	m_startPointTolerance( HK_REAL_EPSILON )
{
	m_alreadyUsed = false;
	m_type = LINEAR_CAST_WHEEL_COLLIDE;
	m_chassisBody = hknpBodyId::INVALID;
}


void hknpVehicleLinearCastWheelCollide::setWheelShapes( const hknpVehicleInstance* vehicle, const hkArray<hknpShape*>& shapes )
{
	const int numWheels = vehicle->getNumWheels();
	HK_WARN_ON_DEBUG_IF( m_wheelStates.getSize(), 0x5efe3917, "Wheel shapes probably already provided." );
	HK_ASSERT2( 0x5efe3917, shapes.getSize() >= numWheels, "Not enough shape pointers provided." );

	// The resizing of the array is taken as an indicating that wheelShapes have been assigned.
	m_wheelStates.setSize( numWheels );

	// Create a wheel shape for each wheel.
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		m_wheelStates[i].m_shape = shapes[i];
		shapes[i]->addReference();
	}
}

namespace
{
	// Stolen from Demo\Demos\DemoCommon\Utilities\GameUtils\NpGameUtils.h
	hknpConvexShape* createCylinderShape( hkReal height, hkReal radius, int numSegments, hkVector4Parameter up )
	{
		hkArray<hkVector4> vertices(2*numSegments);
		{
			hkSimdReal heightUp; heightUp.setFromFloat( 0.5f*height);
			hkSimdReal heightDn; heightDn.setFromFloat(-0.5f*height);
			hkVector4 vertexA; vertexA.setMul( heightUp, up );
			hkVector4 vertexB; vertexB.setMul( heightDn, up );

			hkVector4 axis; axis.setSub(vertexB, vertexA);
			axis.normalize<3>();
			hkVector4 perVec;
			hkVector4Util::calculatePerpendicularVector(axis, perVec);
			perVec.normalize<3>();
			hkSimdReal rrad; rrad.setFromFloat(radius);
			perVec.mul(rrad);

			hkSimdReal inverseNumSegments; inverseNumSegments.setFromFloat(1.f/numSegments);
			hkSimdReal dphi = hkSimdReal_2*hkSimdReal_Pi*inverseNumSegments;
			for(int idx = 0; idx < numSegments; ++idx)
			{
				hkSimdReal fIdx; fIdx.setFromFloat(1.0 * idx);
				hkQuaternion q; q.setAxisAngle_Approximate(axis, dphi*fIdx);
				hkVector4 rotDir; rotDir._setRotatedDir(q, perVec);
				vertices[idx].setAdd(rotDir, vertexA);
				vertices[idx+numSegments].setAdd(rotDir, vertexB);
			}
		}

		hknpConvexShape* cylinderShape = hknpConvexShape::createFromVertices( vertices );
		return cylinderShape;
	}
}

hknpConvexShape* hknpVehicleLinearCastWheelCollide::createWheelShape( hkReal width, hkReal radius )
{
	hkVector4 vertA;
	{
		vertA.set( 0.0f, 0.0f, 0.5f * width );
	}
	hkVector4 vertB;
	{
		vertB.setNeg<3>( vertA );
	}

	const int numCylinderSegments = 32;
	hkVector4 up; up.set(0, 0, 1);
	return createCylinderShape( 0.5f * width, radius, numCylinderSegments, up );
}


void hknpVehicleLinearCastWheelCollide::init( const hknpVehicleInstance* vehicle )
{
	const int numWheels = vehicle->getNumWheels();

	// The wheel states array has zero size unless wheel shapes have been provided.
	if ( m_wheelStates.getSize() == 0 )
	{
		m_wheelStates.setSize( numWheels );
		for ( hkUint8 i = 0; i < numWheels; ++i )
		{
			const hknpVehicleData::WheelComponentParams& wheelParams = vehicle->m_data->m_wheelParams[i];
			m_wheelStates[i].m_shape = createWheelShape( wheelParams.m_width, wheelParams.m_radius );
		}
	}

	HK_ASSERT2( 0x9fee4145, m_wheelStates.getSize() == numWheels, "Wheel states set to the wrong size." );

	// Set up the other wheel state parameters.
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		// Set the other parameters.
		updateWheelState( vehicle, i );
	}

	m_chassisBody = vehicle->m_body;
}


hknpVehicleLinearCastWheelCollide::~hknpVehicleLinearCastWheelCollide()
{
	const int numWheels = m_wheelStates.getSize();
	for ( int i = 0; i < numWheels; ++i )
	{
		WheelState& wheelState = m_wheelStates[i];
		wheelState.m_shape->removeReference();
	}
}


void hknpVehicleLinearCastWheelCollide::collideWheels( const hkReal deltaTime, const hknpVehicleInstance* vehicle, CollisionDetectionWheelOutput* cdInfoOut, hknpWorld* world  )
{
	const hkUint8 numWheels = vehicle->m_data->m_numWheels;
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		CollisionDetectionWheelOutput& cd_wheelInfo = cdInfoOut[i];

		hknpCollisionResult linearCastOutput;

		if ( castSingleWheel( vehicle, i, world, linearCastOutput) )
		{
			getCollisionOutputFromCastResult( vehicle, i, linearCastOutput, world, cd_wheelInfo );
		}
		else
		{
			getCollisionOutputWithoutHit( vehicle, i, cd_wheelInfo );
		}

		wheelCollideCallback( vehicle, i, cd_wheelInfo );
	}
}


void hknpVehicleLinearCastWheelCollide::updateBeforeCollisionDetection( const hknpVehicleInstance* vehicle )
{
	const int numWheels = m_wheelStates.getSize();

	// Set up the wheel states.
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		// Adjust the wheel state parameters.
		updateWheelState( vehicle, i );
	}
}


hkBool hknpVehicleLinearCastWheelCollide::castSingleWheel( const hknpVehicleInstance* vehicle, hkUint8 wheelNum, hknpWorld* world, hknpCollisionResult& linearCastOutput) const
{
	const WheelState& wheelState = m_wheelStates[wheelNum];

	hknpClosestHitCollector target;
	{
		hknpShapeCastQuery query;
		//query.m_filterData = m_wheelCollisionFilterInfo;
		query.m_body = &world->getBody(m_chassisBody);
		query.setStartEnd(wheelState.m_transform.getTranslation(), wheelState.m_to);
		query.m_shape = wheelState.m_shape;

		world->castShape(query, wheelState.m_transform.getRotation(), &target);
	}

	if(target.hasHit())
	{
		linearCastOutput = target.getHits()[0];

		return true;
	}
	else
	{
		return false;
	}
}

void hknpVehicleLinearCastWheelCollide::getCollisionOutputFromCastResult( const hknpVehicleInstance* vehicle, hkUint8 wheelNum,
	const hknpCollisionResult& linearCastOutput, hknpWorld* world, CollisionDetectionWheelOutput& output ) const
{
	const hkReal suspensionLength = vehicle->m_suspension->m_wheelParams[wheelNum].m_length;
	const hknpVehicleInstance::WheelInfo& wheel_info = vehicle->m_wheelsInfo[wheelNum];

	const hknpMaterial& material = world->getMaterialLibrary()->getEntry(
		world->getBody(linearCastOutput.m_hitBodyInfo.m_bodyId).m_materialId);

	{
		output.m_contactPoint.set(linearCastOutput.m_position, linearCastOutput.m_normal, 0);
		output.m_contactFriction = material.m_dynamicFriction;
	}

	output.m_contactBodyId = linearCastOutput.m_hitBodyInfo.m_bodyId;
	// The full shape key hierarchy is not available.
	output.m_contactShapeKey = linearCastOutput.m_hitBodyInfo.m_shapeKey;

	{
		const hkReal wheelRadius = vehicle->m_data->m_wheelParams[wheelNum].m_radius;
		hkReal hitDistance = linearCastOutput.m_fraction * ( suspensionLength + wheelRadius );
		output.m_currentSuspensionLength = hitDistance - wheelRadius;

		hkVector4 contactPointWsPosition; contactPointWsPosition.setAddMul( wheel_info.m_hardPointWs, wheel_info.m_suspensionDirectionWs, hkSimdReal::fromFloat(hitDistance) );
		output.m_contactPoint.setPosition( contactPointWsPosition );
		output.m_contactPoint.setDistance( output.m_currentSuspensionLength );
	}

	// Let theta be the angle between the contact normal and the suspension direction.
	hkSimdReal cosTheta = output.m_contactPoint.getNormal().dot<3>( wheel_info.m_suspensionDirectionWs );
	if ( cosTheta < -hkSimdReal::fromFloat(vehicle->m_data->m_normalClippingAngleCos) )
	{
		//
		// calculate the suspension velocity
		//
		hkVector4 chassis_velocity_at_contactPoint;
		vehicle->getChassisMotion().getPointVelocity(output.m_contactPoint.getPosition(), chassis_velocity_at_contactPoint);

		hkVector4 groundVelocityAtContactPoint;
		const hknpMotion& motion = world->getMotion(world->getBody(output.m_contactBodyId).m_motionId);
		motion.getPointVelocity(output.m_contactPoint.getPosition(), groundVelocityAtContactPoint);

		hkVector4 chassisRelativeVelocity; chassisRelativeVelocity.setSub( chassis_velocity_at_contactPoint, groundVelocityAtContactPoint);

		hkSimdReal projVel = output.m_contactPoint.getNormal().dot<3>( chassisRelativeVelocity );

		hkSimdReal inv; inv.setReciprocal(cosTheta); inv = -inv;
		output.m_suspensionClosingSpeed = (projVel * inv).getReal();
		output.m_suspensionScalingFactor = inv.getReal();
	}
	else if(cosTheta.isLessZero())
	{
		getCollisionOutputWithoutHit( vehicle, wheelNum, output );
	}
	else
	{
		output.m_suspensionClosingSpeed = 0.0f;
		output.m_suspensionScalingFactor = 1.0f / vehicle->m_data->m_normalClippingAngleCos;
	}
}


void hknpVehicleLinearCastWheelCollide::getCollisionOutputWithoutHit( const hknpVehicleInstance* vehicle, hkUint8 wheelNum, CollisionDetectionWheelOutput& cdInfo ) const
{
	const hkReal suspensionLength = vehicle->m_suspension->m_wheelParams[wheelNum].m_length;
	const hknpVehicleInstance::WheelInfo& wheel_info = vehicle->m_wheelsInfo[wheelNum];

	cdInfo.m_contactBodyId = hknpBodyId::INVALID;
	cdInfo.m_currentSuspensionLength = suspensionLength;
	cdInfo.m_suspensionClosingSpeed = 0.0f;
	cdInfo.m_contactPoint.setPosition( wheel_info.m_rayEndPointWs );

	hkVector4 contactPointWsNormal; contactPointWsNormal.setNeg<4>( wheel_info.m_suspensionDirectionWs );
	cdInfo.m_contactPoint.setNormalOnly( contactPointWsNormal );
	cdInfo.m_contactFriction = 0.0f;
	cdInfo.m_suspensionScalingFactor = 1.0f;
	cdInfo.m_contactPoint.setDistance( suspensionLength );
}


void hknpVehicleLinearCastWheelCollide::calcAabbOfWheel( const hknpVehicleInstance* vehicle, hkUint8 wheelNum, hkAabb& aabbOut ) const
{
	const hknpVehicleData::WheelComponentParams& wheelParam = vehicle->m_data->m_wheelParams[wheelNum];

	hkVector4 halfExtents;
	{
		halfExtents.set( wheelParam.m_radius, wheelParam.m_radius, 0.5f * wheelParam.m_width );
	}

	hkTransform trans( m_wheelStates[wheelNum].m_transform );
	{
		trans.setTranslation( m_wheelStates[wheelNum].m_to );
	}

	hkAabbUtil::calcAabb( trans, halfExtents, hkSimdReal_0, aabbOut );
}


void hknpVehicleLinearCastWheelCollide::updateWheelState( const hknpVehicleInstance* vehicle, hkUint8 wheelNum )
{
	const hknpVehicleInstance::WheelInfo& wheelInfo = vehicle->m_wheelsInfo[wheelNum];
	WheelState& wheelState = m_wheelStates[wheelNum];
	hkQuaternion rotation;
	{
		rotation.setMul( vehicle->getChassisMotion().m_orientation, wheelInfo.m_steeringOrientationChassisSpace );

		// Get chassis forward orientation relative to its rigid body
		hkRotation& chassisOrientation = vehicle->m_data->m_chassisOrientation;
		hkRotation rot; rot.setCols( chassisOrientation.getColumn<1>(), chassisOrientation.getColumn<0>(), chassisOrientation.getColumn<2>() );
		hkQuaternion orient; orient.set( rot );

		rotation.setMul( rotation, orient );
	}
	wheelState.m_transform.set( rotation, wheelInfo.m_hardPointWs );
	wheelState.m_to.setAddMul( wheelInfo.m_hardPointWs, wheelInfo.m_suspensionDirectionWs, hkSimdReal::fromFloat(vehicle->m_suspension->m_wheelParams[wheelNum].m_length) );
}


void hknpVehicleLinearCastWheelCollide::addToWorld( hknpWorld* world )
{
	// Nothing needs to happen
}


void hknpVehicleLinearCastWheelCollide::removeFromWorld()
{
	// Nothing needs to happen
}


void hknpVehicleLinearCastWheelCollide::setCollisionFilterInfo( hkUint32 filterInfo )
{
	HK_ASSERT(0x279e4ec3,!"Not implemented");
}


void hknpVehicleLinearCastWheelCollide::wheelCollideCallback( const hknpVehicleInstance* vehicle, hkUint8 wheelIndex, CollisionDetectionWheelOutput& cdInfo )
{
	centerWheelContactPoint( vehicle, wheelIndex, cdInfo );
}


void hknpVehicleLinearCastWheelCollide::centerWheelContactPoint( const hknpVehicleInstance* vehicle, hkUint8 wheelIndex, CollisionDetectionWheelOutput& cdInfo ) const
{
	// Move the contact point position to the center plane of the wheel by translating it to wheel space,
	// zeroing its z coordinate and translating it back to world space.
	const hkTransform& transformToWorld = m_wheelStates[wheelIndex].m_transform;
	hkVector4 positionWheelSpace;
	{
		positionWheelSpace.setTransformedInversePos( transformToWorld, cdInfo.m_contactPoint.getPosition() );
		positionWheelSpace.zeroComponent<2>();
	}
	hkVector4 newPosition;
	{
		newPosition.setTransformedPos( transformToWorld, positionWheelSpace );
	}
	cdInfo.m_contactPoint.setPosition( newPosition );
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
