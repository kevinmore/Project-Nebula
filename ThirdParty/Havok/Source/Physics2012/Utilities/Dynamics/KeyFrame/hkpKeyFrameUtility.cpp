/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Utilities/Dynamics/KeyFrame/hkpKeyFrameUtility.h>

// hkVector4::equals3() default epsilon (1e-3) is potentially too big
#define EQUALITY_EPSILON hkSimdReal::fromFloat(1e-5f)


void hkpKeyFrameUtility::KeyFrameInfo::fastSetUsingPositionOrientationPair( const hkVector4& currentPosition, const hkQuaternion& currentOrientation, const hkVector4& nextPosition, const hkQuaternion& nextOrientation, hkReal invDeltaTime )
{
	hkSimdReal invDeltaTimeSr; invDeltaTimeSr.setFromFloat(invDeltaTime);
	// Get linear velocity required
	{
		m_position = currentPosition;
		m_linearVelocity.setSub(nextPosition, currentPosition);
		m_linearVelocity.setMul(invDeltaTimeSr, m_linearVelocity);
	}

	// Get angular velocity required. Use approx: sin(theta) ~ theta for small theta.
	{
		m_orientation = currentOrientation;
		hkQuaternion quatDif;	quatDif.setMulInverse(nextOrientation, currentOrientation);
		m_angularVelocity.setMul(hkSimdReal::getConstant<HK_QUADREAL_2>() * invDeltaTimeSr, quatDif.getImag());
		m_angularVelocity.setFlipSign(m_angularVelocity, quatDif.getRealPart().lessZero());
	}	
}

hkpKeyFrameUtility::AccelerationInfo::AccelerationInfo()
{
	const hkVector4 one = hkVector4::getConstant<HK_QUADREAL_1>();
	m_linearPositionFactor = one;
	m_angularPositionFactor = one;
	m_linearVelocityFactor = one;
	m_angularVelocityFactor = one;
	m_maxLinearAcceleration = HK_REAL_MAX;
	m_maxAngularAcceleration = HK_REAL_MAX;
	m_maxAllowedDistance = HK_REAL_MAX;
}

void hkpKeyFrameUtility::KeyFrameInfo::setUsingPositionOrientationPair( const hkVector4& currentPosition, const hkQuaternion& currentOrientation, const hkVector4& nextPosition, const hkQuaternion& nextOrientation, hkReal invDeltaTime )
{
	hkSimdReal invDeltaTimeSr; invDeltaTimeSr.setFromFloat(invDeltaTime);
	// Get linear velocity required
	{
		m_position = currentPosition;
		m_linearVelocity.setSub(nextPosition, currentPosition);
		m_linearVelocity.setMul(invDeltaTimeSr, m_linearVelocity);
	}

	// Get angular velocity required.
	{
		m_orientation = currentOrientation;
		hkQuaternion quatDif;	quatDif.setMulInverse(nextOrientation, currentOrientation);

		quatDif.normalize();

		if ( !quatDif.hasValidAxis() )
		{
			m_angularVelocity.setZero();
		}
		else
		{
			const hkSimdReal angle = hkSimdReal::fromFloat(quatDif.getAngle());
			quatDif.getAxis(m_angularVelocity);
			m_angularVelocity.setMul(angle * invDeltaTimeSr, m_angularVelocity);		
		}	
	}
}



void HK_CALL hkpKeyFrameUtility::applySoftKeyFrame( const KeyFrameInfo& keyFrameInfo, AccelerationInfo& accelInfo, hkReal deltaTime,  hkReal invDeltaTime, hkpRigidBody* body )
{
	HK_ASSERT2(0x34182658, accelInfo.m_linearPositionFactor(0) <= invDeltaTime, "SoftKeyframe will be unstable if linear position factor is greater than 1/detaTime");
	HK_ASSERT2(0x4cd1fff6, accelInfo.m_linearPositionFactor(1) <= invDeltaTime, "SoftKeyframe will be unstable if linear position factor is greater than 1/detaTime");
	HK_ASSERT2(0x67023fd1, accelInfo.m_linearPositionFactor(2) <= invDeltaTime, "SoftKeyframe will be unstable if linear position factor is greater than 1/detaTime");
	HK_ASSERT2(0x2b641943, accelInfo.m_angularPositionFactor(0) <= invDeltaTime, "SoftKeyframe will be unstable if angular position factor is greater than 1/detaTime");
	HK_ASSERT2(0x53542f40, accelInfo.m_angularPositionFactor(1) <= invDeltaTime, "SoftKeyframe will be unstable if angular position factor is greater than 1/detaTime");
	HK_ASSERT2(0x659b5db2, accelInfo.m_angularPositionFactor(2) <= invDeltaTime, "SoftKeyframe will be unstable if angular position factor is greater than 1/detaTime");

	HK_ASSERT2(0x63571671, accelInfo.m_linearVelocityFactor(0) <= invDeltaTime, "SoftKeyframe will be unstable if linear velocity factor is greater than 1/detaTime");
	HK_ASSERT2(0x649deeae, accelInfo.m_linearVelocityFactor(1) <= invDeltaTime, "SoftKeyframe will be unstable if linear velocity factor is greater than 1/detaTime");
	HK_ASSERT2(0x7d8c14ac, accelInfo.m_linearVelocityFactor(2) <= invDeltaTime, "SoftKeyframe will be unstable if linear velocity factor is greater than 1/detaTime");
	HK_ASSERT2(0x70b3e40f, accelInfo.m_angularVelocityFactor(0) <= invDeltaTime, "SoftKeyframe will be unstable if angular velocity factor is greater than 1/detaTime");
	HK_ASSERT2(0x46dcedad, accelInfo.m_angularVelocityFactor(1) <= invDeltaTime, "SoftKeyframe will be unstable if angular velocity factor is greater than 1/detaTime");
	HK_ASSERT2(0x3516ed27, accelInfo.m_angularVelocityFactor(2) <= invDeltaTime, "SoftKeyframe will be unstable if angular velocity factor is greater than 1/detaTime");

	hkSimdReal deltaTime4; deltaTime4.setFromFloat(deltaTime);

	// calculate the current delta position/orientations
	hkVector4 deltaPosition;	deltaPosition.setSub(    keyFrameInfo.m_position, body->getPosition() );
	
	//
	// Check whether our distance gets too big, so we have to warp it
	//
	hkSimdReal maxAllowed; maxAllowed.setFromFloat(accelInfo.m_maxAllowedDistance);
	hkSimdReal bodyMaxAngular2;
	hkSimdReal bodyMaxLinear2;
	{
		hkSimdReal bodyMaxAngular; bodyMaxAngular.setFromFloat(body->getMaxAngularVelocity());
		hkSimdReal bodyMaxLinear; bodyMaxLinear.setFromFloat(body->getMaxLinearVelocity());
		bodyMaxAngular2 = bodyMaxAngular * bodyMaxAngular;
		bodyMaxLinear2 = bodyMaxLinear * bodyMaxLinear;
	}

	if ( deltaPosition.lengthSquared<3>() > maxAllowed * maxAllowed )
	{
		deltaPosition.setZero();
		body->setPositionAndRotation( keyFrameInfo.m_position, keyFrameInfo.m_orientation );

		if (keyFrameInfo.m_angularVelocity.lengthSquared<3>() > bodyMaxAngular2 ||
			keyFrameInfo.m_linearVelocity.lengthSquared<3>() > bodyMaxLinear2)
		{
			HK_WARN(0x253b0986, "Target angular or linear velocity exceeds max angular or linear velocity of body, so it will be clipped.");
		}
		body->setAngularVelocity( keyFrameInfo.m_angularVelocity );
		body->setLinearVelocity( keyFrameInfo.m_linearVelocity );
	}


	hkQuaternion quatDif;		quatDif.setMulInverse(keyFrameInfo.m_orientation, body->getRotation());

	hkVector4 deltaOrientation; deltaOrientation.setMul(hkSimdReal::getConstant<HK_QUADREAL_2>() , quatDif.getImag() );
	deltaOrientation.setFlipSign(deltaOrientation, quatDif.getRealPart().lessZero());

	hkVector4 scaledDeltaPosition;    scaledDeltaPosition.setMul( accelInfo.m_linearPositionFactor, deltaPosition );
	hkVector4 scaledDeltaOrientation; scaledDeltaOrientation.setMul( accelInfo.m_angularPositionFactor, deltaOrientation );


		//
		//	Calc the part based on velocity difference
		//
	hkVector4 linVelFactor;  linVelFactor.setMul( deltaTime4, accelInfo.m_linearVelocityFactor );
	hkVector4 angVelFactor;  angVelFactor.setMul( deltaTime4, accelInfo.m_angularVelocityFactor );

	hkVector4 deltaLinVel; deltaLinVel.setSub( keyFrameInfo.m_linearVelocity, body->getLinearVelocity() );
	hkVector4 deltaAngVel; deltaAngVel.setSub( keyFrameInfo.m_angularVelocity, body->getAngularVelocity() );

	deltaLinVel.mul( linVelFactor );
	deltaAngVel.mul( angVelFactor );


		//
		//	Add everything together
		//
	deltaLinVel.add( scaledDeltaPosition );
	deltaAngVel.add( scaledDeltaOrientation );


		//
		//	clip values
		//
	{
		// clip linear
		const hkSimdReal maxLinDelta = deltaTime4 * hkSimdReal::fromFloat(accelInfo.m_maxLinearAcceleration);
		const hkSimdReal deltaLinVelLength2 = deltaLinVel.lengthSquared<3>();

		if ( deltaLinVelLength2 > maxLinDelta*maxLinDelta )
		{
			const hkSimdReal f = maxLinDelta * deltaLinVelLength2.sqrtInverse();
			deltaLinVel.mul( f );
		}

		// clip angular
		const hkSimdReal maxAngDelta = deltaTime4 * hkSimdReal::fromFloat(accelInfo.m_maxAngularAcceleration);
		const hkSimdReal deltaAngVelLength2 = deltaAngVel.lengthSquared<3>();

		if ( deltaAngVelLength2 > maxAngDelta*maxAngDelta )
		{
			const hkSimdReal f = maxAngDelta * deltaAngVelLength2.sqrtInverse();
			deltaAngVel.mul( f );
		}
	}
		//	apply values
		//
	{
		hkVector4 newAngVel; newAngVel.setAdd( body->getAngularVelocity(), deltaAngVel );
		hkVector4 newLinVel; newLinVel.setAdd( body->getLinearVelocity(),  deltaLinVel );

		if (newAngVel.lengthSquared<3>() > bodyMaxAngular2 ||
			newLinVel.lengthSquared<3>() > bodyMaxLinear2)
		{
			HK_WARN(0x7bfd0db2, "Target angular or linear velocity exceeds max angular or linear velocity of body, so it will be clipped.");
		}

		// Avoid setting velocities if possible, as it would activate the body
		const hkSimdReal eps = EQUALITY_EPSILON;
		if( !body->getAngularVelocity().allEqual<3>( newAngVel, eps ) )
		{
			body->setAngularVelocity( newAngVel );
		}
		if( !body->getLinearVelocity().allEqual<3>( newLinVel, eps ) )
		{
			body->setLinearVelocity( newLinVel );
		}
	}
}


void HK_CALL hkpKeyFrameUtility::applyHardKeyFrame( const hkVector4& nextPosition, const hkQuaternion& nextOrientation, hkReal invDeltaTime, hkpRigidBody* body )
{
	hkSimdReal invDeltaTimeSr; invDeltaTimeSr.setFromFloat(invDeltaTime);
	const hkSimdReal eps = EQUALITY_EPSILON;

	// Linear part
	{
		// Get required velocity
		hkVector4 linearVelocity;
		{
			hkVector4 newCenterOfMassPosition;
			newCenterOfMassPosition._setRotatedDir( nextOrientation, body->getCenterOfMassLocal() );
			newCenterOfMassPosition.add( nextPosition );
			linearVelocity.setSub( newCenterOfMassPosition, body->getCenterOfMassInWorld() );

			linearVelocity.setMul( invDeltaTimeSr, linearVelocity );
		}

		hkSimdReal maxLin; maxLin.setFromFloat(body->getMaxLinearVelocity());
		if( linearVelocity.lengthSquared<3>() > maxLin * maxLin )
		{
			HK_WARN(0xc3a91ee, "Target linear velocity exceeds max linear velocity of body, so it will be clipped.");
		}

		// Avoid setting it if possible, as it would activate the body
		if( !body->getLinearVelocity().allEqual<3>( linearVelocity, eps ) )
		{
			body->setLinearVelocity( linearVelocity );
		}
	}

	// Angular part
	{
		// Get required velocity
		hkVector4 angularVelocity;
		{
			hkQuaternion quatDif;
			quatDif.setMulInverse(nextOrientation, body->getRotation());
			quatDif.normalize();

			if( !quatDif.hasValidAxis() )
			{
				angularVelocity.setZero();
			}
			else
			{
				hkSimdReal angle; angle.setFromFloat(quatDif.getAngle());
				quatDif.getAxis(angularVelocity);
				angularVelocity.setMul(angle * invDeltaTimeSr, angularVelocity);		
			}
		}

		hkSimdReal maxAng; maxAng.setFromFloat(body->getMaxAngularVelocity());
		if( angularVelocity.lengthSquared<3>() > maxAng * maxAng )
		{
			HK_WARN(0x29c79a33, "Target angular velocity exceeds max angular velocity of body, so it will be clipped.");
		}

		// Avoid setting it if possible, as it would activate the body
		if( !body->getAngularVelocity().allEqual<3>( angularVelocity, eps ) )
		{
			body->setAngularVelocity( angularVelocity );
		}
	}
}

void HK_CALL hkpKeyFrameUtility::applyHardKeyFrameAsynchronously( const hkVector4& nextPosition, const hkQuaternion& nextOrientation, hkReal invDeltaTime, hkpRigidBody* body)
{
	hkSimdReal invDeltaTimeSr; invDeltaTimeSr.setFromFloat(invDeltaTime);
	const hkSimdReal eps = EQUALITY_EPSILON;

	hkVector4 bodyCOMinWorld;
	hkQuaternion approxRotation;
	{
		// We could call approxTransformAt, but that does some other stuff with the centerShift that we don't need here
		const hkSweptTransform& st = body->getRigidMotion()->m_motionState.getSweptTransform();
		const hkSimdReal dt = (hkSimdReal::fromFloat(body->getWorld()->getCurrentTime()) - st.getBaseTimeSr()) * st.getInvDeltaTimeSr();
		bodyCOMinWorld.setInterpolate( st.m_centerOfMass0, st.m_centerOfMass1, dt);

		approxRotation.m_vec.setInterpolate( st.m_rotation0.m_vec, st.m_rotation1.m_vec, dt );
		approxRotation.m_vec.normalize<4>();
	}

	// Get linear velocity required
	{
		hkVector4 linearVelocity;
		{
			hkVector4 newCenterOfMassPosition;
			newCenterOfMassPosition._setRotatedDir( nextOrientation, body->getCenterOfMassLocal() );
			newCenterOfMassPosition.add( nextPosition );
			linearVelocity.setSub( newCenterOfMassPosition, bodyCOMinWorld );

			linearVelocity.setMul(invDeltaTimeSr, linearVelocity);
		}

		hkSimdReal maxLin; maxLin.setFromFloat(body->getMaxLinearVelocity());
		if( linearVelocity.lengthSquared<3>() > maxLin * maxLin )
		{
			HK_WARN(0x69481829, "Target linear velocity exceeds max linear velocity of body, so it will be clipped.");
		}

		// Avoid setting it if possible, as it would activate the body
		if( !body->getLinearVelocity().allEqual<3>( linearVelocity, eps ) )
		{
			body->setLinearVelocity(linearVelocity);
		}
	}

	// Get angular velocity required
	{
		hkVector4 angularVelocity;
		{
			hkQuaternion quatDif;
			quatDif.setMulInverse(nextOrientation, approxRotation);
			quatDif.normalize();

			if( !quatDif.hasValidAxis() )
			{
				angularVelocity.setZero();
			}
			else
			{
				quatDif.getAxis(angularVelocity);
				hkSimdReal angle; angle.setFromFloat(quatDif.getAngle());
				angularVelocity.setMul(angle * invDeltaTimeSr, angularVelocity);		
			}
		}

		hkSimdReal maxAng; maxAng.setFromFloat(body->getMaxAngularVelocity());
		if( angularVelocity.lengthSquared<3>() > maxAng * maxAng )
		{
			HK_WARN(0x47b26408, "Target angular velocity exceeds max angular velocity of body, so it will be clipped.");
		}

		// Avoid setting it if possible, as it would activate the body
		if( !body->getAngularVelocity().allEqual<3>( angularVelocity, eps ) )
		{
			body->setAngularVelocity( angularVelocity );
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
