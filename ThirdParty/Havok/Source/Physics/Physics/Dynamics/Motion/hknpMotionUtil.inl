/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE int	hknpMotionUtil::convertAngleToAngularTIM( hkReal angle )
{
	return hkMath::clamp((int)(angle * 510/HK_REAL_PI + 0.5f),0,255);
}

HK_FORCE_INLINE void hknpMotionUtil::convertAngleToAngularTIM( hkSimdRealParameter angle, hkInt32* angleOut )
{
	hkSimdReal f; f.setFromFloat( 510/HK_REAL_PI );
	hkSimdReal angTim = angle * f;
	angTim.setMin( angTim, hkSimdReal::getConstant<HK_QUADREAL_255>());
	angTim.storeSaturateInt32( angleOut );
}

HK_FORCE_INLINE int	hknpMotionUtil::convertAngleToAngularTIM( hkVector4Parameter a, hkVector4Parameter b )
{
	hkVector4 cross; cross.setCross( a, b );
	hkSimdReal len = cross.length<3, HK_ACC_12_BIT, HK_SQRT_SET_ZERO>();	// this is > sin(angle)
	hkSimdReal f; f.setFromFloat( 510/HK_REAL_PI );
	hkSimdReal angTim = len * f;
	int result;
	angTim.storeSaturateInt32( &result );
	return result;
}

HK_FORCE_INLINE void hknpMotionUtil::convertDistanceToLinearTIM(
	const hknpSolverInfo& solverInfo, hkSimdRealParameter deltaDist, hkUint16& linearTimOut )
{
	hkSimdReal dd = deltaDist * solverInfo.m_distanceToLinearTim;
	dd.storeSaturateUint16( &linearTimOut );
}

HK_FORCE_INLINE void hknpMotionUtil::convertVelocityToLinearTIM(
	const hknpSolverInfo& solverInfo, hkVector4Parameter velocity, hkUint16& linearTimOut )
{
	hkSimdReal deltaDist = velocity.length<3, HK_ACC_12_BIT, HK_SQRT_SET_ZERO>();
	hkSimdReal f = solverInfo.m_deltaTime * solverInfo.m_distanceToLinearTim;
	deltaDist = f * deltaDist;
	deltaDist.storeSaturateUint16( &linearTimOut );
}

HK_FORCE_INLINE void hknpMotionUtil::_sweepBodyAndCalcAabb(
	hknpBody* HK_RESTRICT body, const hknpShape* shape, const hknpMotion& motion, const hknpBodyQuality& quality,
	hkSimdRealParameter collisionTolerance, const hknpSolverInfo& solverInfo,
	hkAabb* HK_RESTRICT aabbOut )
{
	// Calculate expansion parameters
	hkVector4 expandLin0;
	hkVector4 expandLin1;
	hkSimdReal anyDirectionExpansion;
	hkSimdReal linearDirectionExpansion;
	calcSweepExpansion(
		body, motion, collisionTolerance, solverInfo,
		&expandLin0, &expandLin1, &anyDirectionExpansion, &linearDirectionExpansion );

	// Set the body's max contact distance
	{
		hkSimdReal maxContactDistance = (anyDirectionExpansion + linearDirectionExpansion) * solverInfo.m_distanceToLinearTim;
		maxContactDistance.storeSaturateUint16( &body->m_maxContactDistance );
	}

	// Calculate the swept AABB
	{
		hkAabb shapeAabb;
		shape->calcAabb( body->getTransform(), shapeAabb );
		HK_ASSERT( 0xb82554a2, shapeAabb.isValid() );

		if( quality.m_requestedFlags.anyIsSet( hknpBodyQuality::ENABLE_NEIGHBOR_WELDING ) )
		{
			hkAabbUtil::expandAabbByMotionCircle( shapeAabb, expandLin0, expandLin1, anyDirectionExpansion, *aabbOut );
			
		}
		else if( quality.m_requestedFlags.anyIsSet( hknpBodyQuality::USE_DISCRETE_AABB_EXPANSION ) )
		{
			anyDirectionExpansion.load<1>( &body->getCollisionLookAheadDistance() );
			if( body->m_timAngle > 10 )	// ~3 degrees
			{
				hkAabb futureAabb;
				calcFutureAabb( *body, shape, motion, solverInfo.m_deltaTime, &futureAabb );
				aabbOut->setUnion( shapeAabb, futureAabb );
				aabbOut->expandBy( anyDirectionExpansion );

				return;
			}
			else
			{
				anyDirectionExpansion.setMax( anyDirectionExpansion, collisionTolerance * hkSimdReal_Inv2 );
			}
		}

		hkAabbUtil::expandAabbByMotion( shapeAabb, expandLin0, expandLin1, anyDirectionExpansion, *aabbOut );
	}
}

HK_FORCE_INLINE void hknpMotionUtil::calcStaticBodyAabb(
	const hknpBody& body, hkSimdRealParameter collisionTolance, hkAabb* HK_RESTRICT aabbOut )
{
	const hknpShape* shape = body.m_shape;
	shape->calcAabb( body.getTransform(), *aabbOut );
	HK_ASSERT( 0xb82554a8, aabbOut->isValid() );
	hkSimdReal expansion = collisionTolance * hkSimdReal_Inv2;
	aabbOut->expandBy( expansion );
}

HK_FORCE_INLINE void hknpMotionUtil::_predictMotionTransform(
	hkSimdRealParameter deltaTime, const hknpMotion& motion,
	hkVector4* HK_RESTRICT motionComOut, hkQuaternion* HK_RESTRICT motionOrientationOut )
{
	hkVector4 vLinA = motion.m_linearVelocity;
	hkVector4 vAngA = motion.m_angularVelocity;	// in local space

	// Integrate translation
	motionComOut->setAddMul( motion.getCenterOfMassInWorld(), vLinA, deltaTime );

	// Integrate rotation
	{
		const hkSimdReal halfTimeStep = deltaTime * hkSimdReal_Inv2;

		hkQuaternion dqA;
		dqA.m_vec.setMul( halfTimeStep, vAngA );
		dqA.m_vec.setComponent<3>( hkSimdReal_1 );

		hkQuaternion qComA;
		qComA.setMul( motion.m_orientation, dqA );
		qComA.m_vec.normalize<4,HK_ACC_23_BIT, HK_SQRT_IGNORE>();

		motionOrientationOut[0] = qComA;
	}
}

HK_FORCE_INLINE void hknpMotionUtil::_calculateBodyTransform(
	const hknpBody& body, hkVector4Parameter motionCom, hkQuaternionParameter motionOrientation,
	hkTransform* HK_RESTRICT worldFromBodyOut )
{
	hkQTransform bodyQTmotion;
	body.getMotionToBodyTransformEstimate( &bodyQTmotion );
	bodyQTmotion.m_rotation.normalize<HK_ACC_23_BIT, HK_SQRT_IGNORE>();

	hkQuaternion worldQbody;
	worldQbody.setMulInverse( motionOrientation, bodyQTmotion.m_rotation );

	hkVector4Util::convertQuaternionToRotation( worldQbody, worldFromBodyOut->getRotation() );

	hkVector4 centerShift; centerShift._setRotatedDir( worldFromBodyOut->getRotation(), bodyQTmotion.m_translation );
	hkVector4 newGeomCenter; newGeomCenter.setSub( motionCom, centerShift );
	worldFromBodyOut->setTranslation( newGeomCenter );
}

HK_FORCE_INLINE hkSimdReal hknpMotionUtil::calcEnergy(const hknpMotion& motion, hkVector4Parameter gravity,
												hkSimdRealParameter groundHeight)
{
	const hkVector4& com = motion.getCenterOfMassInWorld();
	hkVector4 up; up.setNeg<3>(gravity);
	hkSimdReal gravityLength = up.normalizeWithLength<3>();
	hkSimdReal height = com.dot<3>(up) - groundHeight;

	// Linear kinetic energy
	hkVector4 inertia; motion.getInertiaLocal(inertia);
	const hkVector4& linearVelocity = motion.getLinearVelocity();
	hkSimdReal mass = inertia.getComponent<3>();
	hkSimdReal linearEnergy = hkSimdReal_Inv2 * mass * linearVelocity.lengthSquared<3>();

	// Angular kinetic energy
	const hkVector4& angularVelocity = motion.getAngularVelocityLocal();
	hkVector4 angularMomentum; angularMomentum.setMul(inertia, angularVelocity);
	hkSimdReal rotationalEnergy = hkSimdReal_Inv2 * angularVelocity.dot<3>(angularMomentum);

	hkSimdReal potentialEnergy = mass * gravityLength * height;

	return linearEnergy + rotationalEnergy + potentialEnergy;
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
