/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotion.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionCinfo.h>

#include <Common/Base/Math/Matrix/hkMatrix3Util.h>
#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>

#define hknpMotionProperties_DEFAULT_REFERENCE_DISTANCE hkReal(0.05f)
#define hknpMotionProperties_MAX_ROTATION_IN_PI_RAD hkReal(0.6f)
#define hknpMotionProperties_DEFAULT_MINIMUM_PATHING_VELOCITY hkReal(0.035f) // 2*hknpMotionProperties_DEFAULT_REFERENCE_DISTANCE * 60/128 * 1.5/2
#define hknpMotionProperties_REFERENCE_DIST_TO_WRAPPING_BLOCK_SIZE_RATIO hkReal(20)

HK_COMPILE_TIME_ASSERT(sizeof(hknpMotion) % 16 == 0);
HK_COMPILE_TIME_ASSERT(sizeof(hknpMotionProperties) % 16 == 0);
#if !defined(HK_REAL_IS_DOUBLE)
HK_COMPILE_TIME_ASSERT(sizeof(hknpMotion) <= 128);
HK_COMPILE_TIME_ASSERT(sizeof(hknpMotion) >= 128);
#endif


void hknpMotion::getInverseInertiaWorld( hkMatrix3& invInertiaOut ) const
{
	hkRotation rotation; rotation.set(m_orientation);

	hkVector4 inertiaLocal; getInverseInertiaLocal( inertiaLocal );
	hkMatrix3 in;

	hkVector4 a; a.setBroadcast<0>( inertiaLocal );
	hkVector4 b; b.setBroadcast<1>( inertiaLocal );
	hkVector4 c; c.setBroadcast<2>( inertiaLocal );
	in.getColumn(0).setMul( a, rotation.getColumn(0) );
	in.getColumn(1).setMul( b, rotation.getColumn(1) );
	in.getColumn(2).setMul( c, rotation.getColumn(2) );

	invInertiaOut.setMulInverse( in, rotation );
}

void hknpMotion::getInertiaWorld( hkMatrix3& inertiaOut ) const
{
	hkRotation rotation; rotation.set(m_orientation);

	hkVector4 inertiaLocal; getInertiaLocal( inertiaLocal );
	hkMatrix3 in;

	hkVector4 a; a.setBroadcast<0>( inertiaLocal );
	hkVector4 b; b.setBroadcast<1>( inertiaLocal );
	hkVector4 c; c.setBroadcast<2>( inertiaLocal );
	in.getColumn(0).setMul( a, rotation.getColumn(0) );
	in.getColumn(1).setMul( b, rotation.getColumn(1) );
	in.getColumn(2).setMul( c, rotation.getColumn(2) );

	inertiaOut.setMulInverse( in, rotation );
}

void hknpMotion::setVelocities(
	hkVector4Parameter linearVelocity, hkVector4Parameter angularVelocity,
	hkVector4Parameter centerOfLinearVelocityInWorld )
{
	m_angularVelocity._setRotatedInverseDir( m_orientation, angularVelocity );
	hkVector4 arm; arm.setSub( centerOfLinearVelocityInWorld, getCenterOfMassInWorld() );
	hkVector4 cross; cross.setCross( arm, angularVelocity );
	m_linearVelocity.setAdd( linearVelocity, cross );
}

void hknpMotion::setPreviousStepVelocities(
	hkVector4Parameter linearVelocity, hkVector4Parameter angularVelocity,
	hkVector4Parameter centerOfLinearVelocityInWorld )
{
	m_previousStepAngularVelocity._setRotatedInverseDir( m_orientation, angularVelocity );
	hkVector4 arm; arm.setSub( centerOfLinearVelocityInWorld, getCenterOfMassInWorld() );
	hkVector4 cross; cross.setCross( arm, angularVelocity );
	m_previousStepLinearVelocity.setAdd( linearVelocity, cross );
}

void hknpMotion::setFromMassProperties( const hkMassProperties& mp, const hkTransform& massPropertiesTransform )
{
	checkConsistency();

	// Save velocities.
	hkVector4 linVel = getLinearVelocity();
	hkVector4 angVel;  getAngularVelocity(angVel);	// get angular velocity on world space

	hkVector4 prevLinVel = getLinearVelocity();		
	hkVector4 prevAngVel;  prevAngVel._setRotatedDir(m_orientation, m_previousStepAngularVelocity);

	// Update center of mass
	hkVector4 com; com._setTransformedPos( massPropertiesTransform, mp.m_centerOfMass );
	setCenterOfMassInWorld( com );

	if( mp.m_mass > 0.0f )
	{
		hkRotation bodyRmotion;
		hkVector4  inertiaDiagonal;
		mp.m_inertiaTensor.diagonalizeSymmetricApproximation( bodyRmotion, inertiaDiagonal, 10 );
		inertiaDiagonal(3) = mp.m_mass;
		inertiaDiagonal.setMax( inertiaDiagonal, hkVector4::getConstant(HK_QUADREAL_EPS));

		hkRotation worldRmotion; worldRmotion.setMul( massPropertiesTransform.getRotation(), bodyRmotion );
		hkQuaternion worldQmotion; worldQmotion.set( (hkRotation&)worldRmotion);
		worldQmotion.normalize();
		m_orientation = worldQmotion;
		hkVector4 inertiaDiagInv; inertiaDiagInv.setReciprocal<HK_ACC_23_BIT, HK_DIV_IGNORE>(inertiaDiagonal);
		inertiaDiagInv.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(m_inverseInertia);
	}
	else
	{
		setInfiniteInertiaAndMass();
	}

	setVelocities(linVel,angVel,getCenterOfMassInWorld());
	setPreviousStepVelocities(prevLinVel, prevAngVel, getCenterOfMassInWorld());

	checkConsistency();
}


void hknpMotion::getAngularVelocity( hkVector4& angularVelWorldOut ) const
{
	_getAngularVelocity( angularVelWorldOut );
}

void hknpMotion::applyPointImpulse( hkVector4Parameter impulse, hkVector4Parameter position )
{
	_applyPointImpulse( impulse, position );
}

void hknpMotion::getPointVelocity( hkVector4Parameter position, hkVector4& velOut ) const
{
	_getPointVelocity( position, velOut );
}

void hknpMotion::applyLinearImpulse( hkVector4Parameter impulse )
{
	_applyLinearImpulse( impulse );
}

void hknpMotion::applyAngularImpulse( hkVector4Parameter angImpulseWorld )
{
	_applyAngularImpulse( angImpulseWorld );
}

void hknpMotion::setAngularVelocity( hkVector4Parameter angVelocityWorld )
{
	_setAngularVelocity( angVelocityWorld );
}

void hknpMotion::buildEffMassMatrixAt( hkVector4Parameter position, hkMatrix3& effMassMatrixOut) const
{
	hkReal massInv = this->getInverseMass().getReal();

	hkReal massInvNonNull = (massInv == 0.f ? 1.f : massInv);

	hkVector4 r; r.setSub( position, this->getCenterOfMassInWorld() );

	hkMatrix3 rhat;		rhat.setCrossSkewSymmetric(r);

	hkMatrix3 inertialInvWorld;	this->getInverseInertiaWorld(inertialInvWorld);

	hkMatrix3Util::_setDiagonal( massInvNonNull, massInvNonNull, massInvNonNull, effMassMatrixOut );

	// calculate: effMassMatrixOut -= (rhat * inertialInvWorld * rhat)
	hkMatrix3 temp;		temp._setMul(rhat, inertialInvWorld);
	hkMatrix3 temp2;	temp2._setMul(temp, rhat);
	effMassMatrixOut.sub(temp2);
	effMassMatrixOut._invertSymmetric();
}


void hknpMotion::setPointVelocity( hkVector4Parameter velocity, hkVector4Parameter position )
{
	hkMatrix3 effMassMatrix; buildEffMassMatrixAt( position, effMassMatrix );

	hkVector4	velocityIn;	_getPointVelocity(position, velocityIn);
	hkVector4	delta;	delta.setSub(velocity,velocityIn);

	hkVector4	impulse; impulse._setRotatedDir(effMassMatrix, delta);

	_applyPointImpulse(impulse, position);
}


void hknpMotion::reintegrate( hkSimdRealParameter t, hkSimdRealParameter deltaTime )
{
	hkSimdReal prevT; prevT.setFromHalf( m_integrationFactor );
	hkSimdReal realT = prevT * t;
	realT.store<1>( &m_integrationFactor );

	hkVector4 linVel = m_previousStepLinearVelocity;
	hkVector4 angVel = m_previousStepAngularVelocity;

	hkSimdReal velFactor = t;	// the m_previousStep*Velocities are already multiplied with m_integrationFactor, so we need to factor this out
	hkSimdReal reverseTime = deltaTime * velFactor - deltaTime;

	m_previousStepLinearVelocity.setMul( velFactor, linVel );
	m_previousStepAngularVelocity.setMul( velFactor, angVel );

	hknpMotionUtil::integrateMotionTransform( linVel, angVel, reverseTime, this );
	HK_ON_DEBUG( checkConsistency() );
}

void hknpMotion::checkConsistency() const
{
#ifdef HK_DEBUG
	HK_ASSERT( 0xf03d349f, m_linearVelocity.isOk<4>());
	HK_ASSERT( 0xf03d349f, m_angularVelocity.isOk<4>());
	HK_ASSERT( 0xf03d349f, m_previousStepLinearVelocity.isOk<4>());
	HK_ASSERT( 0xf03d349f, m_previousStepAngularVelocity.isOk<4>());
	HK_ASSERT( 0xf03d349f, m_centerOfMassAndMassFactor.isOk<4>());
	HK_ASSERT( 0xf03d349f, m_orientation.m_vec.isOk<4>());

	hkVector4 invInertia; getInverseInertiaLocal( invInertia );
	HK_ASSERT( 0xf03d349f, invInertia.isOk<4>());
#endif
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
