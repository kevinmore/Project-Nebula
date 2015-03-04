/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpBoxMotion.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>


hkpBoxMotion::hkpBoxMotion(const hkVector4& position, const hkQuaternion& rotation)
	:hkpMotion( position, rotation )
{
	m_inertiaAndMassInv = hkVector4::getConstant<HK_QUADREAL_1>();
	m_type = MOTION_BOX_INERTIA;
}

void hkpBoxMotion::getInertiaLocal(hkMatrix3& inertia) const
{
	const hkVector4 minIn = hkVector4::getConstant<HK_QUADREAL_EPS>();
	hkVector4 maxI; maxI.setMax( minIn, m_inertiaAndMassInv );
	hkVector4 invI; invI.setReciprocal(maxI);
	hkMatrix3Util::_setDiagonal(invI, inertia);
}


void hkpBoxMotion::setInertiaLocal(const hkMatrix3& inertia)
{
	
	HK_ASSERT(0x7708edc8,  inertia(0,0) >0 );
	HK_ASSERT(0x1ff66a9d,  inertia(1,1) >0 );
	HK_ASSERT(0x51ff9422,  inertia(2,2) >0 );
	hkVector4 diag; hkMatrix3Util::_getDiagonal(inertia, diag);
	hkVector4 invD; invD.setReciprocal(diag);
	m_inertiaAndMassInv.setXYZ(invD);
}

void hkpBoxMotion::getInertiaInvLocal(hkMatrix3& inertiaInv) const
{
	hkMatrix3Util::_setDiagonal(m_inertiaAndMassInv, inertiaInv);
}

void hkpBoxMotion::setInertiaInvLocal(const hkMatrix3& inertiaInv)
{
	hkVector4 diag; hkMatrix3Util::_getDiagonal(inertiaInv, diag);
	m_inertiaAndMassInv.setXYZ( diag );
}

void hkpBoxMotion::getInertiaInvWorld(hkMatrix3& inertiaInv) const
{
	hkVector4 a; a.setMul( m_inertiaAndMassInv.getComponent<0>(), getTransform().getColumn<0>() );
	hkVector4 b; b.setMul( m_inertiaAndMassInv.getComponent<1>(), getTransform().getColumn<1>() );
	hkVector4 c; c.setMul( m_inertiaAndMassInv.getComponent<2>(), getTransform().getColumn<2>() );

	hkMatrix3 in;
	in.setCols(a,b,c);

	inertiaInv.setMulInverse( in, getTransform().getRotation() );
}

void hkpBoxMotion::getInertiaWorld(hkMatrix3& inertia) const
{
	getInertiaLocal(inertia);
	inertia.changeBasis( getTransform().getRotation() );
}


	// Set the mass of the rigid body.
void hkpBoxMotion::setMass(hkReal mass)
{
	HK_ASSERT2(0x16462fdb,  mass > 0.0f, "If you want to set the mass to zero, use a fixed rigid body" );
	hkSimdReal massInv; massInv.setReciprocal(hkSimdReal::fromFloat(mass));
	m_inertiaAndMassInv.setComponent<3>(massInv);
}
void hkpBoxMotion::setMass(hkSimdRealParameter mass)
{
	HK_ASSERT2(0x16462fdb,  mass.isGreaterZero(), "If you want to set the mass to zero, use a fixed rigid body" );
	hkSimdReal massInv; massInv.setReciprocal(mass);
	m_inertiaAndMassInv.setComponent<3>(massInv);
}


void hkpBoxMotion::applyPointImpulse(const hkVector4& imp, const hkVector4& p)
{
	// PSEUDOCODE IS m_linearVelocity += m_massInv * imp;
	// PSEUDOCODE IS m_angularVelocity += getWorldInertiaInv() * (p - centerOfMassWorld).cross(imp);
	m_linearVelocity.addMul(getMassInv(), imp);

	// Can calc inertiaWorld * v, but it's faster to calc r * m_inertiaAndMassInv * r^-1 * v
	// where r is m_localToWorld.getRotation()
	hkVector4 relMassCenter; relMassCenter.setSub( p, m_motionState.getSweptTransform().m_centerOfMass1 );
	hkVector4 crossWs; crossWs.setCross( relMassCenter, imp );

	hkVector4 crossLs; crossLs._setRotatedInverseDir( getTransform().getRotation(), crossWs);
	hkVector4 deltaVelLs; deltaVelLs.setMul( m_inertiaAndMassInv, crossLs);
	hkVector4 deltaVelWs; deltaVelWs._setRotatedDir(getTransform().getRotation(), deltaVelLs);
	m_angularVelocity.add(deltaVelWs);
}


void hkpBoxMotion::applyAngularImpulse(const hkVector4& imp)
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
	hkVector4 impLocal; impLocal._setRotatedInverseDir( getTransform().getRotation(), imp );
	hkVector4 dangVelLocal; dangVelLocal.setMul( m_inertiaAndMassInv, impLocal );
	hkVector4 dangVel; dangVel._setRotatedDir( getTransform().getRotation(), dangVelLocal );
	m_angularVelocity.add(dangVel);
}


void hkpBoxMotion::applyForce( const hkReal deltaTime, const hkVector4& force)
{
	hkVector4 impulse; impulse.setMul( hkSimdReal::fromFloat(deltaTime), force );
	m_linearVelocity.addMul(getMassInv(), impulse);
}

void hkpBoxMotion::applyForce( const hkReal deltaTime, const hkVector4& force, const hkVector4& p)
{
	hkVector4 impulse; impulse.setMul( hkSimdReal::fromFloat(deltaTime), force );
	applyPointImpulse( impulse, p );
}


void hkpBoxMotion::applyTorque( const hkReal deltaTime, const hkVector4& torque)
{
	hkVector4 impulse; impulse.setMul( hkSimdReal::fromFloat(deltaTime), torque );
	applyAngularImpulse( impulse );
}


void hkpBoxMotion::getProjectedPointVelocity(const hkVector4& pos, const hkVector4& normal, hkReal& velOut, hkReal& invVirtMassOut) const
{
	hkVector4 arm;
	hkVector4 relPos; relPos.setSub( pos, getCenterOfMassInWorld() );
	arm.setCross( relPos, normal );
	const hkSimdReal vel = arm.dot<3>(m_angularVelocity) + m_linearVelocity.dot<3>(normal);

	hkVector4 armLocal; armLocal._setRotatedInverseDir(m_motionState.getTransform().getRotation(), arm);
	hkVector4 jacDivMass; jacDivMass.setMul(armLocal, m_inertiaAndMassInv);
	const hkSimdReal invVirtMass = getMassInv() + armLocal.dot<3>(jacDivMass);

	vel.store<1>(&velOut);
	invVirtMass.store<1>(&invVirtMassOut);
}

void hkpBoxMotion::getProjectedPointVelocitySimd(const hkVector4& pos, const hkVector4& normal, hkSimdReal& velOut, hkSimdReal& invVirtMassOut) const
{
	hkVector4 arm;
	hkVector4 relPos; relPos.setSub( pos, getCenterOfMassInWorld() );
	arm.setCross( relPos, normal );
	velOut = arm.dot<3>(m_angularVelocity) + m_linearVelocity.dot<3>(normal);

	hkVector4 armLocal; armLocal._setRotatedInverseDir(m_motionState.getTransform().getRotation(), arm);
	hkVector4 jacDivMass; jacDivMass.setMul(armLocal, m_inertiaAndMassInv);
	invVirtMassOut = getMassInv() + armLocal.dot<3>(jacDivMass);
}


HK_COMPILE_TIME_ASSERT( sizeof(hkpBoxMotion) == sizeof(hkpMotion) );
#if ( HK_POINTER_SIZE == 4 ) && !defined(HK_REAL_IS_DOUBLE)
HK_COMPILE_TIME_ASSERT( sizeof( hkpMotion) == 0x120 );
#endif

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
