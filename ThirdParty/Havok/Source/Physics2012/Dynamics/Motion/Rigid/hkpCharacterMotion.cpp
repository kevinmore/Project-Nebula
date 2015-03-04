/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpCharacterMotion.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>


//
// INERTIA
//

	// Get the inertia tensor in local space
void hkpCharacterMotion::getInertiaLocal(hkMatrix3& inertia) const
{
	const hkSimdReal i = m_inertiaAndMassInv.getComponent<0>();
	if (i.isEqualZero())
	{
		hkMatrix3Util::_setDiagonal(HK_REAL_HIGH, HK_REAL_HIGH, HK_REAL_HIGH, inertia);
	}
	else
	{
		hkSimdReal hkInvDiag; hkInvDiag.setReciprocal(i);
		hkMatrix3Util::_setDiagonal(hkInvDiag, inertia);
	}
}

	// Get the inertia tensor in local space
void hkpCharacterMotion::getInertiaWorld(hkMatrix3& inertia) const
{
	hkSimdReal hkInvDiag; hkInvDiag.setReciprocal(m_inertiaAndMassInv.getComponent<0>());
	hkMatrix3Util::_setDiagonal(hkInvDiag, inertia);
}




//
// TODO - force the inertia to be correct - replace the set inertia functions
// 

	// Sets the inertia tensor of the rigid body. Advanced use only.
void hkpCharacterMotion::setInertiaLocal(const hkMatrix3& inertia)
{
		// Take max value
	hkVector4 diag; hkMatrix3Util::_getDiagonal(inertia, diag);
	const hkSimdReal temp = diag.horizontalMax<3>();
	HK_ASSERT2(0x7163b90d, temp.isGreaterZero(), "Cannot set inertia of Oriented Particle: Diagonal of inertia tensor must have at least one positive element!");

	hkSimdReal invInertia; invInertia.setReciprocal<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(temp);

	m_inertiaAndMassInv.setXYZ(invInertia);
}


	// Sets the inertia tensor of the rigid body by supplying its inverse. Advanced use only.
void hkpCharacterMotion::setInertiaInvLocal(const hkMatrix3& inertiaInv)
{
	hkVector4 diag; hkMatrix3Util::_getDiagonal(inertiaInv, diag);
	const hkSimdReal invInertia = diag.horizontalMin<3>();
	m_inertiaAndMassInv.setXYZ(invInertia);
}


	// Get the inverse inertia tensor in local space
void hkpCharacterMotion::getInertiaInvLocal(hkMatrix3& inertiaInv) const
{
	const hkSimdReal d = m_inertiaAndMassInv.getComponent<0>();
	hkMatrix3Util::_setDiagonal(d, inertiaInv);
}

	// Get the inverse inertia tensor in world space
void hkpCharacterMotion::getInertiaInvWorld(hkMatrix3& inertiaInv) const
{
	const hkSimdReal d = m_inertiaAndMassInv.getComponent<0>();
	hkMatrix3Util::_setDiagonal(d, inertiaInv);
}


//
// IMPULSE APPLICATION
//

void hkpCharacterMotion::applyPointImpulse(const hkVector4& imp, const hkVector4& p)
{
	// PSEUDOCODE IS m_linearVelocity += m_massInv * imp;
	// PSEUDOCODE IS m_angularVelocity += getWorldInertiaInv() * (p - centerOfMassWorld).cross(imp);
	m_linearVelocity.addMul(getMassInv(), imp);


	// Can calc inertiaWorld * v, but it's faster to calc r * m_inertiaInv * r^-1 * v
	// where r is m_localToWorld.getRotation()
	hkVector4 relMassCenter; relMassCenter.setSub( p, this->getCenterOfMassInWorld() );
	hkVector4 crossWs; crossWs.setCross( relMassCenter, imp );
	m_angularVelocity.addMul( m_inertiaAndMassInv, crossWs);
}


void hkpCharacterMotion::applyAngularImpulse(const hkVector4& imp)
{
	// PSEUDOCODE IS m_angularVelocity += m_worldInertiaInv * imp;
	m_angularVelocity.addMul( m_inertiaAndMassInv, imp);
}


//
// FORCE APPLICATION
//


void hkpCharacterMotion::applyForce( const hkReal deltaTime, const hkVector4& force)
{
	hkVector4 impulse; impulse.setMul( hkSimdReal::fromFloat(deltaTime), force );
	m_linearVelocity.addMul(getMassInv(), impulse);
}

void hkpCharacterMotion::applyForce( const hkReal deltaTime, const hkVector4& force, const hkVector4& p)
{
	hkVector4 impulse; impulse.setMul( hkSimdReal::fromFloat(deltaTime), force );
	applyPointImpulse( impulse, p );
}


void hkpCharacterMotion::applyTorque( const hkReal deltaTime, const hkVector4& torque)
{
	hkVector4 impulse; impulse.setMul( hkSimdReal::fromFloat(deltaTime), torque );
	applyAngularImpulse( impulse );
}


void hkpCharacterMotion::getProjectedPointVelocity(const hkVector4& pos, const hkVector4& normal, hkReal& velOut, hkReal& invVirtMassOut) const
{
	hkVector4 arm;
	hkVector4 relPos; relPos.setSub( pos, getCenterOfMassInWorld() );
	arm.setCross( normal, relPos);
	const hkSimdReal vel = arm.dot<3>(m_angularVelocity) + m_linearVelocity.dot<3>(normal);
	vel.store<1>(&velOut);

	getMassInv().store<1>(&invVirtMassOut);
}

void hkpCharacterMotion::getProjectedPointVelocitySimd(const hkVector4& pos, const hkVector4& normal, hkSimdReal& velOut, hkSimdReal& invVirtMassOut) const
{
	hkVector4 arm;
	hkVector4 relPos; relPos.setSub( pos, getCenterOfMassInWorld() );
	arm.setCross( normal, relPos);

	velOut = arm.dot<3>(m_angularVelocity) + m_linearVelocity.dot<3>(normal);
	invVirtMassOut = getMassInv();
}


HK_COMPILE_TIME_ASSERT( sizeof( hkpCharacterMotion) == sizeof( hkpMotion) );

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
