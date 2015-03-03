/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/WheelFriction/hkpWheelFrictionConstraintData.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>

void hkpWheelFrictionConstraintData::init( hkpWheelFrictionConstraintAtom::Axle* axle, hkReal radius )
{
	HK_ASSERT2(0x58a783b7, axle != HK_NULL, "The constraint requires a valid spin velocity pointer");

	m_atoms.m_friction.m_axle = axle;

	m_atoms.m_friction.m_forwardAxis = 1;	// Wheel forward direction
	m_atoms.m_friction.m_sideAxis = 2;		// Wheel side direction
	m_atoms.m_friction.m_maxFrictionForce = HK_REAL_MAX;
	m_atoms.m_friction.m_isEnabled = true;

	m_atoms.m_friction.m_radius = radius;
}

hkBool hkpWheelFrictionConstraintData::isValid() const
{
	// Make sure init() has been called
	return (m_atoms.m_friction.m_axle != HK_NULL);
}

int hkpWheelFrictionConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_CUSTOM;
}

/*static*/ inline hkpWheelFrictionConstraintData::Runtime* hkpWheelFrictionConstraintData::getRuntime( hkpConstraintRuntime* runtime )
{
	return reinterpret_cast<Runtime*>(runtime);
}

void hkpWheelFrictionConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	infoOut.m_numSolverResults = 0;

	if ( wantRuntime )
	{
		infoOut.m_sizeOfExternalRuntime = sizeof( Runtime);
	}
	else
	{
		infoOut.m_sizeOfExternalRuntime = 0;
	}
}

void hkpWheelFrictionConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpWheelFrictionConstraintData::setInWorldSpace( const hkTransform& bodyATransform, const hkTransform& bodyBTransform, const hkVector4& contact, const hkVector4& forward, const hkVector4& side )
{
	HK_ON_DEBUG(hkSimdReal dot = forward.dot<3>(side));
	HK_ASSERT2(0xb53df472, hkMath::fabs(dot.getReal()) < 0.0001f, "forward and side directions should be orthogonal");
	hkVector4 up; up.setCross(forward, side);

	m_atoms.m_transforms.m_transformA.getColumn(0).setRotatedInverseDir(bodyATransform.getRotation(), up);
	m_atoms.m_transforms.m_transformA.getColumn(1).setRotatedInverseDir(bodyATransform.getRotation(), forward);
	m_atoms.m_transforms.m_transformA.getColumn(2).setRotatedInverseDir(bodyATransform.getRotation(), side);
	m_atoms.m_transforms.m_transformA.getTranslation().setTransformedInversePos(bodyATransform, contact);

	m_atoms.m_transforms.m_transformB.getColumn(0).setRotatedInverseDir(bodyBTransform.getRotation(), up);
	m_atoms.m_transforms.m_transformB.getColumn(1).setRotatedInverseDir(bodyBTransform.getRotation(), forward);
	m_atoms.m_transforms.m_transformB.getColumn(2).setRotatedInverseDir(bodyBTransform.getRotation(), side);
	m_atoms.m_transforms.m_transformB.getTranslation().setTransformedInversePos(bodyBTransform, contact);

	HK_ASSERT2(0x3a0a5292, isValid(), "Members of constraint inconsistent after World Space changed.");
}

void hkpWheelFrictionConstraintData::setMaxFrictionForce(hkReal maxFrictionForce)
{
	m_atoms.m_friction.m_maxFrictionForce = maxFrictionForce;
}

void hkpWheelFrictionConstraintData::setTorque(hkReal torque)
{
	m_atoms.m_friction.m_torque = torque;
}

hkReal hkpWheelFrictionConstraintData::getSpinVelocity()
{
	return m_atoms.m_friction.getSpinVelocity();
};

void hkpWheelFrictionConstraintData::setSpinVelocity(hkReal spinVelocity)
{
	m_atoms.m_friction.setSpinVelocity(spinVelocity);
};

void hkpWheelFrictionConstraintData::setInvInertia(hkReal invInertia)
{
	m_atoms.m_friction.m_axle->setInvInertia(invInertia);
};

void hkpWheelFrictionConstraintData::setImpulseScaling(hkReal impulseScaling, hkReal impulseMax)
{
	m_atoms.m_friction.m_axle->setImpulseScaling(impulseScaling, impulseMax);
};

hkReal hkpWheelFrictionConstraintData::getForwardFrictionImpulse()
{
	return m_atoms.m_friction.m_frictionImpulse[0];
}

hkReal hkpWheelFrictionConstraintData::getSideFrictionImpulse()
{
	return m_atoms.m_friction.m_frictionImpulse[1];
}

hkReal hkpWheelFrictionConstraintData::getForwardSlipImpulse()
{
	return m_atoms.m_friction.m_slipImpulse[0];
}

hkReal hkpWheelFrictionConstraintData::getSideSlipImpulse()
{
	return m_atoms.m_friction.m_slipImpulse[1];
}

void hkpWheelFrictionConstraintData::resetSolverData()
{
	m_atoms.m_friction.resetSolverData();
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
