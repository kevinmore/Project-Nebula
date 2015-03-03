/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpKeyframedRigidMotion.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

hkpKeyframedRigidMotion::hkpKeyframedRigidMotion(const hkVector4& position, const hkQuaternion& rotation) :
	hkpMotion( position, rotation)
{
	m_savedMotion = HK_NULL;
	m_savedQualityTypeIndex = 0;

		// The "mass" of a keyframed body is infinite. It must be treated as "unmovable" (though not, obviously, "unmoving"!)
		// by the solver. This actually happens explicitly in the hkRigidMotionUtilApplyForcesAndBuildAccumulators() (which doesn't
		// actually read these values, but we set them anyway for consistency.
	m_inertiaAndMassInv.setZero();
	m_type = MOTION_KEYFRAMED;
}

hkpKeyframedRigidMotion::~hkpKeyframedRigidMotion()
{
	if (m_savedMotion)
	{
		m_savedMotion->removeReference();
	}
}

void hkpKeyframedRigidMotion::setMass(hkReal m)
{
	HK_ASSERT2(0xad67f4d3, 0, "Error: do not call setMass on a fixed or keyframed object(hkpKeyframedRigidMotion)");
}
void hkpKeyframedRigidMotion::setMass(hkSimdRealParameter m)
{
	HK_ASSERT2(0xad67f4d3, 0, "Error: do not call setMass on a fixed or keyframed object(hkpKeyframedRigidMotion)");
}

void hkpKeyframedRigidMotion::setMassInv(hkReal mInv)
{
	HK_ASSERT2(0xad67f4d4, 0, "Error: do not call setMassInv on a fixed or keyframed object(hkpKeyframedRigidMotion)");
}
void hkpKeyframedRigidMotion::setMassInv(hkSimdRealParameter mInv)
{
	HK_ASSERT2(0xad67f4d4, 0, "Error: do not call setMassInv on a fixed or keyframed object(hkpKeyframedRigidMotion)");
}

void hkpKeyframedRigidMotion::getInertiaLocal(hkMatrix3& inertia) const
{
	// The "mass" of a keyframed body is infinite. It must be treated as "unmovable" (though not, obviously "unmoving"!)
	// by the solver. 
	// We create an "invalid" inertia, since it should never be used (would have to be infinite!)
	inertia.setZero();
}

void hkpKeyframedRigidMotion::getInertiaWorld(hkMatrix3& inertia) const
{
	// The "mass" of a keyframed body is infinite. It must be treated as "unmovable" (though not, obviously "unmoving"!)
	// by the solver. 
	// We create an "invalid" inertia, since it should never be used (would have to be infinite!)
	inertia.setZero();
}

void hkpKeyframedRigidMotion::setInertiaLocal(const hkMatrix3& inertia)
{
	HK_ASSERT2(0x28204ab9, 0, "Error: do not call setInertiaLocal on a fixed of keyframed object (hkpKeyframedRigidMotion)");
}


void hkpKeyframedRigidMotion::setInertiaInvLocal(const hkMatrix3& inertiaInv)
{
	HK_ASSERT2(0x7b611123, 0, "Error: do not call setInertiaInvLocal on a fixed or keyframed object (hkpKeyframedRigidMotion)");
}

void hkpKeyframedRigidMotion::getInertiaInvLocal(hkMatrix3& inertiaInv) const
{
	// The "mass" of a keyframed body is infinite. It must be treated as "unmovable" (though not, obviously "unmoving"!)
	// by the solver. 
	inertiaInv.setZero();
}

void hkpKeyframedRigidMotion::getInertiaInvWorld(hkMatrix3& inertiaInvOut) const
{
	// The "mass" of a keyframed body is infinite. It must be treated as "unmovable" (though not, obviously "unmoving"!)
	// by the solver. 
	inertiaInvOut.setZero();
}

void hkpKeyframedRigidMotion::applyLinearImpulse(const hkVector4& imp)
{
}

void hkpKeyframedRigidMotion::applyPointImpulse(const hkVector4& imp, const hkVector4& p)
{
}

void hkpKeyframedRigidMotion::applyAngularImpulse(const hkVector4& imp)
{
}

void hkpKeyframedRigidMotion::applyForce(const hkReal deltaTime, const hkVector4& force)
{
}

void hkpKeyframedRigidMotion::applyForce(const hkReal deltaTime, const hkVector4& force, const hkVector4& p)
{
}

void hkpKeyframedRigidMotion::applyTorque(const hkReal deltaTime, const hkVector4& torque)
{
}

void hkpKeyframedRigidMotion::setStepPosition( hkReal position, hkReal timestep )
{
}

void hkpKeyframedRigidMotion::setStoredMotion( hkpMaxSizeMotion* savedMotion )
{
	if (savedMotion)
	{
		savedMotion->addReference();
	}
	if (m_savedMotion)
	{
		m_savedMotion->removeReference();
	}
	m_savedMotion = savedMotion;
}

void hkpKeyframedRigidMotion::getProjectedPointVelocity(const hkVector4& pos, const hkVector4& normal, hkReal& velOut, hkReal& invVirtMassOut) const
{
	hkVector4 arm;
	hkVector4 relPos; relPos.setSub( pos, getCenterOfMassInWorld() );
	arm.setCross( normal, relPos);
	const hkSimdReal vel = arm.dot<3>(m_angularVelocity) + m_linearVelocity.dot<3>(normal);
	vel.store<1>(&velOut);

	invVirtMassOut = hkReal(0);
}

void hkpKeyframedRigidMotion::getProjectedPointVelocitySimd(const hkVector4& pos, const hkVector4& normal, hkSimdReal& velOut, hkSimdReal& invVirtMassOut) const
{
	hkVector4 arm;
	hkVector4 relPos; relPos.setSub( pos, getCenterOfMassInWorld() );
	arm.setCross( normal, relPos);

	velOut = arm.dot<3>(m_angularVelocity) + m_linearVelocity.dot<3>(normal);
	invVirtMassOut.setZero();
}


HK_COMPILE_TIME_ASSERT( sizeof( hkpKeyframedRigidMotion) <= sizeof( hkpMaxSizeMotion));

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
