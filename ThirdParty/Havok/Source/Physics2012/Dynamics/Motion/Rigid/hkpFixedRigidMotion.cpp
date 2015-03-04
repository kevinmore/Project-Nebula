/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpFixedRigidMotion.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>


hkpFixedRigidMotion::hkpFixedRigidMotion(const hkVector4& position, const hkQuaternion& rotation)
	:hkpKeyframedRigidMotion( position, rotation)
{
	m_type = MOTION_FIXED;
}



void hkpFixedRigidMotion::setStepPosition( hkReal position, hkReal timestep )
{
}



void hkpFixedRigidMotion::getPositionAndVelocities( hkpMotion* motionOut )
{
	motionOut->m_motionState = m_motionState;
	motionOut->m_linearVelocity.setZero();	// zero linear velocity
	motionOut->m_angularVelocity.setZero();	// zero angular velocity
}

void hkpFixedRigidMotion::setLinearVelocity(const hkVector4& newVel)
{
	HK_WARN(0xdbc34901, "Do not call setLinearVelocity on a fixed object(hkpFixedRigidMotion)");
}

void hkpFixedRigidMotion::setAngularVelocity(const hkVector4& newVel)
{
	HK_WARN(0xdbc34902, "Do not call setAngularVelocity on a fixed object(hkpFixedRigidMotion)");
}

HK_COMPILE_TIME_ASSERT( sizeof(hkpFixedRigidMotion) == sizeof( hkpKeyframedRigidMotion) );

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
