/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/VelocityDamper/Default/hknpVehicleDefaultVelocityDamper.h>

hknpVehicleDefaultVelocityDamper::hknpVehicleDefaultVelocityDamper()
{
	m_normalSpinDamping = 0;
	m_collisionSpinDamping = 0;
	m_collisionThreshold = 1.0f;
}

// can't be const because of the changing of one of the members (unless you make that member mutable)
void hknpVehicleDefaultVelocityDamper::applyVelocityDamping(const hkReal deltaTime, hknpVehicleInstance& vehicle )
{
// 	hknpRigidBody*	chassis_motionstate = vehicle.getChassis();
// 	hkVector4 angularVel = chassis_motionstate->getAngularVelocity();
// 	const hkReal spinSqrd = angularVel.lengthSquared<3>().getReal();
//
// 	hkReal exp_time;
// 	if (spinSqrd > m_collisionThreshold * m_collisionThreshold)
// 	{
// 		exp_time = hkMath::max2( hkReal(0.0f), 1.0f - deltaTime * m_collisionSpinDamping );
// 	}
// 	else
// 	{
// 		exp_time = hkMath::max2( hkReal(0.0f), 1.0f - deltaTime * m_normalSpinDamping );
// 	}
//
// 	hkVector4 newAngVel; newAngVel.setMul( hkSimdReal::fromFloat(exp_time), angularVel);
//
// 	chassis_motionstate ->setAngularVelocity( newAngVel);
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
