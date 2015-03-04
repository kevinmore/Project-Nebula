/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/AeroDynamics/Default/hknpVehicleDefaultAerodynamics.h>

hknpVehicleDefaultAerodynamics::hknpVehicleDefaultAerodynamics()
{
	m_airDensity = 0;
	m_frontalArea = 0;
	m_dragCoefficient = 0;
	m_liftCoefficient = 0;
	m_extraGravityws.setZero();
}


hkReal hknpVehicleDefaultAerodynamics::calcAerodynamicDrag(hkReal forwardVelocity) const
{
	const hkReal drag = - 0.5f * m_airDensity * m_dragCoefficient * m_frontalArea * hkMath::fabs(forwardVelocity) * forwardVelocity;

	return drag;
}

hkReal hknpVehicleDefaultAerodynamics::calcAerodynamicLift(hkReal forwardVelocity) const
{
	const hkReal lift = 0.5f * m_airDensity * m_liftCoefficient * m_frontalArea * forwardVelocity * forwardVelocity;
	return lift;
}

void hknpVehicleDefaultAerodynamics::calcAerodynamics(const hkReal deltaTime, const hknpVehicleInstance* vehicle, AerodynamicsDragOutput& dragInfoOut )
{
	const hkTransform &chassis_transform = vehicle->getChassisTransform();
	const hkVector4& forward_cs = vehicle->m_data->m_chassisOrientation.getColumn<1>();
	const hkVector4& up_cs = vehicle->m_data->m_chassisOrientation.getColumn<0>();

	hkVector4 forward_ws;	forward_ws._setRotatedDir(chassis_transform.getRotation(),forward_cs);
	hkVector4 up_ws;		up_ws._setRotatedDir(chassis_transform.getRotation(),up_cs);

	const hknpMotion& motion = vehicle->getChassisMotion();

	//const hkVector4& chassis_velocity_ws = rb->getLinearVelocity();
	const hkVector4& chassis_velocity_ws = motion.m_linearVelocity;

	const hkReal forward_velocity = chassis_velocity_ws.dot<3>(forward_ws).getReal();

	const hkReal drag = calcAerodynamicDrag( forward_velocity );
	const hkReal lift = calcAerodynamicLift( forward_velocity );

	dragInfoOut.m_aerodynamicsForce.setMul( hkSimdReal::fromFloat(drag), forward_ws );
	dragInfoOut.m_aerodynamicsForce.addMul( hkSimdReal::fromFloat(lift), up_ws );

	// add extra gravity
//	dragInfoOut.m_aerodynamicsForce.addMul( hkSimdReal::fromFloat(rb->getMass()), m_extraGravityws);
	dragInfoOut.m_aerodynamicsForce.addMul( motion.getMass(), m_extraGravityws);

	// No calculations are done for torque.
	dragInfoOut.m_aerodynamicsTorque.setZero();
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
