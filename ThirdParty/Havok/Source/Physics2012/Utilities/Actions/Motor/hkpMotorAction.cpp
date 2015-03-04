/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Actions/Motor/hkpMotorAction.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

hkpMotorAction::hkpMotorAction(hkpRigidBody* body, const hkVector4& axis, hkReal spinRate, hkReal gain) :
	hkpUnaryAction(body), m_axis(axis), m_spinRate(spinRate), m_gain(gain), m_active(true)
{
	m_axis.normalize<4>();
}


void hkpMotorAction::applyAction( const hkStepInfo& stepInfo )
{
	// Early exit if inactive.
	if (!m_active)
	{
		return;
	}

	hkpRigidBody* rb = getRigidBody();

	// Work out the current angular velocity in body space.
	const hkTransform& tr = rb->getTransform();
	hkVector4 curr;
	curr.setRotatedInverseDir(tr.getRotation(), rb->getAngularVelocity());
		
	// Calculate the difference between the desired spin rate and the current rate of spin
	// about the desired axis 'm_axis'.
	hkSimdReal currentRate = m_axis.dot<3>(curr);
	hkSimdReal diff = hkSimdReal::fromFloat(m_spinRate) - currentRate;

	// Calculate the newTorque to apply based on the difference and the gain. The newTorque
	// should be proportional to each of difference, gain, and inertia
	// (to make the Action mass-independent).
	hkVector4 newTorque;
	newTorque.setMul(diff * hkSimdReal::fromFloat(m_gain), m_axis);
	hkMatrix3 m;
	rb->getInertiaLocal(m);
	newTorque._setRotatedDir(m, newTorque);

	newTorque.setRotatedDir(tr.getRotation(), newTorque);

	// Apply the new torque.
	rb->applyTorque(stepInfo.m_deltaTime, newTorque);
}

hkpAction* hkpMotorAction::clone( const hkArray<hkpEntity*>& newEntities, const hkArray<hkpPhantom*>& newPhantoms ) const
{
	HK_ASSERT2(0xf5a8efca, newEntities.getSize() == 1, "Wrong clone parameters given to a motor action (needs 1 body).");
	if (newEntities.getSize() != 1) return HK_NULL;

	HK_ASSERT2(0x277857f0, newPhantoms.getSize() == 0, "Wrong clone parameters given to a motor action (needs 0 phantoms).");
	// should have no phantoms.
	if (newPhantoms.getSize() != 0) return HK_NULL;

	hkpMotorAction* ma = new hkpMotorAction( (hkpRigidBody*)newEntities[0], m_axis, m_spinRate, m_gain);
	ma->m_active = m_active;
	ma->m_userData = m_userData;

	return ma;
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
