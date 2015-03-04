/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Actions/AngularDashpot/hkpAngularDashpotAction.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

hkpAngularDashpotAction::hkpAngularDashpotAction(hkpRigidBody* entityA, hkpRigidBody* entityB, hkUlong userData) 
: hkpBinaryAction( entityA, entityB, userData ),
  m_strength( 0.1f),
  m_damping(0.01f)
{
	m_rotation.setIdentity();
}

void hkpAngularDashpotAction::applyAction(const hkStepInfo& stepInfo)
{
	const hkReal dtscale = 200; // to keep constants sensible around 1
	const hkReal dt = dtscale * (stepInfo.m_deltaTime);

	hkpRigidBody* rbA = static_cast<hkpRigidBody*>( m_entityA );
	hkpRigidBody* rbB = static_cast<hkpRigidBody*>( m_entityB );
	HK_ASSERT2(0xf668efca, rbA && rbB, "Bodies not set in angular dashpot.");

	hkQuaternion q;
	q.setMul(rbB->getRotation(),m_rotation);

	hkQuaternion inv;
	inv.setInverse(q);

	hkQuaternion rrot;
	rrot.setMul(rbA->getRotation(),inv);

	hkVector4 avel;
	avel.setSub(rbA->getAngularVelocity(),rbB->getAngularVelocity());

 	const hkReal angle = rrot.getAngle();
	hkVector4 axis; axis.setZero();

	
	if( angle > 0.001f)
	{
		axis.setMul(hkSimdReal::fromFloat(angle), rrot.getImag());
	}

	axis.mul(hkSimdReal::fromFloat(dt * m_strength));
	avel.mul(hkSimdReal::fromFloat(dt * m_damping));

	hkVector4 impulse;
	impulse.setAdd(axis,avel);

	rbB->applyAngularImpulse( impulse);
	impulse.setNeg<4>( impulse );
	rbA->applyAngularImpulse(impulse);

}

// hkpAction clone interface
hkpAction* hkpAngularDashpotAction::clone( const hkArray<hkpEntity*>& newEntities, const hkArray<hkpPhantom*>& newPhantoms ) const
{
	HK_ASSERT2(0xf568efca, newEntities.getSize() == 2, "Wrong clone parameters given to a spring action (needs 2 bodies).");
	// should have two entities as we are a binary action.
	if (newEntities.getSize() != 2) return HK_NULL;

	HK_ASSERT2(0x392f03ac, newPhantoms.getSize() == 0, "Wrong clone parameters given to a spring action (needs 0 phantoms).");
	// should have no phantoms.
	if (newPhantoms.getSize() != 0) return HK_NULL;

	hkpAngularDashpotAction* ada = new hkpAngularDashpotAction( (hkpRigidBody*)newEntities[0], (hkpRigidBody*)newEntities[1], m_userData );
	ada->m_rotation = m_rotation;
	ada->m_strength = m_strength;
	ada->m_damping = m_damping;

	return ada;
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
