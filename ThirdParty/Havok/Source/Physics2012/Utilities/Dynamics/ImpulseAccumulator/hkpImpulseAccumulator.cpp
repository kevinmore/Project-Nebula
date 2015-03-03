/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Utilities/Dynamics/ImpulseAccumulator/hkpImpulseAccumulator.h>

					hkpImpulseAccumulator::hkpImpulseAccumulator(hkpRigidBody* body,hkReal timeStep)
	:	m_body(body)
{
	m_dt.setFromFloat(timeStep);
	m_com = body->getCenterOfMassInWorld();
	m_accLinearImpulses.setZero();
	m_accAngularImpulses.setZero();
}

void				hkpImpulseAccumulator::clearImpulses()
{
	m_accLinearImpulses.setZero();
	m_accAngularImpulses.setZero();
}

void				hkpImpulseAccumulator::flushImpulses()
{
	// Apply linear impulse 
	hkVector4	linearVelocity;
	linearVelocity.setAddMul(m_body->getLinearVelocity(), m_accLinearImpulses, m_body->getRigidMotion()->getMassInv());
	m_body->setLinearVelocity(linearVelocity);

	// Apply angular impulse
	hkVector4	angularVelocity;
	hkMatrix3	invInertia;
	m_body->getInertiaInvWorld(invInertia);
	invInertia.multiplyVector(m_accAngularImpulses, angularVelocity);
	angularVelocity.add(m_body->getAngularVelocity());
	m_body->setAngularVelocity(angularVelocity);

	// Clear accumulated impulses
	clearImpulses();
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
