/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>


hkBool hknpCharacterRigidBody::isActive() const
{
	const hknpBody& body = m_world->getBody(m_bodyId);
	return body.isActive() || m_world->m_deactivationManager->m_deactivatedIslands[body.getDeactivatedIslandIndex()]->m_isMarkedForActivation;
}

const hkVector4& hknpCharacterRigidBody::getLinearVelocity() const
{
	return m_world->getMotion( m_world->getBody(m_bodyId).m_motionId ).getLinearVelocity();
}

void hknpCharacterRigidBody::getAngularVelocity(hkVector4& currentVel) const
{
	const hknpMotion& characterMotion = m_world->getMotion( m_world->getBody(m_bodyId).m_motionId );
	characterMotion._getAngularVelocity(currentVel);
}

const hkVector4& hknpCharacterRigidBody::getPosition() const
{
	return m_world->getBody(m_bodyId).getTransform().getTranslation();
}

const hkTransform& hknpCharacterRigidBody::getTransform() const
{
	return m_world->getBody(m_bodyId).getTransform();
}

hknpBodyId hknpCharacterRigidBody::getBodyId() const
{
	return m_bodyId;
}

const hknpBody& hknpCharacterRigidBody::getSimulatedBody() const
{
	return m_world->getSimulatedBody(m_bodyId);
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
