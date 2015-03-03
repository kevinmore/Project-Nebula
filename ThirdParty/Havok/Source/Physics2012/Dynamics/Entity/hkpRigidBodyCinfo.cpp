/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

hkpRigidBodyCinfo::hkpRigidBodyCinfo()
{
	m_position.setZero();
	m_rotation.setIdentity();
	m_linearVelocity.setZero();
	m_angularVelocity.setZero();
	hkMatrix3Util::_setDiagonal(hkSimdReal_1, m_inertiaTensor);
	m_centerOfMass.setZero();
	m_mass = 1.0f;
	m_linearDamping = 0.0f;
	m_angularDamping = 0.05f;
	m_gravityFactor = 1.0f;
	m_friction = 0.5f;
	m_rollingFrictionMultiplier = 0.0f;
	m_restitution = 0.4f;
	m_maxLinearVelocity = 200.0f;
	m_maxAngularVelocity = 200.0f;
	m_motionType = hkpMotion::MOTION_DYNAMIC;
	m_enableDeactivation = true;
	m_solverDeactivation = SOLVER_DEACTIVATION_LOW;
	m_collisionFilterInfo = 0;
	m_shape = HK_NULL;
	m_timeFactor = 1.0f;
	m_localFrame = HK_NULL;
	m_qualityType = HK_COLLIDABLE_QUALITY_INVALID;
	m_allowedPenetrationDepth = -1.0f;
	m_autoRemoveLevel = 0;
	m_responseModifierFlags = 0;
	m_numShapeKeysInContactPointProperties = 0;
	m_collisionResponse = hkpMaterial::RESPONSE_SIMPLE_CONTACT;
	m_contactPointCallbackDelay = 0xffff;
	m_forceCollideOntoPpu = false;
}


void hkpRigidBodyCinfo::setMassProperties(const hkMassProperties& mp)
{
	m_mass = mp.m_mass;
	m_inertiaTensor = mp.m_inertiaTensor;
	m_centerOfMass = mp.m_centerOfMass;
}

void hkpRigidBodyCinfo::setTransform( const hkTransform& transform )
{
	m_position = transform.getTranslation();
	m_rotation.set( transform.getRotation() );
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
