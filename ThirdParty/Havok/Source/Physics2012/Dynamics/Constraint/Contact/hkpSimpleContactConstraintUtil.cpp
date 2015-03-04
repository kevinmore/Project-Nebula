/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Physics2012/Dynamics/Constraint/Contact/hkpSimpleContactConstraintUtil.h>


hkSimdReal HK_CALL hkpSimpleContactConstraintUtil::calculateSeparatingVelocity(const hkpRigidBody* bodyA, const hkpRigidBody* bodyB, 
																			   hkVector4Parameter centerOfMassInWorldA, hkVector4Parameter centerOfMassInWorldB, 
																			   const hkContactPoint* cp )
{
	hkVector4 velA;
	{
		hkVector4 relPos; relPos.setSub( cp->getPosition(), centerOfMassInWorldA );
		velA.setCross( bodyA->getAngularVelocity(), relPos);
		velA.add( bodyA->getLinearVelocity() );
	}
	hkVector4 velB;
	{
		hkVector4 relPos; relPos.setSub( cp->getPosition(), centerOfMassInWorldB );
		velB.setCross( bodyB->getAngularVelocity(), relPos);
		velB.add( bodyB->getLinearVelocity() );
	}
	hkVector4 deltaVel; deltaVel.setSub( velA, velB );
	return deltaVel.dot<3>( cp->getNormal() );
}

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
