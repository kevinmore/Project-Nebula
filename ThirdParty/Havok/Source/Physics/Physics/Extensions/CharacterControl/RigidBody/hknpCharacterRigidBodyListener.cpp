/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBodyListener.h>

#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBody.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>


hknpCharacterRigidBody::ContactType hknpCharacterRigidBodyListener::processManifold(
	hknpCharacterRigidBody* characterRB, const hknpManifoldProcessedEvent& manifoldEvent,
	hkArray<hknpCharacterRigidBody::ContactPointInfo>& processedContactPoints )
{
	hknpWorld* world = characterRB->m_world;

	// Ensure the contact normals we consider are pointing the correct way.
	const bool characterIsA = ( manifoldEvent.m_bodyIds[0] == characterRB->m_bodyId );
	hkSimdReal mulDir;
	hknpBodyId other;
	if (characterIsA)
	{
		mulDir.setFromFloat(1.0f);
		other = manifoldEvent.m_bodyIds[1];
	}
	else
	{
		mulDir.setFromFloat(-1.0f);
		other = manifoldEvent.m_bodyIds[0];
	}

	const hkSimdReal surfaceVerticalComponent = manifoldEvent.m_manifold.m_normal.dot<3>( characterRB->m_up );
	const hkReal surfaceVerticalComponentFromA = (mulDir * surfaceVerticalComponent).getReal();

	// find closest point
	hkSimdReal distance = manifoldEvent.m_manifold.m_distances.getComponent<0>();
	int cpIndex = 0;
	for (int i =1; i < manifoldEvent.m_numContactPoints; i++ )
	{
		if ( manifoldEvent.m_manifold.m_distances.getComponent(i) < distance )
		{
			distance = manifoldEvent.m_manifold.m_distances.getComponent(i);
			cpIndex = i;
		}
	}
	hkVector4 pos = manifoldEvent.m_manifold.m_positions[cpIndex];

	// If contact is vertical according to max slope add a vertical contact point
	if ((surfaceVerticalComponentFromA > HK_REAL_EPSILON) && (surfaceVerticalComponentFromA < characterRB->m_maxSlopeCosine))
	{
		hknpCharacterRigidBody::ContactPointInfo& contactInfo = processedContactPoints.expandOne();
		contactInfo.m_contactType = hknpCharacterRigidBody::VERTICAL;
		contactInfo.m_bodyIds[0] = manifoldEvent.m_bodyIds[0];
		contactInfo.m_bodyIds[1] = manifoldEvent.m_bodyIds[1];

		// Set vertical plane distance to zero unless the character is penetrating the surface.
		distance.setMin( distance, hkSimdReal_0 );
	#ifdef DEBUG_HKNP_CHARACTER_RIGIDBODY
		pos.setInt24W( hknpCharacterRigidBody::m_magicNumber );
	#endif
		hkVector4 sepN = manifoldEvent.m_manifold.m_normal;
		sepN.addMul( -surfaceVerticalComponent, characterRB->m_up );
		// Normalizing here is safe: m_maxSlopeCosine <= 1, and surfaceVerticalComponentFromA is
		// always positive, thus a vertical cp normal wouldn't have passed the above test.
		sepN.normalize<3,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
		contactInfo.m_contactPoint.setPositionNormalAndDistance(pos, sepN, distance);
	}

	hknpCharacterRigidBody::ContactPointInfo& contactInfo = processedContactPoints.expandOne();
	contactInfo.m_contactType = hknpCharacterRigidBody::HORIZONTAL;
	contactInfo.m_bodyIds[0] = manifoldEvent.m_bodyIds[0];
	contactInfo.m_bodyIds[1] = manifoldEvent.m_bodyIds[1];

	// Return first point of original manifold
	contactInfo.m_contactPoint.setPositionNormalAndDistance( pos, manifoldEvent.m_manifold.m_normal, distance );

	// For dynamic collision partners additionally adjust the collision mass
	const hknpBody& otherBody = world->getBody(other);
	if ( otherBody.isDynamic() )
	{
		contactInfo.m_contactType = hknpCharacterRigidBody::MODIFIED;

#if 0	// TODO: Verify code
		if ( manifoldEvent.m_manifoldCache != HK_NULL )
		{
			hkSimdReal factorChar;

			// Deal with the zero case separately.
			if ( characterRB->m_maxForce > 0.0f )
			{
				hkVector4 contactNormal; contactNormal.setMul( -mulDir,  manifoldEvent.m_manifold.m_normal ) ;

				hkSimdReal currentForce;
				hkSimdReal invBodyMass = world->getMotion(characterRB->m_character).getMassInv();
				currentForce.setReciprocal<HK_ACC_23_BIT,HK_DIV_IGNORE>( invBodyMass );
				{
					// HVK-4666:
					// We'd like to project the acceleration onto the contact normal.
					hkSimdReal lensq = contactNormal.lengthSquared<3>();
					currentForce = currentForce * contactNormal.dot<3>( characterRB->m_acceleration );
					currentForce.div<HK_ACC_23_BIT,HK_DIV_IGNORE>(lensq);
				}

				hkSimdReal maxForce; maxForce.setFromFloat(characterRB->m_maxForce);
				factorChar = hkSimdReal_1;
				if ( currentForce > maxForce )
				{
					factorChar.setDiv<HK_ACC_23_BIT,HK_DIV_IGNORE>(currentForce, maxForce);
				}
			}
			else
			{
				// Since mass changer modifies inverse mass, we need a special case
				// when m_maxForce = 0.0f. For the purposes of this collision, make the other entity
				// infinitely massive.
				factorChar = hkSimdReal_0;
			}

			hknpMxContactJacobian* HK_RESTRICT mxJac = manifoldEvent.m_manifoldCache->m_manifoldSolverInfo.m_contactJacobian;
			int mxId = manifoldEvent.m_manifoldCache->m_manifoldSolverInfo.m_mxJacobianIndex;

			mxJac->m_massChangerData[mxId] = characterIsA ? (factorChar.getReal() - 1.f) : (factorChar.getReal() + 1.f);

		}	// m_manifoldCache
#endif
	} // update dynamic mass

	return contactInfo.m_contactType;
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
