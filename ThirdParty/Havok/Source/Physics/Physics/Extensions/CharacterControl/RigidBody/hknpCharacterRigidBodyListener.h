/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CHARACTER_RIGIDBODY_LISTENER_H
#define HKNP_CHARACTER_RIGIDBODY_LISTENER_H

#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBody.h>

class hkContactPoint;
class hknpSolverData;
struct hknpManifoldProcessedEvent;


/// A listener which lets the user modify the interaction of a hknpCharacterRigidBody's rigid body.
class hknpCharacterRigidBodyListener : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor.
		hknpCharacterRigidBodyListener() {}

		/// Destructor.
		virtual ~hknpCharacterRigidBodyListener() {}

		/// Called during collision detection, for each Manifold Processed event.
		/// Allows you to actively disable the contact point returning the contact type IGNORED, which will adjust the
		/// corresponding Jacobian.
		/// Horizontal contact points added to processedContactPoints will be used in the
		/// hknpCharacterRigidBody::getSupportInfo() call, vertical ones will be used to create immediate constraints in
		/// hknpCharacterRigidBody::onPostCollideSignal().
		virtual hknpCharacterRigidBody::ContactType processManifold(
			hknpCharacterRigidBody* characterRB, const hknpManifoldProcessedEvent& cpEvent,
			hkArray<hknpCharacterRigidBody::ContactPointInfo>& processedContactPoints );

		/// Called once after collide but before solving.
		/// New customized contact points can added simply by storing them in newContactPoints.
		/// All other contact point management must be handled by manipulating the Jacobians in question.
		virtual void onPostCollide(
			hknpCharacterRigidBody* characterRB, hknpSolverData& solverData,
			hkArray<hknpCharacterRigidBody::ContactPointInfo>& newContactPoints )
		{
			// do nothing
		}

		/// Called once after the simulate step. This usually where you should put your control logic.
		virtual void onPostSolve( hknpCharacterRigidBody* characterRB )
		{
			// do nothing
		}
};

#endif // HKNP_CHARACTER_RIGIDBODY_LISTENER_H

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
