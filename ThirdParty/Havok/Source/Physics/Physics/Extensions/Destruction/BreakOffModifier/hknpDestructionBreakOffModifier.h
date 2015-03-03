/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_DESTRUCTION_BREAKOFF_MODIFIER_H
#define HKNP_DESTRUCTION_BREAKOFF_MODIFIER_H

#include <Common/Base/Container/CommandStream/hkCommandStream.h>

#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>

// A break-off modifier used by Destruction
class hknpDestructionBreakOffModifier : public hknpModifier
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DESTRUCTION, hknpDestructionBreakOffModifier);

	public:

		/// Extends the physics collision manifold with additional properties and functions. hkndManifolds are
		/// forwarded from the collision detection stage. At least one of the bodies is breakable.
		struct Manifold : public hknpPpuManifold
		{
			public:

				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DESTRUCTION, hknpDestructionBreakOffModifier::Manifold);

			public:

				/// Estimates the contact points. Returns true if at least one of the contacts occurs before the end of the frame
				hkBool32 estimateContacts(	hkVector4Parameter vLinVelocityA, const hknpShape* HK_RESTRICT shapeA, const hkTransform& worldFromA,
											hkVector4Parameter vLinVelocityB, const hknpShape* HK_RESTRICT shapeB, const hkTransform& worldFromB,
											hkSimdRealParameter frameDeltaTime, hkVector4& vPenDepthOut);

				/// Re-shuffles the manifold vertices so they form a 2D convex hull.
				void convexify();

				/// Computes the force that needs to be applied to body A to fix the penetrations.
				void computePenetrationImpulse(	hkVector4Parameter vLinearVelA, hkSimdRealParameter invMassA,
												hkVector4Parameter vLinearVelB, hkSimdRealParameter invMassB,
												hkVector4Parameter vPenDepth, hkSimdRealParameter invDeltaTime);

			protected:

				/// Estimated contact point. Four potential contacts are estimated by integrating body A's motion until it hits body B.
				/// These are then averaged into a single contact point. The .w component is used for storing the vertex remap table that convexifies the manifold.
				hkVector4			m_contact;

			public:

				///< Estimated penetration impulse for body 0. Based on penetration depths along the manifold normal, i.e. (contactPtB - contactPtA) * normal.
				/// The .w component stores the equivalent mass of the two bodies.
				hkVector4			m_penImpulse;

				const hknpMaterial*	m_materialA;	///< Saved material of body A
				hknpBodyId			m_bodies[2];	///< Rigid bodies in contact
		};

		/// The contact event that will be fired by the modifier
		struct ContactEvent : public hkCommand
		{
			public:

				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DESTRUCTION, hknpDestructionBreakOffModifier::ContactEvent);

				/// The contact event type
				enum { CONTACT = 0, };

			public:

				/// Constructor
				ContactEvent()
				:	hkCommand(TYPE_DESTRUCTION_EVENTS, CONTACT, sizeof(*this))
				{
					m_filterBits = hknpCommandDispatchType::AS_SOON_AS_POSSIBLE;	// Will be called in the first single-threaded section, i.e. after the collide step
				}

				Manifold m_manifold;		///< The manifold
		};

	public:

		/// Constructor
		hknpDestructionBreakOffModifier();

		/// Destructor
		virtual ~hknpDestructionBreakOffModifier();

	public:

		/// We want to process each created manifold
		HK_FORCE_INLINE int getEnabledFunctions()	{	return (1 << FUNCTION_MANIFOLD_PROCESS);	}

		/// Called to process the given manifold
		virtual void manifoldProcessCallback(	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
												const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
												hknpManifold* HK_RESTRICT manifold) HK_OVERRIDE;

/*	protected:

		hkndWorld* m_ndWorld;	///< Destruction world*/
};

#endif	// HKNP_DESTRUCTION_BREAKOFF_MODIFIER_H

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
