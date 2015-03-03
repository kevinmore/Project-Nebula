/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_RESTITUTION_MODIFIER_H
#define HKNP_RESTITUTION_MODIFIER_H

#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>


/// A modifier for handling restitution.
class hknpRestitutionModifier : public hknpModifier
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpRestitutionModifier);

		hknpRestitutionModifier() {};

		HK_FORCE_INLINE int getEnabledFunctions()
		{
			return	( 1 << FUNCTION_MANIFOLD_CREATED_OR_DESTROYED ) |
					( 1 << FUNCTION_POST_CONTACT_JACOBIAN_SETUP );
		}

		virtual void manifoldCreatedCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
			ManifoldCreatedCallbackInput* HK_RESTRICT info ) HK_OVERRIDE;

		virtual void postContactJacobianSetup(
			const hknpSimulationThreadContext& tl,
			const hknpSolverInfo& solverInfo,
			const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
			const hknpManifoldCollisionCache* cache, const hknpManifold* manifold,
			hknpMxContactJacobian* HK_RESTRICT mxJac, int mxJacIdx ) HK_OVERRIDE;

	public:

		/// If the approaching velocity of two objects is less than this * gravity * timestep,
		/// no restitution will be applied. Default = 1.0f
		static const hkSimdReal s_contactRestingVelocityFactor;

		/// Internally the system remembers approaching velocities and applies restitution 1 or more frames later.
		/// To limit the side effects of this delay, the approaching velocities will be damped by this value.
		static const hkSimdReal s_velocityBufferDamping;
};


#endif	//HKNP_RESTITUTION_MODIFIER_H

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
