/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BREAKING_CONTACT_MODIFIER_H
#define HKNP_BREAKING_CONTACT_MODIFIER_H

#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>


/// A modifier which simply takes the materialA.m_maxContactImpulse and use this to
/// set the maximum impulse at the solver.
class hknpContactImpulseClippedEventCreator : public hknpModifier
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpContactImpulseClippedEventCreator );

		//
		// hknpModifier implementation
		//

		HK_FORCE_INLINE int getEnabledFunctions()
		{
			return (1<<FUNCTION_MANIFOLD_CREATED_OR_DESTROYED) | (1<<FUNCTION_POST_CONTACT_IMPULSE_CLIPPED);
		}

		void manifoldCreatedCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
			ManifoldCreatedCallbackInput* HK_RESTRICT info
			) HK_OVERRIDE;

		void postContactImpulseClipped(
			const hknpSimulationThreadContext& tl, const SolverCallbackInput& input,
			const hkReal& sumContactImpulseUnclipped, const hkReal& sumContactImpulseClipped
			);
};


#endif	//HKNP_BREAKING_CONTACT_MODIFIER_H

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
