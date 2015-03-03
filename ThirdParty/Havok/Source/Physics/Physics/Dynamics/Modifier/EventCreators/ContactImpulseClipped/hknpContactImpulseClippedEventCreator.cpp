/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/Modifier/EventCreators/ContactImpulseClipped/hknpContactImpulseClippedEventCreator.h>

#include <Common/Base/Math/Vector/Mx/hkMxVectorUtil.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>

#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>
#include <Physics/Physics/Dynamics/Solver/Event/hknpSolverEvents.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>


void hknpContactImpulseClippedEventCreator::manifoldCreatedCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
	ManifoldCreatedCallbackInput* HK_RESTRICT info
	)
{
	const hknpMaterial& materialA = *cdBodyA.m_material;
	const hknpMaterial& materialB = *cdBodyB.m_material;

	// Copy the max impulse from the materials to the manifold cache
	{
		hkSimdReal impA; impA.setFromFloat( materialA.m_maxContactImpulse );
		hkSimdReal impB; impB.setFromFloat( materialB.m_maxContactImpulse );
		hkSimdReal maxImp; maxImp.setMin( impA, impB );
		maxImp.store<1>( &info->m_collisionCache->m_maxImpulse );
	}

	// Copy the fraction of impulse to clip from the materials to the manifold cache
	{
		hkReal uncompressedFractionA = materialA.m_fractionOfClippedImpulseToApply;
		hkReal uncompressedFractionB = materialB.m_fractionOfClippedImpulseToApply;
		hkReal minFraction = hkMath::min2(uncompressedFractionA, uncompressedFractionB);
		info->m_collisionCache->m_fractionOfClippedImpulseToApply = (hkUint8)(minFraction * 255.0f);
	}
}


void hknpContactImpulseClippedEventCreator::postContactImpulseClipped(
	const hknpSimulationThreadContext& tl, const hknpModifier::SolverCallbackInput& input,
	const hkReal& sumContactImpulseUnclipped, const hkReal& sumContactImpulseClipped )
{
	const hknpMxContactJacobian::ManifoldData& manifoldData = input.m_contactJacobian->m_manifoldData[input.m_manifoldIndex];
	if( manifoldData.m_manifoldType == hknpManifold::TYPE_NORMAL )
	{
		hknpContactImpulseClippedEvent event( manifoldData.m_bodyIdA, manifoldData.m_bodyIdB );
		event.initialize( input );
		event.m_sumContactImpulseUnclipped = sumContactImpulseUnclipped;

		tl.execCommand( event );
	}
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
