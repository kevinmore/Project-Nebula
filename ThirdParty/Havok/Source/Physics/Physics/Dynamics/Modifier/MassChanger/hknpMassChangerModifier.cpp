/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/Modifier/MassChanger/hknpMassChangerModifier.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolver.h>

#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>


void hknpMassChangerModifier::manifoldProcessCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
	hknpManifold* HK_RESTRICT manifold
	)
{
	const hknpMaterial& matA = *cdBodyA.m_material;
	const hknpMaterial& matB = *cdBodyB.m_material;

	if ( matA.m_massChangerCategory == hknpMaterial::MASS_CHANGER_DEBRIS )
	{
		hkReal massChangerHeavyObjectFactor = matB.m_massChangerHeavyObjectFactor;
		HK_ASSERT2( 0xf0cdfde5, massChangerHeavyObjectFactor >= 1.0f, "Your massChangerHeavyObjectFactor must be larger than one");
		hkReal massChangerInternalFactor = (massChangerHeavyObjectFactor-1.0f) / (massChangerHeavyObjectFactor+1.0f);

		manifold->m_massChangerData = massChangerInternalFactor;
	}
	else if ( matB.m_massChangerCategory == hknpMaterial::MASS_CHANGER_DEBRIS )
	{
		hkReal massChangerHeavyObjectFactor = matA.m_massChangerHeavyObjectFactor;	// take the material info from the heavier objects
		HK_ASSERT2( 0xf0cdfde5, massChangerHeavyObjectFactor >= 1.0f, "Your massChangerHeavyObjectFactor must be larger than one");
		hkReal massChangerInternalFactor = (massChangerHeavyObjectFactor-1.0f) / (massChangerHeavyObjectFactor+1.0f);

		manifold->m_massChangerData = -massChangerInternalFactor;
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
