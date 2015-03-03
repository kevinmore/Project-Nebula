/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/Modifier/SoftContact/hknpSoftContactModifier.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolver.h>

#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Collide/hknpCollideSharedData.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobianUtil.h>


namespace
{
	struct SoftContactProperties
	{
		HK_ALIGN_REAL( hkReal m_softContactSeperationVelocity );
		hkReal m_rhsFactor;
		hkReal m_effMassFactor;
	};
}


void hknpSoftContactModifier::manifoldCreatedCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& bodyDataA, const hknpCdBody& bodyDataB,
	ManifoldCreatedCallbackInput* HK_RESTRICT info )
{

	const hkSimdReal invMassA = bodyDataA.m_motion->getInverseMass();
	const hkSimdReal invMassB = bodyDataB.m_motion->getInverseMass();

	int softBodyIndex;
	hkSimdReal invMass;

	// read forceFactor as int and convert 0 to maxInt
// 	unsigned int forceFactor0 = (unsigned int)(bodyDataA->m_childMaterial->m_softContactForceFactor.m_value-1);
// 	unsigned int forceFactor1 = (unsigned int)(bodyDataB->m_childMaterial->m_softContactForceFactor.m_value-1);
#if defined(HK_HALF_IS_FLOAT)
#error not implemented
#endif
	unsigned int forceFactor0 = (unsigned int)(*((hkInt16*)&bodyDataA.m_material->m_softContactForceFactor)-1);
	unsigned int forceFactor1 = (unsigned int)(*((hkInt16*)&bodyDataB.m_material->m_softContactForceFactor)-1);

	const hknpMaterial* material;

	if ( forceFactor0  < forceFactor1 )
	{
		invMass = invMassA;
		softBodyIndex = 0;
		material = bodyDataA.m_material;
	}
	else
	{
		invMass = invMassB;
		softBodyIndex = 1;
		material = bodyDataB.m_material;
	}


	hkSimdReal forceFactor; forceFactor.setFromFloat( material->m_softContactForceFactor );
	hkSimdReal dampFactor;  dampFactor.setFromFloat( material->m_softContactDampFactor );

	SoftContactProperties* props = info->m_collisionCache->allocateProperty<SoftContactProperties>(
		hknpManifoldCollisionCache::SOFT_CONTACT_PROPERTY, HK_REAL_ALIGNMENT );

	hkSimdReal invDamp; invDamp.setReciprocal<HK_ACC_12_BIT, HK_DIV_SET_ZERO>( dampFactor );
	hkSimdReal rhsFactor = forceFactor / dampFactor;

	hkSimdReal softContactSeperationVelocity;
	softContactSeperationVelocity.setFromFloat( material->m_softContactSeperationVelocity );

	softContactSeperationVelocity.store<1>( & props->m_softContactSeperationVelocity );
	rhsFactor.setMin( rhsFactor, hkSimdReal_1 );
	rhsFactor.store<1>( &props->m_rhsFactor );
	dampFactor.store<1>( &props->m_effMassFactor );
}

void hknpSoftContactModifier::manifoldProcessCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
	hknpManifold* HK_RESTRICT manifold )
{
	SoftContactProperties* HK_RESTRICT props = manifold->m_collisionCache->accessProperty<SoftContactProperties>(
		hknpManifoldCollisionCache::SOFT_CONTACT_PROPERTY );
	HK_ASSERT2( 0xf0345458, props, "The collision cache does not have a soft contact property" );

	//
	//	Change some secret constants to remove the springiness
	//

	hkSimdReal sepVel; sepVel.setFromFloat(props->m_softContactSeperationVelocity);
	hkVector4 recoveryVelocity; recoveryVelocity.setAll( sepVel ); // meter/per second

	hkVector4 distances; distances.setAddMul
					( manifold->m_distances, recoveryVelocity, sharedData.m_solverInfo.val()->m_deltaTime );
	hkVector4 zero; zero.setZero();
	hkVector4 distanceOffset; distanceOffset.setMin( manifold->m_collisionCache->m_distanceOffset, zero );
	manifold->m_collisionCache->m_distanceOffset.setMin( distanceOffset, distances );
}

void hknpSoftContactModifier::postContactJacobianSetup(
	const hknpSimulationThreadContext& tl,
	const hknpSolverInfo& solverInfo,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	const hknpManifoldCollisionCache* HK_RESTRICT cache, const hknpManifold* HK_RESTRICT manifold,
	hknpMxContactJacobian* HK_RESTRICT mxJac, int mxJacIdx )
{
	const SoftContactProperties* HK_RESTRICT props = cache->accessProperty<SoftContactProperties>(
		hknpManifoldCollisionCache::SOFT_CONTACT_PROPERTY );
	HK_ASSERT2( 0xf034545a, props != HK_NULL, "The collision cache does not have a soft contact property" );

	hkVector4 effMassScale; effMassScale.setAll( props->m_effMassFactor );
	hkVector4 rhsScale; rhsScale.setAll( props->m_rhsFactor );
	hknpContactJacobianUtil::scaleEffectiveMass( mxJac, mxJacIdx, effMassScale );
	hknpContactJacobianUtil::scalePenetrations ( mxJac, mxJacIdx, rhsScale );
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
