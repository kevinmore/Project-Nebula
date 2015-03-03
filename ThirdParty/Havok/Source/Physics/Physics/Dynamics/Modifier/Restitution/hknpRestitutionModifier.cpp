/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Modifier/Restitution/hknpRestitutionModifier.h>

#include <Common/Base/Math/Vector/Mx/hkMxVector.h>

#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>

// Static variable initialization;
const hkSimdReal hknpRestitutionModifier::s_contactRestingVelocityFactor = hkSimdReal_1;
const hkSimdReal hknpRestitutionModifier::s_velocityBufferDamping = hkSimdReal_1;

void hknpRestitutionModifier::manifoldCreatedCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& bodyDataA, const hknpCdBody& bodyDataB,
	ManifoldCreatedCallbackInput* HK_RESTRICT info )
{
	hkVector4* minVelocities = info->m_collisionCache->allocateProperty<hkVector4>(
		hknpManifoldCollisionCache::RESTITUTION_PROPERTY, HK_REAL_ALIGNMENT );
	minVelocities->setZero();
}

void hknpRestitutionModifier::postContactJacobianSetup(
	const hknpSimulationThreadContext& tl,
	const hknpSolverInfo& solverInfo,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	const hknpManifoldCollisionCache* cache, const hknpManifold* manifold,
	hknpMxContactJacobian* HK_RESTRICT mxJac, int mxId )
{
	if(cache->m_restitution.isZero())
	{
		return;
	}

	hkVector4* HK_RESTRICT minVelocities = cache->accessProperty<hkVector4>(
		hknpManifoldCollisionCache::RESTITUTION_PROPERTY );
	HK_ASSERT2( 0xf0345456, minVelocities != HK_NULL, "The collision cache does not have a restitution property" );

	const hknpMotion* HK_RESTRICT motionA = cdBodyA.m_motion;
	const hknpMotion* HK_RESTRICT motionB = cdBodyB.m_motion;

	mxJac->m_disableNegImpulseClip |= 1<<mxId;

	hkSimdReal effInvMassAtCenter = motionA->getInverseMass() + motionB->getInverseMass();
	const hkVector4 normalWs = manifold->m_normal;

	hkVector4 linVelA = motionA->getLinearVelocity();
	hkVector4 linVelB = motionB->getLinearVelocity();


	hkSimdReal projVelocities[4];

	hkVector4 linDiff; linDiff.setSub( linVelA, linVelB );
	hkVector4 linDiffProjected; linDiffProjected.setMul( linDiff, normalWs );

	hkMxVector< hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD > ang0, ang1;
	mxJac->unpackManifoldAngular( mxId, &ang0, &ang1 );

	hkMxUNROLL_4(
	{
		hkVector4 vel;
		vel.setAddMul( linDiffProjected, motionA->m_angularVelocity, ang0.getVector<hkMxI>() );
		vel.addMul( motionB->m_angularVelocity, ang1.getVector<hkMxI>() );
		projVelocities[hkMxI] = vel.horizontalAdd<3>();
	});

	hkSimdReal negContactRestingVelocity;
	{
		negContactRestingVelocity.load<1>( &solverInfo.m_nominalGravityLength);
		negContactRestingVelocity = -(s_contactRestingVelocityFactor * negContactRestingVelocity * solverInfo.m_deltaTime);
	}

	hkVector4 proj; proj.set( projVelocities[0], projVelocities[1], projVelocities[2], projVelocities[3] );

	//HK_TIMER_SPLIT_LIST( "2" );
	bool restitutionApplied = false;

	int i = 0;
	for ( ; i < manifold->m_numPoints; i++)
	{
		hkSimdReal projectedVelocitySr = projVelocities[i];
		hkSimdReal minVelocitySr;
		minVelocitySr = minVelocities->getComponent(i);

		if ( minVelocitySr < negContactRestingVelocity )
		{
			// apply the restitution only we are seriously slowed down
			hkSimdReal minVelocity2 = minVelocitySr * hkSimdReal_Inv4;
			if ( projectedVelocitySr > minVelocity2 )
			{

				hkSimdReal restitution; restitution.load<1>(&cache->m_restitution);
				hkSimdReal effPointMass; effPointMass.setFromFloat( mxJac->m_contactPointData[i].m_effectiveMass[mxId] );
				hkSimdReal invDamp; invDamp.setReciprocal<HK_ACC_12_BIT, HK_DIV_IGNORE>( solverInfo.m_damping );
				hkSimdReal ratio = effPointMass * effInvMassAtCenter * invDamp;
				// extra experimental terms for newRhs: /*hkSimdReal::fromFloat(mxJac->m_contactPointData[i].m_rhs[mxId])*/ /*projectedVelocitySr * ( hkSimdReal_1 - invDamp  )*/
				hkSimdReal newRhs =  -invDamp * restitution * minVelocitySr * ratio;
				newRhs.store<1>( &mxJac->m_contactPointData[i].m_rhs[mxId] );
				restitutionApplied = true;
				//HK_REPORT( "Restitution" <<cache->m_bodyA.value() << " and " << cache->m_bodyB.value() << " cpId" << i << " vel " << projectedVelocitySr.getReal() << " effMass " << effPointMass.getReal() << " ratio " << ratio.getReal() );
			}
		}
	}

	// fix minimum velocity
	{
		HK_ASSERT2( 0xf043ef65, (s_velocityBufferDamping * solverInfo.m_deltaTime).getReal() < 1.0f, "Your time step or m_velocityBufferDamping is too high");
		hkVector4 minVelocity;
		minVelocity = *minVelocities;
		hkVector4 velocityBufferDamping; velocityBufferDamping.setAll( s_velocityBufferDamping );
		minVelocity.addMul( velocityBufferDamping,  solverInfo.m_deltaTime );
		minVelocity.setMin( minVelocity, proj );
		hkVector4Comparison mask; mask.set( hkVector4ComparisonMask::Mask( hkVector4ComparisonMask::MASK_XYZW >> (4-manifold->m_numPoints)) );
		minVelocity.zeroIfFalse( mask );
		*minVelocities = minVelocity;
	}

	if ( restitutionApplied )
	{
		// if we have restitution,
		// reduce the effect of friction significantly.
		hkReal massFactor = 0.25f;
		minVelocities->setZero();
#if defined(HKNP_MX_FRICTION)
		hkReal fem0 = mxJac->m_frictionEffMass[0][mxId] * massFactor;
		hkReal fem1 = mxJac->m_frictionEffMass[1][mxId] * massFactor;
		hkReal fem2 = mxJac->m_frictionEffMass[2][mxId] * massFactor;
		mxJac->m_frictionRhs[0][mxId] = 0.0f;	// bad performance on spu
		mxJac->m_frictionRhs[1][mxId] = 0.0f;
		mxJac->m_frictionRhs[2][mxId] = 0.0f;
		mxJac->m_frictionEffMass[0][mxId] = fem0;
		mxJac->m_frictionEffMass[1][mxId] = fem1;
		mxJac->m_frictionEffMass[2][mxId] = fem2;
#else
		hknpContactFrictionJacobian* fj = &mxJac->m_friction[mxId];
		hkReal fem0 = fj->m_frictionEffMass[0] * massFactor;
		hkReal fem1 = fj->m_frictionEffMass[1] * massFactor;
		hkReal fem2 = fj->m_frictionEffMass[2] * massFactor;
		fj->m_frictionRhs[0] = 0.0f;	// bad performance on spu
		fj->m_frictionRhs[1] = 0.0f;
		fj->m_frictionRhs[2] = 0.0f;
		fj->m_frictionEffMass[0] = fem0;
		fj->m_frictionEffMass[1] = fem1;
		fj->m_frictionEffMass[2] = fem2;
#endif

		const_cast<hknpManifoldCollisionCache*>(cache)->m_manifoldSolverInfo.m_frictionRhsMultiplier.setZero();
	}

	//HK_TIMER_END_LIST();
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
