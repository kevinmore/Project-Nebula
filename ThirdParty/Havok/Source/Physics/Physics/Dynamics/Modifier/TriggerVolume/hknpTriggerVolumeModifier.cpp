/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Modifier/TriggerVolume/hknpTriggerVolumeModifier.h>

#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>
#include <Physics/Physics/Dynamics/World/Commands/hknpInternalCommands.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>


void hknpTriggerVolumeModifier::manifoldCreatedCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
	ManifoldCreatedCallbackInput* HK_RESTRICT info)
{
	HK_ASSERT2( 0x2920835a,
		cdBodyA.m_material->m_triggerVolumeType != hknpMaterial::TRIGGER_VOLUME_NONE ||
		cdBodyB.m_material->m_triggerVolumeType != hknpMaterial::TRIGGER_VOLUME_NONE,
		"Trigger volume modifier called for non trigger volume materials" );

	hknpManifoldCollisionCache* HK_RESTRICT cache = info->m_collisionCache;

	// No need to calculate full manifolds.
	// (Should have been set when the cache was configured but we do it here just in case.)
	cache->m_qualityFlags.orWith( hknpBodyQuality::FORCE_GSK_SINGLE_POINT_MANIFOLD );

	const hkUint32 bodyFlagsA = cdBodyA.m_body->m_flags.get();
	const hkUint32 bodyFlagsB = cdBodyB.m_body->m_flags.get();

	// If either body wants to raise trigger events
	if( ( bodyFlagsA | bodyFlagsB ) & hknpBody::RAISE_TRIGGER_VOLUME_EVENTS )
	{
		// If either body is a high quality trigger volume
		if( ( ( bodyFlagsA & hknpBody::RAISE_TRIGGER_VOLUME_EVENTS ) &&
			  ( cdBodyA.m_material->m_triggerVolumeType == hknpMaterial::TRIGGER_VOLUME_HIGH_QUALITY ) ) ||
			( ( bodyFlagsB & hknpBody::RAISE_TRIGGER_VOLUME_EVENTS ) &&
			  ( cdBodyB.m_material->m_triggerVolumeType == hknpMaterial::TRIGGER_VOLUME_HIGH_QUALITY ) ) )
		{
			// Allow the solver to process the Jacobian, but not to apply any impulses.
			cache->m_maxImpulse.setZero();
			cache->m_fractionOfClippedImpulseToApply = 0;
		}
		else
		{
			// Don't bother with any solving
			cache->m_bodyAndMaterialFlags |= hknpBody::DONT_BUILD_CONTACT_JACOBIANS;

			// Raise a trigger volume entered event now
			hknpTriggerVolumeEvent event(
				cdBodyA.m_body->m_id, cdBodyA.m_shapeKey,
				cdBodyB.m_body->m_id, cdBodyB.m_shapeKey,
				hknpTriggerVolumeEvent::STATUS_ENTERED );
			tl.execCommand(event);

			cache->m_manifoldSolverInfo.m_flags.orWith( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_ENTERED );
		}
	}
	else
	{
		// Don't do any solving or raise any events
		cache->m_bodyAndMaterialFlags |= hknpBody::DONT_BUILD_CONTACT_JACOBIANS;
	}
}


void hknpTriggerVolumeModifier::manifoldProcessCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
	hknpManifold* HK_RESTRICT manifold)
{
	// Reduce collision detection frequency based on the tolerance settings
	{
		hkSimdReal distance;
		const hkUint32 bodyFlagsA = cdBodyA.m_body->m_flags.get();
		const hkUint32 bodyFlagsB = cdBodyB.m_body->m_flags.get();
		if( ( bodyFlagsA | bodyFlagsB ) & hknpBody::RAISE_TRIGGER_VOLUME_EVENTS )
		{
			hkSimdReal toleranceA; toleranceA.setFromFloat( cdBodyA.m_material->m_triggerVolumeTolerance );
			hkSimdReal toleranceB; toleranceB.setFromFloat( cdBodyB.m_material->m_triggerVolumeTolerance );
			hkSimdReal tolerance;  tolerance.setMin( toleranceA, toleranceB );

			distance = manifold->m_distances.horizontalMin<4>();

			// flag whether the manifold is in a penetrating state or not
			hknpManifoldSolverInfo& msi = manifold->m_collisionCache->m_manifoldSolverInfo;
			if( distance.isLessZero() )
			{
				msi.m_flags.orWith( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_PENETRATING );

				// Raise an enter event if it hasn't already been raised.
				// This can happen if bodies are spawned in an penetrating state.
				if( !msi.m_flags.anyIsSet( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_ENTERED ) )
				{
					hknpTriggerVolumeEvent event(
						cdBodyA.m_body->m_id, cdBodyA.m_shapeKey,
						cdBodyB.m_body->m_id, cdBodyB.m_shapeKey,
						hknpTriggerVolumeEvent::STATUS_ENTERED );
					tl.execCommand(event);

					msi.m_flags.orWith( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_ENTERED );
				}
			}
			else
			{
				msi.m_flags.clear( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_PENETRATING );
			}

			HK_ASSERT( 0x2920835b, manifold->m_numPoints > 1 || distance == manifold->m_distances.getComponent<0>() );
			distance.setAbs( distance );
			distance.add( tolerance );
		}
		else
		{
			// We don't want any events, so no need to keep doing collision detection
			distance = hkSimdReal_Max;
		}

		hknpMotionUtil::convertDistanceToLinearTIM(
			*sharedData.m_solverInfo.val(), distance, manifold->m_collisionCache->m_linearTim );
	}

	// helps optimizing the solver, skips friction
	manifold->m_manifoldType = hknpManifold::TYPE_TRIGGER;
}


void hknpTriggerVolumeModifier::manifoldDestroyedCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	hknpManifoldCollisionCache* HK_RESTRICT cache, hknpCdCacheDestructReason::Enum reason)
{
	const hknpManifoldSolverInfo& msi = cache->m_manifoldSolverInfo;

	if( msi.m_flags.get( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_ENTERED ) )
	{
		hknpTriggerVolumeEvent event(
			cdBodyA.m_body->m_id, cdBodyA.m_shapeKey,
			cdBodyB.m_body->m_id, cdBodyB.m_shapeKey,
			hknpTriggerVolumeEvent::STATUS_EXITED );
		tl.execCommand(event);
	}

	// Clear any cached trigger volume flags.
	msi.m_flags.clear(	hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_ENTERED |
						hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_PENETRATING );

}


void hknpTriggerVolumeModifier::postContactJacobianSetup(
	const hknpSimulationThreadContext& tl,
	const hknpSolverInfo& solverInfo,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	const hknpManifoldCollisionCache* cache, const hknpManifold* manifold,
	hknpMxContactJacobian* HK_RESTRICT mxJac, int mxJacIdx )
{
	// Set the effective mass of all contact points to 1 to make sure impulses are not zeroed during collisions with
	// keyframed or static bodies. cache->m_maxImpulse should have been set to 0 previously in manifoldCreatedCallback
	// to make sure that no actual impulses are applied.
	HK_ASSERT( 0x633f3482, cache->m_maxImpulse.isZero() );
	for( int i = 0; i < manifold->m_numPoints; i++ )
	{
		mxJac->m_contactPointData[i].m_effectiveMass[mxJacIdx] = 1.0f;
	}
}


void hknpTriggerVolumeModifier::postContactSolve(
	const hknpSimulationThreadContext& tl, const hknpModifier::SolverCallbackInput& input,
	hkVector4Parameter contactImpulses, hkReal frictionFactor )
{
	const hknpMxContactJacobian::ManifoldData& manifoldData = input.m_contactJacobian->m_manifoldData[input.m_manifoldIndex];
	const hknpManifoldSolverInfo* msi = &(input.m_collisionCache->m_manifoldSolverInfo);
	HK_ASSERT( 0x633f3483, msi );

	const hkBool32& wasImpulseClipped = input.m_wasImpulseClipped;
	const hkUint8 wasPenetrating = msi->m_flags.get( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_PENETRATING );
	const hkUint8 wasEnterEventRaised = msi->m_flags.get( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_ENTERED );

	if( !wasEnterEventRaised && wasImpulseClipped )
	{
		// The solver tried to apply some impulse, so this is *potentially* a new penetration.
		// Issue a command to examine the pair of bodies for a valid time of impact, post solve.
		hknpValidateTriggerVolumeEventCommand command( manifoldData.m_bodyIdA, manifoldData.m_bodyIdB );
		command.initialize( input );
		tl.execCommand( command );
	}

	// If we had no impulse or initial penetration, raise an exit event.
	// Note that no impulse does not imply no penetration.
	else if( wasEnterEventRaised && !wasImpulseClipped && !wasPenetrating )
	{
		// Raise an exit event.
		hknpTriggerVolumeEvent event(
			manifoldData.m_bodyIdA, manifoldData.m_shapeKeyA,
			manifoldData.m_bodyIdB, manifoldData.m_shapeKeyB,
			hknpTriggerVolumeEvent::STATUS_EXITED );
		tl.execCommand(event);

		// Clear any cached trigger volume flags.
		msi->m_flags.clear(	hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_ENTERED |
							hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_PENETRATING );
	}
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
