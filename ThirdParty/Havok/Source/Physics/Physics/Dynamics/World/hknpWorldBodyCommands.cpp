/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>

#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>

#include <Physics/Physics/Dynamics/World/Commands/hknpCommands.h>
#include <Physics/Physics/Dynamics/World/Commands/hknpInternalCommands.h>

#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>
#include <Physics/Physics/Dynamics/World/CacheManager/hknpCollisionCacheManager.h>


/// Update the AABB of the bodies if the motion has its velocity modified
void hknpWorld_updateBodyAabbOfAcceleratedMotion( hknpWorld* world, hknpBodyId bodyId, const hknpMotion& motion )
{
	const hknpBodyQualityLibrary* qualityLibrary = world->getBodyQualityLibrary();

	// iterate over all attached bodies
	hkUint32 attachedBodyId = bodyId.valueUnchecked();
	do
	{
		hknpBody& attachedBody = world->m_bodyManager.accessBody( hknpBodyId(attachedBodyId) );
		const hknpBodyQuality& quality = qualityLibrary->getEntry( attachedBody.m_qualityId );
		hknpMotionUtil::calcSweptBodyAabb(
			&attachedBody, motion, quality, world->m_collisionTolerance, world->m_solverInfo, world->m_intSpaceUtil );
		attachedBodyId = attachedBody.m_nextAttachedBodyId.valueUnchecked();
	} while( attachedBodyId != bodyId.valueUnchecked() );
}

/// Update the AABB of the bodies if the motion has its velocity modified
void hknpWorld_expandBodyAabb( hknpWorld* world, hknpBodyId bodyId, hkVector4Parameter direction )
{
	// iterate over all attached bodies
	hknpBodyId attachedBodyId = bodyId;
	do
	{
		hkAabb aabb; world->getBodyAabb( attachedBodyId, aabb );
		hkAabbUtil::expandAabbByMotion( aabb, direction, aabb );

		HK_ALIGN16( hkAabb16 aabb16 );
		world->m_intSpaceUtil.convertAabb( aabb, aabb16 );

		hknpBody& attachedBody = world->m_bodyManager.accessBody( attachedBodyId );
		attachedBody.m_aabb = aabb16;

		attachedBodyId = attachedBody.m_nextAttachedBodyId;
	} while( attachedBodyId != bodyId );
}


void hknpWorld::getBodyAabb( hknpBodyId bodyId, hkAabb& aabbOut ) const
{
	const hknpBody& body = m_bodyManager.getBody( bodyId );
	m_intSpaceUtil.restoreAabb( body.m_aabb, aabbOut );
}

void hknpWorld::setBodyMass( hknpBodyId bodyId, hkReal massOrNegativeDensity, RebuildCachesMode cacheBehavior )
{
	checkNotInSimulation();
	addTrace( hknpSetBodyMassCommand(bodyId, massOrNegativeDensity, cacheBehavior) );

	hknpBody &body = m_bodyManager.accessBody( bodyId );
	HK_ASSERT2( 0x043FF90F, body.m_motionId.value() >= hknpMotionId::NUM_PRESETS, "You cannot change the mass of preset bodies" );

	hknpMotion& motion = accessMotionUnchecked( body.m_motionId );
	if( massOrNegativeDensity == 0.0f )
	{
		if( !motion.hasInfiniteMass() )
		{
			//
			// Dynamic to keyframed
			//

			motion.setInfiniteInertiaAndMass();
			motion.m_motionPropertiesId = hknpMotionPropertiesId::KEYFRAMED;
			body.m_flags.orWith( hknpBody::IS_KEYFRAMED );

			// Force a cache rebuild, because the collision dispatcher might want to ignore keyframed overlaps
			rebuildBodyCollisionCaches( bodyId );
		}
	}
	else
	{
		if( motion.hasInfiniteMass() )
		{
			//
			// Keyframed to dynamic
			//

			// We need to calculate appropriate mass properties for the motion
			hkMassProperties massProperties;
			hkSimdReal massFactor;
			{
				// Get/build the shape's mass properties
				
				if( body.m_shape->getMassProperties( massProperties ) == HK_FAILURE )
				{
					HK_WARN( 0x972a1f28, "A shape has no mass properties. Approximating them instead." );
					hknpShape::MassConfig massConfig;
					massConfig.m_quality = hknpShape::MassConfig::QUALITY_LOW;
					hkDiagonalizedMassProperties dmp;
					body.m_shape->buildMassProperties( massConfig, dmp );
					dmp.unpack( &massProperties );
				}
				HK_ASSERT( 0x643FF90F, massProperties.m_mass > 0.0f );

				// Calculate a mass factor to match the desired mass/density
				if( massOrNegativeDensity > 0.0f )
				{
					massFactor.setFromFloat( massOrNegativeDensity / massProperties.m_mass );
				}
				else
				{
					massFactor.setFromFloat( ( -massOrNegativeDensity * massProperties.m_volume ) / massProperties.m_mass );
				}

				// Apply it
				massProperties.m_mass *= massFactor.getReal();
				massProperties.m_inertiaTensor.mul( massFactor );
			}

			// Override body COM
			body.getCenterOfMassLocal( massProperties.m_centerOfMass );

			// Set up the motion
			motion.setFromMassProperties( massProperties, body.getTransform() );
			motion.setMassFactor( massFactor );
			motion.m_motionPropertiesId = hknpMotionPropertiesId::DYNAMIC;

			// Force a cache rebuild, because the collision dispatcher might have ignored keyframed overlaps
			rebuildBodyCollisionCaches( bodyId );
		}
		else
		{
			//
			// Dynamic to dynamic
			//

			// Get the shape's mass properties
			
			hkDiagonalizedMassProperties dmp;
			if( body.m_shape->getMassProperties( dmp ) == HK_FAILURE )
			{
				HK_WARN( 0x972a1f29, "A shape has no mass properties. Approximating them instead." );
				hknpShape::MassConfig massConfig;
				massConfig.m_quality = hknpShape::MassConfig::QUALITY_LOW;
				body.m_shape->buildMassProperties( massConfig, dmp );
			}
			HK_ASSERT( 0x643FF91F, dmp.m_mass > 0.0f );

			// Calculate desired mass and mass factor
			hkSimdReal invMass;
			hkSimdReal massFactor;
			{
				hkSimdReal mass;
				if( massOrNegativeDensity > 0.0f )
				{
					mass.setFromFloat( massOrNegativeDensity );
				}
				else
				{
					mass.setFromFloat( -massOrNegativeDensity * dmp.m_volume );
				}
				invMass.setReciprocal( mass );

				hkSimdReal invShapeMass;
				invShapeMass.setReciprocal( hkSimdReal::fromFloat(dmp.m_mass) );
				massFactor = mass * invShapeMass;
			}

			// Set up the motion
			motion.setInverseMass( invMass );
			motion.setMassFactor( massFactor );
		}

		body.m_flags.clear( hknpBody::IS_KEYFRAMED );
	}

	m_signals.m_bodyChanged.fire( this, bodyId );
}

void hknpWorld::setBodyMotion( hknpBodyId bodyId, hknpMotionId motionId, RebuildCachesMode cacheBehavior )
{
	if( !m_bodyManager.isBodyWaitingToBeAdded(bodyId) )
	{
		// We need to be outside the simulation if the body is 'live'.
		checkNotInSimulation();
	}

	HK_ASSERT( 0x3420b473, bodyId.isValid() && motionId.isValid() );
	addTrace( hknpSetBodyMotionCommand(bodyId, motionId, cacheBehavior) );

	hknpBody &body = m_bodyManager.accessBody(bodyId);
	if( body.m_motionId == motionId )
	{
		return;
	}

	//
	// Static to dynamic
	//

	if( body.m_motionId == hknpMotionId::STATIC )
	{
		body.m_flags.clear( hknpBody::IS_STATIC );
		body.m_flags.orWith( hknpBody::IS_DYNAMIC );

		if( body.isAddedToWorld() )
		{
			m_broadPhase->markBodiesDirty( &bodyId, 1, m_bodyManager.accessBodyBuffer() );
		}

		// Flag the body's motion as invalid (required for addBodyToMotion() to succeed).
		body.m_motionId = hknpMotionId::invalid();
		addBodyToMotion( bodyId, motionId, HK_NULL );

		if( body.isAddedToWorld() )
		{
			registerBodyAtActiveList( bodyId, motionId );
			if( areConsistencyChecksEnabled() )
			{
				m_bodyManager.checkConsistency();
			}

			if( cacheBehavior == REBUILD_COLLISION_CACHES )
			{
				// rebuilding the caches will produce new broad phase pairs, which will wake up all attached bodies.
				rebuildBodyCollisionCaches( bodyId );
			}
			else
			{
				// We need to fix the collision cache for bodies in contact with the static body
				hkInplaceArray<hknpBodyId,32> hits;
				m_broadPhase->queryAabb( body.m_aabb, m_bodyManager.getBodyBuffer(), hits );
				for( int i = 0; i < hits.getSize(); i++ )
				{
					const hknpBodyId otherId = hits[i];

					// Ignore self
					if( otherId == bodyId )
					{
						continue;
					}

					// If colliding with another static body, we have to rebuild the collision cache between the two bodies
					const hknpBody& other = getBody( otherId );
					if( other.isStatic() )
					{
						hknpBodyIdPair staticPair;
						staticPair.m_bodyA = bodyId;
						staticPair.m_bodyB = otherId;
						rebuildBodyPairCollisionCaches( &staticPair, 1 );
					}
					// If colliding with an inactive body, simply wake it up
					else if( other.isInactive() )
					{
						activateBody( otherId );
					}
				}
			}
		}

		m_signals.m_bodySwitchStaticDynamic.fire( this, bodyId, false );
		m_signals.m_bodyChanged.fire( this, bodyId );
		return;
	}

	//
	// Dynamic to static
	//

	if( motionId == hknpMotionId::STATIC )
	{
		const hkBool32 isDynamicButDeactivated = body.isInactive();

		if( isDynamicButDeactivated )
		{
			// we need to wake up the body to clear the collision caches in the island
			activateBody( bodyId );
		}

		body.m_flags.clear( hknpBody::IS_DYNAMIC | hknpBody::IS_KEYFRAMED );
		body.m_flags.orWith( hknpBody::IS_STATIC );

		// set AABB
		{
			hkAabb aabb;
			hknpMotionUtil::calcStaticBodyAabb( body, m_collisionTolerance, &aabb );
			HK_ALIGN16(hkAabb16 aabb16);
			m_intSpaceUtil.convertAabb( aabb, aabb16 );
			hknpBody& bpBody = m_bodyManager.accessBody( bodyId );
			bpBody.m_aabb = aabb16;
		}

		if( body.isAddedToWorld() )
		{
			m_broadPhase->markBodiesDirty( &bodyId, 1, m_bodyManager.accessBodyBuffer() );

			// remove from motion
			if( !isDynamicButDeactivated )
			{
				unregisterBodyAtActiveList( &body );
			}
		}

		removeBodyFromMotion( &body );

		// attach it to the fixed motion
		body.m_motionId = hknpMotionId::STATIC;
		body.syncStaticMotionToBodyTransform();

		if( body.isAddedToWorld() )
		{
			if( cacheBehavior == REBUILD_COLLISION_CACHES )
			{
				rebuildBodyCollisionCaches( bodyId );
			}
			else
			{
				// now we have to make sure to update the previous AABB
				hkAabb16* HK_RESTRICT prevAabb = &m_bodyManager.m_previousAabbs[ bodyId.value() ];
				if ( !prevAabb->isEmpty() && !prevAabb->isEqual( body.m_aabb ))
				{
					m_bodyManager.setScheduledBodyFlags(bodyId, hknpBodyManager::MOVED_STATIC);
				}
			}
		}

		m_signals.m_bodySwitchStaticDynamic.fire( this, bodyId, true );
		m_signals.m_bodyChanged.fire( this, bodyId );
		return;
	}

	//
	// Dynamic to dynamic
	//

	{
		// Make sure it is not deactivated
		if ( body.isAddedToWorld() && body.isInactive() )
		{
			activateBody( bodyId );
			m_deactivationManager->activateMarkedIslands();
		}

		// Detach from the old motion
		if( body.isAddedToWorld() )
		{
			unregisterBodyAtActiveList( &body );
		}
		removeBodyFromMotion( &body );

		// Attach to the new motion
		addBodyToMotion( bodyId, motionId );
		if( body.isAddedToWorld() )
		{
			registerBodyAtActiveList( bodyId, motionId );
			rebuildBodyCollisionCaches( bodyId );
		}

		m_signals.m_bodyChanged.fire( this, bodyId );
		return;
	}
}

void hknpWorld::synchronizeBodiesFromMotion(
	hknpBodyId firstBodyId, hkSimdRealParameter movedDistance, hkSimdRealParameter rotatedAngle )
{
	checkNotInSimulation();

	HK_ON_DEBUG( const hknpBody& body = m_bodyManager.getBody(firstBodyId) );
	HK_ASSERT( 0xf034dfed, body.isValid() );

	hkUint16 linearTim;
	hkUint16 angleTim;
	{
		hknpMotionUtil::convertDistanceToLinearTIM( m_solverInfo, movedDistance, linearTim );
		hkSimdReal scaledAng = rotatedAngle * hkSimdReal::fromFloat(510.0f/HK_REAL_PI) + hkSimdReal::fromFloat(0.999f);
		scaledAng.storeSaturateUint16( &angleTim );
	}

	hknpBodyId attachedBodyId = firstBodyId;
	do
	{
		hknpMotionUtil::updateBodyAfterMotionIntegration( this, attachedBodyId );

		hknpBody& attachedBody = m_bodyManager.accessBody( attachedBodyId );
		attachedBody.m_maxTimDistance = hkMath::min2( hkUint16(linearTim+attachedBody.m_maxTimDistance), 0xffff );
		attachedBody.m_timAngle = static_cast<hkUint8>( hkMath::min2(hkUint32(angleTim+attachedBody.m_timAngle), hkUint32(0xff)) );

		attachedBodyId = attachedBody.m_nextAttachedBodyId;
	} while( attachedBodyId != firstBodyId );
}

void hknpWorld::updateMotionAndAttachedBodiesAfterModifyingTransform(
	hknpBodyId bodyId, const hkQuaternion* orientation, hknpWorld::PivotLocation pivot,
	hknpActivationBehavior::Enum activationBehavior )
{
	hknpBody& body = m_bodyManager.accessBody( bodyId );

	const hkTransform& transform = body.getTransform();

	hknpMotionId motionId = body.m_motionId;

	if( motionId != hknpMotionId::STATIC )
	{
		hknpMotion& motion = accessMotionUnchecked( motionId );
		hkCheckDeterminismUtil::checkMtCrc( 0xf0002d45, &motion );

		hkVector4 massCenterLocal; body.getCenterOfMassLocal( massCenterLocal );
		hkVector4 newMassCenterWorld; newMassCenterWorld._setTransformedPos( transform, massCenterLocal );

		hkSimdReal comMovedDistance; comMovedDistance.setZero();
		hkSimdReal comMovedAngle;    comMovedAngle.setZero();

		if( pivot == PIVOT_BODY_POSITION )
		{
			hkVector4 massCenterMovedDistance; massCenterMovedDistance.setSub( newMassCenterWorld, motion.getCenterOfMassInWorld() );
			motion.setCenterOfMassInWorld( newMassCenterWorld );
			comMovedDistance = massCenterMovedDistance.length<3>();
		}
		else
		{
			HK_ON_DEBUG( hkSimdReal epsilon( motion.getCenterOfMassInWorld().length<3>() ); )
			HK_ON_DEBUG( epsilon.add( hkSimdReal_1 ); )
			HK_ON_DEBUG( epsilon.mul( hkSimdReal::fromFloat(0.0001f) ); )
			HK_ASSERT( 0xaf142e31, motion.getCenterOfMassInWorld().allEqual<3>( newMassCenterWorld, epsilon ) );
		}

		if( orientation )
		{
			hkQuaternion bodyQmotion; body.m_motionToBodyRotation.unpack( &bodyQmotion.m_vec );
			bodyQmotion.normalize();

			hkQuaternion newOrientation; newOrientation.setMul( *orientation, bodyQmotion );
			hkVector4 deltaAngle; motion.m_orientation.estimateAngleTo( newOrientation, deltaAngle );

			motion.m_orientation = newOrientation;

			comMovedAngle = deltaAngle.length<3>();
		}

		hknpMotionUtil::updateCellIdx( this, &motion, motionId );

		// update all attached bodies
		synchronizeBodiesFromMotion( bodyId, comMovedDistance, comMovedAngle );

		if( body.isAddedToWorld() )
		{
			m_deactivationManager->markBodyForActivation( bodyId );
		}
	}
	else
	{
		body.syncStaticMotionToBodyTransform();

		m_signals.m_staticBodyMoved.fire( this, bodyId );

		
		
		if( body.isAddedToWorld() )
		{
			body.m_maxTimDistance = 0xffff;
			body.m_timAngle = 0xff;
			m_bodyManager.setScheduledBodyFlags( bodyId, hknpBodyManager::MOVED_STATIC );
		}

		hkAabb16 prevAabb = body.m_aabb;

		hkAabb aabb; hknpMotionUtil::calcStaticBodyAabb( body, m_collisionTolerance, &aabb );
		HK_ALIGN16(hkAabb16 aabb16); m_intSpaceUtil.convertAabb( aabb, aabb16 );
		if( !aabb16.isEqual( body.m_aabb ) )
		{
			body.m_aabb = aabb16;
			if( body.isAddedToWorld() )
			{
				hknpBody* bodies = m_bodyManager.accessBodyBuffer();

				// Bodies in the static broad phase layer need to be refreshed (to update their AABB)
				m_broadPhase->markBodiesDirty( &bodyId, 1, bodies );

				if( activationBehavior != hknpActivationBehavior::KEEP_DEACTIVATED )
				{
					// iterate over the old pairs and activate if no longer overlapping
					{
						hkInplaceArray<hknpBodyId,32> hitsAtOldPosition;
						
						
						m_broadPhase->queryAabb( prevAabb, m_bodyManager.getBodyBuffer(), hitsAtOldPosition);

						for( int i = 0; i < hitsAtOldPosition.getSize(); i++ )
						{
							hknpBodyId otherId = hitsAtOldPosition[i];
							const hknpBody& other = getBody(otherId);
							if( !other.isInactive() )
							{
								continue;
							}
							// check whether this pair a no longer colliding collision
							if( activationBehavior == hknpActivationBehavior::ACTIVATE || other.m_aabb.disjoint( aabb16 ) )
							{
								m_deactivationManager->markBodyForActivation( otherId );
							}
						}
					}

					{
						// get the new pairs. If partner body:
						//  - isDynamic:  Ignore as this will find the new pairs automatically
						//  - isStatic:   We can ignore the collisions
						//  - isDeactivated:	Activate
						hkInplaceArray<hknpBodyId,32> hitsAtNewPosition;
						m_broadPhase->queryAabb( body.m_aabb, m_bodyManager.getBodyBuffer(), hitsAtNewPosition );

						hkInplaceArray<hknpBodyIdPair,32> pairs;
						for( int i = 0; i < hitsAtNewPosition.getSize(); i++ )
						{
							hknpBodyId otherId = hitsAtNewPosition[i];
							const hknpBody& other = getBody(otherId);
							if( (otherId == bodyId) || !other.isInactive() )
							{
								continue;
							}

							hkBool32 isDisjoint = m_bodyManager.m_previousAabbs[bodyId.value()].disjoint( m_bodyManager.m_previousAabbs[otherId.value()] );
							if( isDisjoint )
							{
								hknpBodyIdPair& np = pairs.expandOne();
								np.m_bodyA = bodyId;
								np.m_bodyB = otherId;
							}
							// check whether this is a new collision
							if( activationBehavior == hknpActivationBehavior::ACTIVATE || isDisjoint )
							{
								m_deactivationManager->markBodyForActivation( otherId );
							}

							// we need to build the collision caches of deactivated bodies,
							// as those bodies will activated after the broad phase call
						}

						if( pairs.getSize() )
						{
							rebuildBodyPairCollisionCaches( pairs.begin(), pairs.getSize() );
						}
					}
				}
			}
		}
	}

	if( areConsistencyChecksEnabled() )
	{
		m_bodyManager.checkConsistency();
	}
}

void hknpWorld::setBodyPosition(
	hknpBodyId bodyId, hkVector4Parameter position, hknpActivationBehavior::Enum activationBehavior )
{
	checkNotInSimulation();
	addTrace( hknpSetBodyPositionCommand( bodyId, position, activationBehavior ) );

	hknpBody& body = m_bodyManager.accessBody( bodyId );
	HK_ASSERT( 0xaf34dfec, body.isValid() );

	hkCheckDeterminismUtil::checkMtCrc( 0xaf001e21, &body.getTransform() );
	hkCheckDeterminismUtil::checkMtCrc( 0xaf001e22, &position );

	hkTransform transform( body.getTransform() );
	if( !transform.getTranslation().allEqual<3>( position, hkSimdReal_Eps ) )
	{
		transform.setTranslation( position );

		body.setTransform( transform );
		updateMotionAndAttachedBodiesAfterModifyingTransform( bodyId, HK_NULL, PIVOT_BODY_POSITION, activationBehavior );
		m_signals.m_bodyChanged._fire( this, bodyId );
	}
}

void hknpWorld::setBodyOrientation(
	hknpBodyId bodyId, const hkQuaternion& orientation, hknpWorld::PivotLocation pivot,
	hknpActivationBehavior::Enum activationBehavior )
{
	checkNotInSimulation();
	addTrace( hknpSetBodyOrientationCommand( bodyId, orientation, pivot, activationBehavior ) );

	hknpBody& body = m_bodyManager.accessBody( bodyId );
	HK_ASSERT( 0xaf34dfec, body.isValid() );

	hkCheckDeterminismUtil::checkMtCrc( 0xaf001e23, &body.getTransform() );
	hkCheckDeterminismUtil::checkMtCrc( 0xaf001e24, &orientation );

	hkRotation orientationMatrix;
	orientationMatrix.set( orientation );	

	hkTransform transform( body.getTransform() );
	if( !transform.getRotation().isApproximatelyEqualSimd( orientationMatrix, hkSimdReal_Eps ) )
	{
		transform.setRotation( orientationMatrix );
		if( pivot == PIVOT_CENTER_OF_MASS )
		{
			const hknpMotion& motion = getMotion( body.m_motionId );

			hkVector4 massCenterLocal; body.getCenterOfMassLocal( massCenterLocal );
			hkVector4 massCenterPostRotationWorldSpace; massCenterPostRotationWorldSpace.setTransformedPos( transform, massCenterLocal );
			hkVector4 deltaMassCenter; deltaMassCenter.setSub( massCenterPostRotationWorldSpace, motion.getCenterOfMassInWorld() );
			hkVector4 updatedPosition; updatedPosition.setSub( transform.getTranslation(), deltaMassCenter );

			transform.setTranslation( updatedPosition );
		}

		body.setTransform( transform );
		updateMotionAndAttachedBodiesAfterModifyingTransform( bodyId, &orientation, pivot, activationBehavior );
		m_signals.m_bodyChanged._fire( this, bodyId );
	}
}

void hknpWorld::setBodyTransform(
	hknpBodyId bodyId, const hkTransform& transform, hknpActivationBehavior::Enum activationBehavior )
{
	// Check for 'live' bodies not being modified during simulation
	if ( !m_bodyManager.isBodyWaitingToBeAdded(bodyId) )
	{
		checkNotInSimulation();
	}

	addTrace( hknpSetBodyTransformCommand( bodyId, transform, activationBehavior ) );

	hknpBody& body = m_bodyManager.accessBody(bodyId);
	HK_ASSERT( 0xf034dfed, body.isValid() );

	hkCheckDeterminismUtil::checkMtCrc( 0xf0002d45, &body.getTransform() );
	hkCheckDeterminismUtil::checkMtCrc( 0xf0002d46, &transform );

	if( !body.getTransform().isApproximatelyEqualSimd( transform, hkSimdReal_Eps ) )
	{
		hkQuaternion orientation;
		orientation.set( transform.getRotation() );	

		body.setTransform( transform );
		updateMotionAndAttachedBodiesAfterModifyingTransform( bodyId, &orientation,
			hknpWorld::PIVOT_BODY_POSITION, activationBehavior );
		m_signals.m_bodyChanged._fire( this, bodyId );
	}
}

void hknpWorld::reintegrateBody( hknpBodyId bodyId, hkReal fraction )
{
	checkNotInSimulation();
	addTrace( hknpReintegrateBodyCommand( bodyId, fraction ) );

	HK_ASSERT2( 0xf045fe45, fraction >= 0.0f && fraction <= 1.0f, "'fraction' has to be [0.0f, 1.0f]" );
	hkSimdReal f; f.setFromFloat(fraction);

	const hknpBody& body = m_bodyManager.getBody( bodyId );
	HK_ASSERT( 0xf034dfed, body.isValid() );

	hknpMotionId motionId = body.m_motionId;
	if( motionId == hknpMotionId::STATIC )
	{
		return;
	}

	hknpMotion& motion = accessMotionUnchecked( motionId );
	motion.reintegrate( f, m_solverInfo.m_deltaTime );

	// update all attached bodies
	// no need to update TIMs as the current TIMs are already over-conservative
	hkSimdReal zeroMovement; zeroMovement.setZero();
	synchronizeBodiesFromMotion( bodyId, zeroMovement, zeroMovement );

	m_signals.m_bodyChanged.fire( this, bodyId );
}

void hknpWorld::getBodyVelocity( hknpBodyId bodyId, hkVector4& linearVelocity, hkVector4& angularVelocity ) const
{
	const hknpBody& body = getBody( bodyId );
	const hknpMotion& motion = getMotion( body.m_motionId );
	linearVelocity = motion.getLinearVelocity();
	motion._getAngularVelocity( angularVelocity );
}

void hknpWorld::setBodyVelocity( hknpBodyId bodyId,
	hkVector4Parameter linearVelocity, hkVector4Parameter angularVelocity )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );
	addTrace( hknpSetBodyVelocityCommand( bodyId, linearVelocity, angularVelocity) );

	const hknpBody& body = getBody(bodyId);
	HK_WARN_ON_DEBUG_IF( body.isStatic(), 0xf034dfed, "Setting velocity on a static body has no effect." );
	if( body.isDynamic() )
	{
		hknpMotion& motion = accessMotionUnchecked( body.m_motionId );

		// If both velocities are unchanged, do nothing
		if( linearVelocity.allEqual<3>( motion.getLinearVelocity(), hkSimdReal_Eps ) )
		{
			hkVector4 currentAngVel;
			motion._getAngularVelocity( currentAngVel );
			if( angularVelocity.allEqual<3>( currentAngVel, hkSimdReal_Eps ) )
			{
				return;
			}
		}

		// Make sure the body is active
		if( body.isInactive() )
		{
			activateBody(bodyId);
		}

		motion.m_linearVelocity = linearVelocity;
		motion.m_angularVelocity._setRotatedInverseDir( motion.m_orientation, angularVelocity );
		hknpWorld_updateBodyAabbOfAcceleratedMotion( this, bodyId, motion );

		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

const hkVector4& hknpWorld::getBodyLinearVelocity( hknpBodyId bodyId ) const
{
	const hknpBody& body = getBody( bodyId );
	const hknpMotion& motion = getMotion( body.m_motionId );
	return motion.getLinearVelocity();
}

void hknpWorld::getBodyAngularVelocity( hknpBodyId bodyId, hkVector4& angularVelocity ) const
{
	const hknpBody& body = getBody( bodyId );
	const hknpMotion& motion = getMotion( body.m_motionId );
	motion._getAngularVelocity( angularVelocity );
}

void hknpWorld::setBodyLinearVelocity( hknpBodyId bodyId, hkVector4Parameter linearVelocity )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );
	addTrace( hknpSetBodyLinearVelocityCommand( bodyId, linearVelocity ) );

	const hknpBody& body = getBody( bodyId );
	HK_WARN_ON_DEBUG_IF( body.isStatic(), 0xafe12132, "Setting linear velocity on a static body has no effect." );
	if( body.isDynamic() )
	{
		hknpMotion& motion = accessMotionUnchecked( body.m_motionId );

		// If velocity is unchanged, do nothing
		if( linearVelocity.allEqual<3>( motion.getLinearVelocity(), hkSimdReal_Eps ) )
		{
			return;
		}

		// Make sure the body is active
		if( body.isInactive() )
		{
			activateBody( bodyId );
		}

		motion.setLinearVelocity( linearVelocity );
		hknpWorld_updateBodyAabbOfAcceleratedMotion( this, bodyId, motion );

		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::setBodyAngularVelocity( hknpBodyId bodyId, hkVector4Parameter angularVelocity )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );
	addTrace( hknpSetBodyAngularVelocityCommand( bodyId, angularVelocity ) );

	const hknpBody& body = getBody( bodyId );
	HK_WARN_ON_DEBUG_IF( body.isStatic(), 0xafe12131, "Setting angular velocity on a static body has no effect." );
	if( body.isDynamic() )
	{
		hknpMotion& motion = accessMotionUnchecked( body.m_motionId );

		// If velocity is unchanged, do nothing
		{
			hkVector4 currentAngVel;
			motion._getAngularVelocity( currentAngVel );
			if( angularVelocity.allEqual<3>( currentAngVel, hkSimdReal_Eps ) )
			{
				return;
			}
		}

		// Make sure the body is active
		if( body.isInactive() )
		{
			activateBody( bodyId );
		}

		motion._setAngularVelocity( angularVelocity );
		hknpWorld_updateBodyAabbOfAcceleratedMotion( this, bodyId, motion );

		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::applyHardKeyFrame( hknpBodyId bodyId, hkVector4Parameter targetPosition,
	const hkQuaternion& targetOrientation, hkReal deltaTime )
{
	HK_ASSERT2( 0x1f9b3a50, deltaTime > 0.0f, "Delta time must be greater than zero." );

	hkSimdReal invDeltaTimeSr;
	invDeltaTimeSr.setReciprocal( hkSimdReal::fromFloat(deltaTime) );

	const hknpBody& body = getBody( bodyId );
	const hknpMotion& motion = getMotion( body.m_motionId );

	// Calculate desired linear velocity
	
	hkVector4 linearVelocity;
	{
		hkVector4 newCenterOfMassPosition;
		hkVector4 comLocal;
		body.getCenterOfMassLocal( comLocal );
		newCenterOfMassPosition._setRotatedDir( targetOrientation, comLocal );
		newCenterOfMassPosition.add( targetPosition );
		linearVelocity.setSub( newCenterOfMassPosition, motion.getCenterOfMassInWorld() );
		linearVelocity.setMul( invDeltaTimeSr, linearVelocity );
	}

	// Calculate desired angular velocity
	hkVector4 angularVelocity;
	{
		hkQuaternion bodyOrient; bodyOrient.set( getBodyTransform(bodyId).getRotation() );
		hkQuaternion quatDif;
		quatDif.setMulInverse( targetOrientation, bodyOrient );
		quatDif.normalize();

		if( !quatDif.hasValidAxis() )
		{
			angularVelocity.setZero();
		}
		else
		{
			hkSimdReal angle; angle.setFromFloat( quatDif.getAngle() );
			quatDif.getAxis( angularVelocity );
			angularVelocity.setMul( angle * invDeltaTimeSr, angularVelocity );
		}
	}

	// Apply them
	setBodyVelocity( bodyId, linearVelocity, angularVelocity );
}

void hknpWorld::applyBodyLinearImpulse( hknpBodyId bodyId, hkVector4Parameter linearImpulse )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );
	addTrace( hknpApplyLinearImpulseCommand(bodyId, linearImpulse) );

	const hknpBody& body = getBody(bodyId);
	if( body.isDynamic() )
	{
		// Make sure the body is active
		if( body.isInactive() )
		{
			activateBody( bodyId );
		}

		hknpMotion& motion = accessMotionUnchecked( body.m_motionId );
		motion._applyLinearImpulse( linearImpulse );
		hknpWorld_updateBodyAabbOfAcceleratedMotion( this, bodyId, motion );

		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::applyBodyAngularImpulse( hknpBodyId bodyId, hkVector4Parameter angularImpulse )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );
	addTrace( hknpApplyAngularImpulseCommand(bodyId, angularImpulse ) );

	const hknpBody& body = getBody(bodyId);
	if( body.isDynamic() )
	{
		// Make sure the body is active
		if( body.isInactive())
		{
			activateBody( bodyId );
		}

		hknpMotion& motion = accessMotionUnchecked( body.m_motionId );
		motion._applyAngularImpulse( angularImpulse );
		hknpWorld_updateBodyAabbOfAcceleratedMotion( this, bodyId, motion );

		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::applyBodyImpulseAt( hknpBodyId bodyId, hkVector4Parameter impulse, hkVector4Parameter position )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );
	addTrace( hknpApplyPointImpulseCommand(bodyId, impulse, position) );

	const hknpBody& body = getBody(bodyId);
	if( body.isDynamic() )
	{
		// Make sure the body is active
		if( body.isInactive() )
		{
			activateBody( bodyId );
		}

		hknpMotion& motion = accessMotionUnchecked( body.m_motionId );
		motion._applyPointImpulse( impulse, position );
		hknpWorld_updateBodyAabbOfAcceleratedMotion( this, bodyId, motion );

		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::setBodyVelocityAt( hknpBodyId bodyId, hkVector4Parameter velocity, hkVector4Parameter position )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );
	addTrace( hknpSetPointVelocityCommand(bodyId, velocity, position) );

	const hknpBody& body = getBody(bodyId);
	if( body.isDynamic() )
	{
		// Make sure the body is active
		if( body.isInactive() )
		{
			activateBody( bodyId );
		}

		hknpMotion& motion = accessMotionUnchecked( body.m_motionId );
		motion.setPointVelocity( velocity, position );
		hknpWorld_updateBodyAabbOfAcceleratedMotion( this, bodyId, motion );

		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::setBodyCenterOfMass( hknpBodyId bodyId, hkVector4Parameter centerOfMass )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	addTrace( hknpSetBodyCenterOfMassCommand(bodyId, centerOfMass) );

	const hknpBody& body = getBody(bodyId);
	if( body.isDynamic() )
	{
		// Make sure the body is active
		if( body.isInactive() )
		{
			activateBody( bodyId );
		}

		// Set motion's COM
		hknpMotion& motion = accessMotionUnchecked( body.m_motionId );
		motion.setCenterOfMassInWorld( centerOfMass );

		// Update offset and bounding radius of all attached bodies
		hknpBodyId attachedBodyId = bodyId;
		do
		{
			hknpBody& attachedBody = m_bodyManager.accessBody( attachedBodyId );
			hkVector4 comMinusPosition; comMinusPosition.setSub( motion.getCenterOfMassInWorld(), attachedBody.getTransform().getTranslation() );
			hkVector4 centerOfMassLocal; centerOfMassLocal._setRotatedInverseDir( attachedBody.getTransform().getRotation(), comMinusPosition );
			attachedBody.setBodyToMotionTranslation( centerOfMassLocal );
			attachedBody.updateComCenteredBoundingRadius( motion );

			attachedBodyId = attachedBody.m_nextAttachedBodyId;
		} while( attachedBodyId != bodyId );

		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::setBodyShape( hknpBodyId bodyId, const hknpShape* shape )
{
	checkNotInSimulation();
	addTrace( hknpSetBodyShapeCommand(bodyId, shape) );

	hknpBody& body = accessBody(bodyId);

	// Make sure the body is active
	if( body.isInactive() )
	{
		activateBody( bodyId );
	}

	// If original shape was mutable, tell the shape manager
	if( body.m_shape->isMutable() )
	{
		m_shapeManager.deregisterBodyWithMutableShape( body );
	}

	body.setShape( shape );

	// If new shape is mutable, tell the shape manager
	if( body.m_shape->isMutable() )
	{
		m_shapeManager.registerBodyWithMutableShape( body );
	}

	if( body.m_motionId != hknpMotionId::STATIC )
	{
		const hknpMotion& motion = getMotion(body.m_motionId );
		body.updateComCenteredBoundingRadius( motion );
		hknpWorld_updateBodyAabbOfAcceleratedMotion( this, bodyId, motion );
	}
	else
	{
		updateMotionAndAttachedBodiesAfterModifyingTransform(
			bodyId, HK_NULL, hknpWorld::PIVOT_CENTER_OF_MASS, hknpActivationBehavior::ACTIVATE );
	}

	// Collision caches must be rebuilt
	rebuildBodyCollisionCaches( bodyId );

	m_signals.m_bodyShapeChanged.fire( this, bodyId );
	m_signals.m_bodyChanged.fire( this, bodyId );
}

void hknpWorld::setBodyMotionProperties( hknpBodyId bodyId, hknpMotionPropertiesId motionPropertiesId )
{
	HK_ASSERT2( 0xf03dfde4, motionPropertiesId.value() > 0 && motionPropertiesId.value() < m_motionPropertiesLibrary->getCapacity(),
		"Motion properties ID out of range" );
	addTrace( hknpSetBodyMotionPropertiesCommand(bodyId, motionPropertiesId) );

	hknpBody& body = accessBody( bodyId );
	hknpMotion& motion = accessMotionUnchecked( body.m_motionId );
	if( motion.m_motionPropertiesId != motionPropertiesId )
	{
		motion.m_motionPropertiesId = motionPropertiesId;
		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::rebuildBodyPairCollisionCaches( hknpBodyIdPair* pairs, int numPairs )
{
	checkNotInSimulation();

	m_collisionCacheManager->m_newUserCollisionPairs.append( pairs, numPairs );
}

void hknpWorld::rebuildBodyCollisionCaches( hknpBodyId bodyId )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	addTrace( hknpRebuildBodyCollisionCachesCommand(bodyId) );

	hknpBody& body = accessBody( bodyId );
	hkAabb16* HK_RESTRICT prevAabb = &m_bodyManager.m_previousAabbs[ bodyId.value() ];
	if( body.isDynamic() )
	{
		if( body.isInactive() )
		{
			// we need to wake up this body immediately
			checkNotInSimulation();
			activateBody( bodyId );
		}
		else
		{
			// body is active, make sure it stays active for a while
			m_deactivationManager->resetDeactivationFrameCounter( body.m_motionId );
		}

		// treat any broad phase overlaps as new ones
		prevAabb->setEmpty();
	}
	else
	{
		if( !prevAabb->isEmpty() )
		{
			prevAabb->setEmpty();
			if( body.isAddedToWorld() )
			{
				m_bodyManager.setScheduledBodyFlags( bodyId, hknpBodyManager::MOVED_STATIC );
			}
		}
	}

	// Kill any existing collision caches
	body.m_flags.orWith( hknpBody::TEMP_REBUILD_COLLISION_CACHES );
}

void hknpWorld::setBodyMaterial( hknpBodyId bodyId, hknpMaterialId materialId, RebuildCachesMode cacheBehavior )
{
	checkNotInSimulation();
	addTrace( hknpSetBodyMaterialCommand( bodyId, materialId, cacheBehavior ) );

	hknpBody& body = accessBody( bodyId );
	if( body.m_materialId != materialId )
	{
		body.m_materialId = materialId;
		if( cacheBehavior == REBUILD_COLLISION_CACHES )
		{
			rebuildBodyCollisionCaches( bodyId );
		}
		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::setBodyQuality( hknpBodyId bodyId, hknpBodyQualityId qualityId, RebuildCachesMode cacheBehavior )
{
	checkNotInSimulation();
	addTrace(hknpSetBodyQualityCommand( bodyId, qualityId, cacheBehavior ) );

	hknpBody& body = accessBody( bodyId );
	if( body.m_qualityId != qualityId )
	{
		body.m_qualityId = qualityId;
		if( cacheBehavior == REBUILD_COLLISION_CACHES )
		{
			rebuildBodyCollisionCaches( bodyId );
		}
		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::setBodyCollisionLookAheadDistance( hknpBodyId bodyId, hkReal distance, hkVector4Parameter tempExpansionVelocity )
{
	checkNotInSimulation();
	addTrace( hknpSetBodyCollisionLookAheadDistanceCommand( bodyId, distance, tempExpansionVelocity ) );

	hknpBody& body = accessBody(bodyId);
	if( ( body.getCollisionLookAheadDistance() == distance ) && tempExpansionVelocity.allExactlyEqualZero<3>() )
	{
		return;
	}
	if( body.isStatic() )
	{
		return;
	}

	if( body.isInactive() )
	{
		activateBody( bodyId );
	}

	if( distance > 0.0f )
	{
		body.setCollisionLookAheadDistance( distance );

		// update motion
		hknpMotion& motion = accessMotionUnchecked( body.m_motionId );
		{
			hkReal maxIntegrationDistance = HK_REAL_HIGH;

			hknpBodyId attachedBodyId = bodyId;
			do
			{
				const hknpBody& attachedBody = m_bodyManager.getBody(attachedBodyId);
				hkReal d = attachedBody.getCollisionLookAheadDistance();
				if( d > 0.0f )
				{
					maxIntegrationDistance = hkMath::min2( d, maxIntegrationDistance );
				}
			} while( attachedBodyId != bodyId );

			motion.m_maxLinearAccelerationDistancePerStep.setReal<false>( maxIntegrationDistance );
		}
		tempExpansionVelocity.store<3>( motion.m_linearVelocityCage );
		hknpWorld_updateBodyAabbOfAcceleratedMotion( this, bodyId, motion );
	}
	else
	{
		hkVector4 direction; direction.setMul( tempExpansionVelocity, m_solverInfo.m_deltaTime );
		hknpWorld_expandBodyAabb( this, bodyId, direction );
	}

	m_signals.m_bodyChanged.fire( this, bodyId );
}

void hknpWorld::setBodyCollisionFilterInfo( hknpBodyId bodyId, hkUint32 collisionFilterInfo, RebuildCachesMode cacheBehavior )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	addTrace( hknpSetBodyCollisionFilterInfoCommand(bodyId, collisionFilterInfo) );

	hknpBody& body = accessBody( bodyId );
	if( body.m_collisionFilterInfo != collisionFilterInfo )
	{
		body.m_collisionFilterInfo = collisionFilterInfo;
		if( cacheBehavior == REBUILD_COLLISION_CACHES )
		{
			rebuildBodyCollisionCaches( bodyId );
		}
		m_signals.m_bodyChanged.fire( this, bodyId );
	}
}

void hknpWorld::checkBodyConsistency( hknpWorld* world, hknpBodyId bodyId ) const
{
#ifdef HK_DEBUG

	HK_ASSERT( 0x30798460, world == this );
	const hknpBody& body = getBody( bodyId );

	// Check transform
	{
		const hkVector4& position = body.getTransform().getTranslation();
		const hkRotation& orientation = body.getTransform().getRotation();

		HK_ASSERT3( 0x30798461, position.isOk<4>(), "Body " << bodyId.value() << "'s position is invalid" );
		HK_ASSERT3( 0x30798462, orientation.isOk(), "Body " << bodyId.value() << "'s orientation is invalid" );

		const hkSimdReal maxBpDistance = m_solverInfo.m_unitScale * hkSimdReal::fromFloat( 1000.0f );
		HK_WARN_ON_DEBUG_IF( hkAabbUtil::distanceSquared( m_intSpaceUtil.m_aabb, position ) > maxBpDistance * maxBpDistance,
			0x30798463, "Body " << bodyId.value() << "'s position is very far outside broad phase AABB" );
	}

	// Check velocities
	if( body.isDynamic() )
	{
		const hknpMotion& motion = getMotion( body.m_motionId );
		const hkVector4& linearVelocity = motion.getLinearVelocity();
		hkVector4 angularVelocity; motion.getAngularVelocity( angularVelocity );

		HK_ASSERT3( 0x30798464, linearVelocity.isOk<4>(), "Body " << bodyId.value() << "'s linear velocity is invalid" );
		HK_ASSERT3( 0x30798466, angularVelocity.isOk<4>(), "Body " << bodyId.value() << "'s angular velocity is invalid" );

		const hkSimdReal maxLinearSpeed = m_solverInfo.m_unitScale * hkSimdReal::fromFloat( 10000.0f );
		HK_WARN_ON_DEBUG_IF( linearVelocity.lengthSquared<3>().isGreater( maxLinearSpeed * maxLinearSpeed ),
			0x30798465, "Body " << bodyId.value() << "'s linear speed is very high" );

		const hkSimdReal maxAngularSpeed = hkSimdReal::fromFloat( 1000.0f );
		HK_WARN_ON_DEBUG_IF( angularVelocity.lengthSquared<3>().isGreater( maxAngularSpeed * maxAngularSpeed ),
			0x30798467, "Body " << bodyId.value() << "'s angular speed is very high" );
	}

#endif // HK_DEBUG
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
