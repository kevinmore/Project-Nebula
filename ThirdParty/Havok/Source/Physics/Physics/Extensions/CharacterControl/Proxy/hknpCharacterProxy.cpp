/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxy.h>
#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxyListener.h>
#include <Physics/Physics/Extensions/CharacterControl/hknpCharacterSurfaceInfo.h>
#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h>
#include <Common/Base/Types/Physics/hkStepInfo.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Internal/SimplexSolver/hkSimplexSolver.h>


#if defined(HKNP_DEBUG_CHARACTER_PROXY)

#include <Common/Visualize/hkDebugDisplay.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Visualize/Shape/hkDisplayConvex.h>

namespace
{
	// Planes in this color show the results of the start point collector
	// These planes are filtered by the character proxy
	const hkColor::Argb HK_DEBUG_STARTPOINT_HIT_POSITION_COLOR = hkColor::WHITE;

	// Planes in this color show the results of the cast collector
	// These planes are filtered by the character proxy
	const hkColor::Argb HK_DEBUG_CASTPOINT_HIT_POSITION_COLOR = hkColor::WHITE;

	// Planes in this color show the +ve distance returned by the start point collector
	// These planes are filtered by the character proxy
	const hkColor::Argb HK_DEBUG_NONPENETRATING_STARTPOINT_DIST_COLOR = hkColor::BLUE;

	// Planes in this color show the -ve distance returned by the start point collector
	// These planes are filtered by the character proxy
	const hkColor::Argb HK_DEBUG_PENETRATING_STARTPOINT_DIST_COLOR = hkColor::RED;

	// Planes in this color show the distance returned by the cast collector
	// These planes are filtered by the character proxy
	const hkColor::Argb HK_DEBUG_CASTPOINT_DIST_COLOR = hkColor::MAGENTA;

	static void debugCast(const hknpAllHitsCollector& startCollector, const hknpAllHitsCollector& castCollector)
	{
		{
			for (int h=0; h < startCollector.getNumHits(); ++h)
			{
				const hknpCollisionResult& hit = startCollector.getHits()[h];
				hkVector4 plane = hit.m_normal;
				plane.zeroComponent<3>();
				hkVector4 pos = hit.m_position;
				HK_DISPLAY_PLANE(plane, pos, 0.5f, HK_DEBUG_STARTPOINT_HIT_POSITION_COLOR);
				pos.addMul(hkSimdReal::fromFloat(hit.m_fraction), hit.m_normal);

				if (hit.m_fraction < 0.0f)
				{
					HK_DISPLAY_PLANE(plane, pos, 0.5f, HK_DEBUG_PENETRATING_STARTPOINT_DIST_COLOR);
				}
				else
				{
					HK_DISPLAY_PLANE(plane, pos, 0.5f, HK_DEBUG_NONPENETRATING_STARTPOINT_DIST_COLOR);
				}
			}
		}

		// Add castCollector plane
		{
			for (int h=0; h < castCollector.getNumHits(); ++h)
			{
				const hknpCollisionResult& hit = castCollector.getHits()[h];
				hkVector4 plane = hit.m_normal;
				plane.zeroComponent<3>();
				hkVector4 pos = hit.m_position;
				HK_DISPLAY_PLANE(plane, pos, 0.5f, HK_DEBUG_CASTPOINT_HIT_POSITION_COLOR);
				pos.addMul(hkSimdReal::fromFloat(hit.m_fraction), hit.m_normal);
				HK_DISPLAY_PLANE(plane, pos, 0.5f, HK_DEBUG_CASTPOINT_DIST_COLOR);
			}
		}
	}
}

#endif	// defined(HKNP_DEBUG_CHARACTER_PROXY)


hknpCharacterProxy::hknpCharacterProxy(const hknpCharacterProxyCinfo& info) : m_shape(HK_NULL), m_bodyId(hknpBodyId::InvalidValue)
{
	HK_ASSERT2(0x29d40ec9, !(info.m_world == HK_NULL), "No world defined");
	HK_ASSERT2(0x29d51ec9, !(info.m_shape == HK_NULL), "No shape for character defined");
	HK_ASSERT2(0x29d52ec9, info.m_up.isOk<3>(), "Up direction incorrectly defined");
	HK_ASSERT2(0x29d52ec9, (0 < info.m_maxSlope) && (info.m_maxSlope <= 0.5f * HK_REAL_PI) , "maxSlope must be between 0 and pi/2 radians");

	m_world = info.m_world; m_world->addReference();
	m_collisionFilterInfo = info.m_collisionFilterInfo;
	m_transform.setRotation(info.m_orientation);
	m_transform.setTranslation(info.m_position);
	m_velocity = info.m_velocity;
	m_dynamicFriction = info.m_dynamicFriction;
	m_staticFriction = info.m_staticFriction;
	m_keepContactTolerance = info.m_keepContactTolerance;
	m_keepDistance = info.m_keepDistance;
	m_contactAngleSensitivity = info.m_contactAngleSensitivity;
	m_userPlanes = info.m_userPlanes;
	m_maxCharacterSpeedForSolver = info.m_maxCharacterSpeedForSolver;
	m_characterStrength = info.m_characterStrength;
	m_characterMass = info.m_characterMass;
	m_maxSlopeCosine = hkMath::cos(info.m_maxSlope);
	m_penetrationRecoverySpeed = info.m_penetrationRecoverySpeed;
	m_maxCastIterations = info.m_maxCastIterations;
	m_refreshManifoldInCheckSupport = info.m_refreshManifoldInCheckSupport;
	m_up = info.m_up;
	m_up.normalize<3>();
	m_lastDisplacement.setZero();

	setShape(info.m_shape);
	if(info.m_presenceInWorld)
	{
		addPhantom();
	}

	HK_SUBSCRIBE_TO_SIGNAL(m_world->m_signals.m_postSolve, this, hknpCharacterProxy);
}


hknpCharacterProxy::~hknpCharacterProxy()
{
	m_world->m_signals.m_postSolve.unsubscribeAll(this);

	for(int i = 0; i < m_listeners.getSize(); ++i)
	{
		m_listeners[i]->removeReference();
	}

	removePhantom();
	m_shape->removeReference();
	m_world->removeReference();
}


void hknpCharacterProxy::checkSupport(hkVector4Parameter direction, hknpCharacterSurfaceInfo& ground)
{
	hknpAllHitsCollector startCollector;
	checkSupportWithCollector(direction, ground, startCollector);
}


void hknpCharacterProxy::checkSupportWithCollector(hkVector4Parameter direction, hknpCharacterSurfaceInfo& ground,
	hknpAllHitsCollector& startCollector)
{
	HK_ASSERT2(0x79d57ec9, direction.isNormalized<3>(1e-5f), "checkSupport Direction should be normalized");

	HK_TIMER_BEGIN("checkSupport", HK_NULL);

	// If enabled, refresh the manifold with the current contacts/closest points
	if (m_refreshManifoldInCheckSupport)
	{
		refreshManifold(startCollector);
	}

	// If the manifold is empty we are definitely unsupported
	if (m_manifold.isEmpty())
	{
		ground.m_supportedState = hknpCharacterSurfaceInfo::UNSUPPORTED;
		HK_TIMER_END();
		return;
	}

	// Check for removed bodies.
	validateManifold();

	// Create surface constraints from the manifold contacts. We must allocate enough space for the constrains
	// created from manifold contacts, user planes add possible additional max slope planes.
	hkLocalArray<hkSurfaceConstraintInfo> constraints(m_manifold.getSize() + m_userPlanes + 10);
	createConstraintsFromManifold(0, constraints);

	// Interactions array - this is the output of the simplex solver
	hkLocalArray<hkSurfaceConstraintInteraction> interactions(constraints.getSize() + m_userPlanes);

	// Stored velocities - used to remember contact velocities to give the correct output contact velocity
	hkLocalArray<hkVector4> storedVelocities(constraints.getSize() + m_userPlanes);

	// Run the simplex solver to calculate the effect of each constraint on the character's movement
	hkSimplexSolverInput input;
	hkSimplexSolverOutput output;
	{
		input.m_position.setZero();
		input.m_constraints = constraints.begin();
		input.m_numConstraints = constraints.getSize();
		input.m_velocity = direction;
		input.m_deltaTime = 1.0f / 60.0f;
		input.m_minDeltaTime = 1.0f / 60.0f;
		input.m_upVector = m_up;
		input.m_maxSurfaceVelocity.setAll(m_maxCharacterSpeedForSolver);

		//
		// Allow the user to do whatever they wish with the surface constraints
		//
		fireConstraintsProcessed(m_manifold, input);

		// Set the sizes of the arrays to be correct
		storedVelocities.setSize(input.m_numConstraints);
		interactions.setSize(input.m_numConstraints);
		constraints.setSize(input.m_numConstraints);

		// Remove velocities and friction to make this a query of the static geometry
		for (int i = 0; i < input.m_numConstraints; ++i )
		{
			storedVelocities[i] = constraints[i].m_velocity;
			constraints[i].m_velocity.setZero();
		}

		output.m_planeInteractions = interactions.begin();

		hkSimplexSolverSolve(input, output);
	}

	// Initialize ground surface
	ground.m_surfaceVelocity.setZero();
	ground.m_surfaceNormal.setZero();
	ground.m_surfaceDistanceExcess = 0;		// not required by the proxy character controller

	// If the constraints did not affect the character movement then it is unsupported and we can finish
	hkSimdReal epsilon; epsilon.setFromFloat(0.001f);
	if (output.m_velocity.allEqual<3>(direction, epsilon))
	{
		ground.m_supportedState = hknpCharacterSurfaceInfo::UNSUPPORTED;
		HK_TIMER_END();
		return;
	}

	// Check how was the input velocity modified to determine if the character is supported or sliding
	if (output.m_velocity.lengthSquared<3>().isLess(epsilon))
	{
		ground.m_supportedState = hknpCharacterSurfaceInfo::SUPPORTED;
	}
	else
	{
		output.m_velocity.normalize<3>();
		const hkReal angleSin = output.m_velocity.dot<3>(direction).getReal();
		const hkReal cosSqr = 1 - angleSin * angleSin;
		if (cosSqr < m_maxSlopeCosine * m_maxSlopeCosine)
		{
			ground.m_supportedState = hknpCharacterSurfaceInfo::SLIDING;
		}
		else
		{
			ground.m_supportedState = hknpCharacterSurfaceInfo::SUPPORTED;
		}
	}

	// Add all supporting constraints to the ground information
	int numTouching = 0;
	for (int i = 0; i < input.m_numConstraints; i++)
	{
		// Check if we touched this plane and it supports the movement direction
		if (interactions[i].m_touched && (constraints[i].m_plane.dot<3>(direction).getReal() < -0.08f))
		{
			ground.m_surfaceNormal.add(constraints[i].m_plane);
			ground.m_surfaceVelocity.add(storedVelocities[i]);
			numTouching++;
		}
	}

	// Average ground information collected from the supporting constraints
	if (numTouching > 0)
	{
		ground.m_surfaceNormal.normalize<3>();
		ground.m_surfaceVelocity.mul(hkSimdReal::fromFloat(1 / (hkReal)numTouching));
	}
	else
	{
		ground.m_supportedState = hknpCharacterSurfaceInfo::UNSUPPORTED;
	}

	HK_TIMER_END();
}


void hknpCharacterProxy::integrate(const hkStepInfo& stepInfo, hkVector4Parameter worldGravity)
{
	hknpAllHitsCollector castCollector;
	hknpAllHitsCollector startCollector;
	integrateWithCollectors(stepInfo, worldGravity, castCollector, startCollector);
}

void hknpCharacterProxy::updateTriggersSeen(const hkLocalArray<TriggerVolumeHit>& triggerHits, hkReal maxFraction,
	hkLocalArray<TriggerVolume>& triggersSeen)
{
	// Mark all seen triggers as non-overlapping
	for (int i = 0; i < triggersSeen.getSize(); ++i)
	{
		triggersSeen[i].m_isOverlapping = false;
	}

	// If the trigger hit has been seen before, mark it as overlapping, otherwise add it as a new seen trigger
	for (int i = 0; i < triggerHits.getSize(); ++i)
	{
		const TriggerVolumeHit& hit = triggerHits[i];

		if (hit.m_fraction < maxFraction)
		{
			int index = triggersSeen.indexOf(hit.m_trigger);
			if (index == -1)
			{
				triggersSeen.pushBack(hit.m_trigger);
			}
			else
			{
				triggersSeen[index].m_isOverlapping = true;
			}
		}
	}
}

void hknpCharacterProxy::updateOverlappingTriggers(const hkLocalArray<TriggerVolume>& triggersSeen)
{
	// Mark all trigger volumes from the last step as non-overlapping
	for (int i = 0; i < m_overlappingTriggers.getSize(); ++i)
	{
		m_overlappingTriggers[i].m_isOverlapping = false;
	}

	// Go over all triggers seen in this step
	for (int i = 0; i < triggersSeen.getSize(); ++i)
	{
		const TriggerVolume& trigger = triggersSeen[i];
		int index = m_overlappingTriggers.indexOf(trigger);

		// Not overlapping in the previous step
		if (index == -1)
		{
			if (trigger.m_isOverlapping)
			{
				fireTriggerVolumeInteraction(trigger.m_bodyId, trigger.m_shapeKey, hknpTriggerVolumeEvent::STATUS_ENTERED);
				m_overlappingTriggers.pushBack(trigger);
			}
			else
			{
				fireTriggerVolumeInteraction(trigger.m_bodyId, trigger.m_shapeKey, hknpTriggerVolumeEvent::STATUS_ENTERED);
				fireTriggerVolumeInteraction(trigger.m_bodyId, trigger.m_shapeKey, hknpTriggerVolumeEvent::STATUS_EXITED);
			}
		}

		// Already overlapping in the previous step
		else
		{
			if (trigger.m_isOverlapping)
			{
				m_overlappingTriggers[index].m_isOverlapping = true;
			}
			else
			{
				m_overlappingTriggers.removeAt(index);
				fireTriggerVolumeInteraction(trigger.m_bodyId, trigger.m_shapeKey, hknpTriggerVolumeEvent::STATUS_EXITED);
			}
		}
	}

	// Remove all remaining non-overlapping triggers
	for (int i = m_overlappingTriggers.getSize() - 1; i >= 0 ; --i)
	{
		const TriggerVolume& trigger = m_overlappingTriggers[i];
		if (!trigger.m_isOverlapping)
		{
			fireTriggerVolumeInteraction(trigger.m_bodyId, trigger.m_shapeKey, hknpTriggerVolumeEvent::STATUS_EXITED);
			m_overlappingTriggers.removeAt(i);
		}
	}
}

void hknpCharacterProxy::integrateWithCollectors(const hkStepInfo& stepInfo, hkVector4Parameter gravity,
												 hknpAllHitsCollector& castCollector,
												 hknpAllHitsCollector& startCollector)
{
	HK_TIMER_BEGIN_LIST("updateCharacter", "Cast");

	hkLocalArray<TriggerVolume> triggersSeen(8);
	hkLocalArray<TriggerVolumeHit> triggerHits(8);

	// Try to move the character with the given input velocity iterating each time we hit a new contact
	hkReal remainingTime = stepInfo.m_deltaTime;
	hkVector4 newVelocity;	newVelocity.setZero();

	// Valid start direction
	m_lastDisplacement.setMul(m_velocity, hkSimdReal::fromFloat(remainingTime));

	for (int iter = 0; (remainingTime > HK_REAL_EPSILON) && (iter < m_maxCastIterations) ; iter++)
	{
		HK_TIMER_SPLIT_LIST("InitialCast");

		// Linear cast with the last displacement
		castCollector.reset();
		startCollector.reset();
		triggerHits.clear();
		worldLinearCast(m_lastDisplacement, castCollector, &startCollector, &triggerHits);

		HK_TIMER_SPLIT_LIST("UpdateManifold");

		// Update the manifold with the linear cast results
		if (castCollector.hasHit())
		{
			castCollector.sortHits();
		}
		updateManifold(startCollector, castCollector);

#if defined(HKNP_DEBUG_CHARACTER_PROXY)
		debugCast(startCollector, castCollector);
#endif

		// Create surface constraints from the manifold contacts. We must allocate enough space for the constrains
		// created from manifold contacts, user planes add possible additional max slope planes.
		hkLocalArray<hkSurfaceConstraintInfo> constraints(m_manifold.getSize() + m_userPlanes + 10);
		createConstraintsFromManifold(stepInfo.m_deltaTime - remainingTime, constraints);

		// Update seen triggers with all trigger hits closer than the first castCollector hit.
		hkReal maxFraction = (castCollector.hasHit() ? castCollector.getHits()[0].m_fraction.val() : HK_REAL_MAX);
		updateTriggersSeen(triggerHits, maxFraction, triggersSeen);

		HK_TIMER_SPLIT_LIST("SlexMove");

		// Use the simplex solver to calculate the allowed displacement in the target direction
		hkReal solverDeltaTime;
		hkVector4 newDisplacement;
		{
			// Set up solver input
			hkSimplexSolverInput input;
			input.m_constraints = constraints.begin();
			input.m_numConstraints = constraints.getSize();
			input.m_velocity = m_velocity;
			input.m_deltaTime = remainingTime;
			if (m_velocity.lengthSquared<3>().isEqualZero())
			{
				input.m_minDeltaTime = 0.0f;
			}
			else
			{
				input.m_minDeltaTime = 0.5f * m_keepDistance * m_velocity.lengthInverse<3>().getReal();
			}
			input.m_position.setZero();
			input.m_upVector = m_up;
			input.m_maxSurfaceVelocity.setXYZ(m_maxCharacterSpeedForSolver);

			// Allow the user to do whatever they wish with the surface constraints
			fireConstraintsProcessed(m_manifold, input);

			// Run the solver
			hkSimplexSolverOutput output;
			hkLocalArray<hkSurfaceConstraintInteraction> interactions(input.m_numConstraints);
			interactions.setSizeUnchecked(input.m_numConstraints);
			output.m_planeInteractions = interactions.begin();
			hkSimplexSolverSolve(input, output);

			// Read results
			solverDeltaTime = output.m_deltaTime;
			newDisplacement = output.m_position;
			newVelocity = output.m_velocity;
		}

		HK_TIMER_SPLIT_LIST("ApplySurf");

		// Apply forces - e.g. Character hits dynamic objects
		resolveContacts(stepInfo, gravity);

		HK_TIMER_SPLIT_LIST("CastMove");

		// Check whether we can walk to the new position the simplex has suggested
		{
			const hknpCollisionResult* newContact = HK_NULL;

			// If the simplex has given an output direction different from the cast guess
			// we re-cast to check we can move there. There is no need to get the start points again.
			if (newDisplacement.lengthSquared<3>().isGreater(hkSimdReal_EpsSqrd) && !m_lastDisplacement.allEqual<3>(newDisplacement, hkSimdReal::fromFloat(0.001f)))
			{
				castCollector.reset();
				triggerHits.clear();
				worldLinearCast(newDisplacement, castCollector, HK_NULL, &triggerHits);

				const int numHits = castCollector.getNumHits();
				if (numHits > 0)
				{
					// Find the first contact that isn't already in the manifold
					castCollector.sortHits();
					const hknpCollisionResult* contacts = castCollector.getHits();
					int contactIdx = 0;
					for (; (contactIdx < numHits) && (findContact(contacts[contactIdx], m_manifold, 0.1f) != -1); ++contactIdx) {}

					// Add contact to the manifold
					if (contactIdx < numHits)
					{
						fireContactAdded(contacts[contactIdx]);
						m_manifold.pushBack(contacts[contactIdx]);
						newContact = contacts + contactIdx;

						// Update triggers seen with all trigger hits closer than the new contact.
						updateTriggersSeen(triggerHits, newContact->m_fraction, triggersSeen);
					}
				}
			}

			// Move character along the newDisplacement direction until it is at m_keepDistance of the new contact
			if ( newContact != HK_NULL )
			{
				// Calculate fraction along the newDisplacement we will move the character to
				const hkSimdReal displacementLengthInv = newDisplacement.lengthInverse<3>();
				HK_ASSERT2(0x5ea11036, hkSimdReal_1.isGreater(displacementLengthInv * hkSimdReal_Eps), "LinearCast of zero length picked up new contact points");
				hkSimdReal angleBetweenMovementAndSurface;
				angleBetweenMovementAndSurface.setMul(newDisplacement.dot<3>(newContact->m_normal), displacementLengthInv);
				hkSimdReal keepDistanceAlongMovement;
				keepDistanceAlongMovement.setDiv(hkSimdReal::fromFloat(m_keepDistance), -angleBetweenMovementAndSurface);
				hkSimdReal distance = hkSimdReal::fromFloat(newContact->m_fraction);
				hkSimdReal fraction; fraction.setSubMul(distance, keepDistanceAlongMovement, displacementLengthInv);
				fraction.setClamped(fraction, hkSimdReal_0, hkSimdReal_1);

				// Update character position
				hkVector4 displacement; displacement.setMul(newDisplacement, fraction);
				hkVector4& position = m_transform.getTranslation();
				position.add(displacement);

				// Update remaining time with the time traveled in the displacement
				remainingTime -= solverDeltaTime * fraction.getReal();
			}

			// Move character all the way until the end of the allowed displacement returned by the solver
			else
			{
				hkVector4& position = m_transform.getTranslation();
				position.add(newDisplacement);
				remainingTime -= solverDeltaTime;
			}

			// Remember last direction for next iteration
			m_lastDisplacement = newDisplacement;
		}
	}

	// Update with the last velocity calculated by the solver
	m_velocity = newVelocity;

	updatePhantom();

	// Update overlapping trigger volumes
	updateOverlappingTriggers(triggersSeen);

	HK_TIMER_END_LIST();
}


void hknpCharacterProxy::resolveContacts(const hkStepInfo& stepInfo, hkVector4Parameter worldGravity)
{
	const hkReal recoveryTau = 0.4f;
	const hkReal dampFactor = 0.9f;

	// Process all current contacts
	for (int i = 0; i < m_manifold.getSize(); i++)
	{
		// Obtain contacted body
		const hknpCollisionResult& contact = m_manifold[i];
		if( !contact.m_hitBodyInfo.m_bodyId.isValid() )
		{
			continue;
		}

		const hknpBody& body = m_world->getBody( contact.m_hitBodyInfo.m_bodyId );

		// If the contact is with another character proxy fire the character interaction callback and skip impulse application.
		const hknpCharacterProxyProperty* prop = m_world->getBodyProperty<hknpCharacterProxyProperty>(
			body.m_id, hknpCharacterProxyProperty::Key );
		if( prop )
		{
			hknpCharacterProxy* otherCharacter = prop->m_proxyController;
			hkContactPoint contactPoint;
			contactPoint.setPosition(contact.m_position);
			contactPoint.setSeparatingNormal(contact.m_normal);
			fireCharacterInteraction(otherCharacter, contactPoint);

			continue;
		}

		// Skip fixed or keyframed bodies as we won't apply impulses to them
		if (body.isStaticOrKeyframed())
		{
			continue;
		}

		const hknpMotion& motion = m_world->getMotion(body.m_motionId);

		// Calculate and apply impulse on contacted body
		{
			hknpBodyId bodyId = contact.m_hitBodyInfo.m_bodyId;

			hknpCharacterObjectInteractionEvent input;
			input.m_position = contact.m_position;
			input.m_normal  = contact.m_normal;
			input.m_bodyId = bodyId;
			input.m_timestep = stepInfo.m_deltaTime;

			// Calculate required velocity change
			hkReal deltaVelocity;
			{
				// Calculate relative normal velocity of the contact point in the contacted body
				hkVector4 pointRelVel;
				motion.getPointVelocity(contact.m_position, pointRelVel);
				pointRelVel.sub(m_velocity);
				input.m_projectedVelocity = pointRelVel.dot<3>(input.m_normal).getReal();

				// Change velocity
				deltaVelocity = - input.m_projectedVelocity * dampFactor;

				// Apply an extra impulse if the collision is actually penetrating. HVK-1903
				const hkReal distance = input.m_normal(3);
				if (distance < 0)
				{
					deltaVelocity += distance * stepInfo.m_invDeltaTime * recoveryTau;
				}
			}

			// Initialize the output result
			hknpCharacterObjectInteractionResult output;
			output.m_impulsePosition = input.m_position;
			output.m_objectImpulse.setZero();

			// Apply impulse if required to keep bodies apart
			if (deltaVelocity < 0.0f)
			{

				//	Calculate the impulse magnitude
				{
					hkMatrix3 invInertia; motion.getInverseInertiaWorld(invInertia);
					hkVector4 r; r.setSub(input.m_position, motion.getCenterOfMassInWorld());
					hkVector4 jacAng; jacAng.setCross(r, input.m_normal);
					hkVector4 rc; rc._setRotatedDir(invInertia, jacAng);

					input.m_objectMassInv = rc.dot<3>(jacAng).getReal();
					input.m_objectMassInv += motion.getInverseMass().getReal();
					input.m_objectImpulse = deltaVelocity / input.m_objectMassInv;
				}

				// Clamp impulse magnitude if required and apply it to the normal direction
				hkReal maxPushImpulse = -m_characterStrength * stepInfo.m_deltaTime;
				if (input.m_objectImpulse < maxPushImpulse)
				{
					input.m_objectImpulse = maxPushImpulse;
				}
				output.m_objectImpulse.setMul(hkSimdReal::fromFloat(input.m_objectImpulse), input.m_normal);
			}

			// No impulse required
			else
			{
				input.m_objectImpulse = 0.0f;
				input.m_objectMassInv = motion.getInverseMass().getReal();
			}

			// Add gravity
			{
				// Calculate effect of gravity on the velocity of the character in the contact normal direction
				hkSimdReal relVelN;
				{
					hkVector4 charVelDown; charVelDown.setMul(hkSimdReal::fromFloat(stepInfo.m_deltaTime), worldGravity);
					relVelN = charVelDown.dot<3>(contact.m_normal);
				}

				// If it is a separating contact subtract the separation velocity
				{
					const hkSimdReal projectedVelocity = hkSimdReal::fromFloat(input.m_projectedVelocity);
					const hkVector4Comparison isSeparatingContact = projectedVelocity.lessZero();
					const hkSimdReal newRelVelN = relVelN - projectedVelocity;
					relVelN.setSelect(isSeparatingContact, newRelVelN, relVelN);
				}

				// If the resulting velocity is negative an impulse is applied to stop the character from falling into
				// the contacted body
				{
					hkVector4 newObjectImpulse = output.m_objectImpulse;
					newObjectImpulse.addMul(relVelN * hkSimdReal::fromFloat(m_characterMass), contact.m_normal);
					output.m_objectImpulse.setSelect(relVelN.less(-hkSimdReal_Eps), newObjectImpulse, output.m_objectImpulse);
				}
			}

			// Fire callback to allow user to change impulse + use the info / play sounds
			fireObjectInteraction(input, output);

			// Apply impulse
			m_world->applyBodyImpulseAt(bodyId, output.m_objectImpulse, output.m_impulsePosition);
		}
	}
}

void hknpCharacterProxy::setShape(const hknpShape* shape)
{
	HK_ASSERT2(0x29d51ec9, shape != HK_NULL, "No shape for character defined");

	shape->addReference();

	// Do we have a phantom? Then update its shape
	if(m_bodyId.isValid())
	{
		m_world->setBodyShape(m_bodyId, shape);
	}
	// Clean up old shape
	if (m_shape)
	{
		m_shape->removeReference();
	}
	m_shape = shape;
	m_shape->calcAabb(hkTransform::getIdentity(), m_aabb);
	for(int i = 0; i < m_manifold.getSize(); ++i)
	{
		fireContactRemoved(m_manifold[i]);
	}
	m_manifold.clear();

	fireShapeChanged(shape);
}

void hknpCharacterProxy::addPhantom()
{
	// Use a special never-colliding (but still queryable) body

	hknpBodyCinfo binfo;
	{
		binfo.m_flags.orWith( hknpBody::DONT_COLLIDE );
		binfo.m_collisionFilterInfo = m_collisionFilterInfo;
		binfo.m_shape = m_shape;
		binfo.m_position = m_transform.getTranslation();
		binfo.m_orientation.set( m_transform.getRotation() );
	}

	hknpMotionCinfo minfo;
	{
		minfo.initializeAsKeyFramed( &binfo, 1 );
	}

	m_bodyId = m_world->createDynamicBody( binfo, minfo );

	// Add a property to identify this body as a character proxy
	{
		hknpCharacterProxyProperty prop;
		prop.m_proxyController = this;
		m_world->setBodyProperty( m_bodyId, hknpCharacterProxyProperty::Key, prop );
	}
}


void hknpCharacterProxy::removePhantom()
{
	// If no phantom is present just return
	if(!m_bodyId.isValid())
	{
		return;
	}

	m_world->destroyBodies( &m_bodyId, 1 );
	m_bodyId = hknpBodyId::invalid();
}


void hknpCharacterProxy::updatePhantom()
{
	// If no phantom is present just return
	if(!m_bodyId.isValid())
	{
		return;
	}

	m_world->setBodyTransform( m_bodyId, m_transform );
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
