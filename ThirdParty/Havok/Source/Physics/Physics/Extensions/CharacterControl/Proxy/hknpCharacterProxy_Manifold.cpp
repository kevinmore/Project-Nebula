/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxy.h>
#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Internal/SimplexSolver/hkSimplexSolver.h>

// Functions to leave trace of relevant simulation data
#if 0//defined(HKNP_DEBUG_CHARACTER_PROXY)
#define PROXY_TRACE(STRING)		HK_TRACE(STRING)
HK_FORCE_INLINE static hkStringBuf toString(hkVector4Parameter vector)
{
	hkStringBuf text;
	text.printf("(%.2f, %.2f, %.2f)", vector(0), vector(1), vector(2));
	return text;
}

HK_FORCE_INLINE static hkStringBuf toString(const hknpCollisionResult* hit)
{
	hkStringBuf text;
	text.printf("pos %s n %s distance %.2f fraction %.2f bodyId %d",
		toString(hit->m_position).cString(), toString(hit->m_normal).cString(), hit->m_normal(3), hit->m_fraction,
		hit->m_hitBodyInfo.m_bodyId);
	return text;
}

static hkStringBuf toString(const hknpAllHitsCollector& collector, const char* name)
{
	hkStringBuf text;
	const hknpCollisionResult* hit = collector.getHits();
	const int numHits = collector.getNumHits();
	for (int i = 0; i < numHits; ++i)
	{
		char* newLine = (i == numHits - 1) ? "" : "\n";
		text.appendPrintf("%s %s%s", name, toString(hit++).cString(), newLine);
	}
	return text;
}
#else
#define PROXY_TRACE(STRING)
#endif


void hknpCharacterProxy::refreshManifold(hknpAllHitsCollector& startCollector)
{
	// Obtain current closest points
	
	hknpAllHitsCollector castCollector;
	startCollector.reset();
	worldLinearCast(hkVector4::getZero(), castCollector, &startCollector);

	// Update the manifold with the query results
	updateManifold(startCollector, castCollector);
}


void hknpCharacterProxy::validateManifold()
{
	for( int i = m_manifold.getSize()-1; i >= 0; --i )
	{
		hknpBodyId bodyId = m_manifold[i].m_hitBodyInfo.m_bodyId;
		if( !bodyId.isValid() || !m_world->isBodyValid(bodyId) || !m_world->isBodyAdded(bodyId) )
		{
			fireContactRemoved(m_manifold[i]);
			m_manifold.removeAt(i);
		}
	}
}

void hknpCharacterProxy::updateManifold(const hknpAllHitsCollector& startCollector,
										const hknpAllHitsCollector& castCollector)
{
	PROXY_TRACE("updateManifold()");
	PROXY_TRACE(toString(startCollector, "START").cString());
	PROXY_TRACE(toString(castCollector, "CAST").cString());

	// Copy the contacts in the start collector and find the closest one
	hkSimdReal minDistance = hkSimdReal::getConstant<HK_QUADREAL_MAX>();
	hkLocalArray<hknpCollisionResult> startContacts(startCollector.getNumHits());
	if (startCollector.getNumHits() > 0)
	{
		const hknpCollisionResult* results = startCollector.getHits();
		for (int i = 0; i < startCollector.getNumHits(); ++i, ++results)
		{
			// Should we only consider fixed or keyframed rigid bodies?
			minDistance.setMin(results->m_normal.getComponent<3>(), minDistance);
			startContacts.pushBack(*results);
		}
	}

	// Iterate over all contacts in the latest manifold trying to match them to any of the current ones
	// Validation of the entries of the current manifold is done implicitly.
	for (int i = m_manifold.getSize() - 1; i >= 0; --i)
	{
		hknpCollisionResult& manifoldContact = m_manifold[i];

		// Look for the manifold contact in the current ones
		const int bestMatch = findContact(manifoldContact, startContacts, 1.1f);

		// If it has been found update the manifold and remove the contact from the start list
		if (bestMatch >= 0)
		{
			const hknpCollisionResult& current = startContacts[bestMatch];
			if( current.m_hitBodyInfo.m_bodyId != manifoldContact.m_hitBodyInfo.m_bodyId ||
				current.m_hitBodyInfo.m_shapeKey != manifoldContact.m_hitBodyInfo.m_shapeKey )
			{
				fireContactRemoved(manifoldContact);
				fireContactAdded(current);
			}
			manifoldContact = current;
			startContacts.removeAt(bestMatch);
		}
		// Else remove the point from the manifold
		else
		{
			fireContactRemoved(manifoldContact);
			m_manifold.removeAt(i);
		}
	}

	// Add the closest current contact if it is not already in the manifold. This is safe as it can never be
	// an unwanted edge.
	{
		int pointIdx = 0;

		// Look for a contact that matches the minimum distance
		for (; pointIdx < startContacts.getSize() && minDistance.isNotEqual(startContacts[pointIdx].m_normal.getComponent<3>()); ++pointIdx);

		// Process the contact
		if (pointIdx < startContacts.getSize())
		{
			const hknpCollisionResult& closestContact = startContacts[pointIdx];

			// Make sure that it is not very similar to any point already in the manifold
			const int bestMatch = findContact(closestContact, m_manifold, 0.1f);

			// If we got a good match update the manifold point with the closest point
			if (bestMatch >= 0)
			{
				hknpCollisionResult& current = m_manifold[bestMatch];
				if( closestContact.m_hitBodyInfo.m_bodyId != current.m_hitBodyInfo.m_bodyId ||
					closestContact.m_hitBodyInfo.m_shapeKey != current.m_hitBodyInfo.m_shapeKey )
				{
					fireContactRemoved(current);
					fireContactAdded(closestContact);
				}
				current = closestContact;
			}
			// Else add the closest contact to the manifold
			else
			{
				fireContactAdded(closestContact);
				m_manifold.pushBack(closestContact);
			}
		}
	}

	// Process linear cast results if any
	if (castCollector.hasHit())
	{
		// Add the closest contact to the manifold if it is not already there
		const hknpCollisionResult& contact = castCollector.getHits()[0];
		const int bestMatch = findContact(contact, m_manifold, 0.1f);
		if (bestMatch == -1)
		{
			fireContactAdded(contact);
			m_manifold.pushBack(contact);
		}

		// NOTE: We used to update the manifold with the new point from the cast collector here, but in fact this
		// is unnecessary and sometimes wrong. All points in the manifold have been correctly updated at this stage
		// by the start point collector so they do not need to be replaced here. If the points are penetrating, then
		// the cast collector will have a distance of 0, which is incorrect, and the negative distance picked up by
		// the start collector is the one that we want.
	}

	// Remove from the manifold contacts that are too similar as the simplex does not handle parallel planes
	for (int c1 = m_manifold.getSize() - 1; c1 > 0; --c1)
	{
		int c2 = c1 - 1;
		for (; c2 >= 0; --c2)
		{
			// If c1 and c2 are the same then we should remove c1
			hkReal fitness = compareContacts(m_manifold[c1], m_manifold[c2]);
			if (fitness < 0.1f)
			{
				break;
			}
		}
		if (c2 >= 0)
		{
			fireContactRemoved( m_manifold[c1] );
			m_manifold.removeAt(c1);
		}
	}
}


void hknpCharacterProxy::createConstraintsFromManifold(hkReal timeTravelled, hkArray<hkSurfaceConstraintInfo>& constraints) const
{
	// Create constraints from the manifold contacts
	const int size = m_manifold.getSize();
	constraints.setSizeUnchecked( size );
	for (int i = 0; i < size; ++i)
	{
		createSurfaceConstraint(m_manifold[i], timeTravelled, constraints[i]);
		addMaxSlopePlane(m_maxSlopeCosine, m_up, i, constraints);
	}

	// Resize array if it is now too small to accommodate the user planes
	if ( (constraints.getCapacity() - constraints.getSize()) < m_userPlanes )
	{
		constraints.reserve(constraints.getSize() + m_userPlanes);
	}
}


void hknpCharacterProxy::createSurfaceConstraint(const hknpCollisionResult& contact, hkReal timeTravelled,
												 hkSurfaceConstraintInfo& constraint) const
{

	// Subtract m_keepDistance from the plane distance as we want the constraint to enforce that separation
	{
		hkSimdReal distance; distance.setSub(contact.m_normal.getComponent<3>(), hkSimdReal::fromFloat(m_keepDistance));
		constraint.m_plane = contact.m_normal;
		constraint.m_plane.setComponent<3>(distance);
	}

	// Friction parameters
	constraint.m_staticFriction = m_staticFriction;
	constraint.m_dynamicFriction = m_dynamicFriction;
	constraint.m_extraUpStaticFriction = 0.f;
	constraint.m_extraDownStaticFriction = 0.f;

	// Assume the velocity of this constraint is 0
	constraint.m_velocity.setZero();

	// Assume this is a low priority constraint
	constraint.m_priority = 0;

	// Set constraint parameters that depend on the body motion and type
	if( contact.m_hitBodyInfo.m_bodyId.isValid() )
	{
		const hknpBody& body = m_world->getBody(contact.m_hitBodyInfo.m_bodyId);
		const hknpMotion& motion = m_world->getMotion(body.m_motionId);

		// HVK-1871. This code gets the point velocity at the collision, based on how far
		// the object actually traveled, rather than the velocity result of the constraint solver.
		// (i.e. the value got from getPointVelocity)
		// When a heavy force pushes a rigid body into a fixed rigid body these values can diverge,
		// which can cause the character controller to penetrate the moving rigid body, as it sees
		// an incorrectly moving plane.
		//
		// Note, this means that velocities will be one frame behind, so for accelerating platforms
		// (HVK-1477) (i.e. For keyframed or fixed objects) we just take the velocity, to make sure the
		// character does not sink in.


		// If the body is static or keyframed use the current contact velocity
		if (body.isStaticOrKeyframed())
		{
			motion.getPointVelocity(contact.m_position, constraint.m_velocity);
		}

		// For dynamic bodies use the velocity of the previous frame
		else
		{
			hkVector4 relPos; relPos.setSub(contact.m_position, motion.getCenterOfMassInWorld());
			constraint.m_velocity.setCross(motion.m_previousStepAngularVelocity, relPos);
			constraint.m_velocity.add(motion.m_previousStepLinearVelocity);
		}

		// Move the plane back based on the velocity and the timeTravelled HVK-1477
		{
			hkSimdReal shift; shift.setMul(constraint.m_velocity.dot<3>(constraint.m_plane), hkSimdReal::fromFloat(timeTravelled));
			shift.setSub(constraint.m_plane.getComponent<3>(), shift);
			constraint.m_plane.setComponent<3>(shift);
		}

		// Increase priority of contacts with fixed and keyframed bodies
		if (body.isStatic())
		{
			constraint.m_priority = 2;
		}
		else if (body.isKeyframed())
		{
			constraint.m_priority = 1;
		}
	}

	// If penetrating we add extra velocity to push the character back out
	{
		hkSimdReal distance = constraint.m_plane.getComponent<3>();
		const hkVector4Comparison isPenetrating = distance.less(-hkSimdReal::getConstant<HK_QUADREAL_EPS>());
		hkVector4 newVelocity;
		newVelocity.setSubMul(constraint.m_velocity, contact.m_normal, distance * hkSimdReal::fromFloat(m_penetrationRecoverySpeed));
		constraint.m_velocity.setSelect(isPenetrating, newVelocity, constraint.m_velocity);
		distance.setSelect(isPenetrating, hkSimdReal::getConstant<HK_QUADREAL_0>(), distance);
		constraint.m_plane.setComponent<3>(distance);
	}
}


hkBool hknpCharacterProxy::addMaxSlopePlane(hkReal maxSlopeCos, hkVector4Parameter up, int index,
											hkArray<hkSurfaceConstraintInfo>& constraints)
{
	const hkSimdReal verticalComponent = constraints[index].m_plane.dot<3>(up);
	hkVector4Comparison shouldAddPlane;
	shouldAddPlane.setAnd(verticalComponent.greater(hkSimdReal::fromFloat(0.01f)), verticalComponent.less(hkSimdReal::fromFloat(maxSlopeCos)));
	if (shouldAddPlane.anyIsSet())
	{
		// Add an additional vertical plane at the end of the constraints array
		hkSurfaceConstraintInfo& newConstraint = constraints.expandOne();

		// Copy original constraint and zero vertical normal component
		newConstraint = constraints[index];
		newConstraint.m_plane.subMul(verticalComponent, up);
		newConstraint.m_plane.normalize<3>();
		return true;
	}

	// No plane added
	return false;
}

hkReal hknpCharacterProxy::compareContacts(const hknpCollisionResult& c1, const hknpCollisionResult& c2) const
{
	HK_ASSERT2(0x12345678, c1.m_normal.isNormalized<3>(), "c1.m_normal must be normalized!");
	HK_ASSERT2(0x12345678, c2.m_normal.isNormalized<3>(), "c2.m_normal must be normalized!");

	// Compare contact normals (small angle approx, changed from cross product HVK-2184)
	hkSimdReal angleSquared;
	{
		angleSquared.setSub(hkSimdReal::getConstant<HK_QUADREAL_1>(), c1.m_normal.dot<3>(c2.m_normal));
		angleSquared.mul(hkSimdReal::fromFloat(m_contactAngleSensitivity*m_contactAngleSensitivity));
	}

	// Compare contact distances
	hkSimdReal planeDistanceSqrd;
	{
		planeDistanceSqrd.setSub(c1.m_normal.getComponent<3>(), c2.m_normal.getComponent<3>());
		planeDistanceSqrd.mul(planeDistanceSqrd);
	}

	// Compare point velocities
	hkSimdReal velocityDiffSqrd;
	{
		hkVector4 p1Vel, p2Vel;
		{
			const hknpBody& body1 = m_world->getBody(c1.m_hitBodyInfo.m_bodyId);
			const hknpBody& body2 = m_world->getBody(c2.m_hitBodyInfo.m_bodyId);
			const hknpMotion& motion1 = m_world->getMotion(body1.m_motionId);
			const hknpMotion& motion2 = m_world->getMotion(body2.m_motionId);
			motion1._getPointVelocity(c1.m_position, p1Vel);
			motion2._getPointVelocity(c2.m_position, p2Vel);
		}
		hkVector4 velDiff; velDiff.setSub(p1Vel, p2Vel);
		velocityDiffSqrd = velDiff.lengthSquared<3>();
	}

	const hkReal fitness = angleSquared.getReal() * 10.0f + velocityDiffSqrd.getReal() * 0.1f + planeDistanceSqrd.getReal();

	return fitness;
}


int hknpCharacterProxy::findContact(const hknpCollisionResult& contact, const hkArray<hknpCollisionResult>& contactList, hkReal fitnessThreshold) const
{
	int bestIdx = -1;
	if( contact.m_hitBodyInfo.m_bodyId.isValid() )
	{
		hkReal bestFitness = fitnessThreshold;
		for (int j = 0; j < contactList.getSize(); ++j)
		{
			hkReal fitness = compareContacts(contact, contactList[j]);
			if (fitness < bestFitness)
			{
				bestFitness = fitness;
				bestIdx = j;
			}
		}
	}
	return bestIdx;
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
