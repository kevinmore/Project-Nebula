/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBody.h>

#include <Common/Base/Types/Physics/hkStepInfo.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Internal/SimplexSolver/hkSimplexSolver.h>

#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBodyListener.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>
#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobianUtil.h>


#ifdef DEBUG_HKNP_CHARACTER_RIGIDBODY

#include <Common/Visualize/hkDebugDisplay.h>

namespace
{
	const hkColor::Argb DEBUG_HKNP_CHARACTER_CONTACT_POINT_COLOR = hkColor::WHITE;
	const hkColor::Argb DEBUG_HKNP_CHARACTER_CONTACT_PLANE_COLOR = hkColor::WHITESMOKE;
	const hkColor::Argb DEBUG_HKNP_CHARACTER_SUPPORT_PLANE_COLOR = hkColor::CYAN;
	const hkColor::Argb DEBUG_HKNP_CHARACTER_VERTICAL_PLANE_COLOR = hkColor::RED;
	const hkColor::Argb DEBUG_HKNP_CHARACTER_SURFACE_DYNAMIC_COLOR = hkColor::GREEN;
	const hkColor::Argb DEBUG_HKNP_CHARACTER_SURFACE_NONDYNAMIC_COLOR = hkColor::YELLOWGREEN;
	const hkColor::Argb DEBUG_HKNP_CHARACTER_SURFACE_VELOCITY_COLOR = hkColor::YELLOW;
}

/*static*/ void HK_CALL hknpCharacterRigidBody::debugContactPoint( hkContactPoint& cp, bool isSupportingPoint )
{
	bool isVerticalContact = ( hknpCharacterRigidBody::m_magicNumber == cp.getPosition().getInt24W() );
	hkColor::Argb color = isSupportingPoint ? DEBUG_HKNP_CHARACTER_SUPPORT_PLANE_COLOR : ( isVerticalContact ? DEBUG_HKNP_CHARACTER_VERTICAL_PLANE_COLOR : DEBUG_HKNP_CHARACTER_CONTACT_PLANE_COLOR );
	hkReal size = isSupportingPoint ? 0.6f : 0.5f;
	// Display contact point
	HK_DISPLAY_STAR( cp.getPosition(),0.2f, color );

	// Display contact plane
	hkVector4 plane = cp.getNormal();
	plane.zeroComponent<3>();
	hkVector4 pos = cp.getPosition();
	HK_DISPLAY_PLANE( plane, pos, size, color );
}

/*static*/ void HK_CALL hknpCharacterRigidBody::debugGround( hkVector4Parameter position, const hknpCharacterSurfaceInfo& ground )
{
	// Display surface normal in center
	HK_DISPLAY_ARROW( position, ground.m_surfaceNormal, (ground.m_isSurfaceDynamic ? DEBUG_HKNP_CHARACTER_SURFACE_DYNAMIC_COLOR : DEBUG_HKNP_CHARACTER_SURFACE_NONDYNAMIC_COLOR) );

	// Display surface velocity
	HK_DISPLAY_ARROW( position, ground.m_surfaceVelocity, DEBUG_HKNP_CHARACTER_SURFACE_VELOCITY_COLOR );
}


#endif	// defined(DEBUG_HKNP_CHARACTER_RIGIDBODY)

// If we hit this assert then the overridetype of the m_additionflags should be changed
HK_COMPILE_TIME_ASSERT(sizeof(hknpWorld::AdditionFlags) == sizeof(hkUint8));

hknpCharacterRigidBodyCinfo::hknpCharacterRigidBodyCinfo()
{
	m_mass = 75.0f;
	m_maxForce = 1000.0f;
	m_dynamicFriction = 0.0f;
	m_staticFriction = 0.0f;
	m_weldingTolerance = 0.1f;
	m_maxSlope = HK_REAL_PI / 3.0f;
	m_up.set( 0,1,0 );
	m_maxSpeedForSimplexSolver = 10.0f;
	m_collisionFilterInfo = 0;
	m_position.setZero();
	m_orientation.setIdentity();
	m_supportDistance = 1.f;
	m_hardSupportDistance = 0.0f;
	m_shape = HK_NULL;
	m_world = HK_NULL;
	m_reservedBodyId = hknpBodyId::invalid();
	m_additionFlags = 0;
}

hknpCharacterRigidBody::hknpCharacterRigidBody( const hknpCharacterRigidBodyCinfo& info )
{
	HK_ASSERT2(0x29d50ec9,  !(&info == HK_NULL), "No info defined");
	HK_ASSERT2(0x29d40ec9,  !(info.m_world == HK_NULL), "No world defined");
	HK_ASSERT2(0x29d51ec9,  !(info.m_shape == HK_NULL), "No shape for character defined");
	HK_ASSERT2(0x29d52ec9,  info.m_up.isOk<3>(), "Up direction incorrectly defined");
	HK_ASSERT2(0x29d52ec9,  ( 0 < info.m_maxSlope ) && ( info.m_maxSlope <= 0.5f * HK_REAL_PI ),
		"maxSlope must be between 0 and pi/2 radians" );

	m_world = info.m_world;

	hknpMaterial material;
	{
		material.m_isExclusive = false; // we want to share as much as possible to allow as many characters as possible.
		material.m_dynamicFriction.setReal<true>(info.m_dynamicFriction);
		material.m_staticFriction.setReal<true>(info.m_staticFriction);
		material.m_frictionCombinePolicy = hknpMaterial::COMBINE_MIN;
		material.m_restitution.setReal<true>(0.0f);
		material.m_weldingTolerance.setReal<true>(info.m_weldingTolerance);
		material.m_massChangerCategory = hknpMaterial::MASS_CHANGER_HEAVY;
		material.m_massChangerHeavyObjectFactor.setOne();
	}

	hknpBodyCinfo bodyCinfo;
	{
		bodyCinfo.m_shape = info.m_shape;
		bodyCinfo.m_materialId = m_world->accessMaterialLibrary()->addEntry(material);
		bodyCinfo.m_collisionFilterInfo = info.m_collisionFilterInfo;
		bodyCinfo.m_flags.orWith( hknpBody::RAISE_MANIFOLD_PROCESSED_EVENTS );
		bodyCinfo.m_reservedBodyId = hknpBodyId::invalid(); // create a new one
		bodyCinfo.m_qualityId = hknpBodyQualityId::CHARACTER;
		bodyCinfo.m_position = info.m_position;
		bodyCinfo.m_orientation = info.m_orientation;
		bodyCinfo.m_reservedBodyId = info.m_reservedBodyId;
	}

	hknpMotionCinfo motionCinfo;
	{
		motionCinfo.initializeWithMass( &bodyCinfo, 1, info.m_mass );
		motionCinfo.m_inverseInertiaLocal.setZero();							// no rotation
		motionCinfo.m_motionPropertiesId = hknpMotionPropertiesId::KEYFRAMED;	// no gravity or damping
	}

	// Create the body
	m_bodyId = m_world->createDynamicBody( bodyCinfo, motionCinfo, info.m_additionFlags );

	m_shape = bodyCinfo.m_shape;
	m_shape->addReference();

	// forward all manifold processed events to our handler function.
	m_world->getEventSignal(hknpEventType::MANIFOLD_PROCESSED, m_bodyId).subscribe(
		this, &hknpCharacterRigidBody::onManifoldEvent, "hknpCharacterRigidBody manifold processed" );

	// subscribe to pre-collide signal. We need that to clear the contact point list.
	HK_SUBSCRIBE_TO_SIGNAL(m_world->m_signals.m_preCollide, this, hknpCharacterRigidBody);

	// Values used for check support
	{
		// Set up direction
		m_up = info.m_up;
		HK_ASSERT2(0x79d58ec9, m_up.isNormalized<3>(), "checkSupport down direction should be normalized");

		// Set max slope cosine
		// It is an angle between UP and an object plane.
		m_maxSlopeCosine = hkMath::cos(info.m_maxSlope);

		// Set limit support distance to detect support state
		m_supportDistance = info.m_supportDistance;
		m_hardSupportDistance = info.m_hardSupportDistance;

		// Set additional parameters for simplex solver
		m_maxSpeedForSimplexSolver = info.m_maxSpeedForSimplexSolver;
	}

	m_acceleration.setZero();

	m_maxForce = info.m_maxForce;

	m_filteredContactPoints.reserve(4);
}

hknpCharacterRigidBody::~hknpCharacterRigidBody()
{
	for(int i = 0; i < m_listeners.getSize(); ++i)
	{
		m_listeners[i]->removeReference();
	}

	// unsubscribe all signal handlers
	m_world->m_signals.m_preCollide.unsubscribeAll(this);
	m_world->m_signals.m_postSolve.unsubscribeAll(this);
	m_world->getEventSignal(hknpEventType::MANIFOLD_PROCESSED, m_bodyId).unsubscribeAll(this);

	// destroy the body
	m_world->destroyBodies(&m_bodyId, 1);

	m_shape->removeReference();
}

void hknpCharacterRigidBody::onManifoldEvent( const hknpEventHandlerInput& input, const hknpEvent& event )
{
	const hknpManifoldProcessedEvent& mpEvent = event.asManifoldProcessedEvent();

	if( mpEvent.m_numContactPoints == 0 )
	{
		return;
	}

	// Ignore collisions with trigger volumes
	if( mpEvent.involvesTriggerVolume() )
	{
		return;
	}

	// record filtered contact points
	if( m_listeners.isEmpty() )
	{
		// Very coarse default support of contact points
		const int index = mpEvent.m_manifold.m_distances.getIndexOfMinComponent<3>();
		hkSimdReal distance = mpEvent.m_manifold.m_distances.getComponent(index);
		hkVector4 position = mpEvent.m_manifold.m_positions[index];
		hknpCharacterRigidBody::ContactPointInfo& vpi = m_filteredContactPoints.expandOne();
		vpi.m_contactPoint.setPositionNormalAndDistance( position, mpEvent.m_manifold.m_normal, distance );
		vpi.m_bodyIds[0] = mpEvent.m_bodyIds[0];
		vpi.m_bodyIds[1] = mpEvent.m_bodyIds[1];
		vpi.m_contactType = hknpCharacterRigidBody::HORIZONTAL;
	}
	else
	{
		bool ignored = true;
		for(int i = 0; i < m_listeners.getSize(); ++i)
		{
			hknpCharacterRigidBody::ContactType type = m_listeners[i]->processManifold(this, mpEvent, m_filteredContactPoints);
			if(type != hknpCharacterRigidBody::IGNORED)
			{
				ignored = false;
			}
		}

		// If contact point is ignored by the user it is ignored from the contact solver as well.
		if (ignored && mpEvent.m_manifoldCache != HK_NULL)
		{
			hknpMxContactJacobian* mxJac = mpEvent.m_manifoldCache->m_manifoldSolverInfo.m_contactJacobian;
			int mxId = mpEvent.m_manifoldCache->m_manifoldSolverInfo.m_manifoldIndex;
			hknpContactJacobianUtil::disableContacts( mxJac, mxId );
		}
	}
}

void hknpCharacterRigidBody::addListener( hknpCharacterRigidBodyListener* listener )
{
	HK_ASSERT2(0xa83f81d5, listener != NULL, "The character rigid body listener is not allowed to be NULL.");
	HK_ASSERT2(0xa83f81d6, m_listeners.indexOf(listener) < 0, "You tried to add the same character listener twice.");

	if(m_listeners.isEmpty())
	{
		// Listen for post collide and post solve signals
		HK_SUBSCRIBE_TO_SIGNAL(m_world->m_signals.m_postCollide, this, hknpCharacterRigidBody);
		HK_SUBSCRIBE_TO_SIGNAL(m_world->m_signals.m_postSolve, this, hknpCharacterRigidBody);
	}
	m_listeners.pushBack(listener);
	m_listeners.back()->addReference();
}

void hknpCharacterRigidBody::removeListener( hknpCharacterRigidBodyListener* listener )
{
	HK_ASSERT2(0xa83f81d5, listener != NULL, "The character rigid body listener is not allowed to be NULL.");

	const int i = m_listeners.indexOf(listener);
	HK_ASSERT2(0xa83f81d7, i >= 0, "You tried to remove a character listener that was never added.");
	m_listeners[i]->removeReference();
	m_listeners.removeAt(i);

	if(m_listeners.isEmpty())
	{
		m_world->m_signals.m_postCollide.unsubscribeAll(this);
		m_world->m_signals.m_postSolve.unsubscribeAll(this);
	}
}

void hknpCharacterRigidBody::onPostCollideSignal( hknpWorld* /*world*/, hknpSolverData* collideOut )
{
	HK_ASSERT2( 0xa83f81d4, !m_listeners.isEmpty(),
		"Should not be a post collide signal when no character rigid body listener is set.");

	hkArray<ContactPointInfo> newContactPoints;
	for(int i = 0; i < m_listeners.getSize(); ++i)
	{
		m_listeners[i]->onPostCollide( this, *collideOut, newContactPoints );
	}

	// Add constraints for vertical contact points
	for (int i = 0; i < m_filteredContactPoints.getSize(); ++i)
	{
		const ContactPointInfo& contactInfo = m_filteredContactPoints[i];
		if (contactInfo.m_contactType != VERTICAL)
		{
			continue;
		}

		const hkContactPoint& contactPoint = contactInfo.m_contactPoint;
		hknpManifold manifold;
		manifold.init(contactPoint.getNormal(), contactPoint.getPosition(), contactPoint.getDistance());

		// We do not wish the character to be stuck on vertical edges
		hknpSolverData::SimpleMaterial material;
		material.m_friction = contactInfo.m_contactType == VERTICAL ? 0.f : 1.f;
		collideOut->addImmediateContactConstraint(contactInfo.m_bodyIds[0], contactInfo.m_bodyIds[1], manifold,
			HK_NULL, HK_NULL, material);
	}

	if (!newContactPoints.isEmpty())
	{
		for(int i = 0; i < newContactPoints.getSize(); ++i)
		{
			const ContactType type = newContactPoints[i].m_contactType;
			if(type == UNCLASSIFIED || type == IGNORED) continue;

			ContactPointInfo& cpi = m_filteredContactPoints.expandOne();
			cpi = newContactPoints[i];
			if(!cpi.m_bodyIds[0].isValid() || !cpi.m_bodyIds[1].isValid())
			{
				cpi.m_bodyIds[0] = m_bodyId;
				cpi.m_bodyIds[1] = hknpBodyId(0);
			}

			hknpManifold manifold;
			{
				const hkVector4& normal = cpi.m_contactPoint.getNormal();
				const hkVector4& point = cpi.m_contactPoint.getPosition();
				const hkSimdReal dist = cpi.m_contactPoint.getDistanceSimdReal();
				manifold.init( normal, point, dist.getReal() );
			}
			hknpSolverData::SimpleMaterial material;
			// We do not wish the character to be stuck on vertical edges
			material.m_friction = type == VERTICAL ? 0.f : 1.f;
			collideOut->addImmediateContactConstraint( cpi.m_bodyIds[0], cpi.m_bodyIds[1], manifold, HK_NULL, HK_NULL, material );
		}
	}
}

void hknpCharacterRigidBody::onPostSolveSignal( hknpWorld* /*world*/ )
{
	HK_ASSERT2( 0xa83f81d4, !m_listeners.isEmpty(),
		"Should not be a post solve signal when no character rigid body listener is set.");

	for(int i = 0; i < m_listeners.getSize(); ++i)
	{
		m_listeners[i]->onPostSolve( this );
	}
}

hknpCharacterSurfaceInfo::SupportedState hknpCharacterRigidBody::getSupportInfo(
	const hkStepInfo& stepInfo, hkArray<SupportInfo>& supportInfo ) const
{
	HK_TIMER_BEGIN("getSupportInfo", HK_NULL);

	// The default size for the local arrays of constraint planes.
	const int localNumConstraintPlanes = 20;

	// Input to the SimplexSolver.
	hkLocalArray<hkSurfaceConstraintInfo> constraints( localNumConstraintPlanes );

	// An array which stores the contact points corresponding to the constraint planes.
	// We will copy those entries which do offer support from here to the supportInfo array.
	hkLocalArray<SupportInfo> collisionInfo( localNumConstraintPlanes );

	// Iterate over all contact points, filling in the constraint and collisionInfo structures.
	for (int i=0; i < m_filteredContactPoints.getSize(); ++i)
	{
		const ContactPointInfo& contactInfo = m_filteredContactPoints[i];
		if (contactInfo.m_contactType == UNCLASSIFIED || contactInfo.m_contactType == VERTICAL)
		{
			continue;
		}

		const bool characterIsA = ( contactInfo.m_bodyIds[0] == m_bodyId );
		hknpBodyId partner = hknpBodyId(hkSelectOther( m_bodyId.value(), contactInfo.m_bodyIds[1].value(), contactInfo.m_bodyIds[0].value() ));

		// Partner body might have been deleted (if this is called from outside the step via checkSupport())
		if( !m_world->isBodyValid(partner) )
		{
			continue;
		}

		// Take first contact point only (all points share same normal, so the first one well defines the contact plane
		{
			SupportInfo& pair = collisionInfo.expandOne();
			pair.m_rigidBody = partner;
			pair.m_point = contactInfo.m_contactPoint;

			// Ensure the contact normal points towards the character.
			// Can happen if this character interacts with another rigid body character!
			if ( !characterIsA )
			{
				pair.m_point.flip();
			}

			hkSurfaceConstraintInfo& constraint = constraints.expandOne();

			constraint.m_plane = pair.m_point.getSeparatingNormal();

			// Assume the velocity of this surface is 0
			constraint.m_velocity.setZero();

			// Friction isn't used for finding support.
			constraint.m_staticFriction = 0.0f;
			constraint.m_dynamicFriction = 0.0f;
			constraint.m_extraUpStaticFriction = 0.0f;
			constraint.m_extraDownStaticFriction = 0.0f;

			// Set the constraint priority.
			const hknpBody& partnerBody = m_world->accessBody(partner);
			if (partnerBody.isStatic())
			{
				constraint.m_priority = 2;
			}
			else if (partnerBody.isKeyframed())
			{
				constraint.m_priority = 1;
			}
			else
			{
				constraint.m_priority = 0;
			}
		}
	}

	// Interactions array - this is the output of the simplex solver
	hkLocalArray<hkSurfaceConstraintInteraction> interactions( constraints.getSize() );

	//
	//	Use simplex solver to discover whether there are supporting constraints.
	//
	hknpCharacterSurfaceInfo::SupportedState supportedState;
	{
		hkVector4 down;
		{
			// Search downwards by the support distance.
			hkSimdReal s; s.setFromFloat(-m_supportDistance * stepInfo.m_invDeltaTime);
			down.setMul( s, m_up );
		}
		hkSimplexSolverOutput output;
		{
			hkSimplexSolverInput input;
			input.m_position.setZero();
			input.m_constraints = constraints.begin();
			input.m_numConstraints = constraints.getSize();
			input.m_velocity = down;
			input.m_deltaTime = stepInfo.m_deltaTime;
			input.m_minDeltaTime = stepInfo.m_deltaTime;
			input.m_upVector = m_up;
			input.m_maxSurfaceVelocity.setAll( m_maxSpeedForSimplexSolver );

			output.m_planeInteractions = interactions.begin();

			// Set the sizes of the arrays to be correct
			interactions.setSize(input.m_numConstraints);
			constraints.setSize(input.m_numConstraints);

			hkSimplexSolverSolve( input, output );
		}

		// Deduce the supported state by comparing input velocity (down) with output velocity.
		if ( output.m_velocity.allEqual<3>( down, hkSimdReal::getConstant<HK_QUADREAL_EPS>() ) )
		{
			supportedState = hknpCharacterSurfaceInfo::UNSUPPORTED;
		}
		else
		{
			const hkSimdReal outputLengthSqr = output.m_velocity.lengthSquared<3>();
			if ( outputLengthSqr.isLess( hkSimdReal::fromFloat(0.001f) ))
			{
				supportedState = hknpCharacterSurfaceInfo::SUPPORTED;
			}
			else
			{
				// Check whether the angle of the output velocity is so steep that we should consider
				// the character as sliding.

				hkSimdReal angleSin = output.m_velocity.dot<3>( m_up );
				hkSimdReal maxSlopeCos; maxSlopeCos.setFromFloat(m_maxSlopeCosine);
				if ( (outputLengthSqr - angleSin * angleSin) < (outputLengthSqr * maxSlopeCos * maxSlopeCos) )
				{
					supportedState = hknpCharacterSurfaceInfo::SLIDING;
				}
				else
				{
					supportedState = hknpCharacterSurfaceInfo::SUPPORTED;
				}
			}
		}
	}

	// If we're not unsupported, copy the supporting points into the supportInfo array.
	if ( supportedState != hknpCharacterSurfaceInfo::UNSUPPORTED )
	{
		hkInt32 numTouching = 0;

		const int numConstraints = constraints.getSize();
		hkSimdReal dotCutoff; dotCutoff.setFromFloat(0.08f);
		for ( int i=0; i < numConstraints; ++i )
		{
			// If we touched this plane (and it is at least slightly non-vertical).
			if ( ( interactions[i].m_touched ) && ( constraints[i].m_plane.dot<3>( m_up ) > dotCutoff ) )
			{
				SupportInfo& supportingPair = supportInfo.expandOne();
				supportingPair.m_point = collisionInfo[i].m_point;
				supportingPair.m_rigidBody = collisionInfo[i].m_rigidBody;
				++numTouching;

#ifdef DEBUG_HKNP_CHARACTER_RIGIDBODY
				debugContactPoint( collisionInfo[i].m_point, true );
			}
			else
			{
				debugContactPoint( collisionInfo[i].m_point, false );
#endif
			}
		}

		if ( numTouching == 0 )
		{
			supportedState = hknpCharacterSurfaceInfo::UNSUPPORTED;
		}
	}
#ifdef DEBUG_HKNP_CHARACTER_RIGIDBODY
	else
	{
		const int numCollisionInfo = collisionInfo.getSize();
		for ( int i=0; i < numCollisionInfo; ++i )
		{
			debugContactPoint( collisionInfo[i].m_point, false );
		}
	}
#endif

	HK_TIMER_END();

	return supportedState;
}

void hknpCharacterRigidBody::getGround(
	const hkArray<SupportInfo>& supportInfo, bool useDynamicBodyVelocities, hknpCharacterSurfaceInfo& ground ) const
{
	HK_TIMER_BEGIN("getGround", HK_NULL);

	ground.m_surfaceVelocity.setZero();
	ground.m_surfaceNormal.setZero();
	ground.m_surfaceDistanceExcess = 0.0f;
	ground.m_isSurfaceDynamic = false;

	const int numSupportInfo = supportInfo.getSize();
	HK_ASSERT2( 0xaef0119b, numSupportInfo, "getGround requires a positive number of supportInfo entries." );
	for ( int i = 0; i < numSupportInfo; ++i )
	{
		const SupportInfo& support = supportInfo[i];
		ground.m_surfaceNormal.add( support.m_point.getNormal() );
		ground.m_surfaceDistanceExcess += support.m_point.getDistance();
		const hknpBody& supportBody = m_world->getSimulatedBody(support.m_rigidBody);
		const hknpMotion& supportMotion = m_world->getMotion( supportBody.m_motionId );
		if ( supportBody.isKeyframed() )
		{
			hkVector4 pointVelocity;
			supportMotion.getPointVelocity( support.m_point.getPosition(), pointVelocity );
			ground.m_surfaceVelocity.add( pointVelocity );
		}
		else if ( supportBody.isDynamic() )
		{
			ground.m_isSurfaceDynamic = true;
			if ( useDynamicBodyVelocities )
			{
				hkVector4 pointVelocity;
				supportMotion.getPointVelocity( support.m_point.getPosition(), pointVelocity );
				ground.m_surfaceVelocity.add( pointVelocity );
			}
		}
	}

	ground.m_surfaceNormal.normalize<3>();

	hkSimdReal portion; portion.setFromInt32(numSupportInfo);
	hkSimdReal invPortion; invPortion.setReciprocal<HK_ACC_23_BIT,HK_DIV_IGNORE>(portion);

	ground.m_surfaceVelocity.mul( invPortion );

	ground.m_surfaceDistanceExcess *= invPortion.getReal();

	if ( ground.m_isSurfaceDynamic )
	{
		// We need to apply the character's weight onto dynamic bodies. We do this by
		// setting a positive surfaceDistanceExcess which the controller should try
		// to reduce by applying gravity.
		ground.m_surfaceDistanceExcess = 0.01f;
	}
	else
	{
		// For fixed and keyframed bodies, we subtract m_hardSupportDistance from the excess
		// to provide an additional gap beneath the character.
		ground.m_surfaceDistanceExcess -= m_hardSupportDistance;
	}

#ifdef DEBUG_HKNP_CHARACTER_RIGIDBODY
	debugGround(m_world->getSimulatedBody(m_bodyId).getTransform().getTranslation(),ground);
#endif

	HK_TIMER_END();
}

void hknpCharacterRigidBody::checkSupport( const hkStepInfo& stepInfo, hknpCharacterSurfaceInfo& ground ) const
{
	HK_TIMER_BEGIN("checkSupport", HK_NULL);

	if(isActive())
	{
		hkArray<SupportInfo> supportInfo; supportInfo.reserve(20);
		ground.m_supportedState = getSupportInfo( stepInfo, supportInfo );
		if ( ground.m_supportedState != hknpCharacterSurfaceInfo::UNSUPPORTED )
		{
			// By default, dynamic body velocities are not used.
			getGround( supportInfo, false, ground );
		}
	}
	else
	{
		// Should we introduce a hknpCharacterSurfaceInfo::DEACTIVATED state, or will it be too implementation specific?
		ground.m_supportedState = hknpCharacterSurfaceInfo::SUPPORTED;
	}

	HK_TIMER_END();
}


void hknpCharacterRigidBody::onPreCollideSignal( hknpWorld* world )
{
	m_filteredContactPoints.clear();
}

void hknpCharacterRigidBody::setLinearVelocity( hkVector4Parameter newVel, hkSimdRealParameter timestep, hkBool forceActivate )
{
	HK_ASSERT2( 0x566e6f55, timestep.isGreaterEqualZero(), "Time step must be >= 0." );

	// To ensure that the character's rigid body can deactivate, we only modify the velocity if it has changed.
	const hknpMotion& characterMotion = m_world->getMotion( m_world->getBody(m_bodyId).m_motionId );
	const hkVector4& currentVel = characterMotion.getLinearVelocity();
	if ( !newVel.allEqual<3>( currentVel, hkSimdReal_Eps ) )
	{
		// calculate and set acceleration for mass factor modifier
		hkVector4 currentAcc; currentAcc.setSub( newVel, currentVel );
		hkSimdReal rtimestep; rtimestep.setReciprocal<HK_ACC_23_BIT,HK_DIV_IGNORE>(timestep);
		currentAcc.mul( rtimestep );

		setLinearAccelerationToMassModifier( currentAcc );

		hkVector4 angVel; characterMotion._getAngularVelocity(angVel);
		m_world->setBodyVelocity(m_bodyId, newVel, angVel);
	}
	else if( forceActivate )
	{
		m_world->activateBody( m_bodyId );
	}
}

void hknpCharacterRigidBody::setAngularVelocity(hkVector4Parameter newAVel)
{
	// To ensure that the character's rigid body can deactivate, we only modify the velocity if it has changed.
	const hknpMotion& characterMotion = m_world->getMotion( m_world->getBody(m_bodyId).m_motionId );
	hkVector4 currentVel; characterMotion._getAngularVelocity(currentVel);
	if ( !newAVel.allEqual<3>( currentVel, hkSimdReal::getConstant<HK_QUADREAL_EPS>() ) )
	{
		hkVector4 linVel = characterMotion.getLinearVelocity();
		m_world->setBodyVelocity(m_bodyId, linVel, newAVel);
	}
}

void hknpCharacterRigidBody::setPosition(hkVector4Parameter position)
{
	hkTransform transform = m_world->getBody(m_bodyId).getTransform();
	transform.setTranslation( position );
	m_world->setBodyTransform( m_bodyId, transform );
}

void hknpCharacterRigidBody::setTransform(const hkTransform& transform)
{
	m_world->setBodyTransform( m_bodyId, transform );
}

void hknpCharacterRigidBody::setLinearAccelerationToMassModifier(hkVector4Parameter newAcc)
{
	m_acceleration = newAcc;
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
