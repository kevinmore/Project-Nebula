/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>

#include <Common/Internal/SimplexSolver/hkSimplexSolver.h>

#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnTrack.h>
#include <Common/Visualize/hkDebugDisplay.h>

#include <Physics2012/Utilities/CharacterControl/CharacterRigidBody/hkpCharacterRigidBody.h>
#include <Physics2012/Utilities/CharacterControl/CharacterRigidBody/hkpCharacterRigidBodyListener.h>

#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>
#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>

#include <Physics2012/Utilities/CharacterControl/hkpCharacterControl.h>

// Includes for collector used by checkSupport.
#include <Physics2012/Collide/Agent/Collidable/hkpCdPoint.h>
#include <Physics2012/Collide/Agent/Query/hkpCdPointCollector.h>

// Enable this to see manifold planes.
#ifdef HK_DEBUG
#define DEBUG_CHARACTER_RIGIDBODY
#endif

#ifdef DEBUG_CHARACTER_RIGIDBODY
#include <Common/Visualize/hkDebugDisplay.h>
static void debugContactPoint( hkContactPoint& cp, hkBool32 isSupportingPoint = false );
static void debugGround(const hkVector4& position, const hkpSurfaceInfo& ground);
#endif

hkpCharacterRigidBody::hkpCharacterRigidBody( const hkpCharacterRigidBodyCinfo& info )
{
	HK_ASSERT2(0x79d50ec9,  !(&info == HK_NULL), "No info defined");
	HK_ASSERT2(0x79d51ec9,  !(info.m_shape == HK_NULL), "No shape for character defined");
	HK_ASSERT2(0x79d52ec9,  info.m_up.isOk<3>(), "Up direction incorrectly defined");
	HK_ASSERT2(0x79d52ec9,  ( 0 < info.m_maxSlope ) && ( info.m_maxSlope <= 0.5f * HK_REAL_PI ) , "maxSlope must be between 0 and pi/2 radians" );
	
	hkpRigidBodyCinfo ci;
	{
		// Set rigid body character shape (usually capsule)
		ci.m_shape = info.m_shape;

		// Set rigid body mass (Default is 100 kg)
		// Remember that all max acceleration and max force are very depended on the mass of character
		ci.m_mass = info.m_mass;

		// Set friction (Default is no friction)
		ci.m_friction = info.m_friction;

		// Set init position and rotation
		ci.m_position = info.m_position;
		ci.m_rotation = info.m_rotation;

		// Set collision filter info
		ci.m_collisionFilterInfo = info.m_collisionFilterInfo;

		// Set restitution no restitution
		ci.m_restitution = 0.0f;

		// Set maximal velocity (Default is 20 m/s)
		// Help to predict unexpected movement (extreme acceleration, character fired out)
		// Remember: Depended on maxWalkSpeed (Influence to if you want to jump and run simultaneously)
		// Increase it in the case of character on fast moving platforms
		ci.m_maxLinearVelocity = info.m_maxLinearVelocity;

		// Set maximal allowed penetration
		ci.m_allowedPenetrationDepth = info.m_allowedPenetrationDepth;
		
		ci.m_motionType = hkpMotion::MOTION_CHARACTER;
		ci.m_qualityType = HK_COLLIDABLE_QUALITY_CHARACTER;
	}

	// Create rigid body
	m_character = new hkpRigidBody(ci);
	{
		// Set to zero inverse local inertia to avoid proxy rotations. Must be done after rigid body creation!
		hkMatrix3 zeroInvInertia; zeroInvInertia.setZero();
		m_character->setInertiaInvLocal(zeroInvInertia);

		// Set rigid body to default transparent red color
		m_character->addProperty(HK_PROPERTY_DEBUG_DISPLAY_COLOR, info.m_vdbColor);
	}

	// Values used for check support
	{
		// Set up direction
		m_up = info.m_up;
		HK_ASSERT2(0x79d58ec9,  hkMath::equal(m_up.length<3>().getReal(), 1.0f), "checkSupport down direction should be normalized");

		// Set max slope cosine
		// It is a angle between UP and horizontal plane ???
		m_maxSlopeCosine = hkMath::cos(info.m_maxSlope);

		// Set limit support distance to detect support state
		m_supportDistance = info.m_supportDistance;
		m_hardSupportDistance = info.m_hardSupportDistance;
		
		// Set additional parameters for simplex solver
		m_maxSpeedForSimplexSolver = info.m_maxSpeedForSimplexSolver;
	}


	m_unweldingHeightOffsetFactor = info.m_unweldingHeightOffsetFactor;

	{
		m_acceleration.setZero();
		m_maxForce = info.m_maxForce;
	}

	m_character->addEntityListener( this );
	m_listener = HK_NULL;
}


void hkpCharacterRigidBody::setListener( hkpCharacterRigidBodyListener* listener )
{
	HK_ASSERT2( 0x84ae29f7, !m_listener, "Character rigid body already has a listener." );
	m_listener = listener;
	m_listener->addReference();
	if ( m_character->isAddedToWorld() )
	{
		m_character->getWorld()->addWorldPostSimulationListener( this );
	}
}


void hkpCharacterRigidBody::removeListener()
{
	m_listener->removeReference();
	m_listener = HK_NULL;
	// If the entity is added to the world, then this object must be removed from the world
	// as a post-simulation listener.
	if ( m_character->isAddedToWorld() )
	{
		m_character->getWorld()->removeWorldPostSimulationListener( this );
	}
}


hkpCharacterRigidBody::~hkpCharacterRigidBody()
{
	if ( m_listener )
	{
		m_listener->removeReference();
	}
	m_character->removeEntityListener( this );
	m_character->removeProperty( HK_PROPERTY_DEBUG_DISPLAY_COLOR );
	m_character->removeReference();
}


void hkpCharacterRigidBody::entityAddedCallback( hkpEntity* entity )
{
	// So long as its rigid body is added to the world, this object
	// must be kept alive.
	addReference();
	if ( m_listener )
	{
		entity->getWorld()->addWorldPostSimulationListener( this );
	}
}


void hkpCharacterRigidBody::entityRemovedCallback( hkpEntity* entity )
{
	if ( m_listener )
	{
		entity->getWorld()->removeWorldPostSimulationListener( this );
	}
	removeReference();
}


void hkpCharacterRigidBody::postSimulationCallback( hkpWorld* world )
{
	HK_ASSERT2( 0xa83f81d4, m_listener, "Should not be a post simulation listener when no character rigid body listener is set.");
	m_listener->characterCallback( world, this );
}


hkpSurfaceInfo::SupportedState hkpCharacterRigidBody::getSupportInfo( const hkStepInfo& stepInfo, hkArray<SupportInfo>& supportInfo ) const
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
	hkArray<struct hkpLinkedCollidable::CollisionEntry> collisionEntriesTmp;
	
	// This is verbose, but necessary to make sure that the const versions are called (because the non-const versions do write-access checks)
	const hkpRigidBody* characterRb = m_character;
	const hkpLinkedCollidable* linkedCollidable = characterRb->getLinkedCollidable();
	linkedCollidable->getCollisionEntriesSorted(collisionEntriesTmp);

	const hkArray<struct hkpLinkedCollidable::CollisionEntry>& entries = collisionEntriesTmp;

	for ( int i = 0; i < entries.getSize(); ++i )
	{
		hkpContactMgr* mgr = entries[i].m_agentEntry->m_contactMgr;

		if ( mgr->m_type == hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR )
		{
			hkpSimpleConstraintContactMgr* constraintMgr = reinterpret_cast<hkpSimpleConstraintContactMgr*>( mgr );			

			// Skip disabled contact constraints.
			if ( !constraintMgr->isConstraintDisabled() )
			{
				// Check if the constraint has any surface velocity modifier atom				
				hkVector4 surfaceVelocity; surfaceVelocity.setZero();				
				{				
					hkpConstraintAtom* atom = constraintMgr->m_constraint.getConstraintModifiers();				
					while (atom != HK_NULL && atom->isModifierType())
					{
						if (atom->getType() == hkpConstraintAtom::TYPE_MODIFIER_MOVING_SURFACE)
						{						
							surfaceVelocity = static_cast<hkpMovingSurfaceModifierConstraintAtom*>(atom)->m_velocity;
							break;
						}
						atom = static_cast<hkpModifierConstraintAtom*>(atom)->m_child;
					}
				}

				// Loop over all contact points
				for ( int j = 0; j < constraintMgr->m_contactConstraintData.getNumContactPoints(); ++j )
				{
					const int cpId = constraintMgr->m_contactConstraintData.getContactPointIdAt( j );
					hkpContactPointProperties *const properties = constraintMgr->m_contactConstraintData.getContactPointProperties( cpId );

					// skip disabled contact points.
					if ( properties->m_flags & hkContactPointMaterial::CONTACT_IS_DISABLED )
					{
						continue;
					}

					hkpAgentNnEntry *const entry = entries[i].m_agentEntry;
					const hkBool characterIsA = ( entry->getCollidableA() == m_character->getCollidable() );
					hkpRigidBody *const partner = hkpGetRigidBody( characterIsA ? entry->getCollidableB() : entry->getCollidableA() );
					
					// Check that the partner is a rigid body.
					if ( !partner )
					{
						continue;
					}

					SupportInfo& pair = collisionInfo.expandOne();
					pair.m_rigidBody = partner;
					pair.m_point = constraintMgr->m_contactConstraintData.getContactPoint( cpId );
					pair.m_surfaceVelocity = surfaceVelocity;
					
					// Ensure the contact normal points towards the character.
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
					switch( partner->getMotionType() )
					{
						case hkpMotion::MOTION_FIXED:
							constraint.m_priority = 2;
							break;

						case hkpMotion::MOTION_KEYFRAMED:
							constraint.m_priority = 1;
							break;

						default: // any dynamic body
							constraint.m_priority = 0;
							break;
					}
				}
			}
		}
	}

	// Interactions array - this is the output of the simplex solver
	hkLocalArray<hkSurfaceConstraintInteraction> interactions( constraints.getSize() );

	//
	//	Use simplex solver to discover whether there are supporting constraints.
	//
	hkpSurfaceInfo::SupportedState supportedState;
	{
		hkVector4 down;
		{
			// Search downwards by the support distance.
			down.setMul( hkSimdReal::fromFloat(-m_supportDistance / stepInfo.m_deltaTime), m_up );
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

		if ( output.m_velocity.allEqual<3>( down, hkSimdReal::fromFloat(1e-3f) ) )
		{
			supportedState = hkpSurfaceInfo::UNSUPPORTED;
		}
		else
		{
			const hkSimdReal outputLengthSqr = output.m_velocity.lengthSquared<3>();

			if ( outputLengthSqr < hkSimdReal::fromFloat(0.001f) )
			{
				supportedState = hkpSurfaceInfo::SUPPORTED;
			}
			else
			{
				// Check whether the angle of the output velocity is so steep that we should consider
				// the character as sliding.

				const hkSimdReal angleSin = output.m_velocity.dot<3>( m_up );

				if ( outputLengthSqr - angleSin * angleSin < outputLengthSqr * hkSimdReal::fromFloat(m_maxSlopeCosine * m_maxSlopeCosine) )
				{
					supportedState = hkpSurfaceInfo::SLIDING;
				}
				else
				{
					supportedState = hkpSurfaceInfo::SUPPORTED;
				}
			}
		}
	}

	// If we're not unsupported, copy the supporting points into the supportInfo array.
	if ( supportedState != hkpSurfaceInfo::UNSUPPORTED )
	{
		hkInt32 numTouching = 0;

		const int numConstraints = constraints.getSize();
		for ( int i=0; i < numConstraints; ++i )
		{
			// If we touched this plane (and it is at least slightly non-vertical).
			if ( ( interactions[i].m_touched ) && constraints[i].m_plane.dot<3>( m_up ).getReal() > 0.08f )
			{
				SupportInfo& supportingPair = supportInfo.expandOne();
				supportingPair.m_point = collisionInfo[i].m_point;
				supportingPair.m_rigidBody = collisionInfo[i].m_rigidBody;
				supportingPair.m_surfaceVelocity = collisionInfo[i].m_surfaceVelocity;
				++numTouching;

#ifdef DEBUG_CHARACTER_RIGIDBODY
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
			supportedState = hkpSurfaceInfo::UNSUPPORTED;
		}
	}
#ifdef DEBUG_CHARACTER_RIGIDBODY
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

void hkpCharacterRigidBody::getGround( const hkArray<SupportInfo>& supportInfo, hkBool useDynamicBodyVelocities, hkpSurfaceInfo& ground ) const
{
	HK_TIMER_BEGIN("getGround", HK_NULL);
	
	ground.m_surfaceVelocity.setZero();
	ground.m_surfaceNormal.setZero();
	ground.m_surfaceDistanceExcess = 0.0f;
	ground.m_surfaceIsDynamic = false;

	const int numSupportInfo = supportInfo.getSize();
	HK_ASSERT2( 0xaef0119b, numSupportInfo, "getGround requires a positive number of supportInfo entries." );
	for ( int i = 0; i < numSupportInfo; ++i )
	{
		const SupportInfo& support = supportInfo[i];

		ground.m_surfaceVelocity.add(support.m_surfaceVelocity);
		ground.m_surfaceNormal.add( support.m_point.getNormal() );
		ground.m_surfaceDistanceExcess += support.m_point.getDistance();
		const hkpMotion::MotionType motionType = support.m_rigidBody->getMotionType();
		if ( motionType == hkpMotion::MOTION_KEYFRAMED )
		{
			hkVector4 pointVelocity;
			support.m_rigidBody->getPointVelocity( support.m_point.getPosition(), pointVelocity );			
			ground.m_surfaceVelocity.add( pointVelocity );
		}
		else if ( motionType != hkpMotion::MOTION_FIXED )
		{
			ground.m_surfaceIsDynamic = true;
			if ( useDynamicBodyVelocities )
			{
				hkVector4 pointVelocity;
				support.m_rigidBody->getPointVelocity( support.m_point.getPosition(), pointVelocity );
				ground.m_surfaceVelocity.add( pointVelocity );
			}
		}
	}

	ground.m_surfaceNormal.normalize<3>();
	hkSimdReal portion; portion.setFromInt32(numSupportInfo); portion.setReciprocal(portion);

	ground.m_surfaceVelocity.mul( portion );

	ground.m_surfaceDistanceExcess *= portion.getReal();

	if ( ground.m_surfaceIsDynamic )
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

#ifdef DEBUG_CHARACTER_RIGIDBODY
	debugGround(m_character->getPosition(),ground);
#endif

	HK_TIMER_END();
}


void hkpCharacterRigidBody::checkSupport( const hkStepInfo& stepInfo, hkpSurfaceInfo& ground ) const 
{
	HK_TIMER_BEGIN("checkSupport", HK_NULL);

	hkArray<SupportInfo> supportInfo;
	ground.m_supportedState = getSupportInfo( stepInfo, supportInfo );
	if ( ground.m_supportedState != hkpSurfaceInfo::UNSUPPORTED )
	{
		// By default, dynamic body velocities are not used.
		getGround( supportInfo, false, ground );
	}

	HK_TIMER_END();
}


// ////////////////////////////////////////////////////////////////////////

hkpRigidBody* hkpCharacterRigidBody::getRigidBody() const
{
	return m_character;
}

void hkpCharacterRigidBody::setLinearAccelerationToMassModifier(const hkVector4& newAcc)
{
	m_acceleration = newAcc;
}

void hkpCharacterRigidBody::setMaxForce( hkReal maxForce )
{
	m_maxForce = maxForce;
}

void hkpCharacterRigidBody::setLinearVelocity( const hkVector4& newVel, hkReal timestep )
{
	HK_ASSERT2( 0x566e6f55, timestep>=0, "Timestep must be >= 0." );		
	
	// To ensure that the character's rigid body can deactivate, we only modify the velocity if it has changed.
	if ( !newVel.allEqual<3>( m_character->getLinearVelocity(), hkSimdReal::fromFloat(1e-3f) ) )
	{		
		HK_WARN_ON_DEBUG_IF((newVel.lengthSquared<3>().getReal() > (m_character->getMaxLinearVelocity() * m_character->getMaxLinearVelocity())), 0x54dd8a78, "Character Rigid Body target velocity is higher than the maximum allowed in the rigid body");

		// calculate and set acceleration for mass factor modifier
		hkVector4 currentVel; currentVel = m_character->getLinearVelocity();

		hkVector4 currentAcc; currentAcc.setSub( newVel, currentVel );
		hkSimdReal invTimestep; invTimestep.setFromFloat(timestep); invTimestep.setReciprocal(invTimestep);
		currentAcc.mul( invTimestep );

		setLinearAccelerationToMassModifier( currentAcc );
		
		m_character->setLinearVelocity( newVel );
	}
}

const hkVector4& hkpCharacterRigidBody::getLinearVelocity() const
{
	return m_character->getLinearVelocity();
}

void hkpCharacterRigidBody::setAngularVelocity(const hkVector4& newAVel)
{
	// To ensure that the character's rigid body can deactivate, we only modify the velocity if it has changed.
	if ( !newAVel.allEqual<3>( m_character->getAngularVelocity(), hkSimdReal::fromFloat(1e-3f) ) )
	{
		m_character->setAngularVelocity(newAVel);
	}
}

const hkVector4& hkpCharacterRigidBody::getAngularVelocity() const
{
	return m_character->getAngularVelocity();
}

const hkVector4& hkpCharacterRigidBody::getPosition() const
{
	return m_character->getPosition();
}

#ifdef DEBUG_CHARACTER_RIGIDBODY

// This color represent the contact point
const hkColor::Argb HK_DEBUG_CONTACT_POINT_COLOR = hkColor::WHITE;

// Planes in this color represent the contact point plane
const hkColor::Argb HK_DEBUG_CONTACT_PLANE_COLOR = hkColor::WHITESMOKE;

const hkColor::Argb HK_DEBUG_SUPPORT_PLANE_COLOR = hkColor::CYAN;
const hkColor::Argb HK_DEBUG_VERTICAL_PLANE_COLOR = hkColor::RED;

static void debugContactPoint( hkContactPoint& cp, hkBool32 isSupportingPoint )
{
	bool isVerticalContact = ( hkpCharacterRigidBody::m_magicNumber == cp.getPosition().getInt24W() );
	hkColor::Argb color = isSupportingPoint ? HK_DEBUG_SUPPORT_PLANE_COLOR : ( isVerticalContact ? HK_DEBUG_VERTICAL_PLANE_COLOR : HK_DEBUG_CONTACT_PLANE_COLOR );
	hkReal size = isSupportingPoint ? 0.6f : 0.5f;
	// Display contact point
	HK_DISPLAY_STAR( cp.getPosition(),0.2f, color );

	// Display contact plane
	hkVector4 plane = cp.getNormal();
	plane.zeroComponent<3>();
	hkVector4 pos = cp.getPosition();
	HK_DISPLAY_PLANE( plane, pos, size, color );
}

// This color represent the normal of dynamic surface
const hkColor::Argb HK_DEBUG_SURFACE_DYNAMIC_COLOR = hkColor::GREEN;

// This color represent the normal of fixed or keyframed surface
const hkColor::Argb HK_DEBUG_SURFACE_NONDYNAMIC_COLOR = hkColor::YELLOWGREEN;

// This color represent the surface velocity
const hkColor::Argb HK_DEBUG_SURFACE_VELOCITY_COLOR = hkColor::YELLOW;

static void debugGround( const hkVector4& position, const hkpSurfaceInfo& ground )
{
	// Display surface normal in center
	if ( ground.m_surfaceIsDynamic )
	{
		HK_DISPLAY_ARROW( position, ground.m_surfaceNormal, HK_DEBUG_SURFACE_DYNAMIC_COLOR );
	}
	else
	{
		HK_DISPLAY_ARROW( position, ground.m_surfaceNormal, HK_DEBUG_SURFACE_NONDYNAMIC_COLOR );
	}

	// Display surface velocity
	HK_DISPLAY_ARROW( position, ground.m_surfaceVelocity, HK_DEBUG_SURFACE_VELOCITY_COLOR );
}

#endif

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
