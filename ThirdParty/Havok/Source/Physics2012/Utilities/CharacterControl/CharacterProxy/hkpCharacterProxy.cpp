/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/hkpCharacterProxy.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/hkpCharacterProxyCinfo.h>
#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Common/Internal/SimplexSolver/hkSimplexSolver.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Physics2012/Collide/Query/CastUtil/hkpLinearCastInput.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

// For multithreading
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/hkpCharacterProxyJobs.h>
#if defined(HK_PLATFORM_HAS_SPU)
	#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Spu/hkpSpuCharacterProxyConfig.h>
#endif
#if defined(HK_PLATFORM_SPU)
	#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Spu/hkpSpuCharacterProxyUtil.h>
	#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Spu/hkpSpuWorldLinearCastCollector.h>

	#define HK_MANIFOLD_PUSH_BACK( X, Y ) if ( X.tryPushBack(Y) ) { fireContactAdded( Y ); } else { HK_WARN(0xaf354ee2, "No room to expand manifold array."); }
	#define HK_TRIGGER_VOLUME_PUSH_BACK( X, Y ) if ( !X.tryPushBack(Y) ) { HK_WARN(0xaf354ee2, "No room to expand manifold array."); }

#else
	#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Cpu/hkpCpuCharacterProxyUtil.h>
	#include <Physics2012/Utilities/Collide/TriggerVolume/hkpTriggerVolume.h>
	#include <Physics2012/Dynamics/Phantom/hkpShapePhantom.h>

	#define HK_MANIFOLD_PUSH_BACK( X, Y ) X.pushBack( Y ); fireContactAdded( Y );
	#define HK_TRIGGER_VOLUME_PUSH_BACK( X, Y ) if ( !X.tryPushBack(Y) ) { HK_WARN(0xaf354ee2, "No room to expand trigger volume array."); }
#endif

/// Because the code path is the same, it's easier to adopt a hard-limit for single-threaded processing too.
#define HK_MAX_NUM_TRIGGER_VOLUME_ST 16

// Setting and getting the flag and pointer for trigger volume management.
#define HK_TRIGGER_VOLUME_TOUCHED( X )		reinterpret_cast<hkpTriggerVolume*>( hkUlong( X ) | 1 )
#define HK_TRIGGER_VOLUME_LEFT( X )			hkClearBits(X, 1)
#define HK_TRIGGER_VOLUME_GET_PTR( X )		hkClearBits(X, 1)
#define HK_TRIGGER_VOLUME_GET_TOUCHED( X )	reinterpret_cast<hkpTriggerVolume*>( hkUlong( X ) & 1 )

// Enable this to see manifold planes.
#if defined(HK_DEBUG)
	#if !defined(HK_PLATFORM_SPU)
		#define DEBUG_CHARACTER_CONTROLLER
	#endif
#endif

static bool addMaxSlopePlane( hkReal maxSlopeCos, const hkVector4& up, int index, hkArray<hkSurfaceConstraintInfo>& surfaceConstraintsInOut)
{
	const hkReal surfaceVerticalComponent = surfaceConstraintsInOut[index].m_plane.dot<3>(up).getReal();

	if ( surfaceVerticalComponent > 0.01f && surfaceVerticalComponent < maxSlopeCos )
	{

#if defined(HK_PLATFORM_SPU)
		if( surfaceConstraintsInOut.getSize() < surfaceConstraintsInOut.getCapacity() )
#endif
		{
			// Add an additional vertical plane at the end of the constraints array
			hkSurfaceConstraintInfo& newConstraint = surfaceConstraintsInOut.expandOne();

			// Copy original info
			newConstraint = surfaceConstraintsInOut[index];

			// Reorient this new plane so it is vertical
			newConstraint.m_plane.addMul( -hkSimdReal::fromFloat(surfaceVerticalComponent), up );
			newConstraint.m_plane.normalize<3>();
			return true;
		}
#if defined(HK_PLATFORM_SPU)
		else
		{
			HK_WARN( 0xc948b719, "Cannot resize on SPU. The surfaceConstraintsInOut capacity cannot accomodate the max slope plane. \
								 This may lead to unpredictable results." );
		}
#endif
	}

	// No plane added
	return false;
}

#if defined(DEBUG_CHARACTER_CONTROLLER)
// Planes in this colour are passed directly to the simplex solver
// These planes actually restrict character movement
const hkColor::Argb HK_DEBUG_CLEAN_SIMPLEX_MANIFOLD_COLOR = hkColor::GREEN;

// Planes in this color are additional planes the user had added
// Typically these are added in the processConstraintsCallback
// These planes actually restrict character movement
const hkColor::Argb HK_DEBUG_USER_SIMPLEX_MANIFOLD_COLOR = 0xffffdf00;

// Planes in this color are additional vertical planes added to restrict
// character movement up steep slopes
const hkColor::Argb HK_DEBUG_VERTICAL_SIMPLEX_MANIFOLD_COLOR = hkColor::CYAN;


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

static void debugCast(const hkpAllCdPointCollector& startCollector, const hkpAllCdPointCollector& castCollector)
{
	{
		for (int h=0; h < startCollector.getHits().getSize(); h++)
		{
			const hkpRootCdPoint& hit = startCollector.getHits()[h];
			hkVector4 plane = hit.m_contact.getNormal();
			plane.zeroComponent<3>();
			hkVector4 pos = hit.m_contact.getPosition();
			HK_DISPLAY_PLANE(plane, pos, 0.5f, HK_DEBUG_STARTPOINT_HIT_POSITION_COLOR);
			pos.addMul(hit.m_contact.getDistanceSimdReal(), hit.m_contact.getNormal());

			if (hit.m_contact.getDistanceSimdReal().isLessZero())
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
		for (int h=0; h < castCollector.getHits().getSize(); h++)
		{
			const hkpRootCdPoint& hit = castCollector.getHits()[h];
			hkVector4 plane = hit.m_contact.getNormal();
			plane.zeroComponent<3>();
			hkVector4 pos = hit.m_contact.getPosition();
			HK_DISPLAY_PLANE(plane, pos, 0.5f, HK_DEBUG_CASTPOINT_HIT_POSITION_COLOR);
			pos.addMul(hit.m_contact.getDistanceSimdReal(), hit.m_contact.getNormal());
			HK_DISPLAY_PLANE(plane, pos, 0.5f, HK_DEBUG_CASTPOINT_DIST_COLOR);
		}
	}
}
#endif //#if defined(DEBUG_CHARACTER_CONTROLLER)


#if !defined(HK_PLATFORM_SPU)
hkpCharacterProxy::hkpCharacterProxy(const hkpCharacterProxyCinfo& info)
: m_shapePhantom(HK_NULL)
{	
	updateFromCinfo( info );
}

void hkpCharacterProxy::getCinfo(hkpCharacterProxyCinfo& info) const
{
	info.m_position = getPosition();
	info.m_velocity = m_velocity;
	info.m_shapePhantom = m_shapePhantom;
	info.m_dynamicFriction = m_dynamicFriction;
	info.m_staticFriction = m_staticFriction;
	info.m_keepContactTolerance = m_keepContactTolerance;
	info.m_up = m_up;
	info.m_extraUpStaticFriction = m_extraUpStaticFriction;
	info.m_extraDownStaticFriction = m_extraDownStaticFriction;
	info.m_keepDistance = m_keepDistance;
	info.m_contactAngleSensitivity = m_contactAngleSensitivity;
	info.m_userPlanes = m_userPlanes;
	info.m_maxCharacterSpeedForSolver = m_maxCharacterSpeedForSolver;
	info.m_characterStrength = m_characterStrength;
	info.m_characterMass = m_characterMass;
	info.m_maxSlope = hkMath::acos(m_maxSlopeCosine);
	info.m_penetrationRecoverySpeed = m_penetrationRecoverySpeed;
	info.m_maxCastIterations = m_maxCastIterations;
	info.m_refreshManifoldInCheckSupport = m_refreshManifoldInCheckSupport;
}

void hkpCharacterProxy::updateFromCinfo( const hkpCharacterProxyCinfo& info )
{
	HK_ASSERT2(0x1e81f814, info.m_shapePhantom != HK_NULL, "Shape phantom can not be NULL");

	info.m_shapePhantom->addReference();
	if (m_shapePhantom)
	{
		m_shapePhantom->removeReference();
	}

	m_velocity = info.m_velocity;
	m_dynamicFriction = info.m_dynamicFriction;
	m_staticFriction = info.m_staticFriction;
	m_keepContactTolerance = info.m_keepContactTolerance;
	m_extraUpStaticFriction = info.m_extraUpStaticFriction;
	m_extraDownStaticFriction = info.m_extraDownStaticFriction;
	m_keepDistance = info.m_keepDistance;
	m_shapePhantom = info.m_shapePhantom;
	m_contactAngleSensitivity = info.m_contactAngleSensitivity;
	m_userPlanes = info.m_userPlanes;
	m_maxCharacterSpeedForSolver = info.m_maxCharacterSpeedForSolver;
	m_characterStrength = info.m_characterStrength;
	m_characterMass = info.m_characterMass;
	m_maxSlopeCosine = hkMath::cos( info.m_maxSlope );
	m_penetrationRecoverySpeed = info.m_penetrationRecoverySpeed;
	m_maxCastIterations = info.m_maxCastIterations;
	m_refreshManifoldInCheckSupport = info.m_refreshManifoldInCheckSupport;

	m_up = info.m_up;
	m_up.normalize<3>();

	// On construction we cannot guarantee that the phantom
	// has been added to the world already.
	// This often depends on the users specific toolchain
	if (m_shapePhantom->getWorld() != HK_NULL)
	{
		setPosition(info.m_position);
	}
	else
	{
		HK_WARN(0x6cee9071, "Shape phantom has not yet been added to the world. Initial position has been ignored");
	}

	m_oldDisplacement.setZero();

	{
		for (int i=0; i< m_bodies.getSize(); i++)
		{
			m_bodies[i]->removeEntityListener(this);
		}
		m_bodies.clear();

		for (int j=0; j< m_phantoms.getSize(); j++)
		{
			m_phantoms[j]->removePhantomListener(this);
		}
		m_phantoms.clear();
	}
#if defined(HK_PLATFORM_HAS_SPU)
	m_manifold.reserve( HK_MANIFOLD_CAPACITY_ON_SPU );
#endif
	m_manifold.clear();

	// Add a property that allows us to identify this as a character proxy later
	if ( !m_shapePhantom->hasProperty( HK_PROPERTY_CHARACTER_PROXY ) )
	{
		m_shapePhantom->addProperty( HK_PROPERTY_CHARACTER_PROXY, this );
	}
}

// #include <stdio.h>
// static void printManifold(hkArray<hkpRootCdPoint>& manifold)
// {
// 	printf("\nManifold: \n");
// 	for (int i = 0; i < manifold.getSize(); ++i)
// 	{
// 		printf("position = %f %f %f/ \nnormal = %f %f %f\ndist = %f\n",	manifold[i].m_contact.getPosition()(0),
// 																		manifold[i].m_contact.getPosition()(1),
// 																		manifold[i].m_contact.getPosition()(2),
// 																		manifold[i].m_contact.getSeparatingNormal()(0),
// 																		manifold[i].m_contact.getSeparatingNormal()(1),
// 																		manifold[i].m_contact.getSeparatingNormal()(2),
// 																		manifold[i].m_contact.getSeparatingNormal()(3) );
// 	}
// }

hkpCharacterProxy::~hkpCharacterProxy()
{
	for (int i=0; i< m_bodies.getSize(); i++)
	{
		m_bodies[i]->removeEntityListener(this);
	}
	m_bodies.clear();

	for (int j=0; j< m_phantoms.getSize(); j++)
	{
		m_phantoms[j]->removePhantomListener(this);
	}
	m_phantoms.clear();

	{
		const int numTriggerVolumes = m_overlappingTriggerVolumes.getSize();
		for ( int i = 0; i < numTriggerVolumes; ++i )
		{
			m_overlappingTriggerVolumes[i]->getTriggerBody()->removeEntityListener( this );
		}
	}

	m_shapePhantom->removeProperty(HK_PROPERTY_CHARACTER_PROXY);
	m_shapePhantom->removeReference();
}


void hkpCharacterProxy::integrate( const hkStepInfo& stepInfo, const hkVector4& worldGravity )
{
	hkpAllCdPointCollector castCollector;
	hkpAllCdPointCollector startPointCollector;
	integrateImplementation( stepInfo, worldGravity, HK_NULL, castCollector, startPointCollector );
}

void hkpCharacterProxy::integrateWithCollectors(const hkStepInfo& stepInfo,
												const hkVector4& gravity,
												hkpAllCdPointCollector& castCollector,
												hkpAllCdPointCollector& startPointCollector)
{
	integrateImplementation(stepInfo, gravity, HK_NULL, castCollector, startPointCollector );
}

const hkArray<hkpRootCdPoint>& hkpCharacterProxy::getManifold() const
{
	return m_manifold;
}

void hkpCharacterProxy::refreshManifold( hkpAllCdPointCollector& startPointCollector )
{
	// Update the start point collector
	{
		startPointCollector.reset();
		hkpWorld* world = m_shapePhantom->getWorld();
		HK_ASSERT2(0x54e32e12, world, "Character proxy must be added to world before calling refreshManifold");

		hkpCollisionInput input = *world->getCollisionInput();
		input.m_tolerance = input.m_tolerance + (m_keepDistance + m_keepContactTolerance);
		m_shapePhantom->getClosestPoints(startPointCollector, &input);
	}

	// Update the manifold so that it is correct, so that checkSupport() returns the correct answer.
	hkpAllCdPointCollector castCollector;
	updateManifold(startPointCollector, castCollector, m_manifold, m_bodies, m_phantoms );
}

void hkpCharacterProxy::checkSupport(const hkVector4& direction, hkpSurfaceInfo& ground)
{
	hkpAllCdPointCollector startPointCollector;
	checkSupportWithCollector(direction, ground, startPointCollector);
}

void hkpCharacterProxy::checkSupportWithCollector(const hkVector4& direction, hkpSurfaceInfo& ground, hkpAllCdPointCollector& startPointCollector)
{
	HK_ASSERT2(0x79d57ec9,  hkMath::equal(direction.length<3>().getReal(), 1.0f), "checkSupport Direction should be normalized");
	HK_TIMER_BEGIN("checkSupport", HK_NULL);

	if (m_refreshManifoldInCheckSupport)
	{
		refreshManifold( startPointCollector );
	}

	hkLocalArray<hkSurfaceConstraintInfo> constraints(m_manifold.getSize() + m_userPlanes + 10);
	constraints.setSizeUnchecked(m_manifold.getSize());
	{
		for (int i=0; i < m_manifold.getSize() ; i++)
		{
			extractSurfaceConstraintInfo(m_manifold[i], constraints[i], 0);
			addMaxSlopePlane( m_maxSlopeCosine, m_up, i, constraints);
		}
	}
	// Resize array if it is now too small to accomodate the user planes.
	if (constraints.getCapacity() - constraints.getSize() < m_userPlanes )
	{
		constraints.reserve(constraints.getSize() + m_userPlanes);
	}

	// Interactions array - this is the output of the simplex solver
	hkLocalArray<hkSurfaceConstraintInteraction> interactions(constraints.getSize() + m_userPlanes);

	// Stored velocities - used to remember surface velocities to give the correct output surface velocity
	hkLocalArray<hkVector4> storedVelocities( constraints.getSize() + m_userPlanes );

	//
	//	Test Direction
	//
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

		input.m_maxSurfaceVelocity.setAll( m_maxCharacterSpeedForSolver );
		output.m_planeInteractions = interactions.begin();

		//
		// Allow the user to do whatever they wish with the surface constraints
		//
		fireConstraintsProcessed( m_manifold, input );

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

		hkSimplexSolverSolve( input, output );
	}

	ground.m_surfaceVelocity.setZero();
	ground.m_surfaceNormal.setZero();

	if ( output.m_velocity.allEqual<3>( direction, hkSimdReal::fromFloat(1e-3f) ) )
	{
		ground.m_supportedState = hkpSurfaceInfo::UNSUPPORTED;
	}
	else
	{
		if ( output.m_velocity.lengthSquared<3>().getReal() < .001f )
		{
			ground.m_supportedState = hkpSurfaceInfo::SUPPORTED;
		}
		else
		{
			output.m_velocity.normalize<3>();
			const hkSimdReal angleSin = output.m_velocity.dot<3>(direction);

			const hkSimdReal cosSqr = hkSimdReal_1 - angleSin * angleSin;

			if (cosSqr < hkSimdReal::fromFloat(m_maxSlopeCosine * m_maxSlopeCosine) )
			{
				ground.m_supportedState = hkpSurfaceInfo::SLIDING;
			}
			else
			{
				ground.m_supportedState = hkpSurfaceInfo::SUPPORTED;
			}
		}
	}

	if ( ground.m_supportedState != hkpSurfaceInfo::UNSUPPORTED )
	{
		int numTouching = 0;

		for (int i=0; i < input.m_numConstraints; i++)
		{
			// If we touched this plane and it supports our direction
			if ((interactions[i].m_touched) && constraints[i].m_plane.dot<3>(direction).getReal() < -0.08f)
			{
				ground.m_surfaceNormal.add( constraints[i].m_plane );
				ground.m_surfaceVelocity.add( storedVelocities[i] );
				numTouching++;
			}
		}

#ifdef DEBUG_CHARACTER_CONTROLLER
		HK_DISPLAY_ARROW(getPosition(),ground.m_surfaceNormal, 0xffffffff);
#endif

		if (numTouching > 0)
		{
			ground.m_surfaceNormal.normalize<3>();
			hkSimdReal nT; nT.setFromInt32(numTouching); nT.setReciprocal(nT);
			ground.m_surfaceVelocity.mul(nT);
		}
		else
		{
			ground.m_supportedState = hkpSurfaceInfo::UNSUPPORTED;
		}
	}

	// Not required by the proxy character controller.
	ground.m_surfaceDistanceExcess = 0.0f;

	HK_TIMER_END();
}

// Check and see if the character is supported in the give direction
void hkpCharacterProxy::checkSupportDeprecated(const hkVector4& direction, hkpSurfaceInfoDeprecated& ground) const
{
	HK_ASSERT2(0x79d57ec9,  hkMath::equal(direction.length<3>().getReal(), 1.0f), "checkSupport Direction should be normalized");
	HK_TIMER_BEGIN("checkSupport", HK_NULL);

	// We zero the velocity of all the planes, making this call a static geometric query
	hkLocalArray<hkSurfaceConstraintInfo> constraints(m_manifold.getSize() + m_userPlanes + 10);
	constraints.setSizeUnchecked(m_manifold.getSize());
	{
		for (int i=0; i < m_manifold.getSize() ; i++)
		{
			extractSurfaceConstraintInfo(m_manifold[i], constraints[i], 0);
			addMaxSlopePlane( m_maxSlopeCosine, m_up, i, constraints);
		}
	}

	// Resize array if it is now too small to accomodate the user planes.
	if (constraints.getCapacity() - constraints.getSize() < m_userPlanes )
	{
		constraints.reserve(constraints.getSize() + m_userPlanes);
	}

	// Magnitude of our velocity in the current direction
	// ToDo: Work with our current velocity
	hkLocalArray<hkSurfaceConstraintInteraction> interactions(constraints.getSize() + m_userPlanes);
	interactions.setSize(constraints.getSize() + m_userPlanes);


	//
	//	Test Direction
	//
	hkSimplexSolverInput input;
	hkSimplexSolverOutput output;
	{
		input.m_position.setZero();
		input.m_constraints = constraints.begin();
		input.m_numConstraints = m_manifold.getSize();
		input.m_velocity.setAdd(m_velocity,direction);
		input.m_deltaTime = 1.0f / 60.0f;
		input.m_minDeltaTime = 1.0f / 60.0f;
		input.m_upVector = m_up;

		input.m_maxSurfaceVelocity.setAll( m_maxCharacterSpeedForSolver );
		output.m_planeInteractions = interactions.begin();

		//
		// Allow the user to do whatever they wish with the surface constraints
		//
		fireConstraintsProcessed( m_manifold, input );

		hkSimplexSolverSolve( input, output );
	}

	output.m_velocity.sub(m_velocity);

	const hkSimdReal projectedVelocity = output.m_velocity.dot<3>(direction);

	// If our velocity was reduced then we hit something
	// HVK-2402
	ground.m_isSupported = projectedVelocity < hkSimdReal::fromFloat(0.99f);

	hkVector4 resultant;
	resultant.setAddMul(output.m_velocity, direction, -projectedVelocity);
	ground.m_isSliding = resultant.lengthSquared<3>() > hkSimdReal::fromFloat(0.01f);

	ground.m_surfaceVelocity.setZero();
	ground.m_surfaceNormal.setZero();

	int numTouching = 0;

	for (int i=0; i < input.m_numConstraints; i++)
	{
		// If we touched this plane and it supports our direction
		if ((interactions[i].m_touched) && constraints[i].m_plane.dot<3>(direction).getReal() < -0.08f)
		{
			ground.m_surfaceNormal.add( constraints[i].m_plane );
			ground.m_surfaceVelocity.add( constraints[i].m_velocity );
			numTouching++;
		}
	}

	if (numTouching > 0)
	{
		ground.m_surfaceNormal.normalize<3>();
		hkSimdReal nT; nT.setFromInt32(numTouching); nT.setReciprocal(nT);
		ground.m_surfaceVelocity.mul(nT);
	}
	else
	{
		ground.m_isSupported = false;
	}

	HK_TIMER_END();
}

const hkVector4& hkpCharacterProxy::getPosition() const
{
	return m_shapePhantom->getCollidable()->getTransform().getTranslation();
}

void hkpCharacterProxy::setPosition(const hkVector4& position)
{
	// Tolerance should be the same as the castInput.m_startPointTolerance used in integrateWithCollectors
	m_shapePhantom->setPosition(position, m_keepDistance + m_keepContactTolerance);
}


const hkVector4&	hkpCharacterProxy::getLinearVelocity() const
{
	return m_velocity ;
}

void hkpCharacterProxy::setLinearVelocity( const hkVector4& vel )
{
	m_velocity = vel;
}

hkpShapePhantom*	hkpCharacterProxy::getShapePhantom()
{
	return m_shapePhantom;
}

const hkpShapePhantom* hkpCharacterProxy::getShapePhantom() const
{
	return m_shapePhantom;
}

void hkpCharacterProxy::entityRemovedCallback( hkpEntity* entity )
{
	entity->removeEntityListener( this );

	if ( entity->hasProperty( HK_PROPERTY_TRIGGER_VOLUME ) )
	{
		hkpTriggerVolume* triggerVolume = static_cast<hkpTriggerVolume*>( entity->getProperty( HK_PROPERTY_TRIGGER_VOLUME ).getPtr() );
		const int index = m_overlappingTriggerVolumes.indexOf( triggerVolume );
		HK_ASSERT2( 0x49c85c9a, index != -1, "Could not find triggerVolume in overlaps array");
		m_overlappingTriggerVolumes.removeAt( index );
	}
	else
	{
		const int size = m_manifold.getSize();
		for (int p1 = size-1; p1 >= 0; p1--)
		{
			hkpRigidBody* body =  hkpGetRigidBody(m_manifold[p1].m_rootCollidableB);
			if (body == entity)
			{
				m_manifold.removeAt(p1);
			}
		}

		hkpRigidBody* oldBody = static_cast<hkpRigidBody*>(entity);
		const int index = m_bodies.indexOf( oldBody );
		HK_ASSERT2(0x49c85c9f, index != -1, "Could not find entity in manifold");
		if ( index != -1 )
		{
			m_bodies.removeAt( index );
		}
	}
}

void hkpCharacterProxy::phantomRemovedCallback( hkpPhantom* phantom )
{
	phantom->removePhantomListener( this );

	const int size = m_manifold.getSize();
	for (int p1 = size-1; p1 >= 0; p1--)
	{
		hkpPhantom* manifoldPhantom = hkpGetPhantom(m_manifold[p1].m_rootCollidableB);
		if (manifoldPhantom == phantom)
		{
			m_manifold.removeAt(p1);
		}
	}

	hkpPhantom* oldPhantom = static_cast<hkpPhantom*>(phantom);
	HK_ASSERT2(0x49c85c8f, m_phantoms.indexOf( oldPhantom )!=-1, "Could not find phantom in manifold");
	m_phantoms.removeAt( m_phantoms.indexOf(oldPhantom) );
}

void hkpCharacterProxy::addCharacterProxyListener( hkpCharacterProxyListener* cpl)
{
	HK_ASSERT2(0x5efeeea3, m_listeners.indexOf( cpl ) < 0, "You tried to add  a character proxy listener listener twice" );
	m_listeners.pushBack( cpl );
}

void hkpCharacterProxy::removeCharacterProxyListener( hkpCharacterProxyListener* cpl)
{
	const int i = m_listeners.indexOf( cpl );
	HK_ASSERT2(0x2c6b3925, i >= 0, "You tried to remove a character proxy listener, which was never added" );
	m_listeners.removeAt(i);
}

void hkpCharacterProxy::processTriggerVolumes( hkpTriggerVolume** triggerVolumeAndFlags, const int maxNumTriggerVolumes )
{
	hkArray<hkpTriggerVolume*> confirmedThisFrame;
	// Handle trigger volumes which the character has touched this frame.
	{
		int i = 0;
		while ( ( i < maxNumTriggerVolumes ) && ( *triggerVolumeAndFlags != HK_NULL ) )
		{
			// We've touched this trigger volume.
			{
				hkpTriggerVolume *const triggerVolume = HK_TRIGGER_VOLUME_GET_PTR( *triggerVolumeAndFlags );
				const int index = m_overlappingTriggerVolumes.indexOf( triggerVolume );
				if ( index == -1 )
				{
					if ( HK_TRIGGER_VOLUME_GET_TOUCHED( *triggerVolumeAndFlags ) )
					{
						confirmedThisFrame.pushBack( triggerVolume );
						triggerVolume->getTriggerBody()->addEntityListener( this );
						triggerVolume->triggerEventCallback( this, hkpTriggerVolume::ENTERED_EVENT );
					}
					else
					{
						triggerVolume->triggerEventCallback( this, hkpTriggerVolume::ENTERED_AND_LEFT_EVENT );
					}
				}
				else
				{
					m_overlappingTriggerVolumes.removeAt( index );
					if ( HK_TRIGGER_VOLUME_GET_TOUCHED( *triggerVolumeAndFlags ) )
					{
						confirmedThisFrame.pushBack( triggerVolume );
					}
					else
					{
						triggerVolume->getTriggerBody()->removeEntityListener( this );
						triggerVolume->triggerEventCallback( this, hkpTriggerVolume::LEFT_EVENT );
					}
				}
			}
			++i;
			++triggerVolumeAndFlags;
		}
		HK_WARN_ON_DEBUG_IF( ( maxNumTriggerVolumes > 0 ) && ( i == maxNumTriggerVolumes ), 0xb3ba2991, "Trigger volume storage is full, so some trigger volumes may have been missed during character proxy processing" );
	}
	// Handle trigger volumes which the character stopped touching this frame.
	{
		const int numRemainingTriggerVolumes = m_overlappingTriggerVolumes.getSize();
		for ( int i = 0; i < numRemainingTriggerVolumes; ++i )
		{
			m_overlappingTriggerVolumes[i]->getTriggerBody()->removeEntityListener( this );
			m_overlappingTriggerVolumes[i]->triggerEventCallback( this, hkpTriggerVolume::LEFT_EVENT );
		}
	}
	// Put the new overlaps into their array.
	m_overlappingTriggerVolumes.swap( confirmedThisFrame );
}

#endif //#if !defined(HK_PLATFORM_SPU)


void hkpCharacterProxy::fireCharacterInteraction( hkpCharacterProxy* otherProxy, const hkContactPoint& contact)
{	
#if defined(HK_PLATFORM_SPU)
	if( hkpSpuCharacterProxyUtil::s_characterInteractionCallback )
	{
		hkpSpuCharacterProxyUtil::s_characterInteractionCallback( this, otherProxy, contact );
	}
#else
	for ( int i = m_listeners.getSize()-1; i >= 0; i-- )
	{		
		m_listeners[i]->characterInteractionCallback( this, otherProxy, contact );
	}	
#endif
}

void hkpCharacterProxy::fireObjectInteraction( const hkpCharacterObjectInteractionEvent& input, hkpCharacterObjectInteractionResult& output)
{
#if defined(HK_PLATFORM_SPU)
	if ( hkpSpuCharacterProxyUtil::s_objectInteractionCallback )
	{
		hkpSpuCharacterProxyUtil::s_objectInteractionCallback( this, input, output );
	}
#else
	for ( int i = m_listeners.getSize()-1; i >= 0; i-- )
	{
		m_listeners[i]->objectInteractionCallback( this, input, output );
	}	
#endif
}

void hkpCharacterProxy::fireConstraintsProcessed( const HK_MANIFOLD_ARRAY_TYPE& manifold, hkSimplexSolverInput& input ) const
{	
#if defined(HK_PLATFORM_SPU)
	if ( hkpSpuCharacterProxyUtil::s_constraintsProcessedCallback )
	{
		hkpSpuCharacterProxyUtil::s_constraintsProcessedCallback( this, manifold, input );
	}
#else
	for ( int i = m_listeners.getSize()-1; i >= 0; i-- )
	{
		m_listeners[i]->processConstraintsCallback( this, m_manifold, input );
	}
#endif
}

void hkpCharacterProxy::fireContactAdded( const hkpRootCdPoint& point ) const
{
#if defined(HK_PLATFORM_SPU)
	if( hkpSpuCharacterProxyUtil::s_contactAddedCallback )
	{
		hkpSpuCharacterProxyUtil::s_contactAddedCallback( this, point );
	}
#else
	for ( int i = m_listeners.getSize()-1; i >= 0; i-- )
	{
		m_listeners[i]->contactPointAddedCallback( this, point );
	}
#endif
}

void hkpCharacterProxy::fireContactRemoved( const hkpRootCdPoint& point ) const
{
#if defined(HK_PLATFORM_SPU)
	if( hkpSpuCharacterProxyUtil::s_contactRemovedCallback )
	{
		hkpSpuCharacterProxyUtil::s_contactRemovedCallback( this, point );
	}
#else
	for ( int i = m_listeners.getSize()-1; i >= 0; i-- )
	{
		m_listeners[i]->contactPointRemovedCallback( this, point );
	}
#endif
}

// Main character update loop:
//	Loop until timestep complete
//		Collect information about my current position
//		update the manifold
//		solve the simplex to find new position and velocity
//		cast to new desired position
//		if (hit)
//			jump to hit position
//		else
//			jump to new desired position
//	grab velocity from simplex

void hkpCharacterProxy::integrateImplementation( const hkStepInfo& stepInfo
	, const hkVector4& gravity
	, hkpCharacterProxyIntegrateCommand* command
	HK_CPU_ARG( hkpAllCdPointCollector& castCollector )
	HK_CPU_ARG( hkpAllCdPointCollector& startCollector )
	HK_SPU_ARG( const hkpBroadPhase* broadphase )
	HK_SPU_ARG( hkpSpuWorldLinearCastCollector* broadphaseCollector )
	)
{
	HK_TIMER_BEGIN_LIST("updateCharacter", "Cast");

	hkpLinearCastInput castInput;
	{
		castInput.m_startPointTolerance = m_keepDistance + m_keepContactTolerance;
		castInput.m_maxExtraPenetration = 0.01f;
	}

#if defined(HK_PLATFORM_SPU)

	HK_ASSERT2( 0xefb3c8a9, command, "Only character proxy integrate jobs should call this function on SPU so there should be a command given." );
	HK_ASSERT2( 0xaf15e148, broadphase, "This job was not given a broadphase.");

	// Delimit the number of colliding bodies
	command->m_objectInteraction->m_collidingBody = HK_NULL;

	// Initialize arrays on SPU
	hkpSpuCharacterProxyUtil::ArrayData arrayData(this);
	hkArraySpu<hkpRootCdPoint>& manifold = arrayData.m_manifoldOnSpu;

	// Set command dependent broadphaseCollector parameters
	{
		broadphaseCollector->m_input.m_maxExtraPenetration	= castInput.m_maxExtraPenetration;
	}

	// Use the transform of the castCollidable from the broadphaseCollector.
	// Note: This collidable will later be used during the linear casting phase so its transform will be updated as the position changes.
	hkTransform& transform = const_cast<hkTransform&>( broadphaseCollector->m_castCollidableOnSpu->getTransform() );

	hkpFixedBufferCdPointCollector& castCollector = *broadphaseCollector->m_castCollector;
	hkpFixedBufferCdPointCollector& startCollector = *broadphaseCollector->m_startCollector;

	hkpSpuCharacterProxyCollisionCache collisionCache( broadphaseCollector->m_collidableCache );

	hkArrayBase<hkpTriggerVolume*> triggerVolumeAndFlags( command->m_triggerVolumeAndFlags, 0, command->m_maxTriggerVolumes );

#else
	hkTransform transform = m_shapePhantom->getTransform();

	const bool isMultithreaded = ( command != HK_NULL );
	hkpCollidable* collidable = HK_NULL;
	if( isMultithreaded )
	{
		// Delimit the number of colliding bodies
		command->m_objectInteraction->m_collidingBody = HK_NULL;

		// It is not thread safe to change the shape phantom's position, so create a local collidable whose position we may update later.
		// This will be used for world linear casts with an hkpCpuCharacterProxyCollector which filters out the character's collidable
		collidable = new hkpCollidable( m_shapePhantom->getCollidable()->getShape(), &transform );
		collidable->setCollisionFilterInfo( m_shapePhantom->getCollidable()->getCollisionFilterInfo() );
	}


	hkArray<hkpTriggerVolume*>::Temp triggerVolumeAndFlags;
	if ( isMultithreaded )
	{
		triggerVolumeAndFlags.setDataUserFree( command->m_triggerVolumeAndFlags, 0, command->m_maxTriggerVolumes);
	}
	else
	{
		triggerVolumeAndFlags.reserve( HK_MAX_NUM_TRIGGER_VOLUME_ST );
	}

	hkArray<hkpRootCdPoint>& manifold = m_manifold;
	hkArray<hkpRigidBody*>& bodies = m_bodies;
	hkArray<hkpPhantom*>& phantoms = m_phantoms;
#endif

	hkVector4& position = transform.getTranslation();

	hkReal remainingTime = stepInfo.m_deltaTime;
	
	hkSimplexSolverOutput output;

	for ( int iter = 0; (remainingTime > HK_REAL_EPSILON) && (iter < m_maxCastIterations) ; iter++ )
	{
		HK_TIMER_SPLIT_LIST("InitialCast");
		//
		//	Cast in a direction, hopefully the right one.
		//  We actually only really need the start point collector information.
		//  That means we are free to use any direction as a cast direction
		//  For optimizations, we cast into the direction of the old displacement
		//  (just a guess). That means we might be able to avoid a second cast;
		//  see below.
		//
		{
			castInput.m_to.setAdd( position, m_oldDisplacement );

			castCollector.reset();
			startCollector.reset();

			//	Call the caster
#if !defined(HK_PLATFORM_SPU)
			if ( isMultithreaded )
			{
				if ( iter )
				{
					hkpCpuCharacterProxyUtil::linearCastWorldCast( m_shapePhantom, castInput, collidable, castCollector, &startCollector );
				}
				else
				{
					hkpCpuCharacterProxyUtil::linearCastPhantomCast( this, castInput, castCollector, &startCollector );
				}
			}
			else
			{
				hkpCpuCharacterProxyUtil::linearCastSetPositionAndCast( m_shapePhantom, castInput, castCollector, &startCollector, position );
			}
#else
			hkpSpuCharacterProxyUtil::linearCast( castInput, broadphase, m_oldDisplacement, broadphaseCollector, true );
#endif
		}

		//
		// Handle trigger volumes if there are any.
		//
		HK_TIMER_SPLIT_LIST("TriggerVolumes");
		{
			const int numHits = startCollector.getNumHits();
			hkLocalArray<hkpTriggerVolume*> touchedTriggers( numHits );
			hkLocalArray<hkpRootCdPoint> triggerHits( numHits );
			removeTriggerVolumes( startCollector, touchedTriggers, triggerHits );
			updateTriggerVolumes( touchedTriggers, triggerVolumeAndFlags );
		}
	
		//
		// Maintain the internal manifold of contact points
		// See the method for detailed rules
		//
		{
			const int numHits = castCollector.getNumHits();
			hkLocalArray<hkpTriggerVolume*> touchedTriggers( numHits );
			hkLocalArray<hkpRootCdPoint> triggerHits( numHits );
			removeTriggerVolumes( castCollector, touchedTriggers, triggerHits );

			// Change the hit point so it reflects the real distance rather than a fraction
			if ( castCollector.hasHit() )
			{
				castCollector.sortHits();
				convertFractionToDistance( castCollector, m_oldDisplacement );
			}

			HK_TIMER_SPLIT_LIST("UpdateManifold");
			updateManifold( startCollector, castCollector, manifold HK_CPU_ARG( bodies ) HK_CPU_ARG( phantoms ) HK_CPU_ARG( isMultithreaded ) HK_SPU_ARG( &collisionCache ) );

			if ( castCollector.hasHit() )
			{
				// Add any trigger volumes in the cast collector nearer than the first cast hit.
				updateNearTriggerVolumes( touchedTriggers, triggerHits, triggerVolumeAndFlags, castCollector.getHits()[0].m_contact.getDistance() );
			}
		}

		//
		// Convert manifold points to plane equations
		//
#if defined DEBUG_CHARACTER_CONTROLLER
		debugCast(startCollector, castCollector);
#endif

		hkLocalArray<hkSurfaceConstraintInfo> surfaceConstraints(manifold.getSize() + m_userPlanes + 10);
		surfaceConstraints.setSizeUnchecked(manifold.getSize());
		{
			for( int i = 0; i < manifold.getSize() ; i++)
			{
				extractSurfaceConstraintInfo( manifold[i], surfaceConstraints[i], stepInfo.m_deltaTime - remainingTime HK_SPU_ARG( &collisionCache ) );
				addMaxSlopePlane( m_maxSlopeCosine, m_up, i, surfaceConstraints);
			}

#if defined DEBUG_CHARACTER_CONTROLLER
			for (int j=0; j < surfaceConstraints.getSize() ; j++)
			{
				HK_DISPLAY_PLANE(surfaceConstraints[j].m_plane, position, 0.5f, (j < m_manifold.getSize()) ? HK_DEBUG_CLEAN_SIMPLEX_MANIFOLD_COLOR : HK_DEBUG_VERTICAL_SIMPLEX_MANIFOLD_COLOR);
			}
#endif

		}

		// Resize array if it is now too small to accommodate the user planes.
		if (surfaceConstraints.getCapacity() - surfaceConstraints.getSize() < m_userPlanes )
		{
#if !defined(HK_PLATFORM_SPU)
			surfaceConstraints.reserve(surfaceConstraints.getSize() + m_userPlanes);
#else
			HK_WARN(0x38573269, "Cannot resize on SPU. Not all user planes will be used.");
#endif
		}

		//
		// Solve the simplex
		//
		HK_TIMER_SPLIT_LIST("SlexMove");
		hkSimplexSolverInput input;

		hkLocalArray<hkSurfaceConstraintInteraction> surfaceInteractions(surfaceConstraints.getSize() + m_userPlanes);
		surfaceInteractions.setSizeUnchecked(surfaceConstraints.getSize() + m_userPlanes);
		{
			input.m_constraints = surfaceConstraints.begin();
			input.m_numConstraints = surfaceConstraints.getSize();
			input.m_velocity = m_velocity;
			input.m_deltaTime = remainingTime;
			if( m_velocity.lengthSquared<3>().isEqualZero() )
			{
				input.m_minDeltaTime = 0.0f;
			}
			else
			{
				input.m_minDeltaTime = 0.5f * (m_keepDistance * m_velocity.lengthInverse<3>().getReal());
			}
			input.m_position.setZero();
			input.m_upVector = m_up;
			input.m_maxSurfaceVelocity.setZero(); input.m_maxSurfaceVelocity.setXYZ( m_maxCharacterSpeedForSolver );

			//
			// Allow the user to do whatever they wish with the surface constraints
			//
			fireConstraintsProcessed( manifold, input );

			output.m_planeInteractions = surfaceInteractions.begin();

			// Solve Simplex
			hkSimplexSolverSolve( input, output );
		}

#if defined DEBUG_CHARACTER_CONTROLLER
		{
			int extraConstraints = input.m_numConstraints - surfaceConstraints.getSize();
			// Display user planes at the centre of the character shape
			for (int i = 0; i < extraConstraints; i++)
			{
				HK_DISPLAY_PLANE(input.m_constraints[m_manifold.getSize()+i].m_plane, position , 0.5f, HK_DEBUG_USER_SIMPLEX_MANIFOLD_COLOR);
			}
		}
#endif
		//
		// Apply forces - e.g. Character hits dynamic objects
		//
		HK_TIMER_SPLIT_LIST("ApplySurf");

		applySurfaceInteractions( stepInfo, gravity, command, manifold HK_SPU_ARG( &collisionCache ) );
		
		//
		// Check whether we can walk to the new position the simplex has suggested
		//

		HK_TIMER_SPLIT_LIST("CastMove");
		{
			const hkpRootCdPoint* newHit = HK_NULL;

			// If the simplex has given an output direction different from the cast guess
			// we re-cast to check we can move there. There is no need to get the start points again.
			if( !m_oldDisplacement.allEqual<3>(output.m_position,hkSimdReal::fromFloat(1e-3f)) )
			{
				castInput.m_to.setAdd( position, output.m_position);
				castCollector.reset();

#if defined DEBUG_CHARACTER_CONTROLLER
				HK_ON_DEBUG( HK_DISPLAY_LINE(position, castInput.m_to, hkColor::CYAN) );
#endif		

#if !defined(HK_PLATFORM_SPU)
				if ( isMultithreaded )
				{
					hkpCpuCharacterProxyUtil::linearCastWorldCast( m_shapePhantom, castInput, collidable, castCollector, HK_NULL );
				}
				else
				{
					hkpCpuCharacterProxyUtil::linearCastSetPositionAndCast( m_shapePhantom, castInput, castCollector, HK_NULL, position );
				}
#else
				hkpSpuCharacterProxyUtil::linearCast( castInput, broadphase, output.m_position, broadphaseCollector, false );
#endif

				const int numHits = castCollector.getNumHits();

				if ( numHits )
				{
					// Extract trigger volumes from the cast collector.
					hkLocalArray<hkpTriggerVolume*> touchedTriggers( numHits );
					hkLocalArray<hkpRootCdPoint> triggerHits( numHits );
					removeTriggerVolumes( castCollector, touchedTriggers, triggerHits );

					// Have to sort *after* triggers have been removed.
					castCollector.sortHits();
					convertFractionToDistance( castCollector, output.m_position );

					//
					// If the new cast hit something we have to check if this is a new
					// surface, or one we have already passed to the simplex
					//
				
					// Find the first hit that isn't already in the manifold.
					while ( castCollector.hasHit() )
					{
						const hkpRootCdPoint& hit = castCollector.getHits()[0];
						if ( findSurface( hit, manifold HK_SPU_ARG( &collisionCache ) ) != -1 )
						{
							// We found it, so the simplex already took it into account
							// We ignore this point and test the others
#if defined(HK_PLATFORM_SPU)
							castCollector.removeFirstHitAndCopy();
#else
							castCollector.getHits().removeAtAndCopy(0);
#endif
						}
						else
						{
							// We haven't seen it before.
							HK_MANIFOLD_PUSH_BACK( manifold, hit );
#if !defined(HK_PLATFORM_SPU)
							// If we are not refreshing the manifold in check support, we need
							// to add the new phantom or entity as a listener if it is
							// not already in the manifold.
							if ( !m_refreshManifoldInCheckSupport && !isMultithreaded )
							{
								addEntityOrPhantom( hit.m_rootCollidableB, bodies, phantoms );
							}
#endif
							// Add any trigger volumes nearer than the hit.
							updateNearTriggerVolumes( touchedTriggers, triggerHits, triggerVolumeAndFlags, hit.m_contact.getDistance() );
							newHit = &hit;
							break;
						}
					}
				}
			}
			if( newHit )
			{
				remainingTime -= moveToLinearCastHitPosition(output, *newHit, castInput, position);
			}
			else
			{
				position.add(output.m_position);
				remainingTime -= output.m_deltaTime;
			}
			
			m_oldDisplacement = output.m_position;
		}
	}

	// Update with the output velocity from the simplex.
	m_velocity = output.m_velocity;

	// Delimit the trigger volumes, if there's room.
	if ( triggerVolumeAndFlags.getSize() < triggerVolumeAndFlags.getCapacity() )
	{
		*triggerVolumeAndFlags.end() = HK_NULL;
	}

	// Update the phantom with the new position
	// Note: If running multithreaded, we do not update the broadphase or position here as it is not thread safe.
	// The broadphase and position should be updated after the multithread phase.
#if defined(HK_PLATFORM_SPU)
	command->m_position = position;
#else
	if( isMultithreaded )
	{
		delete collidable;
		command->m_position = position;		
	}
	else
	{
		m_shapePhantom->setPosition(position, castInput.m_startPointTolerance);

		// If we're single-threaded, then we process the trigger volume hits now.
		processTriggerVolumes( triggerVolumeAndFlags.begin(), HK_MAX_NUM_TRIGGER_VOLUME_ST );
	}
#endif
	
	// We have to clear the wrapper array to avoid an assert.
	triggerVolumeAndFlags.clear();

	HK_TIMER_END_LIST();
}

// Verify that the surface info is not the same as one we previously
// have in the manifold.

int hkpCharacterProxy::findSurface(const hkpRootCdPoint& info
	, HK_MANIFOLD_ARRAY_TYPE& manifold
	HK_SPU_ARG( hkpSpuCharacterProxyCollisionCache* collisionCache )
	) const
{
	int bestIndex = -1;
	hkReal minDistance = 0.1f;
	for (int i=0; i< manifold.getSize(); i++)
	{
		hkReal distance = surfaceDistance( info, manifold[i] HK_SPU_ARG( collisionCache ) );
		
		if ( distance < minDistance )
		{
			minDistance = distance;
			bestIndex = i;
		}
	}
	return bestIndex;
}

void hkpCharacterProxy::removeTriggerVolumes( HK_COLLECTOR_TYPE& collector, hkLocalArray<hkpTriggerVolume*>& touchedTriggers, hkLocalArray<hkpRootCdPoint>& triggerHits )
{
#if defined (HK_PLATFORM_SPU)
	const hkpRootCdPoint* hit = &collector.getHits()[0];
#else
	const hkpRootCdPoint* hit = collector.getHits().begin();
#endif
	for ( int hitIndex = collector.getNumHits()-1; hitIndex >= 0; --hitIndex )
	{
#if defined( HK_PLATFORM_SPU )
		hkpTriggerVolume* triggerVolume = static_cast<hkpTriggerVolume*>( hkpSpuCharacterProxyUtil::getPointerFromWorldObjectProperty( hit[hitIndex].m_rootCollidableB, HK_PROPERTY_TRIGGER_VOLUME ) );
#else
		hkpTriggerVolume* triggerVolume = HK_NULL;
		hkpWorldObject* worldObject = reinterpret_cast<hkpWorldObject*>( hit[hitIndex].m_rootCollidableB->getOwner() );
		if ( worldObject->hasProperty( HK_PROPERTY_TRIGGER_VOLUME ) )
		{
			triggerVolume = static_cast<hkpTriggerVolume*>( worldObject->getProperty( HK_PROPERTY_TRIGGER_VOLUME ).getPtr() );
		}
#endif

		if ( triggerVolume )
		{
			if ( touchedTriggers.indexOf( triggerVolume ) == -1 )
			{
				touchedTriggers.pushBackUnchecked( triggerVolume );
				triggerHits.pushBackUnchecked( hit[hitIndex] );
			}
			// Remove element from collector.
#if defined( HK_PLATFORM_SPU )
			collector.removeHit(hitIndex);
#else
			collector.getHits().removeAt(hitIndex);
#endif
		}
	}
}

void hkpCharacterProxy::updateTriggerVolumes( hkLocalArray<hkpTriggerVolume*>& touchedTriggers, hkArrayBase<hkpTriggerVolume*>& triggerVolumeAndFlags )
{
	const int numCastTriggers = touchedTriggers.getSize();
	const int numVolumes = triggerVolumeAndFlags.getSize();

	// Identify those trigger volumes which have been touched.
	for ( int i = 0; i < numCastTriggers; ++i )
	{
		hkpTriggerVolume* triggerVolume = touchedTriggers[i];
		triggerVolumeTouched( triggerVolume, triggerVolumeAndFlags );
	}
	
	// Identify those trigger volumes which have been left.
	// We don't need to check the volumes we've added, so can use the old numVolumes.
	for ( int i = 0; i < numVolumes; ++i )
	{
		if ( touchedTriggers.indexOf( HK_TRIGGER_VOLUME_GET_PTR( triggerVolumeAndFlags[i] ) ) == -1 )
		{
			triggerVolumeAndFlags[i] = HK_TRIGGER_VOLUME_LEFT( triggerVolumeAndFlags[i] );
		}
	}
}

void hkpCharacterProxy::triggerVolumeTouched( hkpTriggerVolume* triggerVolume, hkArrayBase<hkpTriggerVolume*>& triggerVolumeAndFlags )
{
	hkBool found = false;
	const int numVolumes = triggerVolumeAndFlags.getSize();
	for ( int j = 0; j < numVolumes; ++j )
	{
		if ( HK_TRIGGER_VOLUME_GET_PTR( triggerVolumeAndFlags[j] ) == triggerVolume )
		{
			triggerVolumeAndFlags[j] = HK_TRIGGER_VOLUME_TOUCHED( triggerVolumeAndFlags[j] );
			found = true;
		}
	}
	if ( !found )
	{
		HK_TRIGGER_VOLUME_PUSH_BACK( triggerVolumeAndFlags, HK_TRIGGER_VOLUME_TOUCHED( triggerVolume ) );
	}
}

void hkpCharacterProxy::updateNearTriggerVolumes( hkLocalArray<hkpTriggerVolume*>& touchedTriggers, hkLocalArray<hkpRootCdPoint>& triggerHits, hkArrayBase<hkpTriggerVolume*>& triggerVolumeAndFlags, hkReal distance )
{
	const int numTriggers = triggerHits.getSize();
	for ( int i = 0; i < numTriggers; ++i )
	{
		const hkpRootCdPoint& hit = triggerHits[i];
		if ( hit.m_contact.getDistance() < distance )
		{
			triggerVolumeTouched( touchedTriggers[i], triggerVolumeAndFlags );
		}
	}
}

// Maintain the manifold of plane equations we pass on to the simple solver
// This forms the bridge between the collision detector and the simplex solver
//
// Manifold Rules:
//	- All moving planes found at start are kept.
// - All penetrating planes found at start are kept.
// - Cast Plane is always kept.
// - All other currently held planes must verify they are present in the start collector.
HK_DISABLE_OPTIMIZATION_VS2008_X64
void hkpCharacterProxy::updateManifold(	const HK_COLLECTOR_TYPE& startCollector
	, const HK_COLLECTOR_TYPE&	castCollector
	, HK_MANIFOLD_ARRAY_TYPE& manifold
	HK_CPU_ARG( hkArray<hkpRigidBody*>& bodies )
	HK_CPU_ARG( hkArray<hkpPhantom*>& phantoms )
	HK_CPU_ARG( hkBool isMultithreaded )
	HK_SPU_ARG( hkpSpuCharacterProxyCollisionCache* collisionCache )
	)
{
	// Remove listener from all bodies and phantoms.
	// Note: adding/removing listeners from bodies and phantoms is not thread safe.
#if !defined(HK_PLATFORM_SPU)
	if( !isMultithreaded )
	{
		for (int i=0 ; i < bodies.getSize(); i++)
		{
			bodies[i]->removeEntityListener(this);
		}		

		for (int j=0; j< phantoms.getSize(); j++)
		{
			phantoms[j]->removePhantomListener(this);
		}
		bodies.clear();
		phantoms.clear();
	}
#endif

	// This is used to add the closest point always to the manifold.
	hkSimdReal minimumPenetration = hkSimdReal_Max;

	//
	// Copy hits from start point collector
	//
	const int numHits = startCollector.getNumHits();
	hkLocalArray<hkpRootCdPoint> startPointHits(numHits);
	startPointHits.setSizeUnchecked(numHits);
	
	for (int i=0; i< startPointHits.getSize(); i++)
	{
		startPointHits[i] = startCollector.getHits()[i];

		// We only consider fixed or keyframed rigid bodies.
		if (startPointHits[i].m_contact.getDistanceSimdReal() < minimumPenetration )
		{
			minimumPenetration = startPointHits[i].m_contact.getDistanceSimdReal();
		}
	}

	//
	//	For each existing point in the manifold -
	//		find it in the start point collector
	//		if found, copy the start point collector point over the manifold point
	//		otherwise drop the manifold point
	//
	{
		for (int s = manifold.getSize() - 1; s >= 0; s--)
		{
			int bestIndex = -1;
			hkReal minDistance = 1.1f;
			hkpRootCdPoint& current = manifold[s];

			// Find the best match for this plane
			for (int h=0; h < startPointHits.getSize(); h++)
			{
				hkpRootCdPoint& surface = startPointHits[h];
				hkReal distance = surfaceDistance( surface, current HK_SPU_ARG( collisionCache ) );
				if ( distance < minDistance )
				{
					minDistance = distance;
					bestIndex = h;
				}
			}

			// Plane already exists in manifold so we update and remove from the collector
			if ( bestIndex >= 0 )
			{
				hkpRootCdPoint& surface = startPointHits[bestIndex];
				if (surface.m_rootCollidableB != current.m_rootCollidableB)
				{
					fireContactRemoved( current);
					fireContactAdded( surface );
				}

				current = surface;
				startPointHits.removeAt( bestIndex );
			}
			else
			{
				//
				//	No matching plane in start point collector - we remove this from the manifold
				//
				fireContactRemoved( manifold[s] );
				manifold.removeAt(s);
			}
		}
	}

	//
	// Add the most penetrating point from the start point collector if it is still in the
	// collector (i.e. if it is not already in the manifold). This is safe, as the closest
	// point can never be an unwanted edge.
	//
	{
		for ( int h = 0; h < startPointHits.getSize(); h++ )
		{
			hkpRootCdPoint& surface = startPointHits[h];
			// Keep the plane if it is the the most penetrating plane
			if ( surface.m_contact.getDistanceSimdReal().isEqual(minimumPenetration) )
			{
				//
				// Find existing plane
				//
				int index = findSurface( surface, manifold HK_SPU_ARG( collisionCache ) );
				if (index >= 0)
				{
					hkpRootCdPoint& current = manifold[index];

					if (surface.m_rootCollidableB != current.m_rootCollidableB)
					{
						fireContactRemoved( current );
						fireContactAdded( surface );
					}

					current = surface;
				}
				else 
				{
					HK_MANIFOLD_PUSH_BACK( manifold, startPointHits[h] );
				}
			}
		}
	}

	//
	// Add castCollector plane
	//
	if (castCollector.hasHit())
	{
		const hkpRootCdPoint& surface = castCollector.getHits()[0];
		int index = findSurface( surface, manifold HK_SPU_ARG( collisionCache ) );

		if ( index == -1 )
		{
			HK_MANIFOLD_PUSH_BACK( manifold, surface );
		}

		// NOTE: We used to update the manifold with the new point from the cast collector here, but in fact this
		// is unnecessary and sometimes wrong. All points in the manifold have been correctly updated at this stage
		// by the start point collector so they do not need to be replaced here. If the points are penetrating, then
		// the cast collector will have a distance of 0, which is incorrect, and the negative distance picked up by
		// the start collector is the one that we want.
	}

	//
	// Cross check the manifold for clean planes
	// The simplex does not handle parallel planes
	//
	{
		for (int p1 = manifold.getSize()-1; p1 > 0; p1--)
		{
			hkBool remove = false;
			for (int p2 = p1-1; p2 >= 0; p2--)
			{
				// If p1 and p2 are the same then we should remove p1
				const hkReal minDistance = 0.1f;
				hkReal distance = surfaceDistance( manifold[p1], manifold[p2] HK_SPU_ARG( collisionCache ) );
				if ( distance < minDistance )
				{
					remove = true;
					break;
				}
			}
			if ( remove )
			{
				fireContactRemoved( manifold[p1] );
				manifold.removeAt( p1 );
			}
		}
	}

#if !defined( HK_PLATFORM_SPU )
	// add bodies/phantoms and listeners from the manifold
	if ( !isMultithreaded )
	{
		for( int i = 0; i < manifold.getSize(); i++ )
		{
			addEntityOrPhantom( manifold[i].m_rootCollidableB, bodies, phantoms );
		}
	}
#endif
}

HK_RESTORE_OPTIMIZATION_VS2008_X64

void hkpCharacterProxy::applySurfaceInteractions( const hkStepInfo& stepInfo
	, const hkVector4& worldGravity
	, hkpCharacterProxyIntegrateCommand* command
	, HK_MANIFOLD_ARRAY_TYPE& manifold
	HK_SPU_ARG( hkpSpuCharacterProxyCollisionCache* collisionCache )
	)
{
#if defined(HK_PLATFORM_SPU)
	hkpCharacterProxyInteractionResults* objectInteractionPtr = command->m_objectInteraction;
#else	
	const hkBool isMultithreaded = ( command != HK_NULL );
	hkpCharacterProxyInteractionResults* objectInteractionPtr = isMultithreaded ? command->m_objectInteraction : HK_NULL;
#endif

#if !defined(HK_PLATFORM_SPU)	
	if( isMultithreaded )
#endif
	{
		if( command->m_maxInteractions > 0 )
		{
			while( objectInteractionPtr->m_collidingBody )
			{
				objectInteractionPtr++;
			}
		}
	}

	hkpCharacterObjectInteractionEvent input;
	input.m_timestep = stepInfo.m_deltaTime;

	for ( int s = 0; s < manifold.getSize() ; s++)
	{
		//
		// Check if we have hit another character
		//				
		hkpCharacterProxy* otherChar = HK_NULL;
#if defined(HK_PLATFORM_SPU)
//		otherChar = hkpSpuCharacterProxyUtil::getCharacterProxy( manifold[s].m_rootCollidableB );
		otherChar = static_cast<hkpCharacterProxy*>( hkpSpuCharacterProxyUtil::getPointerFromWorldObjectProperty( manifold[s].m_rootCollidableB, HK_PROPERTY_CHARACTER_PROXY ) );
#else
		hkpWorldObject* object = hkpGetWorldObject( manifold[s].m_rootCollidableB );
		if(object->hasProperty(HK_PROPERTY_CHARACTER_PROXY))
		{
			otherChar = reinterpret_cast<hkpCharacterProxy*>( object->getProperty(HK_PROPERTY_CHARACTER_PROXY).getPtr() );
		}
#endif

		if( otherChar )
		{
			//
			// Callback to indicate characters have collided
			//
			fireCharacterInteraction( otherChar, manifold[s].m_contact);
			continue;
		}

		// If multithreaded, check if we have any storage left for interactions
#if !defined(HK_PLATFORM_SPU)
		if( isMultithreaded )
#endif
		{
			if( command->m_maxInteractions == 0 )
			{
				continue;
			}
		}		

		//
		// Check if we have hit a rigid body
		//
#if defined(HK_PLATFORM_SPU)		
// 		hkpSpuCharacterProxyUtil::CollidableData collisionDataOnSpu( manifold[s].m_rootCollidableB );
// 		hkpRigidBody* body =  collisionDataOnSpu.m_bodyOnSpu;
		hkpRigidBody* body = collisionCache->getRigidBodyOnSpu( manifold[s].m_rootCollidableB );
#else		
		hkpRigidBody* body = hkpGetRigidBody( manifold[s].m_rootCollidableB );
#endif

		if ( ( body != HK_NULL ) && ( !body->isFixedOrKeyframed() ) )
		{
			//
			//	Some constants used
			//
			const hkSimdReal recoveryTau = hkSimdReal::fromFloat(0.4f);
			const hkSimdReal dampFactor = hkSimdReal::fromFloat(0.9f);

			input.m_position.setXYZ_W( manifold[s].m_contact.getPosition(), manifold[s].m_contact.getDistanceSimdReal() );
			input.m_normal  = manifold[s].m_contact.getNormal();
#if defined(HK_PLATFORM_SPU)
//			input.m_body = collisionDataOnSpu.m_bodyOnPpu;
			input.m_body = collisionCache->getRigidBodyOnPpu( manifold[s].m_rootCollidableB );
#else
			input.m_body = body;
#endif

			//
			//	Calculate required velocity change
			//
			hkSimdReal deltaVelocity;
			{
				hkVector4 pointVel; body->getPointVelocity( input.m_position, pointVel );
				pointVel.sub( m_velocity );

				const hkSimdReal projectedVelocity = pointVel.dot<3>( input.m_normal );
				projectedVelocity.store<1,HK_IO_NATIVE_ALIGNED>(&input.m_projectedVelocity);

				deltaVelocity = -projectedVelocity * dampFactor;

				// Only apply an extra impulse if the collision is actually penetrating. HVK-1903
				if ( input.m_position.getW().isLessZero() )
				{
					deltaVelocity.add( input.m_position.getW() * hkSimdReal::fromFloat(stepInfo.m_invDeltaTime) * recoveryTau );
				}
			}

			//
			// Initialize the output result
			//
			hkpCharacterObjectInteractionResult output;
			output.m_impulsePosition = input.m_position;
			output.m_objectImpulse.setZero();


			if (deltaVelocity.isLessZero())
			{
				//
				//	Calculate the impulse required
				//
				{
					hkMatrix3 inertiaInv;
#if defined(HK_PLATFORM_SPU)
					hkpSpuCharacterProxyUtil::getInertiaInvWorld( body, inertiaInv );
#else
					body->getInertiaInvWorld( inertiaInv );
#endif

					hkVector4 r; r.setSub( input.m_position, body->getCenterOfMassInWorld() );
					hkVector4 jacAng; jacAng.setCross( r, input.m_normal );
					hkVector4 rc; rc._setRotatedDir( inertiaInv, jacAng );

					const hkSimdReal objMassInv = rc.dot<3>( jacAng ) + body->getRigidMotion()->getMassInv();
					input.m_objectMassInv = objMassInv.getReal();
					input.m_objectImpulse = (deltaVelocity / objMassInv).getReal();
				}

				hkReal maxPushImpulse = - m_characterStrength * stepInfo.m_deltaTime;
				if (input.m_objectImpulse < maxPushImpulse)
				{
					input.m_objectImpulse = maxPushImpulse;
				}

				output.m_objectImpulse.setMul(hkSimdReal::fromFloat(input.m_objectImpulse), input.m_normal );
			}
			else
			{
				input.m_objectImpulse = 0.0f;
				input.m_objectMassInv = body->getMassInv();
			}


			// Add gravity
			{
				hkReal deltaTime = stepInfo.m_deltaTime;
				hkVector4 charVelDown; charVelDown.setMul(hkSimdReal::fromFloat(deltaTime), worldGravity);

				// Normal points from object to character
				hkReal relVel = charVelDown.dot<3>(input.m_normal).getReal();

				if (input.m_projectedVelocity < 0 ) // if objects are separating
				{
					relVel -= input.m_projectedVelocity;
				}

				if (relVel < -HK_REAL_EPSILON)
				{
					output.m_objectImpulse.addMul(hkSimdReal::fromFloat(relVel * m_characterMass), input.m_normal);
				}
			}


			//
			// Callback to allow user to change impulse + use the info / play sounds
			//
			fireObjectInteraction( input, output );
			

			//
			//	Apply impulse based on callback result
			//
			if( !output.m_objectImpulse.lengthSquared<3>().approxEqual(hkSimdReal_0,hkSimdReal::fromFloat(1e-7f) ) )
			{
#if !defined(HK_PLATFORM_SPU)
				if( isMultithreaded )
#endif
				{
#if defined(HK_PLATFORM_SPU)
//					objectInteractionPtr->m_collidingBody = collisionDataOnSpu.m_bodyOnPpu;
					objectInteractionPtr->m_collidingBody = collisionCache->getRigidBodyOnPpu( manifold[s].m_rootCollidableB );
#else
					objectInteractionPtr->m_collidingBody = body;					
#endif
					objectInteractionPtr->m_impulsePosition = output.m_impulsePosition;
					objectInteractionPtr->m_objectImpulse = output.m_objectImpulse;
					objectInteractionPtr++;

					command->m_maxInteractions--;
					if( command->m_maxInteractions )
					{	
						// Set the current colliding body pointer to null to delimit the number of interactions if it is less than the max.
						objectInteractionPtr->m_collidingBody = HK_NULL;
					}
					else
					{
						continue;
					}
				}
#if !defined(HK_PLATFORM_SPU)
				else
				{
					body->applyPointImpulse( output.m_objectImpulse, output.m_impulsePosition );
				}
#endif
#if defined DEBUG_CHARACTER_CONTROLLER
				HK_ON_DEBUG( HK_DISPLAY_ARROW(input.m_position, output.m_objectImpulse, 0xffffffff) );
#endif
			}
		}
	}
}

// this function subtracts the keep distance (projected onto the cast vector) from the cast output.
hkReal hkpCharacterProxy::moveToLinearCastHitPosition( const hkSimplexSolverOutput& output, const hkpRootCdPoint& hit, const hkpLinearCastInput& castInput, hkVector4& position )
{
	// Only walk as far as the cast allows.
	const hkVector4& displacement = output.m_position;
	const hkSimdReal displacementLength = displacement.length<3>();
	HK_ASSERT2(0x5ea11036, displacementLength.getReal() > HK_REAL_EPSILON, "LinearCast of zero length picked up new contact points");
	const hkContactPoint& firstContact = hit.m_contact;

	const hkSimdReal angleBetweenMovementAndSurface = displacement.dot<3>( firstContact.getNormal() ) / displacementLength;
	const hkSimdReal keepDistanceAlongMovement = hkSimdReal::fromFloat(m_keepDistance) / -angleBetweenMovementAndSurface;
	hkSimdReal fraction = firstContact.getPosition().getW() - (keepDistanceAlongMovement / displacementLength);
	// If an object is within m_keepDistance e.g. embedded in the character,
	// then we can't move forward but we also do not wish the character to
	// move backwards.
	fraction.setMax(hkSimdReal_0, fraction);
	fraction.setMin(hkSimdReal_1, fraction);

	position.setInterpolate( position, castInput.m_to, fraction );

	hkReal timeTravelled = output.m_deltaTime * fraction.getReal();
	return timeTravelled;
}



void hkpCharacterProxy::extractSurfaceConstraintInfo(const hkpRootCdPoint& hit
	, hkSurfaceConstraintInfo& surface
	, hkReal timeTravelled
	HK_SPU_ARG( hkpSpuCharacterProxyCollisionCache* collisionCache )
	) const
{
	surface.m_plane = hit.m_contact.getSeparatingNormal();

	// Contract the planes by the keep distance
	surface.m_plane(3) -= m_keepDistance;

	surface.m_staticFriction = m_staticFriction;
	surface.m_dynamicFriction = m_dynamicFriction;
	surface.m_extraUpStaticFriction = m_extraUpStaticFriction;
	surface.m_extraDownStaticFriction = m_extraDownStaticFriction;

	// Assume the velocity of this surface is 0
	surface.m_velocity.setZero();

	// Assume this is a low priority surface
	surface.m_priority = 0;

	// Grab body information
	
#if defined(HK_PLATFORM_SPU)
// 	hkpSpuCharacterProxyUtil::CollidableData collisionDataOnSpu( hit.m_rootCollidableB );
// 	hkpRigidBody* body =  collisionDataOnSpu.m_bodyOnSpu;
	hkpRigidBody* body = collisionCache->getRigidBodyOnSpu( hit.m_rootCollidableB );
#else
	hkpRigidBody* body = hkpGetRigidBody(hit.m_rootCollidableB);
#endif

	if (body)
	{
		// Extract point velocity

		// HVK-1871. This code gets the point velocity at the collision, based on how far
		// the object actually travelled, rather than the velocity result of the constraint solver.
		// (i.e. the value got from getPointVelocity)
		// When a heavy force pushes a rigid body into a fixed rigid body these values can diverge,
		// which can cause the character controller to penetrate the moving rigid body, as it sees
		// an incorrectly moving plane.

		// Note, this means that velocities will be one frame behind, so for accelerating platforms
		// (HVK-1477) (i.e. For keyframed or fixed objects) we just take the velocity, to make sure the
		// character does not sink in.
		if (body->isFixedOrKeyframed())
		{
			body->getPointVelocity(hit.m_contact.getPosition(), surface.m_velocity);
		}
		else
		{
			hkVector4 linVel;
			hkVector4 angVel;
			hkSweptTransformUtil::getVelocity(*body->getRigidMotion()->getMotionState(), linVel, angVel);
			hkVector4 relPos; relPos.setSub( hit.m_contact.getPosition(), body->getCenterOfMassInWorld() );
			surface.m_velocity.setCross( angVel, relPos);
			surface.m_velocity.add( linVel );
		}


		// Move the plane by the velocity, based on the timeTravelled HVK-1477
		surface.m_plane(3) -= surface.m_velocity.dot<3>(surface.m_plane).getReal() * timeTravelled;


		// Extract priority
		// - Static objects have highest priority
		// - Then keyframed
		// - Then dynamic
		if (body->getMotionType() == hkpMotion::MOTION_FIXED)
		{
			// Increase the priority
			surface.m_priority = 2;
		}

		if (body->getMotionType() == hkpMotion::MOTION_KEYFRAMED)
		{
			// Increase the priority
			surface.m_priority = 1;
		}
	}

	// If penetrating we add extra velocity to push the character back out
	if ( surface.m_plane.getW() < -hkSimdReal_Eps)
	{
#if defined DEBUG_CHARACTER_CONTROLLER
		HK_ON_DEBUG( HK_DISPLAY_ARROW(hit.m_contact.getPosition(),  hit.m_contact.getNormal(), hkColor::RED) );
#endif
		surface.m_velocity.addMul(-surface.m_plane.getW() * hkSimdReal::fromFloat(m_penetrationRecoverySpeed), hit.m_contact.getNormal());
		surface.m_plane.zeroComponent<3>();
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
