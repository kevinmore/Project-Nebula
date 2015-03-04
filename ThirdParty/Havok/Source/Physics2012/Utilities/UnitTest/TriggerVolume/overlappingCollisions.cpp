/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Utilities/Collide/TriggerVolume/hkpTriggerVolume.h>
#include <Physics2012/Collide/Dispatch/hkpAgentRegisterUtil.h>

#include <Physics2012/Utilities/Collide/hkpShapeGenerator.h>

class OverlappingBodiesTestTriggerVolume : public hkpTriggerVolume
{
	public:
		OverlappingBodiesTestTriggerVolume ( hkpRigidBody* triggerBody ) 
			: hkpTriggerVolume( triggerBody )
		{

		}
		
		using hkpTriggerVolume::triggerEventCallback;

		virtual void triggerEventCallback( hkpRigidBody* body, EventType type )
		{
			if ( type & ENTERED_EVENT )
			{
				m_bodiesTouched.pushBack( body );
			}
			if ( type & LEFT_EVENT )
			{
				m_bodiesLeft.pushBack( body );
			}
		}

	public:
		hkArray<hkpRigidBody*> m_bodiesTouched;
		hkArray<hkpRigidBody*> m_bodiesLeft;
};

// Creates a rigid body for use in this test.
static hkpRigidBody* createOverlappingBody( hkReal width, hkReal height, hkReal x, hkReal y )
{
	hkpRigidBody* body;

	hkpShape* shape;
	{
		hkVector4 halfExtents;
		halfExtents.set( width * 0.5f, height * 0.5f, 1.0f );
		shape = hkpShapeGenerator::createConvexVerticesBox( halfExtents );
	}

	hkpRigidBodyCinfo info;
	info.m_shape = shape;
	info.m_mass = 1.0f;
	info.m_motionType = hkpMotion::MOTION_BOX_INERTIA;
	info.m_position.set( x + ( width * 0.5f ), y + ( height * 0.5f ), 0.0f );
	info.m_rotation.setIdentity();
	info.m_qualityType = HK_COLLIDABLE_QUALITY_CRITICAL;
	info.m_gravityFactor = 0.0f;
	info.m_enableDeactivation = false;

	body = new hkpRigidBody( info );

	shape->removeReference();

	return body;
}


static void testOverlappingCollisions( hkpWorld* world, hkReal deltaTime )
{
	world->lock();
	// Create a set of bodies moving slowly downwards.
	const hkReal heightUnit = 1.0f;
	const hkReal widthUnit = 1.0f;
	const hkReal maxHeight = 4.0f;
	const hkReal heightOfTriggerVolume = heightUnit;
	const hkReal gap = heightUnit;
	const hkReal distance = ( maxHeight * 2.0f ) + ( gap * 2.0f ) + heightOfTriggerVolume;
	const int numSteps = 400;
	const hkReal speed = distance / ( numSteps * deltaTime );

	hkReal x = 0.0f;
	hkReal currentHeight = heightUnit;

	hkArray<hkpRigidBody*> overlappingBodies;

	while ( currentHeight <= maxHeight )
	{
		hkReal y = 0.0f;
		while ( y + currentHeight <= maxHeight )
		{
			// Create the body.
			{
				hkpRigidBody *const body = createOverlappingBody( widthUnit, currentHeight, x, y );
				world->addEntity( body );
				hkVector4 velocity;
				velocity.set( 0.0f, speed, 0.0f );
				body->setLinearVelocity( velocity );
				overlappingBodies.pushBack( body );
			}

			x += 2.0f * widthUnit;
			y += heightUnit * 0.5f;
		}
		currentHeight += heightUnit;
	}
	
	// Create a long thin trigger volume that they all pass through.
	hkpRigidBody *const triggerBody = createOverlappingBody( x, heightOfTriggerVolume, 0.0f, maxHeight + gap );
	OverlappingBodiesTestTriggerVolume* triggerVolume = new OverlappingBodiesTestTriggerVolume( triggerBody );
	world->addEntity( triggerBody );

	const int numBodies = overlappingBodies.getSize();

	for ( int i = 0; i < numSteps; ++i )
	{
		world->unlock();
		world->stepDeltaTime( deltaTime );
		world->lock();
		for ( int j = 0; j < numBodies; ++j )
		{
			// Confirm the objects are neither deflected nor rotated.
			HK_TEST( hkMath::fabs( overlappingBodies[j]->getLinearVelocity()( 0 ) ) < 0.01f );
			HK_TEST( hkMath::fabs( overlappingBodies[j]->getLinearVelocity()( 1 ) - speed ) < 0.01f );
			HK_TEST( hkMath::fabs( overlappingBodies[j]->getLinearVelocity()( 2 ) ) < 0.01f );
			HK_TEST( overlappingBodies[j]->getAngularVelocity().lengthSquared<3>().getReal() < 0.01f );
		}
	}

	// Check that all bodies got the correct collisions
	for ( int i = 0; i < numBodies; ++i )
	{
		{
			const int indexTouched = triggerVolume->m_bodiesTouched.indexOf( overlappingBodies[i] );
			HK_TEST( indexTouched != -1 );
			if ( indexTouched != -1 )
			{
				triggerVolume->m_bodiesTouched.removeAt( indexTouched );
			}
		}
		{
			const int indexLeft = triggerVolume->m_bodiesLeft.indexOf( overlappingBodies[i] );
			HK_TEST( indexLeft != -1 );
			if ( indexLeft != -1 )
			{
				triggerVolume->m_bodiesLeft.removeAt( indexLeft );
			}
		}
	}
	HK_TEST( triggerVolume->m_bodiesTouched.getSize() == 0 );
	HK_TEST( triggerVolume->m_bodiesLeft.getSize() == 0 );

	// Check our reference count for the bodies in accurate.
	for ( int i = 0; i < numBodies; ++i )
	{
		HK_TEST( overlappingBodies[i]->getReferenceCount() == 2 );
	}

	world->removeEntity( triggerBody );
	triggerVolume->removeReference();
	triggerBody->removeReference();
	for ( int i = 0; i < numBodies; ++i )
	{
		world->removeEntity( overlappingBodies[i] );
		overlappingBodies[i]->removeReference();
	}
	world->unlock();
}

static void testOverlappingBodiesCollisionsWithShapeUpdates( hkpWorld* world, hkReal deltaTime )
{
	// Checks that:
	// [HVK-6268] - Calling hkpRigidBody::updateShape() or hkpRigidBody::setShape() on a body inside
	// a trigger volume should keep the body as 'Overlapping' and should not Assert/Crash.

	world->lock();

	const int numSteps = 100; // Number of 'hkpWorld::stepDeltaTime()' that will be taken in this test.
	
	const hkReal heightUnit = 1.0f;
	const hkReal widthUnit = 1.0f;
	const hkReal triggerVolumeScale = 20.0f;
	const hkReal distance = -2.5f * triggerVolumeScale * heightUnit; // distance ran by the bodies we'll create.
	const hkReal speed = distance / ( numSteps * deltaTime ); // speed of the bodies we'll create.

	// Create a set of bodies moving slowly downwards as well as a OverlappingBodiesTestTriggerVolume.
	hkArray<hkpRigidBody*> overlappingBodies;
	OverlappingBodiesTestTriggerVolume* triggerVolume = HK_NULL;
	{
		const hkReal heightDelta = 0.25f;


		hkReal heightIncrement = 0.0f;
		for (hkReal x = -10.0f; x < 10.0f; x += 2.0f)
		{
			hkpRigidBody *const body = createOverlappingBody( widthUnit, heightUnit, x, 0.5f * triggerVolumeScale * heightUnit + heightIncrement );

			world->addEntity( body );
			hkVector4 velocity;
			velocity.set( 0.0f, speed, 0.0f );
			body->setLinearVelocity( velocity );
			overlappingBodies.pushBack( body );
			heightIncrement += heightDelta;
		}

		// Create a long thin trigger volume which they will all pass through.
		hkpRigidBody *const triggerBody = createOverlappingBody( triggerVolumeScale * widthUnit, triggerVolumeScale * heightUnit, -10.0f, -10.0f);
		triggerVolume = new OverlappingBodiesTestTriggerVolume( triggerBody );

		world->addEntity( triggerBody );
	}
	
	// Create a number of Shapes that will be used to update the shapes of the different rigid bodies found in overlappingBodies.
	hkVector4 shapeExtents; 
	shapeExtents.set(0.5f, 0.5f, 0.5f);

	hkpShape* shapes [] = { hkpShapeGenerator::createConvexShape(shapeExtents, hkpShapeGenerator::BOX, HK_NULL),
							hkpShapeGenerator::createConvexShape(shapeExtents, hkpShapeGenerator::SPHERE, HK_NULL),
							hkpShapeGenerator::createConvexShape(shapeExtents, hkpShapeGenerator::CAPSULE, HK_NULL),
						  };

	const int numBodies = overlappingBodies.getSize();
	const int numShapes = sizeof(shapes) / sizeof(shapes[0]);

	int lastShapeUsed = 0; // Used to determine which shape to use next.

	for ( int i = 0; i < numSteps; ++i )
	{
		world->unlock();
		world->stepDeltaTime( deltaTime );
		world->lock();

		for ( int j = 0; j < numBodies; ++j ) // Go through each body to potentially update their shape.
		{
			hkpRigidBody* body = overlappingBodies[j];
			if ( (j & 0x3) == 0) // if j is a multiple of 4.
			{
				if ( (lastShapeUsed & 0x1) == 0) // if lastShapeUsed is an even number
				{
					const hkpShape* previousShape = body->getCollidable()->getShape();

					// Handle reference counting here.
					body->getCollidableRw()->setShape( shapes[lastShapeUsed] );
					shapes[lastShapeUsed]->addReference();

					if (previousShape)
					{
						previousShape->removeReference();
					}
					
					// Let the hkpRigidBody know that it's shape has changed.
					body->updateShape();
				}
				else
				{
					body->setShape( shapes[lastShapeUsed] );
				}

				// Update the lastShape used, rolling back to 0 if we have used them all.
				++lastShapeUsed;
				if (lastShapeUsed == numShapes)
				{
					lastShapeUsed = 0;
				}
			}
			// Confirm the objects are neither deflected nor rotated.
			HK_TEST( hkMath::fabs( overlappingBodies[j]->getLinearVelocity()( 0 ) ) < 0.01f );
			HK_TEST( hkMath::fabs( overlappingBodies[j]->getLinearVelocity()( 1 ) - speed ) < 0.01f );
			HK_TEST( hkMath::fabs( overlappingBodies[j]->getLinearVelocity()( 2 ) ) < 0.01f );
			HK_TEST( overlappingBodies[j]->getAngularVelocity().lengthSquared<3>().getReal() < 0.01f );
		}
	}

	// Check that all bodies got the correct collisions (We know they will have all entered and left the TriggerVolume).
	while ( !triggerVolume->m_bodiesTouched.isEmpty() )
	{
		hkpRigidBody* bodyTouched = triggerVolume->m_bodiesTouched.back();

		const int overlappingBodyTouched = overlappingBodies.indexOf( bodyTouched );
		HK_TEST( overlappingBodyTouched != -1 );

		{
			const int indexLeft = triggerVolume->m_bodiesLeft.indexOf( bodyTouched );
			HK_TEST( indexLeft != -1 );
			if ( indexLeft != -1 )
			{
				triggerVolume->m_bodiesLeft.removeAt( indexLeft );
			}
		}
		triggerVolume->m_bodiesTouched.popBack();

	}

	HK_TEST( triggerVolume->m_bodiesTouched.getSize() == 0 );
	HK_TEST( triggerVolume->m_bodiesLeft.getSize() == 0 );

	// Check our reference count for the bodies in accurate.
	for ( int i = 0; i < numBodies; ++i )
	{
		HK_TEST( overlappingBodies[i]->getReferenceCount() == 2 );
	}

	world->removeEntity( triggerVolume->m_triggerBody );
	triggerVolume->removeReference();
	triggerVolume->m_triggerBody->removeReference();

	for ( int i = 0; i < numBodies; ++i )
	{
		world->removeEntity( overlappingBodies[i] );
		overlappingBodies[i]->removeReference();
	}

	for ( int i = 0; i < numShapes; ++i )
	{
		// Check our reference count for the shapes is accurate.
		HK_TEST( shapes[i]->getReferenceCount() == 1 );
		shapes[i]->removeReference();
	}

	world->unlock();
}

static int overlappingCollisions_main()
{
	hkpWorldCinfo info;
	{
		info.m_gravity.setZero();
		info.setupSolverInfo(hkpWorldCinfo::SOLVER_TYPE_4ITERS_MEDIUM); 
		info.setBroadPhaseWorldSize( 100.0f );
		info.m_enableDeactivation = false;
	}

	// Discrete simulation.
	{
		info.m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_DISCRETE;

		hkpWorld *const world = new hkpWorld( info );

		const hkReal deltaTime = 1.0f / 60.0f;

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testOverlappingCollisions( world, deltaTime );
		testOverlappingBodiesCollisionsWithShapeUpdates(world, deltaTime);
		
		world->removeReference();
	}

	// Continuous simulation.
	{
		info.m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS;

		hkpWorld *const world = new hkpWorld( info );

		const hkReal deltaTime = 1.0f / 60.0f;

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testOverlappingCollisions( world, deltaTime );
		testOverlappingBodiesCollisionsWithShapeUpdates(world, deltaTime);

		world->removeReference();
	}

	// Multithreaded simulation.
	{
		info.m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED;

		hkpWorld *const world = new hkpWorld( info );

		const hkReal deltaTime = 1.0f / 60.0f;

		world->lock();
		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );
		world->unlock();

		testOverlappingCollisions( world, deltaTime );
		testOverlappingBodiesCollisionsWithShapeUpdates(world, deltaTime);

		world->markForWrite();
		world->removeReference();
	}
	

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(overlappingCollisions_main, "Slow", "Physics2012/Test/UnitTest/Utilities/", __FILE__     );

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
