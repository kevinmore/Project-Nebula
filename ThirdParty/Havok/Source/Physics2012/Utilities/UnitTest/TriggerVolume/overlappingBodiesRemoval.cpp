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

#include <Physics2012/Utilities/UnitTest/TriggerVolume/unitTestTriggerVolume.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>

#define NUM_SPHERES 3
static const hkReal sphereRadius = 0.5f;
static const hkReal sphereMass = 10.f;
static const hkReal boxExtend = 4.f;

// This series of unit test checks whether the issue in HVK-6155 is 
// correctly handled, i.e. that overlapping bodies removed
// from the world are instantly removed from the TriggerVolume as well.


class EmptyQueueTriggerVolume : public hkpTriggerVolume
{
	public:
		EmptyQueueTriggerVolume( hkpRigidBody* triggerBody ) :
			hkpTriggerVolume(triggerBody)
		{
		
		}

		~EmptyQueueTriggerVolume()
		{
			// Checks if the event queue is empty before hkpTriggerVolume's 
			// destructor has the chance to empty it.
			HK_TEST( m_eventQueue.isEmpty() );
		}
};

// This TriggerVolume is used to check 'hkpWorld::removeEntity()' works as expected when called
// from 'void triggerEventCallback()';.
class RemoveRandomBodyOnOverlapTriggerVolume : public hkpTriggerVolume
{
public:
	RemoveRandomBodyOnOverlapTriggerVolume( hkpRigidBody* triggerBody, hkpWorld* world ) :
	hkpTriggerVolume(triggerBody), 
	m_removedBodies(),
	m_world(world),
	m_numCallbackCalled(0)
	
	{

	}
	
	virtual void triggerEventCallback( hkpCharacterProxy* proxy, EventType type ) {}

	virtual void triggerEventCallback( hkpRigidBody* body, EventType type )
	{	
		bool shouldRemoveABody = false;
		// Let's remove a body alternatively between receiving an ENTERED_EVENT and LEFT_EVENT as follows:
		// If 'hkpTriggerVolume::triggerEventCallback()' is called with ENTERED_EVENT and m_numCallbackCalled is an odd number
		// Or if 'hkpTriggerVolume::triggerEventCallback()' is called with LEFT_EVENT and m_numCallbackCalled is an even number.
		if( ( (m_numCallbackCalled & 0x1) == 0 && (type & ENTERED_EVENT) ) ||
			( (m_numCallbackCalled & 0x1) == 1 && (type & LEFT_EVENT) ) )
		{
			shouldRemoveABody = true;
		}

		// Test that 'hkpShape::setShape()' and 'hkpRigidBody::updateShape()' doesn't crash/assert
		// when called in 'triggerEventCallback()'
		if ( (m_numCallbackCalled % 7) == 0 )
		{
			if ( type & ENTERED_EVENT )
			{
				body->setShape( body->getCollidable()->getShape() );
			}
			else if (type & LEFT_EVENT)
			{
				body->updateShape();
			}
		}

		if ( shouldRemoveABody )
		{
			// We know from the implementation of 'hkpTriggerVolume::entityRemovedCallback(..);' that
			// entities are removed differently depending on whether they have already been retrieved from the calling
			// 'hkpTriggerVolume::postSimulationCallback(..);'/'hkpTriggerVolume::updateOverlaps()' functions or not.
			// We will consequently either remove an element that is situated before, at, or after bodyToRemove in 
			// m_overlappingBodies.
			hkpRigidBody* bodyToRemove = HK_NULL;
			if ((m_numCallbackCalled % 3) == 0)
			{
				bodyToRemove = body;
			}
			else if ( !m_overlappingBodies.isEmpty() )
			{
				if ( (m_numCallbackCalled % 3) == 1 ) 
				{
					hkInt32 bodyIndex = 0;
					do 
					{
						bodyToRemove = m_overlappingBodies[bodyIndex];
						++bodyIndex;
					}
					while ( !bodyIsInPhysicsWorld( bodyToRemove ) && bodyIndex < m_overlappingBodies.getSize() );

				}
				else if ((m_numCallbackCalled % 3) == 2 )
				{
					hkInt32 bodyIndex = m_overlappingBodies.getSize() - 1;
					do 
					{
						bodyToRemove = m_overlappingBodies[bodyIndex];
						--bodyIndex;
					} while ( !bodyIsInPhysicsWorld( bodyToRemove ) && bodyIndex >= 0);
				}
			}

			if ( bodyIsInPhysicsWorld( bodyToRemove )  ) 
			{
				removeBodyWithTests( body, bodyToRemove );
			}
		}

		// Remove a body from the world if it leaves the TriggerVolume but is still active.
		if ( (type & LEFT_EVENT) && bodyIsInPhysicsWorld( body ) )
		{
			removeBodyWithTests( body, body );
		}

		++m_numCallbackCalled;
	}

	//Returns whether body is still attached to a hkpWorld or not.
	bool bodyIsInPhysicsWorld(hkpRigidBody* body)
	{
		return body && body->getWorld() && (m_removedBodies.indexOf(body) == -1);
	}

	void removeBodyWithTests(hkpRigidBody* bodyFromCallback, hkpRigidBody* bodyToRemove)
	{
		// We know that bodyFromCallback and bodyToRemove are from m_overlappingBodies
		// which is always assumed as sorted as defined by 'bodyOrderForArrays()'.
		bool bodyToRemoveWasAlreadySelected = bodyOrderForArrays(bodyFromCallback, bodyToRemove) 
											|| bodyFromCallback == bodyToRemove;

		hkUint32 currentNumberOfEvents = m_eventQueue.getSize();

		// Compute the following elements which will be checked against
		// once body has been removed from the world.
		hkInt32 numEventsUnattachedToBody = 0;

		for (hkUint32 eventIndex = 0; eventIndex < currentNumberOfEvents; ++eventIndex)
		{
			if (m_eventQueue[eventIndex].m_body != bodyToRemove)
			{
				++numEventsUnattachedToBody;
			}
		}

		m_removedBodies.pushBack(bodyToRemove);
		m_world->removeEntity(bodyToRemove);

		// The number of events might have changed from 'hkpWorld::removeEntity()' calling
		// 'hkpTriggerVolume::entityRemovedCallback()'.
		currentNumberOfEvents = m_eventQueue.getSize(); 
		// If the body to remove was already selected as a new overlap, then it will not clear previous events
		// related to that body as they have already been processed.
		hkUint32 checkForUnremovedEventFrom = bodyToRemoveWasAlreadySelected ? currentNumberOfEvents : 0;
		
		// Make sure that removing bodyToRemove did not invalidate other events.
		for (hkUint32 eventIndex = 0; eventIndex < currentNumberOfEvents; ++eventIndex)
		{
			if (eventIndex >= checkForUnremovedEventFrom)
			{
				HK_TEST2(	m_eventQueue[eventIndex].m_body != bodyToRemove,
							"There is a dangling event to body which was just removed from the world." );
			}

			if ( m_eventQueue[eventIndex].m_body != bodyToRemove )
			{
				--numEventsUnattachedToBody;
			}
		}
		
		HK_TEST2( 	numEventsUnattachedToBody == 0,
					"Some Events not attached to body have been added/removed since removing body from the world." );
	}

public: 
	// Used to avoid calling 'hkpWorld::removeEntity()' twice with the same entity. Note that a reference to those bodies are kept 
	// in 'testTVOverlappingRemoval();'
	hkArray<hkpRigidBody*> m_removedBodies;
private:
	// Used to call 'hkpWorld::removeEntity()' to remove each requested entity.
	hkpWorld* m_world;
	// Number of times 'void triggerEventCallback()' was called. It is used to
	// randomize which overlapping body to remove next.
	int m_numCallbackCalled; 
};

static hkpRigidBody* createTriggerVolumeBody(void)
{
	// Create a box-shaped trigger volume.
	hkpRigidBodyCinfo tvInfo;
	hkVector4 tvSize; tvSize.set( boxExtend, 2.f, 2.f );
	hkpConvexShape* tvShape = new hkpBoxShape( tvSize ); 

	tvInfo.m_motionType = hkpMotion::MOTION_FIXED;
	tvInfo.m_shape =  tvShape;

	hkpRigidBody * tvBody = new hkpRigidBody( tvInfo );
	tvShape->removeReference();

	return tvBody;
}

static hkpTriggerVolume* createTriggerVolume( hkpWorld* world )
{
	hkpRigidBody * tvBody = createTriggerVolumeBody(); 

	world->addEntity( tvBody );

	hkpTriggerVolume* tv = new hkpTriggerVolume( tvBody );

	tvBody->removeReference();
	
	return tv;
}

static hkpRigidBody* createSphere( hkpWorld* world, hkReal xPos = 0.f, hkReal yPos = 2.1f )
{
	hkpRigidBodyCinfo sphereInfo;
	hkpConvexShape* sphereShape = new hkpSphereShape( sphereRadius );

	sphereInfo.m_shape = sphereShape;
	sphereInfo.m_position.set( xPos, yPos, 0.f );
	sphereInfo.m_motionType = hkpMotion::MOTION_SPHERE_INERTIA;
	sphereInfo.m_mass = sphereMass;

	hkMassProperties massProperties;
	hkpInertiaTensorComputer::computeSphereVolumeMassProperties( sphereRadius, sphereInfo.m_mass, massProperties );

	sphereInfo.m_inertiaTensor = massProperties.m_inertiaTensor;
	sphereInfo.m_centerOfMass = massProperties.m_centerOfMass;
	sphereInfo.m_mass = massProperties.m_mass;

	hkpRigidBody* sphere =  new hkpRigidBody( sphereInfo );

	sphereShape->removeReference();	

	return sphere;
}


static void testOverlappingListener( hkpWorld* world, hkReal deltaTime )
{
	world->lock();

	hkpTriggerVolume* tv = createTriggerVolume( world );
	hkpRigidBody* sphere = createSphere( world );
	world->addEntity( sphere );
	
	const int numSteps = 80;

	for ( int i = 0; i < numSteps; ++i )
	{
		world->unlock();
		world->stepDeltaTime( deltaTime );
		world->lock();
		


		if( tv->m_overlappingBodies.getSize() > 0 )
		{

			// Tests if the TriggerVolume listens to the entity.
			HK_TEST( sphere->getEntityListeners().indexOf(tv) != -1 );
			break;
		}	
	}

	for ( int i = 0; i < numSteps; ++i )
	{
		world->unlock();
		world->stepDeltaTime( deltaTime );
		world->lock();

		if( tv->m_overlappingBodies.getSize() == 0 )
		{
			// Tests if the TriggerVolume stopped listening to the entity.
			HK_TEST( sphere->getEntityListeners().indexOf(tv) == -1 );
			break;
		}	
	}
	
	world->removeEntity( sphere );
	sphere->removeReference();
	
	world->unlock();
	tv->removeReference();
}


static void testSingleOverlappingRemoval( hkpWorld* world, hkReal deltaTime)
{
	world->lock();

	hkpTriggerVolume* tv = createTriggerVolume( world );
	hkpRigidBody* sphere = createSphere( world );
	world->addEntity( sphere );

	const int numSteps = 20;

	for ( int i = 0; i < numSteps; ++i )
	{
		world->unlock();
		world->stepDeltaTime( deltaTime );
		world->lock();
		
		if(tv->m_overlappingBodies.getSize() > 0)
		{
			world->removeEntity( sphere );
			
			// Tests if the entity is now completely removed (except for the reference that this function holds).
			HK_TEST( sphere->getReferenceCount() == 1 );
			sphere->removeReference();
			
			HK_TEST( tv->m_overlappingBodies.getSize() == 0 );

			world->unlock();
			break;
		}	
	}
	
	tv->removeReference();
}




static void testMultipleOverlappingRemoval( hkpWorld* world, hkReal deltaTime, bool batch )
{

	// If batch is true, the spheres will be added and removed from the world as a batch
	// instead of one by one.

	hkpTriggerVolume* tv = createTriggerVolume( world );
	
	hkpEntity* spheres[NUM_SPHERES];
	
	hkReal xPos = 0.f;
	hkReal sign = 0.f;
	
	// Adds sphere evenly along the X axis starting from origin.
	for ( int i = 0 ; i < NUM_SPHERES ; ++i )
	{
		if (i%2)
		{
			xPos += ( 2*sphereRadius + 0.1f );
			sign = 1.f;
		} else {
			sign = -1.f;
		}
		HK_ASSERT2( 0x4f2f986d, xPos - sphereRadius < boxExtend ,"Too many spheres !" );
		
		spheres[i] = createSphere( world, sign * xPos );
		if ( !batch  )
		{
			world->addEntity( spheres[i] );
		}
	}  
	
	if( batch )
	{
		world->addEntityBatch( (spheres), NUM_SPHERES );
	}

	world->lock();

	const int numSteps = 20;

	for ( int n = 0; n < numSteps; ++n )
	{
		world->unlock();
		world->stepDeltaTime( deltaTime );
		world->lock();

		if( tv->m_overlappingBodies.getSize() == NUM_SPHERES )
		{		
			if ( batch )
			{
				world ->removeEntityBatch( spheres, NUM_SPHERES );
			}
						
			for (int i = 0 ; i < NUM_SPHERES ; ++i)
			{
				if( !batch ) {
					world->removeEntity( spheres[i] );
					HK_TEST( tv->m_overlappingBodies.getSize() == NUM_SPHERES - i - 1 );
				}
				HK_TEST( spheres[i]->getReferenceCount() == 1 );
				spheres[i]->removeReference();
			}
						
			HK_TEST( tv->m_overlappingBodies.getSize() == 0 );
			world->unlock();
			break;
		}	
	}

	tv->removeReference();
}

static void testTVOverlappingRemoval( hkpWorld* world, hkReal deltaTime)
{
	// Checks that:
	// [HVK-6268] - Calling hkpWorld::removeEntity() on a body inside
	// a trigger volume through hkpTriggerVolume::triggerEventCallback() 
	// should remove the entity from the 'Overlapping' List and should not
	// Assert/Crash.

	world->lock();

	// Create and add a RemoveRandomBodyOnOverlapTriggerVolume to the world.
	RemoveRandomBodyOnOverlapTriggerVolume* tv;
	{
		hkpRigidBody * tvBody = createTriggerVolumeBody(); 

		world->addEntity( tvBody );
		 
		tv = new RemoveRandomBodyOnOverlapTriggerVolume( tvBody, world );
		
		tvBody->removeReference();
	}

	// Spawn a number of spheres which will be removed from the world through tv.
	hkReal xPos = -0.4f;
	hkReal yPos = 0.0f;
	hkVector4 spheresVelocity; 
	spheresVelocity.set(0.0f, -2.5f, 0.0f);
	
	const int numberSpheresRow    = 6;
	const int numberSpheresPerRow = 4;
	const int numberOfSpheres	  = numberSpheresPerRow * numberSpheresRow;
	hkpRigidBody* spheres[numberOfSpheres];

	for (int spheresRowIndex = 0; spheresRowIndex < numberSpheresRow; spheresRowIndex++)
	{
		for (int sphereIndexInRow = 0; sphereIndexInRow < numberSpheresPerRow; ++sphereIndexInRow)
		{
			const hkUint32 sphereIndexInArray = spheresRowIndex * numberSpheresPerRow + sphereIndexInRow;
			spheres[sphereIndexInArray] = createSphere( world, xPos, yPos);
			spheres[sphereIndexInArray]->setLinearVelocity(spheresVelocity);
			world->addEntity( spheres[sphereIndexInArray] );

			xPos += 1.2f;
		}

		yPos += 1.0f;
		xPos = -0.4f;
	}
	
	const int numSteps = 65;

	for ( int i = 0; i < numSteps; ++i )
	{
		world->unlock();
		world->stepDeltaTime( deltaTime );
		world->lock();

		// Checks that the state of overlapping and removed bodies is as expected after calls of 'hkpWorld::removeEntity()'
		// were done from 'hkpTriggerVolume::postSimulationCallback(..);'.
		const hkUint32 numOverlappingBodies = tv->m_overlappingBodies.getSize();
		for (hkUint32 overlapBodyIndex = 0; overlapBodyIndex < numOverlappingBodies; overlapBodyIndex++)
		{
			hkpRigidBody* overlapBody = tv->m_overlappingBodies[overlapBodyIndex];

			HK_TEST2(	overlapBody->getWorld() != HK_NULL, 
						"A body we removed from the world in the last step is still marked as overlapping with the TriggerVolume." );
		}
		// Ensure all spheres have been correctly removed from the world and unattached from tv.
		const hkUint32 numRemovedBodies = tv->m_removedBodies.getSize();
		for (hkUint32 removedBodyIndex = 0 ; removedBodyIndex < numRemovedBodies ; ++removedBodyIndex)
		{
			hkpRigidBody* removedBody = tv->m_removedBodies[removedBodyIndex];

			HK_TEST2( removedBody->getEntityListeners().indexOf(tv) == -1, "The TriggerVolume is still listening to the entity we just removed." );
		}
	}

	// Ensure all spheres have been correctly removed from the world and unattached from tv.
	for (int sphereIndex = 0 ; sphereIndex < numberOfSpheres ; ++sphereIndex)
	{
		hkpRigidBody* sphereBody = spheres[sphereIndex];

		HK_TEST2( sphereBody->getReferenceCount() == 1, "This function should have the last body reference of each sphere." ); 
		sphereBody->removeReference();
	}

	world->unlock();

	tv->removeReference();
}


static int overlappingBodiesRemoval_main() 
{
	hkpWorldCinfo info;
	{
		info.setupSolverInfo( hkpWorldCinfo::SOLVER_TYPE_4ITERS_MEDIUM ); 
		info.setBroadPhaseWorldSize( 100.0f );
		info.m_enableDeactivation = false;
	}
	
	const hkReal deltaTime = 1.f / 60.f;
	
	
	// Test checking if the entity is listened to.
	{
		hkpWorld * const world = new hkpWorld( info );

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testOverlappingListener( world, deltaTime );

		world->removeReference();
	}
	
	// Test adding and removing a single body.
	{
		hkpWorld * const world = new hkpWorld( info );
	
		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );
	
		testSingleOverlappingRemoval( world, deltaTime );
	
		world->removeReference();
	}

	// Test adding and removing several bodies.
	{
		hkpWorld * const world = new hkpWorld( info );

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testMultipleOverlappingRemoval( world, deltaTime, false );

		world->removeReference();
	}
	
	// Test adding and removing several bodies with batch.
	{
		hkpWorld * const world = new hkpWorld( info );

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testMultipleOverlappingRemoval( world, deltaTime, true );

		world->removeReference();
	}
	// Test removing several bodies from the 'hkpTriggerVolume::triggerEventCallback()' callback
	{
		hkpWorld * const world = new hkpWorld( info );

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testTVOverlappingRemoval( world, deltaTime );

		world->removeReference();
	}
	
	
	return 0;
}

#if	defined	(	HK_COMPILER_MWERKS	)
	#	pragma	fullpath_file	on
#	endif

HK_TEST_REGISTER(overlappingBodiesRemoval_main, "Slow", "Physics2012/Test/UnitTest/Utilities/", __FILE__);

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
