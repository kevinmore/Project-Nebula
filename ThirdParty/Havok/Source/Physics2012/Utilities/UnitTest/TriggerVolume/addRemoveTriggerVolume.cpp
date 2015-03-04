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
#include <Physics2012/Utilities/Collide/hkpShapeGenerator.h>


static hkpRigidBody* createRigidBody( hkReal x, hkBool isFixed )
{
	hkpRigidBody* body;

	hkpShape* shape;
	{
		hkVector4 halfExtents;
		halfExtents.set( 1.0f, 1.0f, 1.0f );
		shape = hkpShapeGenerator::createConvexVerticesBox( halfExtents );
	}

	hkpRigidBodyCinfo info;
	info.m_shape = shape;
	info.m_mass = 1.0f;
	info.m_motionType = hkpMotion::MOTION_BOX_INERTIA;
	info.m_position.set( x, 0.0f, 0.0f );
	info.m_rotation.setIdentity();
	info.m_qualityType = HK_COLLIDABLE_QUALITY_CRITICAL;
	//info.m_gravityFactor = 0.0f;
	//info.m_enableDeactivation = false;

	body = new hkpRigidBody( info );
	
	shape->removeReference();

	return body;
}

static void testTriggerVolumeInWorld( hkpWorld* world, hkReal deltaTime )
{
	// Trigger volume in world and body added and removed in subsequent frames.
	// We expect to pick up the event.
	{
		hkpRigidBody *const box = createRigidBody( 0.0f, false );
		hkpRigidBody *const triggerBody = createRigidBody( 1.0f, true );

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };
		( new UnitTestTriggerVolume( triggerBody, 2, expectedEvents ) )->removeReference();

		{
			world->lock(); 
			world->addEntity( triggerBody );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );

			world->lock();
			world->addEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			world->removeEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			world->removeEntity( triggerBody );
			world->unlock();

			world->stepDeltaTime( deltaTime );
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}


	// Trigger volume in world and body added and removed without any stepping.
	// We don't expect to pick up the event.
	{
		hkpRigidBody *const box = createRigidBody( 0.0f, false );
		hkpRigidBody *const triggerBody = createRigidBody( 1.0f, true );

		( new UnitTestTriggerVolume( triggerBody, 0, HK_NULL ) )->removeReference();

		{
			world->lock();
			world->addEntity( triggerBody );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			world->addEntity( box );
			world->removeEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			world->removeEntity( triggerBody );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}

	// Body in world and trigger volume added and removed.
	// We expect to pick up the event.
	{
		hkpRigidBody *const box = createRigidBody( 0.0f, false );
		hkpRigidBody *const triggerBody = createRigidBody( 1.0f, true );

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::TRIGGER_BODY_LEFT_EVENT };
		( new UnitTestTriggerVolume( triggerBody, 2, expectedEvents ) )->removeReference();

		{
			world->lock();
			world->addEntity( box );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
			
			world->lock();
			world->addEntity( triggerBody );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
			
			world->lock();
			world->removeEntity( triggerBody );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
			
			world->lock();
			world->removeEntity( box );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}

	// Body in world and trigger volume added and removed without any stepping.
	// We don't expect to pick up the event.
	{
		hkpRigidBody *const box = createRigidBody( 0.0f, false );
		hkpRigidBody *const triggerBody = createRigidBody( 1.0f, true );

		( new UnitTestTriggerVolume( triggerBody, 0, HK_NULL ) )->removeReference();

		{
			world->lock();
			world->addEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			world->addEntity( triggerBody );
			world->removeEntity( triggerBody );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
			
			world->lock();
			world->removeEntity( box );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}

	// Two bodies in world and one becomes a trigger volume.
	{
		hkpRigidBody *const box = createRigidBody( 0.0f, false );
		hkpRigidBody *const triggerBody = createRigidBody( 1.0f, true );

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };

		{
			world->lock();
			world->addEntity( box );
			world->addEntity( triggerBody );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			{
				hkpTriggerVolume* tv = new UnitTestTriggerVolume( triggerBody, 2, expectedEvents );
				tv->updateOverlaps();
				tv->removeReference();
			}
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			world->removeEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			world->removeEntity( triggerBody );
			world->unlock();

			world->stepDeltaTime( deltaTime );
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}


	// Using recollide and updateOverlaps to identify a body leaving the trigger volume
	// outside the step.
	{
		hkpRigidBody *const box = createRigidBody( 0.0f, false );
		hkpRigidBody *const triggerBody = createRigidBody( 1.0f, true );
		hkpEntity* triggerPtr = triggerBody;
		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };
		hkpTriggerVolume* tv = new UnitTestTriggerVolume( triggerBody, 2, expectedEvents );
		{
			world->lock();
			world->addEntity( box );
			world->addEntity( triggerBody );

			// Recollide the entities first.
			world->reintegrateAndRecollideEntities( &triggerPtr, 1, hkpWorld::RR_MODE_RECOLLIDE_NARROWPHASE );
			tv->updateOverlaps();

			world->removeEntity( box );

			// Only recollide the trigger volume.
			world->reintegrateAndRecollideEntities( &triggerPtr, 1, hkpWorld::RR_MODE_RECOLLIDE_NARROWPHASE );
			tv->updateOverlaps();

			world->removeEntity( triggerBody );
			world->unlock();
		}
		tv->removeReference();

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}

	// Redundant calls to updateOverlaps should have no effect.
	{
		hkpRigidBody *const box = createRigidBody( 0.0f, false );
		hkpRigidBody *const triggerBody = createRigidBody( 1.0f, true );

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };
		hkpTriggerVolume* tv = new UnitTestTriggerVolume( triggerBody, 2, expectedEvents );
		{
			world->lock();
			world->addEntity( triggerBody );
			tv->updateOverlaps();
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			world->addEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			tv->updateOverlaps();

			world->removeEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			tv->updateOverlaps();

			world->removeEntity( triggerBody );
			world->unlock();

			world->stepDeltaTime( deltaTime );
		}
		tv->removeReference();

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}
}

static void testTriggerVolumeInWorldStack( hkpWorld* world, hkReal deltaTime )
{
	// Trigger volume on stack - tests destruction prior to trigger body destruction.
	{
		hkpRigidBody *const box = createRigidBody( 0.0f, false );

		hkVector4 halfExtents;
		halfExtents.set( 1.0f, 1.0f, 1.0f );
		hkpShape* shape = hkpShapeGenerator::createConvexVerticesBox( halfExtents );

		hkpRigidBodyCinfo info;
		{
			info.m_shape = shape;
			info.m_mass = 1.0f;
			info.m_motionType = hkpMotion::MOTION_BOX_INERTIA;
			info.m_position.set( 1.0f, 0.0f, 0.0f );
			info.m_rotation.setIdentity();
			info.m_qualityType = HK_COLLIDABLE_QUALITY_CRITICAL;
			//info.m_gravityFactor = 0.0f;
			//info.m_enableDeactivation = false;
		}
		hkpRigidBody triggerBodyOnStack( info );
		shape->removeReference();
		hkpRigidBody* triggerBody = &triggerBodyOnStack;

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };
		UnitTestTriggerVolume triggerVolume( triggerBody, 2, expectedEvents );

		{
			world->lock(); 
			world->addEntity( triggerBody );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			world->addEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			world->removeEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			world->removeEntity( triggerBody );
			world->unlock();

			world->stepDeltaTime( deltaTime );
		}

		HK_TEST( box->getReferenceCount() == 1 );

		box->removeReference();
	}
}

static int addRemoveTriggerVolume_main()
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

		testTriggerVolumeInWorld( world, deltaTime );
		testTriggerVolumeInWorldStack( world, deltaTime );

		world->removeReference();
	}

	// Continuous simulation.
	{
		info.m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS;

		hkpWorld *const world = new hkpWorld( info );

		const hkReal deltaTime = 1.0f / 60.0f;

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testTriggerVolumeInWorld( world, deltaTime );
		testTriggerVolumeInWorldStack( world, deltaTime );

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

		testTriggerVolumeInWorld( world, deltaTime );
		testTriggerVolumeInWorldStack( world, deltaTime );

		world->markForWrite();
		world->removeReference();
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(addRemoveTriggerVolume_main, "Slow", "Physics2012/Test/UnitTest/Utilities/", __FILE__     );

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
