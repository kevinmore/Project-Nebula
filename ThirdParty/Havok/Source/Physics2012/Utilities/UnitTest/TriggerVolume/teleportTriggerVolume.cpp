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

static hkpRigidBody* createRigidBody( const hkVector4& pos, hkBool isFixed )
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
	info.m_position = pos;
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
	hkVector4 boxOverlapPosition;
	hkVector4 triggerOverlapPosition;
	hkVector4 boxAwayPosition;
	hkVector4 triggerAwayPosition;
	{
		boxAwayPosition.setZero();
		triggerAwayPosition.set( 10.0f, 0.0f, 0.0f );
		boxOverlapPosition.set( 5.0f, 0.0f, 0.0f );
		triggerOverlapPosition.set( 6.0f, 0.0f, 0.0f );
	}

	// Teleport a body into the trigger body and then away.
	{
		hkpRigidBody *const box = createRigidBody( boxAwayPosition, false );
		hkpRigidBody *const triggerBody = createRigidBody( triggerOverlapPosition, true );

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };
		( new UnitTestTriggerVolume( triggerBody, 2, expectedEvents ) )->removeReference();

		{
			world->lock(); 
			world->addEntity( triggerBody );
			world->addEntity( box );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );

			world->lock();
			box->setPosition( boxOverlapPosition );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			box->setPosition( boxAwayPosition );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			world->removeEntity( triggerBody );
			world->removeEntity( box );
			world->unlock();
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}


	// Teleport the trigger body into another body and then away.
	{
		hkpRigidBody *const box = createRigidBody( boxOverlapPosition, false );
		hkpRigidBody *const triggerBody = createRigidBody( triggerAwayPosition, true );

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };
		( new UnitTestTriggerVolume( triggerBody, 2, expectedEvents ) )->removeReference();

		{
			world->lock();
			world->addEntity( triggerBody );
			world->addEntity( box );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			triggerBody->setPosition( triggerOverlapPosition );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			triggerBody->setPosition( triggerAwayPosition );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
		
			world->lock();
			world->removeEntity( triggerBody );
			world->removeEntity( box );
			world->unlock();
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}

	// Teleport the trigger into and away from each other.
	{
		hkpRigidBody *const box = createRigidBody( boxAwayPosition, false );
		hkpRigidBody *const triggerBody = createRigidBody( triggerAwayPosition, true );

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };
		( new UnitTestTriggerVolume( triggerBody, 2, expectedEvents ) )->removeReference();

		{
			world->lock();
			world->addEntity( box );
			world->addEntity( triggerBody );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
			
			world->lock();
			box->setPosition( boxOverlapPosition );
			triggerBody->setPosition( triggerOverlapPosition );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );
			
			world->lock();
			box->setPosition( boxAwayPosition );
			triggerBody->setPosition( triggerAwayPosition );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );

			world->lock();
			world->removeEntity( triggerBody );
			world->removeEntity( box );
			world->unlock();
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}

	// Teleport out and in one frame -- ignored.
	{
		hkpRigidBody *const box = createRigidBody( boxOverlapPosition, false );
		hkpRigidBody *const triggerBody = createRigidBody( triggerOverlapPosition, true );

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };
		( new UnitTestTriggerVolume( triggerBody, 2, expectedEvents ) )->removeReference();

		{
			world->lock();
			world->addEntity( box );
			world->addEntity( triggerBody );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			box->setPosition( boxAwayPosition );
			box->setPosition( boxOverlapPosition );
			world->unlock();
			
			world->stepDeltaTime( deltaTime );

			world->lock();
			box->setPosition( boxAwayPosition );
			triggerBody->setPosition( triggerAwayPosition );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			world->removeEntity( box );
			world->removeEntity( triggerBody );
			world->unlock();
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}

	// Teleport in and out in one frame - ignored.
	{
		hkpRigidBody *const box = createRigidBody( boxAwayPosition, false );
		hkpRigidBody *const triggerBody = createRigidBody( triggerOverlapPosition, true );

		( new UnitTestTriggerVolume( triggerBody, 0, HK_NULL ) )->removeReference();

		{
			world->lock();
			world->addEntity( box );
			world->addEntity( triggerBody );
			world->unlock();

			world->stepDeltaTime( deltaTime );
			
			world->lock();
			box->setPosition( boxOverlapPosition );
			box->setPosition( boxAwayPosition );
			world->unlock();

			world->stepDeltaTime( deltaTime );

			world->lock();
			world->removeEntity( triggerBody );
			world->removeEntity( box );
			world->unlock();
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}
}

static int teleportTriggerVolume_main()
{
	hkpWorldCinfo info;
	{
		info.m_gravity.setZero();
		info.setupSolverInfo(hkpWorldCinfo::SOLVER_TYPE_4ITERS_MEDIUM); 
		info.setBroadPhaseWorldSize( 100.0f );
		info.m_enableDeactivation = false;
	}

	// Discrete simulation.
	/*
	{
		info.m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_DISCRETE;

		hkpWorld *const world = new hkpWorld( info );

		const hkReal deltaTime = 1.0f / 60.0f;

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testTriggerVolumeInWorld( world, deltaTime );
		
		world->removeReference();
	}
	*/

	// Continuous simulation.
	{
		info.m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS;

		hkpWorld *const world = new hkpWorld( info );

		const hkReal deltaTime = 1.0f / 60.0f;

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testTriggerVolumeInWorld( world, deltaTime );
		
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
		
		world->markForWrite();
		world->removeReference();
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(teleportTriggerVolume_main, "Slow", "Physics2012/Test/UnitTest/Utilities/", __FILE__     );

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
