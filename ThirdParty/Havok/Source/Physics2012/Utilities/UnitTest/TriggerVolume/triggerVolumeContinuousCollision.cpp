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
	hkVector4 boxStartPosition;
	hkVector4 triggerPosition;
	hkVector4 boxFinishPosition;
	{
		boxStartPosition.set( -5.0f, 0.0f, 0.0f );
		triggerPosition.setZero();
		boxFinishPosition.set( 5.0f, 0.0f, 0.0f );
	}

	// Slowly - enter and leave in separate frames.
	{
		hkpRigidBody *const box = createRigidBody( boxStartPosition, false );
		hkpRigidBody *const triggerBody = createRigidBody( triggerPosition, true );

		const int numFrames = 10;
		hkVector4 boxVelocity;
		{
			boxVelocity.setSub( boxFinishPosition, boxStartPosition );
			const hkSimdReal distance = boxVelocity.normalizeWithLength<3>();
			const hkSimdReal numFramesR = hkSimdReal::fromInt32(numFrames);
			boxVelocity.mul( distance / ( numFramesR * hkSimdReal::fromFloat(deltaTime) ) );
		}

		hkpTriggerVolume::EventType expectedEvents[2] = { hkpTriggerVolume::ENTERED_EVENT, hkpTriggerVolume::LEFT_EVENT };
		( new UnitTestTriggerVolume( triggerBody, 2, expectedEvents ) )->removeReference();

		{
			world->lock(); 
			world->addEntity( triggerBody );
			world->addEntity( box );
			box->setLinearVelocity( boxVelocity );
			world->unlock();
			
			for ( int i = 0; i < numFrames; ++i )
			{
				world->stepDeltaTime( deltaTime );
			}
			
			world->lock();
			world->removeEntity( triggerBody );
			world->removeEntity( box );
			world->unlock();
		}

		HK_TEST( box->getReferenceCount() == 1 );

		triggerBody->removeReference();
		box->removeReference();
	}

	// Quickly - enter and leave in the same frame.
	{
		hkpRigidBody *const box = createRigidBody( boxStartPosition, false );
		hkpRigidBody *const triggerBody = createRigidBody( triggerPosition, true );

		const int numFrames = 1;
		hkVector4 boxVelocity;
		{
			boxVelocity.setSub( boxFinishPosition, boxStartPosition );
			const hkSimdReal distance = boxVelocity.normalizeWithLength<3>();
			const hkSimdReal numFramesR = hkSimdReal::fromInt32(numFrames);
			boxVelocity.mul( distance / ( numFramesR * hkSimdReal::fromFloat(deltaTime) ) );
		}

		hkpTriggerVolume::EventType expectedEvents[1] = { hkpTriggerVolume::ENTERED_AND_LEFT_EVENT };
		( new UnitTestTriggerVolume( triggerBody, 1, expectedEvents ) )->removeReference();

		{
			world->lock(); 
			world->addEntity( triggerBody );
			world->addEntity( box );
			box->setLinearVelocity( boxVelocity );
			world->unlock();

			for ( int i = 0; i < numFrames; ++i )
			{
				world->stepDeltaTime( deltaTime );
			}

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

static int triggerVolumeContinuousCollision_main()
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

		// Use a low fps
		const hkReal deltaTime = 1.0f / 5.0f;

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testTriggerVolumeInWorld( world, deltaTime );
		
		world->removeReference();
	}
	*/

	// Continuous simulation.
	{
		info.m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS;

		hkpWorld *const world = new hkpWorld( info );

		// Use a low fps
		const hkReal deltaTime = 1.0f / 5.0f;

		hkpAgentRegisterUtil::registerAllAgents( world->getCollisionDispatcher() );

		testTriggerVolumeInWorld( world, deltaTime );
		
		world->removeReference();
	}

	// Multithreaded simulation.
	{
		info.m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_MULTITHREADED;

		hkpWorld *const world = new hkpWorld( info );
		
		// Use a low fps
		const hkReal deltaTime = 1.0f / 5.0f;

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
HK_TEST_REGISTER(triggerVolumeContinuousCollision_main, "Slow", "Physics2012/Test/UnitTest/Utilities/", __FILE__     );

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
