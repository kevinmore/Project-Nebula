/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpKeyframedRigidMotion.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

// Test that applying forces and vecs activates a body

static void deactivate(hkpRigidBody* rb, hkpWorld* w)
{
	HK_TEST2( rb->isActive(), "Rigid body was not active." );
	rb->setLinearVelocity(hkVector4::getZero());
	rb->setAngularVelocity(hkVector4::getZero());
	rb->deactivate();
	w->unlock();
	w->stepDeltaTime(0.016f);
	w->lock();
	HK_ASSERT(0x67e3b9cd,  rb->isActive() == false );
}

static void requestDeactivation(hkpRigidBody* rb, hkpWorld* w)
{
	HK_TEST2( rb->isActive(), "Rigid body was not active." );
	rb->setLinearVelocity(hkVector4::getZero());
	rb->setAngularVelocity(hkVector4::getZero());
	rb->requestDeactivation();
	w->unlock();
	w->stepDeltaTime(0.016f);
	w->lock();
	HK_ASSERT(0x67e3b9cd,  rb->isActive() == false );
}


static int activation_main()
{
	hkpWorld::IgnoreForceMultithreadedSimulation ignoreForceMultithreaded;
	hkDisableError disable0xf034546f( 0xf034546f );	// "You are calling hkpRigidBody::setTransform of a RigidBody, which tries to deactivate.". We do this intentionally here.
	hkDisableError disable0xf0ff0099( 0xf0ff0099 );	// "hkpEntity::deactivate() is deprecated. Please use hkpEntity::requestDeactivation() instead.". We do this intentionally here.

	hkpRigidBody* rigidBody = HK_NULL;
//	hkpRigidBody* base = HK_NULL;
	hkpWorld* world = HK_NULL;

	// Create the world.
	{
		hkpWorldCinfo info;
		
		// Set gravity to zero so body floats.
		info.m_gravity.setZero();
		info.setBroadPhaseWorldSize( 100.0f );
		world = new hkpWorld(info);
		world->lock();
	}

	// Create the shape and a rigid body to view it.
	{
		// Data specific to this shape.
		hkVector4 halfExtents; halfExtents.set(1.0f, 1.0f, 1.0f);
		
		hkpBoxShape* shape = new hkpBoxShape(halfExtents, 0 );
		hkpRigidBodyCinfo rigidBodyInfo;
		rigidBodyInfo.m_shape = shape;
		rigidBodyInfo.m_position.setZero();
		rigidBodyInfo.m_angularDamping = 0.0f;
		rigidBodyInfo.m_linearDamping = 0.0f;
		rigidBodyInfo.m_inertiaTensor.setIdentity();
		rigidBodyInfo.m_mass = 1;
		rigidBodyInfo.m_motionType = hkpMotion::MOTION_BOX_INERTIA;

		rigidBody = new hkpRigidBody(rigidBodyInfo);
		world->addEntity(rigidBody);

		rigidBodyInfo.m_position.set( 0, -5, 0);
		rigidBodyInfo.m_mass = 0;
		rigidBodyInfo.m_inertiaTensor.setZero();
		rigidBodyInfo.m_motionType = hkpMotion::MOTION_FIXED;

	//	base = new hkpRigidBody(rigidBodyInfo);
	//	world->addEntity(base);

		shape->removeReference();
	}

	hkStepInfo stepInfo( hkTime(0.0f), hkTime(0.1f) );
	hkVector4 vec; vec.set(1,1,1);
	hkQuaternion rot; rot.setIdentity();
	hkTransform trans; trans.setIdentity();

	// Test that the requestDeactivation() functions work as advertised.
	// These will pass only in single threaded mode, as otherwise they island split gets delayed.
	{
		rigidBody->setPosition(vec);
		requestDeactivation(rigidBody, world);

		rigidBody->setRotation(rot);
		requestDeactivation(rigidBody, world);

		rigidBody->setPositionAndRotation(vec, rot);
		requestDeactivation(rigidBody, world);

		rigidBody->setTransform(trans);
		requestDeactivation(rigidBody, world);

		rigidBody->setLinearVelocity(vec);
		requestDeactivation(rigidBody, world);

		rigidBody->setAngularVelocity(vec);
		requestDeactivation(rigidBody, world);

		rigidBody->applyPointImpulse(vec, vec);
		requestDeactivation(rigidBody, world);

		rigidBody->applyForce(stepInfo.m_deltaTime, vec);
		requestDeactivation(rigidBody, world);

		rigidBody->applyForce(stepInfo.m_deltaTime, vec);
		requestDeactivation(rigidBody, world);

		rigidBody->applyTorque(stepInfo.m_deltaTime, vec);
		requestDeactivation(rigidBody, world);
	}

	// Test that the deactivate() functions work as advertised.
	{
		rigidBody->setPosition(vec);		
		deactivate(rigidBody, world);

		rigidBody->setRotation(rot);
		deactivate(rigidBody, world);

		rigidBody->setPositionAndRotation(vec, rot);
		deactivate(rigidBody, world);

		rigidBody->setTransform(trans);
		deactivate(rigidBody, world);

		rigidBody->setLinearVelocity(vec);
		deactivate(rigidBody, world);

		rigidBody->setAngularVelocity(vec);
		deactivate(rigidBody, world);

		rigidBody->applyPointImpulse(vec, vec);		
		deactivate(rigidBody, world);

		rigidBody->applyForce(stepInfo.m_deltaTime, vec);
		deactivate(rigidBody, world);

		rigidBody->applyForce(stepInfo.m_deltaTime, vec);
		deactivate(rigidBody, world);

		rigidBody->applyTorque(stepInfo.m_deltaTime, vec);
		deactivate(rigidBody, world);
	}
	
	// clean up
	{
		rigidBody->removeReference();
		world->unlock();
		world->removeReference();
	}
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER( activation_main, "Fast", "Physics2012/Test/UnitTest/Dynamics/", __FILE__ );

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
