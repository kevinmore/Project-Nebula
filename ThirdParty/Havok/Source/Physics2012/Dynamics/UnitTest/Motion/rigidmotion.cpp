/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics2012/Collide/Dispatch/hkpAgentRegisterUtil.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>

#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactListener.h>
#include <Common/Internal/ConvexHull/hkGeometryUtility.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

class rigidMotionTestContactListener : public hkReferencedObject, public hkpContactListener
{
		public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DEMO );

		virtual void contactPointCallback( const hkpContactPointEvent& event ){ m_collisionDetected = true;}

		public:
			hkBool m_collisionDetected;

};

void zeroOrderTest(int testType)
{
	//some objects we need access to for the life-time of the test
	hkpRigidBody* movingBody;
	hkpWorld* world;
	rigidMotionTestContactListener* rmtcl = new rigidMotionTestContactListener;
	rmtcl->m_collisionDetected = false;

	//
	// Create the world
	//

	{
		hkpWorldCinfo info;

		info.m_gravity.set(0.0f, -9.8f, 0.0f);
		info.setupSolverInfo(hkpWorldCinfo::SOLVER_TYPE_4ITERS_MEDIUM); 
		info.m_collisionTolerance = 0.01f;
		info.m_simulationType = hkpWorldCinfo::SIMULATION_TYPE_CONTINUOUS;

		world = new hkpWorld( info );
		world->lock();

		// Register ALL agents (though some may not be necessary)
		hkpAgentRegisterUtil::registerAllAgents(world->getCollisionDispatcher());
	}

	{
		//
		//	The dimensions of the convexShapes etc.
		//

		const unsigned int numVertices = 8;

		hkVector4 vertices[numVertices]; 
		{
			vertices[0].set(-0.3f, 0.5f, 0.5f);
			vertices[1].set( 0.1f, 0.5f, 0.2f);
			vertices[2].set( 0.5f,-0.5f, 0.5f);
			vertices[3].set(-0.5f,-0.6f, 0.1f);
			vertices[4].set(-0.5f, 0.5f,-0.5f);
			vertices[5].set( 0.5f, 0.9f,-0.5f);
			vertices[6].set( 0.8f,-0.5f,-0.5f);
			vertices[7].set(-0.5f,-0.9f,-0.5f);
		}


		hkpConvexVerticesShape::BuildConfig	config;
		config.m_shrinkByConvexRadius =	false;
		const hkpShape* convexShape = new hkpConvexVerticesShape(hkStridedVertices(vertices,numVertices),config);		

		hkMassProperties massProperties;
		hkpInertiaTensorComputer::computeShapeVolumeMassProperties(convexShape, 100.0f, massProperties);
	
		hkpRigidBodyCinfo convexShapeInfo;

		convexShapeInfo.m_inertiaTensor = massProperties.m_inertiaTensor;
		convexShapeInfo.m_mass = massProperties.m_mass;
		convexShapeInfo.m_shape = convexShape;
		convexShapeInfo.m_motionType = hkpMotion::MOTION_KEYFRAMED;
		convexShapeInfo.m_qualityType = HK_COLLIDABLE_QUALITY_MOVING; // fixed / auto will not gen contacts
		

		// convexShapes, number 1
		{
			hkVector4 convexShapePos;
			convexShapePos.set( -0.8f, 0.0f, 0.0f, 0.0f );
			convexShapeInfo.m_position = convexShapePos;
			hkpRigidBody* convexShapeRigidBody = new hkpRigidBody(convexShapeInfo);

			world->addEntity( convexShapeRigidBody );
			convexShapeRigidBody->removeReference();	// Remove reference, since we no longer want to remember this
		}

		// convexShapes, number 2
		{
			hkVector4 convexShapePos;
			convexShapePos.set(0.8f,0.0f,0.0f,0.0f);
			convexShapeInfo.m_position = convexShapePos;
			hkpRigidBody* convexShapeRigidBody = new hkpRigidBody(convexShapeInfo);
			movingBody = convexShapeRigidBody;

			convexShapeRigidBody->addContactListener(rmtcl);

			world->addEntity( convexShapeRigidBody );
			convexShapeRigidBody->removeReference();	// Remove reference, since we no longer want to remember this
		}

		convexShape->removeReference();
	}
	//now step the world to test for collision detection

	hkError::getInstance().setEnabled( 0xad16c0e6, false ); // Creating an agent between two fixed or keyframed objects

	//a collision is guaranteed to be generated within 100 steps
	for (int step = 0; step < 100; step++)
	{
		switch(testType)
		{
			case 0:	{
						hkVector4 pos; pos = movingBody->getPosition();
						hkVector4 mpos; mpos.set(0.01f,0.0f,0.0f,0.0f);
						pos.setSub(pos,mpos);
						movingBody->setPosition(pos);
						break;
					}
			case 1: {
						float r = step * 0.01f;
						hkQuaternion rot;	rot.m_vec.set(0.0f,1.0f,0.0f,r);
						rot.normalize();
						movingBody->setRotation(rot);
						break;
					}
			case 2: {
						float r = step * 0.01f;
						hkQuaternion rot;	rot.m_vec.set(0.0f,1.0f,0.0f,r);
						rot.normalize();

						hkVector4 pos( movingBody->getPosition() );
						hkVector4 tmp; tmp.set( 0.01f, 0.0f, 0.0f, 0.0f );
						pos.setSub( pos, tmp);

						movingBody->setPositionAndRotation(pos,rot);
						break;
					}
			case 3: {
						float r = step * 0.01f;
						hkQuaternion rot;	rot.m_vec.set(0.0f,1.0f,0.0f,r);
						rot.normalize();

						hkVector4 pos( movingBody->getPosition() );
						hkVector4 tmp; tmp.set(0.01f,0.0f,0.0f,0.0f);
						pos.setSub( pos, tmp );

						hkTransform t;
						t.setTranslation(pos);
						t.setRotation(rot);
						
						movingBody->setTransform(t);
						break;
					}
			default:{
						break;
					}
		}
		world->unlock();
		world->stepDeltaTime(0.016f);
		world->lock();
	}

	hkError::getInstance().setEnabled( 0xad16c0e6, true ); // Creating an agent between two fixed or keyframed objects

	HK_TEST2( rmtcl->m_collisionDetected , "No collision detected, TIMs may be wrong. Iteration " << testType  );
	rmtcl->m_collisionDetected = false;
	world->unlock();
	world->removeReference();
	rmtcl->removeReference();
}

// Test that setPosition and setVelocity can be called on a keyframed body

int rigidmotion_main()
{
	hkpWorld::IgnoreForceMultithreadedSimulation ignoreForceMultithreaded;

	zeroOrderTest(0);
	zeroOrderTest(1);
	zeroOrderTest(2);
	zeroOrderTest(3);

	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(rigidmotion_main, "Fast", "Physics2012/Test/UnitTest/Dynamics/", __FILE__     );

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
