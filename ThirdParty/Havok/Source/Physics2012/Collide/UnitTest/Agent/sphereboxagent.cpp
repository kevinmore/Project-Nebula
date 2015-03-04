/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

// This checks the box-sphere agent, both the linearCast() and getPenetrations() methods.
#include <Physics2012/Collide/hkpCollide.h>




#include <Common/Base/UnitTest/hkUnitTest.h>
 // Large include
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics2012/Collide/Agent/ConvexAgent/SphereBox/hkpSphereBoxAgent.h>
#include <Physics2012/Collide/Agent/hkpCollisionInput.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#include <Physics2012/Collide/Query/Collector/BodyPairCollector/hkpFlagCdBodyPairCollector.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBodyCinfo.h>

#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>


#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>


	// This is a structure basically containing where the sphere and box are, and their radius
struct sphereBoxTestParams
{
			// Box details
	hkVector4 m_halfExtentsBox;
	hkVector4 m_positionBox;
	hkQuaternion m_rotationBox;
	hkReal m_radiusBox;

		// Sphere details
	hkReal m_radiusSphere;
	hkVector4 m_positionSphere;
	hkQuaternion m_rotationSphere;

		// Environment params: Can make this large so linearCast() returns contacts even when objects are distant.
	hkReal m_tolerance;
};

	// This structure contains the "expected" results (worked out by hand) given that we know where the bodies are.
struct sphereBoxTestExpectedResults
{
	hkBool m_colliding;
	hkBool m_penetrating;
	hkReal m_dist;
	hkVector4 m_normal;
	hkVector4 m_point;
};



// This is just a user function to help create a box in one line
hkpRigidBody* createBox(const hkVector4 &halfExtents, const hkReal radius, const hkReal mass, const hkVector4 &position, const hkQuaternion &rotation)
{
	hkpBoxShape* cube = new hkpBoxShape(halfExtents, 0 );	// Note: We use HALF-extents for boxes

	cube->setRadius(radius);

	//
	// Create a rigid body construction template
	//
	hkpRigidBodyCinfo boxInfo;

	if(mass != 0.0f)
	{
		boxInfo.m_mass = mass;
		hkMassProperties massProperties;
		hkpInertiaTensorComputer::computeBoxVolumeMassProperties(halfExtents, mass, massProperties);
		boxInfo.m_inertiaTensor = massProperties.m_inertiaTensor;
		boxInfo.m_motionType = hkpMotion::MOTION_BOX_INERTIA;
	}
	else
	{		
		boxInfo.m_motionType = hkpMotion::MOTION_FIXED;
	}
	boxInfo.m_rotation = rotation;
	boxInfo.m_shape = cube;

	boxInfo.m_position = position;
	hkpRigidBody* boxRigidBody = new hkpRigidBody(boxInfo);


	cube->removeReference();

	return boxRigidBody;
}


// This is just a user function to help create a sphere in one line
hkpRigidBody* createSphere(const hkReal radius, const hkReal mass, const hkVector4& position, const hkQuaternion &rotation)
{
	hkpSphereShape* sphere = new hkpSphereShape(radius);

	//
	// Create a rigid body construction template
	//
	hkpRigidBodyCinfo sphereInfo;

	if(mass != 0.0f)
	{
			// Haven't yet added sphere functionality to hkpInertiaTensorComputer, so here's explicit code.
		sphereInfo.m_mass = mass;
		hkMassProperties massProperties;
		hkpInertiaTensorComputer::computeSphereVolumeMassProperties(radius, mass, massProperties);
		sphereInfo.m_inertiaTensor = massProperties.m_inertiaTensor;
		sphereInfo.m_motionType = hkpMotion::MOTION_SPHERE_INERTIA;
	}
	else
	{		
		sphereInfo.m_motionType = hkpMotion::MOTION_FIXED;
	}
	sphereInfo.m_rotation = rotation;
	sphereInfo.m_shape = sphere;


	sphereInfo.m_position = position;
	hkpRigidBody* sphereRigidBody = new hkpRigidBody(sphereInfo);

	sphere->removeReference();

	return sphereRigidBody;
}

	// "randomly" rotate, translate both bodies (by maintain the relative positions, rotations)
	// This ensures we don't get any "it only works at the origin" bugs
void setRandomTransformsForBodies(hkpRigidBody* rb0, hkpRigidBody* rb1, hkTransform& transformUsed)
{
	hkPseudoRandomGenerator prng(32487);
	hkVector4 trans;
	trans.set( prng.getRandReal01() * 10.0f, prng.getRandReal01() * 10.0f, prng.getRandReal01()  * 10.0f);


	hkVector4 axis; 
    axis.set( prng.getRandReal01() * 10.0f, prng.getRandReal01()* 10.0f, prng.getRandReal01() * 10.0f);
	axis.normalize<3>();

	hkQuaternion rot; rot.setAxisAngle(axis, prng.getRandReal01() * HK_REAL_PI);

	hkTransform t; t.set(rot, trans);

	{
		hkTransform temp;
		temp.setMul( t, rb0->getTransform());
		rb0->setTransform(temp);
	}
	{
		hkTransform temp;
		temp.setMul( t, rb1->getTransform());
		rb1->setTransform(temp);
	}

	transformUsed = t;
}



	// Given params, and expected results: Check if agent calculates things correctly
	// Do an additional "penetration" check for the getPenetrations() method
void testBodies(const sphereBoxTestParams& params, const sphereBoxTestExpectedResults& requiredResults)
{
	hkReal mass = 10.0f;

	hkpRigidBody* box = createBox(params.m_halfExtentsBox, params.m_radiusBox, mass, params.m_positionBox, params.m_rotationBox);
	hkpRigidBody* sphere = createSphere(params.m_radiusSphere, mass, params.m_positionSphere, params.m_rotationSphere);

			// This "transforms" the scene, but maintains the relative transform (hence collision result)
	hkTransform transform = hkTransform::getIdentity();
	//setRandomTransformsForBodies(box, sphere, transform);

	hkpClosestCdPointCollector closestPointCollector;
	{
		hkpCollisionInput input;
		input.setTolerance( params.m_tolerance );
		

			// Do an additional "penetration" check for the getPenetrations() method
		hkpFlagCdBodyPairCollector checker;
		hkpSphereBoxAgent::staticGetPenetrations( *sphere->getCollidable(), *box->getCollidable(), input, checker);
		HK_TEST2(checker.hasHit() == requiredResults.m_penetrating," Agent incorectly reported penetration state ");

		hkpSphereBoxAgent::staticGetClosestPoints( *sphere->getCollidable(), *box->getCollidable(), input, closestPointCollector);
	}

	HK_TEST2( closestPointCollector.hasHit() == requiredResults.m_colliding," Agent incorrectly reported collision state ");

	if(   closestPointCollector.hasHit() )
	{
		const hkpRootCdPoint& closestPoint = closestPointCollector.getHit();
		hkReal dist = closestPointCollector.getHit().m_contact.getDistance();
		if( !hkMath::equal( dist, requiredResults.m_dist, 1e-4f))
		{
			HK_TEST2(0,"Distance:" << dist <<" is not target dist:" << requiredResults.m_dist);
		}

			// Rotate normal back
		hkVector4 normalLocal; normalLocal.setRotatedInverseDir(transform.getRotation(), closestPoint.m_contact.getNormal());
		if ( !normalLocal.allEqual<3>( requiredResults.m_normal, hkSimdReal::fromFloat(1e-4f)))
		{
			HK_TEST2(0,"Normal: " << normalLocal << " does not match target normal " << requiredResults.m_normal);
		}


			// Transform world position back.
		hkVector4 posLocal; posLocal.setTransformedInversePos(transform, closestPoint.m_contact.getPosition());
		if ( ! posLocal.allEqual<3>(requiredResults.m_point, hkSimdReal::fromFloat(1e-4f)))
		{
			HK_TEST2(0,"Position: " << posLocal << " does not match target position " << requiredResults.m_point);
		}
	}

	box->removeReference();
	sphere->removeReference();
	
}






	// Check various configurations, both penetrating and non-penetrating.
int spherebox_main()
{
	sphereBoxTestParams params;
	
	// Test sphere just on the surface of the box
	{
			// Box details
		params.m_halfExtentsBox.set(1,1,1);
		params.m_positionBox.set(0,0,0);
		params.m_rotationBox.setIdentity();
		params.m_radiusBox = 0.0f;


			// Sphere details
		params.m_radiusSphere = 1.0f;
		params.m_positionSphere.set( 0.f,1.f,0.f);
		params.m_rotationSphere.setIdentity();

		params.m_tolerance = 1000.0f; // So will be FORCED to return collision

		sphereBoxTestExpectedResults expectedResults;
		expectedResults.m_penetrating = true;
		expectedResults.m_colliding = true;
		expectedResults.m_dist = -1.f;
		expectedResults.m_normal.set(0,1,0);

			// Calc point as on sphere
		expectedResults.m_point.set( 0.f, 1.f, 0.f);


		testBodies(params, expectedResults);
	}
		// Test sphere 5 units above box, hence closest vector is straight down
	
	{
			// Box details
		params.m_halfExtentsBox.set(1,2,3);
		params.m_positionBox.set(0,0,0);
		hkVector4 axisBox; axisBox.set(0,1,0);
		params.m_rotationBox.setAxisAngle(axisBox, 0.0f);
		params.m_radiusBox = 1.0f;


			// Sphere details
		params.m_radiusSphere = 1.0f;
		params.m_positionSphere.set(-0.4f,5,0.7f);
		hkVector4 axisSphere; axisSphere.set(0,1,0);
		params.m_rotationSphere.setAxisAngle(axisSphere, 0.0f);

		params.m_tolerance = 1000.0f; // So will be FORCED to return collision

		sphereBoxTestExpectedResults expectedResults;
		expectedResults.m_penetrating = false;
		expectedResults.m_colliding = true;
		expectedResults.m_dist = 5.0f - params.m_radiusSphere - params.m_radiusBox;
		expectedResults.m_dist -= params.m_halfExtentsBox(1);
		expectedResults.m_normal.set(0,1,0);

			// Calc point as on sphere
		{
			expectedResults.m_point = params.m_positionSphere;
			expectedResults.m_point.addMul(hkSimdReal::fromFloat(- params.m_radiusSphere - expectedResults.m_dist), expectedResults.m_normal);
		}


		testBodies(params, expectedResults);
	}

		// Test sphere 10,10,-10 units away , hence closest vector is to the corner (1,2,-3)
	{
			// Box details
		params.m_halfExtentsBox.set(1,2,3);
		params.m_positionBox.set(0,0,0);
		hkVector4 axisBox; axisBox.set(0,1,0);
		params.m_rotationBox.setAxisAngle(axisBox, 0.0f);
		params.m_radiusBox = 1.0f;


			// Sphere details
		params.m_radiusSphere = 1.0f;
		params.m_positionSphere.set(10,10,-10);
		hkVector4 axisSphere; axisSphere.set(0,1,0);
		params.m_rotationSphere.setAxisAngle(axisSphere, 0.0f);

		params.m_tolerance = 1000.0f; // So will be FORCED to return collision

		sphereBoxTestExpectedResults expectedResults;
		expectedResults.m_penetrating = false;
		expectedResults.m_colliding = true;

		{
			hkVector4 dir;
			hkVector4 closestB; closestB.set(1,2,-3);
			dir.setSub(params.m_positionSphere, closestB);
			expectedResults.m_dist = dir.length<3>().getReal();
			expectedResults.m_dist -= params.m_radiusSphere;
			expectedResults.m_dist -= params.m_radiusBox;
			dir.normalize<3>();
			expectedResults.m_normal = dir;
		}

				// Calc point as on sphere
		{
			expectedResults.m_point = params.m_positionSphere;
			expectedResults.m_point.addMul(hkSimdReal::fromFloat(- params.m_radiusSphere - expectedResults.m_dist), expectedResults.m_normal);
		}

		testBodies(params, expectedResults);
	}
	

			// Test sphere 0,-10,15 units away , hence closest vector is to the edge (0, -2, 3)
	{
			// Box details
		params.m_halfExtentsBox.set(1,2,3);
		params.m_positionBox.set(0,0,0);
		hkVector4 axisBox; axisBox.set(0,1,0);
		params.m_rotationBox.setAxisAngle(axisBox, 0.0f);
		params.m_radiusBox = 1.0f;


			// Sphere details
		params.m_radiusSphere = 1.0f;
		params.m_positionSphere.set(0,-10,15);
		hkVector4 axisSphere; axisSphere.set(0,1,0);
		params.m_rotationSphere.setAxisAngle(axisSphere, 0.0f);

		params.m_tolerance = 1000.0f; // So will be FORCED to return collision

		sphereBoxTestExpectedResults expectedResults;
		expectedResults.m_penetrating = false;
		expectedResults.m_colliding = true;

		{
			hkVector4 dir;
			hkVector4 closestB; closestB.set(0,-2,3);
			dir.setSub(params.m_positionSphere, closestB);
			expectedResults.m_dist = dir.length<3>().getReal();
			expectedResults.m_dist -= params.m_radiusSphere;
			expectedResults.m_dist -= params.m_radiusBox;

			dir.normalize<3>();
			expectedResults.m_normal = dir;
		}

				// Calc point as on sphere
		{
			expectedResults.m_point = params.m_positionSphere;
			expectedResults.m_point.addMul(hkSimdReal::fromFloat(- params.m_radiusSphere - expectedResults.m_dist), expectedResults.m_normal);
		}


		testBodies(params, expectedResults);
	}
	


			// Test sphere -0.5, 1.1, 1.3 units away, penetrating , hence closest vector is to the edge (-1, 1.1, 1.3)
	{
			// Box details
		params.m_halfExtentsBox.set(1,2,3);
		params.m_positionBox.set(0,0,0);
		hkVector4 axisBox; axisBox.set(0,1,0);
		params.m_rotationBox.setAxisAngle(axisBox, 0.0f);
		params.m_radiusBox = 1.0f;


			// Sphere details
		params.m_radiusSphere = 1.0f;
		params.m_positionSphere.set(-0.5f, 1.1f, 1.3f);
		hkVector4 axisSphere; axisSphere.set(0,1,0);
		params.m_rotationSphere.setAxisAngle(axisSphere, 0.0f);

		params.m_tolerance = 1000.0f; // So will be FORCED to return collision

		sphereBoxTestExpectedResults expectedResults;
		expectedResults.m_penetrating = true;
		expectedResults.m_colliding = true;

		{
			hkVector4 dir;
			hkVector4 closestB; closestB.set(-1, 1.1f, 1.3f);
			dir.setSub(params.m_positionSphere, closestB);
			expectedResults.m_dist = -dir.length<3>().getReal() - params.m_radiusSphere - params.m_radiusBox;
			dir.normalize<3>();
			dir.setNeg<4>(dir);
			expectedResults.m_normal = dir;
		}

				// Calc point as on sphere
		{
			expectedResults.m_point = params.m_positionSphere;
			expectedResults.m_point.addMul(hkSimdReal::fromFloat(- params.m_radiusSphere - expectedResults.m_dist), expectedResults.m_normal);
		}


		testBodies(params, expectedResults);
	}
	


		// Test sphere -1 - eps, -2 - 5 *eps, -3 - 2 * eps units away, hence closest vector is to the corner (-1, -2 , -3)
		// and penetrating.
	hkReal eps = 0.04363246f;
	{
		params.m_halfExtentsBox.set(1,2,3);
		params.m_positionBox.set(0,0,0);
		hkVector4 axisBox; axisBox.set(0,1,0);
		params.m_rotationBox.setAxisAngle(axisBox, 0.0f);
		params.m_radiusBox = 1.0f;


			// Sphere details
		params.m_radiusSphere = 1.0f;
		params.m_positionSphere.set(-1 - eps, -2 - 5 * eps, -3 - 2 * eps);
		hkVector4 axisSphere; axisSphere.set(0,1,0);
		params.m_rotationSphere.setAxisAngle(axisSphere, 0.0f);

		params.m_tolerance = 1000.0f; // So will be FORCED to return collision

		sphereBoxTestExpectedResults expectedResults;
		expectedResults.m_penetrating = true;
		expectedResults.m_colliding = true;

		{
			hkVector4 dir;
			hkVector4 closestB; closestB.set(-1, -2, -3);
			dir.setSub(params.m_positionSphere, closestB);
			expectedResults.m_dist = dir.length<3>().getReal();
			expectedResults.m_dist -= params.m_radiusSphere;
			expectedResults.m_dist -= params.m_radiusBox;

			dir.normalize<3>();
			expectedResults.m_normal = dir;
		}

				// Calc point as on sphere
		{
			expectedResults.m_point = params.m_positionSphere;
			expectedResults.m_point.addMul(hkSimdReal::fromFloat(- params.m_radiusSphere - expectedResults.m_dist), expectedResults.m_normal);
		}


		testBodies(params, expectedResults);
	}
	

		// NOT colliding! (tolerance = 0.1f, but very separated!)
	{
		params.m_halfExtentsBox.set(1,2,3);
		params.m_positionBox.set(0,0,0);
		hkVector4 axisBox; axisBox.set(0,1,0);
		params.m_rotationBox.setAxisAngle(axisBox, 0.0f);
		params.m_radiusBox = 1.0f;


			// Sphere details
		params.m_radiusSphere = 1.0f;
		params.m_positionSphere.set(3,4,5);
		hkVector4 axisSphere; axisSphere.set(0,1,0);
		params.m_rotationSphere.setAxisAngle(axisSphere, 0.0f);

		params.m_tolerance = 0.1f;

		sphereBoxTestExpectedResults expectedResults;
		expectedResults.m_penetrating = false;
		expectedResults.m_colliding = false;

	

		testBodies(params, expectedResults);
	}
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(spherebox_main, "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__     );

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
