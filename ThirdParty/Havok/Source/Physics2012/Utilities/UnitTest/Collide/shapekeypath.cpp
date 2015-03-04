/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Types/Geometry/hkStridedVertices.h>

/// Dynamics support
#include <Physics2012/Dynamics/Common/hkpProperty.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactListener.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>

/// Collide support
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>
#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>

#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastInput.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Dispatch/hkpAgentRegisterUtil.h>

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeKeyPath/hkpShapeKeyPath.h>

////////////////////////////////////////////////////////////
// Test of the shape key iterator. 

////////////////////////////////////////////////////////////
// Utility functions.
// 

// Creates a placeholder rigid body.
static hkpRigidBody* createRB( const hkpShape* shape )
{
	hkpRigidBodyCinfo info;
	{
		info.m_shape = shape;
		info.m_motionType = hkpMotion::MOTION_FIXED;
		info.m_numShapeKeysInContactPointProperties = -1;
	}

	hkpRigidBody* body = new hkpRigidBody( info );
	return body;
}

// Create a mesh shape.
static hkpSimpleMeshShape* createMesh()
{

	hkpSimpleMeshShape* simpleMesh = new hkpSimpleMeshShape(0.0f);

	simpleMesh->m_vertices.expandOne().set(-1.f,-1.f,0.f);
	simpleMesh->m_vertices.expandOne().set( 1.f,-1.f,0.f);
	simpleMesh->m_vertices.expandOne().set(-1.f, 1.f,0.f);
	simpleMesh->m_vertices.expandOne().set( 1.f, 1.f,0.f);

	hkpSimpleMeshShape::Triangle triA = {0,1,2,0};
	hkpSimpleMeshShape::Triangle triB = {1,3,2,0};
	simpleMesh->m_triangles.pushBack(triA);
	simpleMesh->m_triangles.pushBack(triB);
	return simpleMesh;
}

// Creates a convex shape choosing between sphere, box, capsule, convex vertices.
static const int NUM_CONVEX_SHAPES = 4; 
hkpConvexShape* createConvexShape()
{
	switch( hkUnitTest::rand() % NUM_CONVEX_SHAPES)
	{
		case 0:
			return new hkpSphereShape(1.f);	
		case 1:
		{
			hkVector4 halfExts;
			halfExts.set(1.f,1.f,1.f);
			return new hkpBoxShape( halfExts );
		}
		case 2:
		{
			hkVector4 a,b;
			a.set(1.f,0.f,0.f);
			b.set(-1.f,0.f,0.f);
			return new hkpCapsuleShape( a , b , 0.2f );
		}
		case 3:
        {
		    // Data specific to this shape.
            int numVertices = 4;

			int stride = hkSizeOf(hkVector4);
			hkReal v3 = hkMath::sqrt(3.f);

            hkReal vertices[] = { // 4 vertices plus padding
                -1.0f, 0.0f,  2.0f, 0.0f, // v0
                -1.0f,   v3, -1.0f, 0.0f, // v1
                -1.0f,  -v3, -1.0f, 0.0f, // v2
                 3.0f, 0.0f,  0.0f, 0.0f  // v3
            };
            
            hkStridedVertices stridedVerts;
            {
                stridedVerts.m_numVertices = numVertices;
                stridedVerts.m_striding = stride;
                stridedVerts.m_vertices = vertices;
            }
			return new hkpConvexVerticesShape(stridedVerts);
		}
		default:
			return HK_NULL;
	}	
}

// Creates a hkpConvexTransform with a random rotation.
static hkpConvexShape* wrapShapeWithRandomTransform(hkpConvexShape* shape)
{
	hkQuaternion randomQuat;
	hkTransform transform;
	randomQuat.setFromEulerAngles(hkUnitTest::randRange(0,HK_REAL_PI) ,hkUnitTest::randRange(0,HK_REAL_PI), hkUnitTest::randRange(0,HK_REAL_PI));
	transform.set(randomQuat, hkVector4::getZero());
	
	hkpConvexShape* result = new hkpConvexTransformShape( shape, transform );
	return result ;
}

////////////////////////////////////////////////////////////
// Helper classes.
// 

//Test case data.
static const int NUM_TEST_CASES = 5;
class TestCase 
{
	public:
        TestCase(int testCaseNum);
        ~TestCase();

	public:
        // Root shape hierarchy
        hkpShape* m_shape;
        // Array of shape types that should be returned by the iterator in root-to leaf order
        hkArray<hkcdShape::ShapeType> m_resultShapeTypes;
        
        // Size of the hierarchy array for each test case
        static const int hierarchySize[NUM_TEST_CASES] ;
};

const int TestCase::hierarchySize[] = {1, 4 , 2, 2 , 5 } ;

TestCase::TestCase( int testCaseNum )
{

	HK_ASSERT( 0x3196a8bb, testCaseNum < NUM_TEST_CASES );

	m_shape = HK_NULL;
	m_resultShapeTypes.setSize(hierarchySize[testCaseNum]);
	
	switch (testCaseNum)
	{
		case 0: // A simple convex shape.
		{
			m_shape = createConvexShape();	
			m_resultShapeTypes[0] = m_shape->getType();
			break;
		}
		
		case 1: // A series of 3 transforms. 
		{
			
			//TODO remake this

			hkpConvexShape* shape = createConvexShape();	
			m_resultShapeTypes[3] = shape->getType();
			for (int i = 0 ; i < 3 ; ++i)
			{
				hkpConvexShape* newShape = wrapShapeWithRandomTransform(shape);
				m_resultShapeTypes[2-i] = newShape->getType();
				shape->removeReference();
				shape = newShape;
			}
			m_shape = shape;
			break;

		}

		case 2: // A list with 3 shapes in it.
		{
			hkpShape* shapes[3];
			hkVector4 translation;

			// Create 2 shapes translated far away so they won't be hit.
			hkpConvexShape* shape0 = createConvexShape();
			translation.set(0.f,10.f,0.f); 
			shapes[0] = new hkpConvexTranslateShape( shape0, translation ); 
			shape0->removeReference();

			hkpConvexShape* shape1 = createConvexShape();
			translation.set(0.f,-10.f,0.f); 
			shapes[1] = new hkpConvexTranslateShape( shape1, translation ); 
			shape1->removeReference();

			// Now create the shape that will get hit.
			shapes[2] = createConvexShape();
            m_resultShapeTypes[1] = shapes[2]->getType();

			m_shape = new hkpListShape( shapes, 3 ); 
			// Cleaning.
            m_resultShapeTypes[0] = m_shape->getType();
			for (int i = 0; i < 3; ++i)
			{
				shapes[i]->removeReference();
			}
			
			break;
		}

		case 3: // A static compound with a mesh shape.
				 // SCS flattens its children by  one level so the array will be [SCS, TRIANGLE]
		{
			hkpStaticCompoundShape* scs = new hkpStaticCompoundShape;
			hkVector4 translation;
			hkQsTransform transform;
			translation.set(0.f,10.f,0.f); 
			transform.set( translation, hkQuaternion::getIdentity());
			
			// Create a convex shape translated far away so it won't get hit.		
			hkpConvexShape* shape1 = createConvexShape();

			scs->addInstance(shape1, transform);
			shape1->removeReference();

			// Create a simple mesh.
			hkpSimpleMeshShape* meshShape = createMesh();
			scs->addInstance( meshShape, hkQsTransform::getIdentity() );
			meshShape->removeReference();
			
			// Fill the result array.
			hkpShapeKey key = meshShape->getContainer()->getFirstKey();
			hkpShapeBuffer buf;
			m_resultShapeTypes[1]  = meshShape->getContainer()->getChildShape(key, buf)->getType();		
	
			scs->bake();

			m_shape = scs ;
			m_resultShapeTypes[0] = scs->getType();
			break;
		}

		case 4: // SCS with a complex hierarchy
				 // SCS -> Convex
				 //     -> Convex
				 //     -> List	-> Translate -> Mesh
				 //				-> Transform -> Transform -> Transform- > Convex  
				 // The SCS will ignore the list since it flattens the hierarchy by one level
				 // the result should be [SCS, TRANSFORM, TRANSFORM, TRANSFORM, CONVEX]
		{
		
			hkpStaticCompoundShape* scs = new hkpStaticCompoundShape;

			// Create the 2 first-level convex shapes, far away.
			hkpConvexShape* shape1 = createConvexShape();
			hkQsTransform transform1;
			hkVector4 translation1;
			translation1.set(0.f,-10.f,0.f);
			transform1.set(translation1, hkQuaternion::getIdentity());

			//scs->addInstance(shape1, transform1);
			shape1->removeReference();


			hkpConvexShape* shape2 = createConvexShape();
			hkQsTransform transform2;
			hkVector4 translation2;
			translation1.set(0.f,-10.f,0.f);
			transform2.set(translation2, hkQuaternion::getIdentity());

			//scs->addInstance(shape2, transform2);
			shape2->removeReference();
			
			// Create the list.
			{
				hkpShape* shapeArray[2];

				// We add a mesh, far away.
				hkpSimpleMeshShape* meshShape = createMesh();
				hkTransform transform3;
				hkVector4 translation3;
				translation3.set(0.f,20.f,0.f);
				transform3.set(hkQuaternion::getIdentity(), translation3);

				hkpShape* tMeshShape = new hkpTransformShape( meshShape, transform3 ); 
				shapeArray[0] = tMeshShape;
				meshShape->removeReference();

				hkpConvexShape* shape = createConvexShape();	
				
				m_resultShapeTypes[4] = shape->getType();

				// Create the three successive transforms.
				for (int i = 0 ; i < 3 ; ++i)
				{
					hkpConvexShape* newShape = wrapShapeWithRandomTransform(shape);
					shape->removeReference();
					shape = newShape;
                    m_resultShapeTypes[3-i] = shape->getType();
				}		
				
				shapeArray[1] = shape;
				hkpListShape* listShape = new hkpListShape ( shapeArray, 2);
				
				scs->addInstance( listShape, hkQsTransform::getIdentity() );

				shapeArray[0]->removeReference();
				shapeArray[1]->removeReference();
				listShape->removeReference();
			}

			scs->bake();
			m_shape = scs;
			m_resultShapeTypes[0] = scs->getType();
			break;
		} 
	}// End of switch
}
TestCase::~TestCase()
{
	m_shape->removeReference();
}

// Custom contact listener class, responsible for testing the iterator with a ContactPointEvent
class TestContactListener : public hkReferencedObject, public hkpContactListener
{
	public:
		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DEMO );
		TestContactListener( const TestCase& testCase );
		virtual void contactPointCallback( const hkpContactPointEvent& event );

	public:	
		const TestCase& m_testCase;
		bool m_hasHit;
};

TestContactListener::TestContactListener( const TestCase& testCase) : m_testCase(testCase), m_hasHit(false)
{
}

// Tests the iterator when the collision is detected.
void TestContactListener::contactPointCallback( const hkpContactPointEvent& event )
{
	HK_ASSERT2(  0x71da8897, event.getBody(event.m_source)->getCollidable()->getShape() == m_testCase.m_shape, 
	"This listener was not attached to the expected body" );

	hkpShapeKeyPath path(event, event.m_source);

	int j = 0 ;
	for (hkpShapeKeyPath::Iterator it = path.getIterator(); it.isValid(); it.next(), ++j)
	//while( it.isValid() )
	{

		HK_TEST(m_testCase.m_resultShapeTypes[j] == it.getShape()->getType());
	}

	// Test the getShapes() function

    const int MAX_SHAPES = 10;
    const hkpShape* shapes[MAX_SHAPES];
    hkpShapeBuffer buffers[MAX_SHAPES];

    int numShapes = 0;
    path.getShapes( MAX_SHAPES, buffers, shapes, numShapes );

    HK_TEST( numShapes == m_testCase.m_resultShapeTypes.getSize() );

    const hkpShape* leafShape = shapes[numShapes - 1];

    HK_TEST( leafShape->isConvex() );

    hkpShapeType type = leafShape->getType();

    HK_TEST( type != hkcdShapeType::CONVEX_TRANSFORM );
    HK_TEST( type != hkcdShapeType::CONVEX_TRANSLATE );

	m_hasHit = true;
}

////////////////////////////////////////////////////////////
// Test functions.
//

static void testWorldRaycast( const TestCase& test )
{
		hkpWorldCinfo info;
        {
            info.m_gravity.setZero();
            info.setBroadPhaseWorldSize( 100.0f );
            info.m_enableDeactivation = false;
        }

		hkpRigidBody* body = createRB( test.m_shape );
		
		hkpWorld* world = new hkpWorld( info );
		world->addEntity( body );

		hkpWorldRayCastInput input ;
		input.m_from.set( 0.f, 0.f,  2.f) ;
		input.m_to.set  ( 0.f, 0.f, -2.f) ;

		hkpWorldRayCastOutput output ;
		
		world->castRay(input,output);

		HK_TEST( output.hasHit());
		if ( output.hasHit() )
		{
			hkpShapeKeyPath path( output );
			int j = 0 ;
			for (hkpShapeKeyPath::Iterator it = path.getIterator(); it.isValid(); it.next(), ++j)
			//while( it.isValid() )
			{
				HK_TEST( test.m_resultShapeTypes[j] == it.getShape()->getType() );
			}
		} 
		
		world->removeReference();
		body->removeReference();
}

static void testShapeRaycast( const TestCase& test )
{
		hkpShapeRayCastInput input ;
		input.m_from.set( 0.f, 0.f,  2.f) ;
		input.m_to.set  ( 0.f, 0.f, -2.f) ;

		hkpShapeRayCastOutput output ;
		
		test.m_shape->castRay( input, output );

		HK_TEST( output.hasHit());
		if ( output.hasHit() )
		{
			hkpShapeKeyPath path( test.m_shape, output );
			int j = 0 ;
			for (hkpShapeKeyPath::Iterator it = path.getIterator(); it.isValid(); it.next(), ++j)
			//while( it.isValid() )
			{
				HK_TEST( test.m_resultShapeTypes[j] == it.getShape()->getType() );
			}
		} 
		
}

static void testCollision( const TestCase& test)
{
        hkpWorldCinfo info;
        {
            info.m_gravity.setZero();
            info.setBroadPhaseWorldSize( 100.0f );
            info.m_enableDeactivation = false;
        }

		hkpRigidBody* body = createRB(test.m_shape);
		TestContactListener* listener = new TestContactListener(test);		
		body->addContactListener(listener);
		
		hkpWorld* world = new hkpWorld(info);

		world->lock();
		
		hkpAgentRegisterUtil::registerAllAgents(world->getCollisionDispatcher());
		world->addEntity( body );

		// Create the projectile triggering the contact
		hkpSphereShape* projectileShape = new hkpSphereShape (0.5f) ;
		hkpRigidBodyCinfo projectileInfo ;
		projectileInfo.m_shape = projectileShape;
		
		projectileInfo.m_mass = 1.f;
		hkMassProperties massprops;
		hkpInertiaTensorComputer::computeShapeVolumeMassProperties( projectileInfo.m_shape, 1.f, massprops );
		projectileInfo.m_motionType = hkpMotion::MOTION_SPHERE_INERTIA;
		projectileInfo.m_inertiaTensor = massprops.m_inertiaTensor;

		hkpRigidBody* projectile = new hkpRigidBody( projectileInfo );
		
		hkVector4 pos, vel;
		pos.set(0.f,0.f,-3.f);
		vel.set(0.f,0.f,1.f);
		projectile->setPosition( pos ); 
		projectile->setLinearVelocity( vel );
		
		world->addEntity( projectile );
		projectile->removeReference();
		projectileShape->removeReference();

		world->unlock();
		
		int iterations = 0;
		while ( !(listener->m_hasHit) )
		{
			world->stepDeltaTime( 1.f/60.f );
			++iterations;
			HK_ASSERT2( 0x432a074b, iterations < 1000, " Too many iterations" );
		}

		world->removeReference();
		listener->removeReference();
		body->removeReference();
}


/////////////////////////////////////////////////////////////////////////////////////
static int shapekeypath_main()
{

	for (int  i = 0 ; i < NUM_TEST_CASES ; ++i)
	{
		TestCase test(i) ;
		testWorldRaycast(test);
		testShapeRaycast(test);
		testCollision(test);
	}
	return 0;

}   

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(shapekeypath_main, "Slow", "Physics2012/Test/UnitTest/Utilities/", __FILE__ );

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
