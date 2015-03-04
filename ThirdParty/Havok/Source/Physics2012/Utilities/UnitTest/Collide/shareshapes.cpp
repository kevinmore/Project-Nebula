/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeSharing/hkpShapeSharingUtil.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>
#include <Common/Internal/ConvexHull/hkGeometryUtility.h>

/// Dynamics support
#include <Physics2012/Dynamics/Common/hkpProperty.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

/// Collide support
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppUtility.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>

#include <Physics2012/Utilities/Collide/hkpShapeGenerator.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

// Test the hkpShapeSharingUtil

hkArray<hkVector4>* g_convexVerticesVerts = HK_NULL;
hkArray<hkVector4>* g_convexVerticesPlanes = HK_NULL;

hkArray<hkVector4>* g_meshVertices = HK_NULL;
hkArray<hkpSimpleMeshShape::Triangle>* g_meshTriangles = HK_NULL;

static void initGlobals()
{
	// Reusable convex vertices
	{
		g_convexVerticesPlanes = new hkArray<hkVector4>;
		g_convexVerticesVerts = new hkArray<hkVector4>;

		hkPseudoRandomGenerator generator(20);
		hkVector4 a; a.set(-10.0f,-10.0f,-10.f);
		hkVector4 b; b.set(10.0f,10.0f,1.0f);
		const hkpConvexVerticesShape* cvsShape = hkpShapeGenerator::createRandomConvexVerticesShape(a,b, 10,&generator);

		cvsShape->getOriginalVertices(*g_convexVerticesVerts);
		*g_convexVerticesPlanes = cvsShape->getPlaneEquations();

		cvsShape->removeReference();
	}

	// Reusable mesh
	{
		g_meshVertices = new hkArray<hkVector4>;
		g_meshTriangles = new hkArray<hkpSimpleMeshShape::Triangle>;

		const int numVerts = 20;
		for (int i=0; i<numVerts; i++)
		{
			hkVector4 a; a.set(hkUnitTest::rand01()*10-5, hkUnitTest::rand01()*10-5, hkUnitTest::rand01()*10-5);
			g_meshVertices->pushBack(a);
			if (i>=2)
			{
				hkpSimpleMeshShape::Triangle triangle;
				triangle.m_a = i-2;
				triangle.m_b = i-1;
				triangle.m_c = i;
				triangle.m_weldingInfo = 0;

				g_meshTriangles->pushBack(triangle);
			}
		}
	}
}

static void releaseGlobals()
{
	delete g_convexVerticesPlanes;
	delete g_convexVerticesVerts;

	delete g_meshTriangles;
	delete g_meshVertices;

}

static bool _isShapeTypeConvex (hkpShapeType shapeType)
{
	switch (shapeType)
	{
		case hkcdShapeType::SPHERE:
		case hkcdShapeType::BOX:
		case hkcdShapeType::CAPSULE:
		case hkcdShapeType::CYLINDER:
		case hkcdShapeType::CONVEX_VERTICES:
			return true;
		case hkcdShapeType::TRIANGLE_COLLECTION:
			return false;
		default:
			HK_ASSERT2(0x6345ba0b,0,"Unexpected shape type");
			return false;
	}
}

static void _getRandomRotation (hkRotation& rotationOut)
{
	hkVector4 rand_first_row; rand_first_row.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
	rand_first_row.normalize<3>();
	hkVector4 rand_second_row;
	hkVector4Util::calculatePerpendicularVector( rand_first_row, rand_second_row);
	rand_second_row.normalize<3>();

	hkVector4 rand_third_row;
	rand_third_row.setCross( rand_first_row, rand_second_row );

	rotationOut.setRows( rand_first_row, rand_second_row, rand_third_row );

}

static void _getRandomTranslation (hkVector4& translationOut)
{
	translationOut.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
}

static void _deleteRbs(hkArray<hkpRigidBody*>& rigidBodies)
{
	for (int j=0; j<rigidBodies.getSize(); j++)
	{
		rigidBodies[j]->removeReference();
	}
	rigidBodies.clear();
}

//create a placeholder rigid body
static hkpRigidBody* shareshapes_createRB( const hkpShape* shape )
{
	hkpRigidBodyCinfo info;
	{
		info.m_shape = shape;

		info.m_position.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );			
		hkVector4 axis; axis.set( 0.0f, 1.0f, 0.0f );
		info.m_rotation.setAxisAngle( axis, HK_REAL_PI / ((hkUnitTest::rand01() * 4.0f) + 0.01f) );
		info.m_linearVelocity.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		info.m_angularVelocity.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );

		// [1,1,1] vector has length sqrt(3) not 1
		info.m_maxAngularVelocity = hkUnitTest::randRange( 1.733f, 100.0f );
		info.m_maxLinearVelocity = hkUnitTest::randRange( 1.733f, 100.0f );

		info.m_linearDamping = hkUnitTest::rand01();
		info.m_angularDamping = hkUnitTest::rand01();

		info.m_friction = hkUnitTest::rand01();
		info.m_restitution = hkUnitTest::rand01();			
		info.m_motionType = hkpMotion::MOTION_DYNAMIC;							
		info.m_qualityType = HK_COLLIDABLE_QUALITY_FIXED;
	}

	hkpRigidBody* body = new hkpRigidBody( info );
	return body;
}

static const hkpShape* createSphereShape (float e, bool)
{
	return new hkpSphereShape(5.0f+e);
}

static const hkpShape* createBoxShape (float e,bool)
{
	hkVector4 a; a.set(1.0f+e, 1.0f+e,1.0f+e);
	return new hkpBoxShape (a, hkConvexShapeDefaultRadius+e);
}

static const hkpShape* createCapsule (float e, bool)
{
	const bool flip = hkUnitTest::rand01() >= 0.5f;

	hkVector4 pa; pa.set (-2.0f-e, -2.0f-e, -2.0f-e);
	hkVector4 pb; pb.set (2.0f+e, 2.0f+e, 2.0f+e);

	const hkReal radius = 3.0f + e;

	if (flip)
	{
		return new hkpCapsuleShape (pb, pa, radius);
	}
	else
	{
		return new hkpCapsuleShape (pa, pb, radius);
	}
}

static const hkpShape* createCylinder (float e, bool)
{
	const bool flip = hkUnitTest::rand01() >= 0.5f;

	hkVector4 pa; pa.set (-2.0f-e, -2.0f-e, -2.0f-e);
	hkVector4 pb; pb.set (2.0f+e, 2.0f+e, 2.0f+e);

	const hkReal radius = 2.0f+e;
	const hkReal cvxRadius = hkConvexShapeDefaultRadius+e;

	if (flip)
	{
		return new hkpCylinderShape (pb, pa, radius, cvxRadius );
	}
	else
	{
		return new hkpCylinderShape (pa, pb, radius, cvxRadius );
	}
}

template<typename T>
static void _shuffleArray(hkArray<T>& array)
{
	if (array.getSize()<2) return;

	// swap the last two items to ensure that the output is different from input
	hkAlgorithm::swap(array[array.getSize()-2],array[array.getSize()-1]);

	// then shuffle the n-2 remaining elements
	const int size = array.getSize()-2;

	const int numShuffles = (int) (hkUnitTest::rand01()*size*0.4f);
	for (int i=0; i<numShuffles; i++)
	{
		const int orig = (int) (hkUnitTest::rand01()*size);
		const int dest = (int) (hkUnitTest::rand01()*size);

		T temp = array[orig];
		array[orig] = array[dest];
		array[dest] = temp;
	}
}

static void _perturbVectorArray(hkArray<hkVector4>& array, const float error)
{
	for (int i=0; i<array.getSize(); i++)
	{
		// Perturb a random x,y or z component
		int component = (int) (hkUnitTest::rand01()*3.0f);
		array[i](component)+=error;
	}
}

static const hkpShape* createConvexVertices (float e, bool permute)
{
	hkArray<hkVector4> vertsCopy; vertsCopy =  *g_convexVerticesVerts;

	hkArray<hkVector4> planesCopy; planesCopy = *g_convexVerticesPlanes;

	// Shuffle 
	if (permute) 
	{
		_shuffleArray(vertsCopy);
		_shuffleArray(planesCopy);
	}


	const hkReal cvxRadius = hkConvexShapeDefaultRadius + e;

	hkStridedVertices stridedVerts;
	stridedVerts.m_numVertices = vertsCopy.getSize();
	stridedVerts.m_striding = sizeof(hkVector4);
	stridedVerts.m_vertices = &(vertsCopy[0](0));

	const hkpConvexVerticesShape* newShape = new hkpConvexVerticesShape(stridedVerts, planesCopy, cvxRadius);
	return newShape;

}

static const hkpShape* createSimpleMesh (float e, bool permute)
{
	hkArray<hkVector4> vertsCopy; vertsCopy = *g_meshVertices;
	_perturbVectorArray(vertsCopy, e);

	hkArray<hkpSimpleMeshShape::Triangle> triangCopy; triangCopy = *g_meshTriangles;

	// Shuffle triangles
	if (permute) _shuffleArray(triangCopy);

	const hkReal cvxRadius = hkConvexShapeDefaultRadius + e;

	hkpSimpleMeshShape* simpleMeshShape = new hkpSimpleMeshShape(cvxRadius);

	simpleMeshShape->m_triangles = triangCopy;
	simpleMeshShape->m_vertices = vertsCopy;


	const bool addMopp = permute && (hkUnitTest::rand01()>=0.5f);

	if (addMopp)
	{
		hkpMoppCompilerInput mci;
		#ifdef HK_PLATFORM_PS3
		mci.m_enableChunkSubdivision = true;
		#endif

		hkpMoppCode* moppCode = hkpMoppUtility::buildCode(simpleMeshShape, mci);

		hkpMoppBvTreeShape* moppBvTreeShape = new hkpMoppBvTreeShape(simpleMeshShape, moppCode);
		simpleMeshShape->removeReference();
		moppCode->removeReference();

		return moppBvTreeShape;
	}
	else
	{
		return simpleMeshShape;
	}
}

typedef const hkpShape* (*_ShapeCreator) (float,bool);

struct _ShapeCreationFuncPair
{
	hkpShapeType m_shapeType;
	_ShapeCreator m_creator;
};

_ShapeCreationFuncPair g_shapeCreators[] =
{
	{hkcdShapeType::SPHERE, createSphereShape},
	{hkcdShapeType::BOX, createBoxShape},
	{hkcdShapeType::CAPSULE, createCapsule},
	{hkcdShapeType::CYLINDER, createCylinder},
	{hkcdShapeType::CONVEX_VERTICES, createConvexVertices},
	{hkcdShapeType::TRIANGLE_COLLECTION, createSimpleMesh},
};

static const int g_numShapeCreators = sizeof(g_shapeCreators) / sizeof(_ShapeCreationFuncPair);


/*  
	RB1->S1
	RB2->S2
	..
	RBn->Sn  

	where Sx==Sy 
*/

static int test_rigidBodiesSharingShape()
{
	// Create 2-10 rigid bodies, all sharing the same cloned shape
	for (int i=0; i<g_numShapeCreators; i++)
	{
		for (int nr=2; nr<10; nr++)
		{
			hkArray<hkpRigidBody*> rigidBodies;
			for (int rbi=0; rbi<nr; rbi++)
			{
				const hkpShape* theShape = g_shapeCreators[i].m_creator(0.0f, true);
				hkpRigidBody* rb = shareshapes_createRB(theShape);
				theShape->removeReference();
				rigidBodies.pushBack(rb);
			}

			hkpShapeSharingUtil::Options options;
			hkpShapeSharingUtil::Results results;
			hkpShapeSharingUtil::shareShapes(rigidBodies,options,results);

			HK_TEST(results.m_numSharedShapes == (nr-1));

			for (int j=0; j<nr; j++)
			{
				delete rigidBodies[j];
			}
		}
	}
	return 0;
}


/*
	RB1->T1->S1
	RB2->T2->S2
	...
	RBx->Tn->Sn 

	where Sx==Sy but Tx!=Ty
*/


/*
	RB1->Ta->Tb->...->S
	RB2->Tc->S
	...

	different depths all pointing to the same shape at the end
*/

/*
	RB1->L1->(S1,S2,..,Sn)
	RB2->L2->(S1',S2'...,Sn')
	...
	RBm->Lm->(S1''',S2''', ..., Sn''')

	where S1==S1'==S1'''
*/
static int test_listshapeSharingShape()
{
	for (int nr=2; nr<10; nr++)
	{
		hkArray<hkpRigidBody*> rigidBodies;
		for (int rbi=0; rbi<nr; rbi++)
		{
			hkArray<const hkpShape*> childrenShapes;

			for (int st=0; st<g_numShapeCreators; st++)
			{
				const hkpShape* childShape = g_shapeCreators[st].m_creator(0.0f, true);
				childrenShapes.pushBack(childShape);
			}

			_shuffleArray(childrenShapes);

			const hkpListShape* listShape = new hkpListShape(childrenShapes.begin(), childrenShapes.getSize());
			for (int j=0; j<childrenShapes.getSize(); j++)
			{
				childrenShapes[j]->removeReference();
			}

			hkpRigidBody* rb = shareshapes_createRB(listShape);
			listShape->removeReference();

			rigidBodies.pushBack(rb);
		}

		hkpShapeSharingUtil::Options options;
		hkpShapeSharingUtil::Results results;
		hkpShapeSharingUtil::shareShapes(rigidBodies, options, results);

		HK_TEST(results.m_numSharedShapes == (g_numShapeCreators+1)*(nr-1));

		// Test that all rigid bodies share the same root shape (the list)
		{
			for (int j=1; j<nr; j++)
			{
				const hkpShape* prevShape = rigidBodies[j-1]->getCollidable()->getShape();
				const hkpShape* thisShape = rigidBodies[j]->getCollidable()->getShape();
				HK_TEST (prevShape==thisShape);
			}
		}

		// Delete rigid bodies
		{
			for (int j=0; j<nr; j++)
			{
				delete rigidBodies[j];
			}
		}
	}
	return 0;
}


/*
	RB1->S1
	RB2->S2
	..
	RBb->Sn

	where Sx = S1+0.1f*x difference

	test that different thresholds will return different amount of shared shapes
*/
static int test_equalityThreshold()
{
	const float errors[] = { 0.0f, 0.01f, 0.004f, 0.5f };
	const int numErrors = sizeof(errors)/sizeof(float);

	for (int ei1=0; ei1<numErrors; ei1++) 
	{
		for (int ei2=0; ei2<numErrors; ei2++)
		{
			if (ei1==ei2) continue;

			const float error = errors[ei1];
			const float threshold = errors[ei2];

			for (int st=0; st<g_numShapeCreators; st++) 
			{
				hkArray<hkpRigidBody*> rigidBodies;

				// Create 2 rigid bodies, one with error, one with no error
				for (int rbi=0; rbi<2; rbi++)
				{
					const hkpShape* theShape = g_shapeCreators[st].m_creator(error*rbi, true);
					hkpRigidBody* rb = shareshapes_createRB(theShape);
					theShape->removeReference();
					rigidBodies.pushBack(rb);
				}

				hkpShapeSharingUtil::Options options;
				options.m_equalityThreshold = threshold;
				hkpShapeSharingUtil::Results results;
				hkpShapeSharingUtil::shareShapes(rigidBodies,options,results);

				// No shapes should be shared if above threshold
				if (error<threshold)
				{
					HK_TEST(results.m_numSharedShapes == 1);
				}
				else
				{
					HK_TEST(results.m_numSharedShapes == 0);
				}

				_deleteRbs(rigidBodies);
			}
		}

	}

	return 0;
}

/*
	Test the m_detectPermutedComponents option
*/
static int test_permuteOnOff()
{
	bool doPermute = false;
	while (1) // doPermute = false, true
	{
		bool checkPermute = true;
		while (1) // checkPermute = false, true
		{
			for (int st=0; st<g_numShapeCreators; st++) 
			{
				hkArray<hkpRigidBody*> rigidBodies;

				// Create 2 rigid bodies, one not permuted, second one possibly permuted
				for (int rbi=0; rbi<2; rbi++)
				{
					const bool permuted = (rbi==0) ? false : doPermute;
					const hkpShape* theShape = g_shapeCreators[st].m_creator(0.0f, permuted);
					hkpRigidBody* rb = shareshapes_createRB(theShape);
					theShape->removeReference();
					rigidBodies.pushBack(rb);
				}

				hkpShapeSharingUtil::Options options;
				options.m_detectPermutedComponents =  checkPermute;
				hkpShapeSharingUtil::Results results;
				hkpShapeSharingUtil::shareShapes(rigidBodies,options,results);

				bool shouldShare = true;
				switch (g_shapeCreators[st].m_shapeType)
				{
					case hkcdShapeType::SPHERE:
					case hkcdShapeType::BOX:
					case hkcdShapeType::CAPSULE:
					case hkcdShapeType::CYLINDER:
						shouldShare = true ;// as permutations are always detected
						break;
					case hkcdShapeType::CONVEX_VERTICES:
					case hkcdShapeType::TRIANGLE_COLLECTION:
						shouldShare = !doPermute || checkPermute;
						break;
					default:
						break;
				}

				if (shouldShare)
				{
					HK_TEST(results.m_numSharedShapes == 1);
				}
				else
				{
					HK_TEST(results.m_numSharedShapes == 0);
				}

				_deleteRbs(rigidBodies);
			}

			checkPermute = !checkPermute;
			if (checkPermute==false) break;
		}

		doPermute = !doPermute;
		if (doPermute==false) break;
	}

	return 0;
}

/*
    RB1->box
	RB2->sphere
	RB3->capsule
	..

	test completely different shapes (no sharing)
*/
static int test_differentShapes()
{
	hkArray<hkpRigidBody*> rigidBodies;

	for (int st=0; st<g_numShapeCreators; st++)
	{
		const hkpShape* theShape = g_shapeCreators[st].m_creator(0.0f,true);
		hkpRigidBody* rb = shareshapes_createRB(theShape);
		theShape->removeReference();
		rigidBodies.pushBack(rb);
	}

	hkpShapeSharingUtil::Options options;
	hkpShapeSharingUtil::Results results;
	hkpShapeSharingUtil::shareShapes(rigidBodies,options,results);

	// Nothing should be shared
	HK_TEST(results.m_numSharedShapes==0);

	_deleteRbs(rigidBodies);

	return 0;
}

/*
	Test that if shapes are already shared we don't do anything
*/
static int test_noWorkIfAlreadyShared()
{
	hkArray<hkpRigidBody*> rigidBodies;
	hkpShapeSharingUtil::Options options;
	hkpShapeSharingUtil::Results results;

	for (int st=0; st<g_numShapeCreators; st++)
	{

		const hkpShape* theShape = g_shapeCreators[st].m_creator(0.0f,true);
		hkpRigidBody* rb1 = shareshapes_createRB(theShape);
		hkpRigidBody* rb2 = shareshapes_createRB(theShape);
		theShape->removeReference();
		rigidBodies.pushBack(rb1);
		rigidBodies.pushBack(rb2);

		hkpShapeSharingUtil::shareShapes(rigidBodies, options, results);

		HK_TEST(results.m_numSharedShapes == 0);

		_deleteRbs(rigidBodies);
	}

	return 0;
}


/*
	Test same shapes with different user data
*/
static int test_userData()
{
	const void *userData[4] = { "Blah", &hkError::getInstance(), g_meshTriangles, HK_NULL };

	for (int ud1=0; ud1<4; ud1++) 
	{
		for (int ud2=0; ud2<4; ud2++)
		{
			for (int st=0; st<g_numShapeCreators; st++) 
			{
				hkArray<hkpRigidBody*> rigidBodies;

				for (int rbi=0; rbi<2; rbi++)
				{
					const hkpShape* theShape = g_shapeCreators[st].m_creator(0.0f, false);
				
					{
						hkpShape* nonconstShape = const_cast<hkpShape*> (theShape);
						nonconstShape->setUserData( reinterpret_cast<hkUlong> (rbi==0 ? userData[ud1] : userData[ud2]) );
					}

					hkpRigidBody* rb = shareshapes_createRB(theShape);
					theShape->removeReference();
					rigidBodies.pushBack(rb);
				}

				hkpShapeSharingUtil::Options options;
				hkpShapeSharingUtil::Results results;
				hkpShapeSharingUtil::shareShapes(rigidBodies,options,results);

				if (ud1==ud2)
				{
					HK_TEST(results.m_numSharedShapes == 1);
				}
				else
				{
					HK_TEST(results.m_numSharedShapes == 0);
				}

				_deleteRbs(rigidBodies);
			}
		}
	}

	return 0;
}


static int shareshapes_main()
{
	initGlobals();

	hkError::getInstance().setEnabled(0x2ff8c16c, false); 
	hkError::getInstance().setEnabled(0x2ff8c16d, false); 
	hkError::getInstance().setEnabled(0x2ff8c16e, false); 
	hkError::getInstance().setEnabled(0x2ff8c16f, false); 
	hkError::getInstance().setEnabled(0x6e8d163b, false); 

	test_rigidBodiesSharingShape();
	test_listshapeSharingShape();
	test_equalityThreshold();
	test_permuteOnOff();
	test_differentShapes();
	test_noWorkIfAlreadyShared();
	test_userData();

	hkError::getInstance().setEnabled(0x2ff8c16c, true); 
	hkError::getInstance().setEnabled(0x2ff8c16d, true); 
	hkError::getInstance().setEnabled(0x2ff8c16e, true); 
	hkError::getInstance().setEnabled(0x2ff8c16f, true); 
	hkError::getInstance().setEnabled(0x6e8d163b, true); 

	releaseGlobals();

	return 0;

}   

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(shareshapes_main, "Slow", "Physics2012/Test/UnitTest/Utilities/", __FILE__     );

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
