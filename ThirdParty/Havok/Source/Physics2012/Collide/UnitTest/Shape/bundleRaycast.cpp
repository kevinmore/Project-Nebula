/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
 // Large include
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppUtility.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastOutput.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayBundleCastInput.h>
#include <Physics2012/Collide/Filter/Null/hkpNullCollisionFilter.h>

#include <Common/Internal/ConvexHull/hkGeometryUtility.h>


//
// Do both regular and bundled raycast, compare the results
//
void compareRaycasts( const hkpShape* shape, const hkpShapeRayBundleCastInput& input, 
						const char* description, hkVector4Comparison::Mask _mask = hkVector4ComparisonMask::MASK_XYZW)
{
	hkpShapeRayBundleCastOutput outRegular, outBundle;
	hkVector4Comparison mask;
	mask.set(_mask);

	shape->castRayBundle(input, outBundle, mask);
	shape->hkpShape::castRayBundle(input, outRegular, mask); // call the fallback implementation

	const hkReal tolerance = 0.0001f;

	for (int i=0; i<4; i++)
	{
		hkpShapeRayCastOutput& A = outBundle.m_outputs[i];
		hkpShapeRayCastOutput& B = outRegular.m_outputs[i];
		HK_TEST2( hkMath::equal(A.m_hitFraction, B.m_hitFraction, tolerance), description);
		if (A.hasHit())
		{
			HK_TEST2(A.m_normal.allEqual<3>(B.m_normal, hkSimdReal::fromFloat(1e-3f)), description);
		}
	}
}

//
// bundle_raycast_tests method, actually does the ray tests
// The rays here were taken from raycast.cpp unit test, to make sure that both versions give the same results
//
void bundle_raycast_tests( const hkpShape* shape, const char* desciption, hkBool isConvex )
{
	// all rays are 0.5 units long
	hkpNullCollisionFilter nullCollisionFilter;
	
	{
		hkpShapeRayBundleCastInput bundleRay;
		hkpShapeRayCastInput ray[4];

		// ray outside cube
		ray[0].m_from.set( 0.0f, 1.5f, 0.0f );
		ray[0].m_to  .set( 0.0f, 1.0f, 0.0f );
		ray[0].m_rayShapeCollectionFilter = &nullCollisionFilter;	// We need this for MOPPs

		// ray penetrating cube, start outside cube
		ray[1].m_from.set( 0.0f, 0.9f, 0.0f );
		ray[1].m_to  .set( 0.0f, 0.4f, 0.0f );
		ray[1].m_rayShapeCollectionFilter = &nullCollisionFilter;

		// ray outside cube, end touching surface
		ray[2].m_from.set( 0.0f, 1.0f, 0.0f );
		ray[2].m_to  .set( 0.0f, 0.5f, 0.0f );
		ray[2].m_rayShapeCollectionFilter = &nullCollisionFilter;

		// ray outside cube, parallel to a face
		ray[3].m_from.set( 0.5f, 1.5f, 0.0f );
		ray[3].m_to  .set( 0.5f, 1.0f, 0.0f );
		ray[3].m_rayShapeCollectionFilter = &nullCollisionFilter;

		hkBundleRays(ray, bundleRay);
		compareRaycasts(shape, bundleRay, desciption);
	}

	{
		hkpShapeRayBundleCastInput bundleRay;
		hkpShapeRayCastInput ray[4];

		// ray inside cube, start touching surface
		ray[0].m_from.set( 0.0f, 0.5f, 0.0f );
		ray[0].m_to  .set( 0.0f, 0.0f, 0.0f );
		ray[0].m_rayShapeCollectionFilter = &nullCollisionFilter;	// We need this for MOPPs

		// zero length ray on the surface
		ray[1].m_from.set( 0.0f, 0.5f, 0.0f );
		ray[1].m_to  .set( 0.0f, 0.5f, 0.0f );
		ray[1].m_rayShapeCollectionFilter = &nullCollisionFilter;

		// ray completely inside cube
		ray[2].m_from.set( 0.0f, 0.25f, 0.0f );
		ray[2].m_to  .set( 0.0f,-0.25f, 0.0f );
		ray[2].m_rayShapeCollectionFilter = &nullCollisionFilter;

		// ray inside cube, end touching surface
		ray[3].m_from.set( 0.0f, 0.0f, 0.0f );
		ray[3].m_to  .set( 0.0f,-0.5f, 0.0f );
		ray[3].m_rayShapeCollectionFilter = &nullCollisionFilter;

		hkBundleRays(ray, bundleRay);
		compareRaycasts(shape, bundleRay, desciption);
	}

	// Only 3 raycasts here, so try masking off one of the inputs too
	{
		hkpShapeRayBundleCastInput bundleRay;
		bundleRay.m_rayShapeCollectionFilter = &nullCollisionFilter;	// We need this for MOPPs
		hkpShapeRayCastInput ray[4];

		// raycast penetrating cube, start inside cube
		ray[0].m_from.set( 0.0f,-0.1f, 0.0f );
		ray[0].m_to  .set( 0.0f,-0.6f, 0.0f );
		ray[0].m_rayShapeCollectionFilter = &nullCollisionFilter;	// We need this for MOPPs

		// deliberately skip ray[1] (y component)
		ray[1].m_rayShapeCollectionFilter = &nullCollisionFilter;
		hkVector4Comparison::Mask mask = hkVector4ComparisonMask::MASK_XZW;

		// raycast outside cube, start touching cube
		ray[2].m_from.set( 0.0f,-0.5f, 0.0f );
		ray[2].m_to  .set( 0.0f,-1.0f, 0.0f );
		ray[2].m_rayShapeCollectionFilter = &nullCollisionFilter;

		// start and end outside cube but penetrating it twice
		ray[3].m_from.set( 0.0f, 1.0f, 0.0f );
		ray[3].m_to  .set( 0.0f,-1.0f, 0.0f );
		ray[3].m_rayShapeCollectionFilter = &nullCollisionFilter;

		hkBundleRays(ray, bundleRay);
		compareRaycasts(shape, bundleRay, desciption, mask);
	}
}



//
// createMoppShape helper method
//
static hkpMoppBvTreeShape* createMoppBvTreeShape( const hkVector4* vertices, const int numVertices,
												 const short* indices, const int numTriangles, const int indicesStride )
{
	// build storage mesh shape, will be owned by the MoppBVTreeShape
	hkpSimpleMeshShape* meshShape = new hkpSimpleMeshShape(0.0001f);

	meshShape->m_vertices.setSize(numVertices);
	meshShape->m_triangles.setSize(numTriangles);
	for ( int i = 0; i < numVertices; i++ )
	{
		meshShape->m_vertices[i] = vertices[i];
	}

	int curT = 0;
	const int numIndicesPerTriangle = indicesStride / sizeof( short );
	for ( unsigned int j = 0; curT < numTriangles; j += numIndicesPerTriangle )
	{
		meshShape->m_triangles[curT].m_a = indices[j];
		meshShape->m_triangles[curT].m_b = indices[j+1];
		meshShape->m_triangles[curT].m_c = indices[j+2];
		curT++;
	}

	// build MOPP code, will be owned by the MoppBVTreeShape
	hkpMoppCompilerInput mci;
	mci.m_enableChunkSubdivision = true;
	// Usually MOPPs are not built at run time but preprocessed instead. We disable the performance warning
	bool wasEnabled = hkError::getInstance().isEnabled(0x6e8d163b); 
	hkError::getInstance().setEnabled(0x6e8d163b, false); // hkpMoppUtility.cpp:18
	hkpMoppCode* code = hkpMoppUtility::buildCode( meshShape , mci);
	hkError::getInstance().setEnabled(0x6e8d163b, wasEnabled);

	// build MOPP BV Tree shape
	hkpMoppBvTreeShape* moppShape = new hkpMoppBvTreeShape( meshShape, code );
	code->removeReference();
	meshShape->removeReference();

	// return our new shape
	return moppShape;
}


//
// createListOfTrianglesShape helper method
//
static hkpListShape* createListOfTrianglesShape( const hkVector4* vertices, const unsigned int numVertices,
												const unsigned int numTriangles,
												const short* indices, const unsigned int numIndices, const unsigned int indicesStride )
{
	// create new triangle shapes
	hkArray<hkpShape*> triangles;
	for ( unsigned int i = 0; i < numIndices; i += ( indicesStride / sizeof( short ) ) )
	{
		triangles.pushBack( new hkpTriangleShape( vertices[indices[i]],
			vertices[indices[i + 1]],
			vertices[indices[i + 2]] ) );
	}

	// build list shape and add triangle shapes to it (copies triangle shapes)
	hkpListShape* listShape = new hkpListShape( triangles.begin(), triangles.getSize() );

	// clean up
	for ( unsigned int j = 0; j < numTriangles; j++ )
	{
		triangles[j]->removeReference();
	}
	triangles.clear();

	// return our new shape
	return listShape;
}



//
// Havok2 raycast test
//
int bundle_raycast_test()
{

	//
	// Create an axis aligned unit cube centred around the origin
	//
	const unsigned int numVertices = 8;
	hkVector4 halfExtents; halfExtents.set( 0.5f, 0.5f, 0.5f );

	hkVector4 vertices[numVertices]; 
	{
		vertices[0].set(-0.5, 0.5, 0.5);
		vertices[1].set( 0.5, 0.5, 0.5);
		vertices[2].set( 0.5,-0.5, 0.5);
		vertices[3].set(-0.5,-0.5, 0.5);
		vertices[4].set(-0.5, 0.5,-0.5);
		vertices[5].set( 0.5, 0.5,-0.5);
		vertices[6].set( 0.5,-0.5,-0.5);
		vertices[7].set(-0.5,-0.5,-0.5);
	}
	
	const unsigned int numVerticesPerTriangle = 3;
	const unsigned int numTriangles = 12;
	const unsigned int numIndices = numTriangles * numVerticesPerTriangle;
	const unsigned int indicesStride = numVerticesPerTriangle * sizeof( short );
	const short indices[numIndices] =
	{
		3,2,1,
		3,1,0,
		6,7,4,
		6,4,5,
		4,7,3,
		4,3,0,
		2,6,5,
		2,5,1,
		7,6,2,
		7,2,3,
		1,5,4,
		1,4,0
	};


	//
	// create shapes and test
	//
	const hkBool CONVEX = true;
	const hkBool CONCAVE = false;

	// MOPP BV tree shape
	
/*	{
		const hkpMoppBvTreeShape* moppBvTreeShape = createMoppBvTreeShape( vertices, numVertices, indices, numTriangles, indicesStride );
		bundle_raycast_tests( moppBvTreeShape, "MoppBvTreeShape", CONCAVE );
		moppBvTreeShape->removeReference();
	}*/

	
	// sphere
	{
		const hkReal radius = 0.5f;
		hkpSphereShape sphereShape( radius );
		bundle_raycast_tests( &sphereShape, "SphereShape", CONVEX );
	}

	// create list shape of triangle shapes, run test, delete it again
	{
		const hkpListShape* listOfTrianglesShape = createListOfTrianglesShape( vertices, numVertices, numTriangles, indices, numIndices, indicesStride );
		bundle_raycast_tests( listOfTrianglesShape, "ListOfTrianglesShape", CONCAVE);
		listOfTrianglesShape->removeReference();
	}

	// convex vertices shape
	{

		hkpConvexVerticesShape convexVerticesShape(hkStridedVertices(vertices,numVertices));


		bundle_raycast_tests( &convexVerticesShape, "ConvexVerticesShape_planeEquations", CONVEX);
	}

	// convex vertices shape - no plane equations
	{

		hkpConvexVerticesShape convexVerticesShape(hkStridedVertices(vertices,numVertices));
		convexVerticesShape.m_planeEquations.setSize(0);
		
		bundle_raycast_tests( &convexVerticesShape, "ConvexVerticesShape_MPR", CONVEX);
	}

	// implicit box shape
	{
		hkpBoxShape boxShape( halfExtents, 0 );
		bundle_raycast_tests( &boxShape, "BoxShape", CONVEX );
	}
	return 0;
}


//
// test registration
//
#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(bundle_raycast_test, "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__     );

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
