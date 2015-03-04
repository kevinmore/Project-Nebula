/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Internal/hkpInternal.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>
#include <Physics2012/Collide/Query/Collector/RayCollector/hkpAllRayHitCollector.h>
#include <Physics2012/Collide/Filter/Group/hkpGroupFilter.h>


void setupIcosahedron( hkpConvexVerticesShape& shape, hkQsTransform& transform ) 
{
	// Data specific to this shape.
	int numVertices = 12;

	// 16 = 4 (size of "each float group", 3 for x,y,z, 1 for padding) * 4 (size of float)
	int stride = hkSizeOf(hkVector4);

	// dodecahedron vertices
	const hkReal phi = hkReal(hkMath::sqrt(5.0f)/2.0f + 0.5f);
	HK_ALIGN_REAL(hkReal vertices[]) = {
		0,  1,  phi, 0, 
		0, -1,  phi, 0, 
		0,  1, -phi, 0, 
		0, -1, -phi, 0,

		 1,  phi, 0, 0, 
		-1,  phi, 0, 0, 
		 1, -phi, 0, 0, 
		-1, -phi, 0, 0, 

		 phi, 0,  1, 0, 
		-phi, 0,  1, 0, 
		 phi, 0, -1, 0, 
		-phi, 0, -1, 0 
	};


	hkStridedVertices stridedVerts;
	{
		stridedVerts.m_numVertices = numVertices;
		stridedVerts.m_striding = stride;
		stridedVerts.m_vertices = vertices;
	} 

	// set build config for hkpConvexVertices
	hkpConvexVerticesShape::BuildConfig build;
	{
		build.m_convexRadius = 0;
	}

	// transform vertices
	for (int i=0; i<numVertices; ++i)
	{
		int j = i*4;
		hkVector4 v; v.set(vertices[j], vertices[j+1], vertices[j+2], vertices[j+3]);
		v.setTransformedPos(transform, v);
		vertices[j] = v(0); vertices[j+1] = v(1); vertices[j+2] = v(2); vertices[j+3] = v(3);
	}

	// create the new shape in-place
	new (&shape) hkpConvexVerticesShape(stridedVerts, build);
}


void checkRayCastNormalTransformation()
{
	//////////////////////////////////////////////////////////////////////////
	// adds a single shape to the static compound shape, and
	// compares against an external shape with the same transform
	//////////////////////////////////////////////////////////////////////////

	// basic icosahedron
	hkpConvexVerticesShape ico;
	{
		hkQsTransform identity; identity.setIdentity();
		setupIcosahedron(ico, identity);
	}

	// some transform
	hkQsTransform transform;
	{
		transform.setIdentity();
		hkVector4 scale; scale.set(1.2f,-0.2f,1.3f); transform.setScale(scale);
		hkVector4 translation; translation.set(4.0f,-2.3f,5.7f); transform.setTranslation(translation);
		hkVector4 axis; axis.set(5.3f,7.2f,-5.1f); axis.normalize<3>();
		hkQuaternion rotation; rotation.setAxisAngle(axis,1.2353f); transform.setRotation(rotation);
	}

	// insert into the static compound shape
	hkpStaticCompoundShape shape1;
	{
		shape1.addInstance(&ico, transform );
		shape1.bake();
	}

	// create a transformed vertices shape using the same
	// transform as the one used with static compound shape
	hkpConvexTransformShape shape2(&ico, transform);

	// setup the input for the ray casts
	hkpShapeRayCastInput input;
	{
		shape2.getCentre( input.m_to);
		input.m_from.set(6,2,-4);
	}

	// perform ray casts
	hkpShapeRayCastOutput output1; hkpShapeRayCastOutput output2;
	{
		shape1.castRay(input,output1);
		shape2.castRay(input,output2);
	}

	// compare hit fractions 			
	hkReal test1 = hkMath::abs(output1.m_hitFraction-output2.m_hitFraction); 

	// compare normals
	hkReal test2 = output1.m_normal.distanceTo(output2.m_normal).getReal(); 

	HK_TEST2( test1 < 1e-4, "Hit fractions must be the same or transformation is not consistent");
	HK_TEST2( test2 < 1e-4, "Hit normals are expected to be the same");




	// repeat this test using a collector
	{
		// cd body to be used for collector calls
		hkpCdBody body(&shape1,&hkTransform::getIdentity());

		hkpAllRayHitCollector collector; 
		shape1.castRayWithCollector(input,body,collector);

		// test 1
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect one hit
		HK_TEST2( results.getSize() == 1, "Expecting exactly one hit from collector"); 

		// check normal of hit
		hkVector4 normal = results[0].m_normal;
		hkReal test3 = normal.distanceTo(output2.m_normal).getReal(); 
		HK_TEST2( test3 < 1e-4, "Hit normals are expected to be the same");
	}


	// do the test again using a collector, but with a non-unit hkpCdBody transform
	{
		// some rigid body transform
		hkTransform rigid;
		{
			rigid.setIdentity();
			hkVector4 axis; axis.set(1,1,1); axis.normalize<3>();
			hkRotation rotation; rotation.setAxisAngle(axis, 0.58763f); rigid.setRotation(rotation);
		}

		// cd body to be used for collector calls
		hkpCdBody body(&shape1,&rigid);

		hkpAllRayHitCollector collector; 
		shape1.castRayWithCollector(input,body,collector);

		// test 1
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect one hit
		HK_TEST2( results.getSize() == 1, "Expecting exactly one hit from collector"); 

		// check normal of hit
		hkVector4 normalFromCollector = results[0].m_normal;

		// transform the normal of the transformed ico by the
		// rigid body transform. (this is the normal direction we expect)
		hkVector4 normalFromConvexShape;
		{
			normalFromConvexShape.setRotatedDir(rigid.getRotation(), output2.m_normal);
		}

		hkReal test3 = normalFromCollector.distanceTo(normalFromConvexShape).getReal(); 
		HK_TEST2( test3 < 1e-4, "Hit normals are expected to be the same");
	}
}

void simpleRayCastDisableEnableTest()
{
	//////////////////////////////////////////////////////////////////////////
	// sets up a hkpStaticCompoundShape with a single box and tests that
	// castRay() respects the box being disabled and enabled
	//////////////////////////////////////////////////////////////////////////

	// setup a box
	hkVector4 v; v.set(1,1,1);
	const hkpBoxShape box(v,0.0f);

	// setup a compound shape with a single box inside
	hkpStaticCompoundShape shape; int id;
	{
		id = shape.addInstance(&box, hkQsTransform::getIdentity());
		shape.bake();
	}

	// a ray that aims to the center of
	// the box
	hkpShapeRayCastInput input; hkpShapeRayCastOutput out;
	{
		box.getCentre( input.m_to);
		input.m_from.set(6,2,-4);
	}

	// cast the ray
	hkBool hit = shape.castRay(input, out);

	// expect a hit
	HK_TEST(hit);

	// disable the box
	shape.setInstanceEnabled(id,false);

	// cast the ray
	out.reset();
	hit = shape.castRay(input, out);

	// expect no hit
	HK_TEST(!hit);

	// re-enable the box
	shape.setInstanceEnabled(id,true);

	// cast the ray
	out.reset();
	hit = shape.castRay(input, out);

	// expect hit
	HK_TEST(hit);
}

void simpleRayCastWithCollectorDisableEnableTest()
{
	//////////////////////////////////////////////////////////////////////////
	// sets up a hkpStaticCompoundShape with two boxes. The two boxes are
	// disabled and enabled in various combinations and compared to the outcome
	// of calling castRayWithCollector on the compound shape. Setup is two boxes
	// (1,1,1) at (-1,0,0) and (1,0,0). Hit fraction for b1 is 1/6 and 1/2 for b2
	//
	//      |----|----|
	//   *--x b1 x b2 |-->     ray from (-3,0,0) to (3,0,0)
	//      |----|----|
	//  -3 -2    0    2  3
	//
	//////////////////////////////////////////////////////////////////////////

	// setup a box
	hkVector4 v; v.set(1,1,1);
	const hkpBoxShape box1(v,0.0f);
	const hkpBoxShape box2(v,0.0f);

	// setup a compound shape with a single box inside
	hkpStaticCompoundShape shape; int id1; int id2;
	{
		hkQsTransform t1; t1.setIdentity(); hkVector4 tr1; tr1.set(-1,0,0); t1.setTranslation(tr1);
		id1 = shape.addInstance(&box1,t1);
		hkQsTransform t2; t2.setIdentity(); hkVector4 tr2; tr2.set(1,0,0); t2.setTranslation(tr2);
		id2 = shape.addInstance(&box2, t2);
		shape.bake();
	}

	//  ray from (-3,0,0) to (3,0,0)
	hkpShapeRayCastInput input; hkpShapeRayCastOutput out;
	{
		input.m_to.set(3,0,0);
		input.m_from.set(-3,0,0);
	}
	
	// cd body to be used for collector calls
	hkpCdBody body(&shape,&hkTransform::getIdentity());

	// first test, non disabled
	{
		hkpAllRayHitCollector collector; 
		shape.castRayWithCollector(input,body,collector);

		// test 1
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect two hits
		HK_TEST2( results.getSize() == 2, "Got the wrong number of hits form collector"); 
		// expect hit at 1/6 and 1/2
		HK_TEST2( hkMath::abs( results[0].m_hitFraction - 1.0f/6.0f ) < 1e-4, "Did not get the expected hit fractions from collector" ); 
		HK_TEST2( hkMath::abs( results[1].m_hitFraction - 1.0f/2.0f ) < 1e-4, "Did not get the expected hit fractions from collector" ); 

	}

	// second test, first box disabled
	{
		shape.setInstanceEnabled(id1, false);

		hkpAllRayHitCollector collector; 
		shape.castRayWithCollector(input,body,collector);

		// test 1
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect one hit
		HK_TEST2( results.getSize() == 1, "Got the wrong number of hits form collector"); 
		// expect hit at 1/6 and 1/2
		HK_TEST2( hkMath::abs( results[0].m_hitFraction - 1.0f/2.0f ) < 1e-4, "Did not get the expected hit fractions from collector" ); 

		// re-enable
		shape.setInstanceEnabled(id1, true);
	}

	// third test, second box disabled
	{
		shape.setInstanceEnabled(id2, false);

		hkpAllRayHitCollector collector; 
		shape.castRayWithCollector(input,body,collector);

		// test 1
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect one hit
		HK_TEST2( results.getSize() == 1, "Got the wrong number of hits from collector"); 
		// expect hit at 1/6 and 1/2
		HK_TEST2( hkMath::abs( results[0].m_hitFraction - 1.0f/6.0f ) < 1e-4, "Did not get the expected hit fractions from collector" ); 

		// re-enable
		shape.setInstanceEnabled(id2, true);
	}


	// fourth test, both disabled
	{
		shape.setInstanceEnabled(id1, false);
		shape.setInstanceEnabled(id2, false);

		hkpAllRayHitCollector collector; 
		shape.castRayWithCollector(input,body,collector);
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect no hits
		HK_TEST2( results.getSize() == 0, "Got the wrong number of hits form collector"); 

		// re-enable
		shape.setInstanceEnabled(id1, true);
		shape.setInstanceEnabled(id2, true);
	}
}

void verySimpleRayCastFilteringTest()
{
	//////////////////////////////////////////////////////////////////////////
	// adds a single box to the compound shape and tests a filter where all 
	// collisions are disabled
	//////////////////////////////////////////////////////////////////////////

	// setup a box
	hkVector4 v; v.set(1,1,1);
	const hkpBoxShape box(v,0.0f);

	// setup a compound shape with a single box inside
	hkpStaticCompoundShape shape; int id;
	{
		id = shape.addInstance(&box, hkQsTransform::getIdentity());
		shape.bake();
	}

	// setup a group filter
	hkpGroupFilter filter;
	{
		// disable all layers
		filter.disableCollisionsUsingBitfield(0xfffffffe, 0xfffffffe);
	}

	// a ray that aims to the center of the box
	hkpShapeRayCastInput input; hkpShapeRayCastOutput out;
	{
		box.getCentre( input.m_to);
		input.m_from.set(6,6,6);
		input.m_rayShapeCollectionFilter = &filter; 
		input.m_filterInfo = hkpGroupFilter::calcFilterInfo(1);
	}

	// set layer 1
	shape.setInstanceFilterInfo(id,hkpGroupFilter::calcFilterInfo(1));

	// cast the ray
	hkBool hit = shape.castRay(input, out);

	// expect a miss
	HK_TEST2(!hit, "Ray hit when it was supposed to miss");
}


void simpleRayCastFilteringTest()
{
	//////////////////////////////////////////////////////////////////////////
	// adds a single box to the compound shape and tests simple filtering
	//////////////////////////////////////////////////////////////////////////

	// setup a box
	hkVector4 v; v.set(1,1,1);
	const hkpBoxShape box(v,0.0f);

	// setup a compound shape with a single box inside
	hkpStaticCompoundShape shape; int id;
	{
		id = shape.addInstance(&box, hkQsTransform::getIdentity());
		shape.bake();
	}

	// setup a group filter
	hkpGroupFilter filter;
	{
		// disable all layers
		filter.disableCollisionsUsingBitfield(0xfffffffe, 0xfffffffe);
	}

	// enable collisions for layer 1
	filter.enableCollisionsBetween(1,1);


	// a ray that aims to the center of the box
	hkpShapeRayCastInput input; hkpShapeRayCastOutput out;
	{
		box.getCentre( input.m_to);
		input.m_from.set(6,6,6);
		input.m_rayShapeCollectionFilter = &filter; 
		input.m_filterInfo = hkpGroupFilter::calcFilterInfo(1);
	}

	// set layer 1
	shape.setInstanceFilterInfo(id,hkpGroupFilter::calcFilterInfo(1));

	// cast the ray
	hkBool hit = shape.castRay(input, out);

	// expect a hit
	HK_TEST2(hit, "Ray missed when it was expected to hit");

	// set layer two	
	shape.setInstanceFilterInfo(id,hkpGroupFilter::calcFilterInfo(2));

	// cast the ray
	hit = shape.castRay(input, out);

	// expect a miss
	HK_TEST2(!hit, "Ray hit when it was supposed to miss");
}

void simpleRayCastWithCollectorFilteringTest()
{
	//////////////////////////////////////////////////////////////////////////
	// sets up a hkpStaticCompoundShape with a single box and tests that
	// castRay() respects the box being disabled and enabled
	//////////////////////////////////////////////////////////////////////////

	// setup a box
	hkVector4 v; v.set(1,1,1);
	const hkpBoxShape box(v,0.0f);

	// setup a compound shape with a single box inside
	hkpStaticCompoundShape shape; int id;
	{
		id = shape.addInstance(&box, hkQsTransform::getIdentity());
		shape.bake();
	}

	// setup a group filter
	hkpGroupFilter filter;
	{
		// disable all layers
		filter.disableCollisionsUsingBitfield(0xfffffffe, 0xfffffffe);
	}

	// enable collisions for layer 1 and disable 1-2
	filter.enableCollisionsBetween(1,1);
	filter.disableCollisionsBetween(1,2);

	// a ray that aims to the center of the box
	hkpShapeRayCastInput input; hkpShapeRayCastOutput out;
	{
		box.getCentre( input.m_to);
		input.m_from.set(6,7,8);
		input.m_rayShapeCollectionFilter = &filter; 
		input.m_filterInfo = hkpGroupFilter::calcFilterInfo(1);
	}

	// set layer 1
	shape.setInstanceFilterInfo(id,hkpGroupFilter::calcFilterInfo(1));

	{
		hkpAllRayHitCollector collector; 
		hkpCdBody body(&shape,&hkTransform::getIdentity());
		shape.castRayWithCollector(input,body,collector);
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect one hits
		HK_TEST2( results.getSize() == 1, "Got the wrong number of hits"); 
	}

	// set layer 2
	shape.setInstanceFilterInfo(id,hkpGroupFilter::calcFilterInfo(2));
	{
		hkpAllRayHitCollector collector; 
		hkpCdBody body(&shape,&hkTransform::getIdentity());
		shape.castRayWithCollector(input,body,collector);
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect no hits because the box is in layer 2 (ray in layer 1)
		HK_TEST2( results.getSize() == 0, "Got the wrong number of hits"); 
	}
}

void gridRayCastWithCollectorFilteringTest()
{
	//////////////////////////////////////////////////////////////////////////
	// set up a 16 by 16 grid of boxes, and assign different layers to them.
	// shoot rays at the grid and test that filtering works as expected 
	//////////////////////////////////////////////////////////////////////////

	//  X    1   2   3   4   5   6   7   8    ...   16
	//
	//     |---|---|---|---|---|---|---|---|       |---|
	//     |   |   |   |   |   |   |   |   |  ...  |   |   16
	//     |---|---|---|---|---|---|---|---|       |---| 
    //       .           .               .           .
    //       .       .                   .           .
	//       .   .                       .           .
	//     |---|---|---|---|---|---|---|---|       |---| 
	//     |0,2|2,2|...|   |   |   |   |   |  ...  |   |    2
	//     |---|---|---|---|---|---|---|---|       |---|
	// o-->|0,0|2,0|...|   |   |   |   |   |  ...  |   |->  1
	//     |---|---|---|---|---|---|---|---|       |---| 
    //                                                      Y (layer)

	// setup 1-1-1 box
	hkVector4 v; v.set(1,1,1);
	const hkpBoxShape box(v,0.0f);

	// create a grid of boxes in the XY plane
	hkpStaticCompoundShape shape; int id[16][16];
	{
		for (int i=0;i<16;++i)
		{
			for (int j=0;j<16;++j)
			{
				hkQsTransform transform; transform.setIdentity(); 
				hkVector4 translation; translation.set(i*2.0f,j*2.0f,0); transform.setTranslation(translation);
				id[i][j] = shape.addInstance(&box, transform);
			}
		}

		shape.bake();
	}

	//go through array and set layers (set layer by the Y tick plus one)
	for (int i=0;i<16;++i)
		for (int j=0;j<16;++j)
			shape.setInstanceFilterInfo(id[i][j], hkpGroupFilter::calcFilterInfo(j+1));


	// setup a group filter
	hkpGroupFilter filter;
	{
		// disable all layers
		filter.disableCollisionsUsingBitfield(0xfffffffe, 0xfffffffe);
	}

	// enable collisions for layers (1,1)
	filter.enableCollisionsBetween(1,1);

	// setup a ray that goes through the first row of boxes. The ray is set to be in layer 1 
	hkpShapeRayCastInput input; 
	{
		input.m_to.set(32+1,0,0);
		input.m_from.set(-5,0,0);
		input.m_rayShapeCollectionFilter = &filter; 
		input.m_filterInfo = hkpGroupFilter::calcFilterInfo(1);
	}

	// shoot the ray through the first row of boxes
	{
		hkpAllRayHitCollector collector; 
		hkpCdBody body(&shape,&hkTransform::getIdentity());
		shape.castRayWithCollector(input,body,collector);
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect 16 hits
		HK_TEST2( results.getSize() == 16, "Got the wrong number of hits"); 
	}

	// disable (1,1) 
	filter.disableCollisionsBetween(1,1);

	// shoot the ray through the first row of boxes
	{
		hkpAllRayHitCollector collector; 
		hkpCdBody body(&shape,&hkTransform::getIdentity());
		shape.castRayWithCollector(input,body,collector);
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect 0 hits
		HK_TEST2( results.getSize() == 0, "Ray cast returned hits, even though collisions are disabled for the ray and boxes"); 
	}

	// set the ray to run through the first column vertically.
	// set the filter to allow collision between 1,1 and 1,2 
	{
		input.m_to.set(0,32+1,0);
		input.m_from.set(0,-5,0);

		filter.enableCollisionsBetween(1,1);
		filter.enableCollisionsBetween(1,2);
	}

	// shoot the ray through the first column of boxes
	{
		hkpAllRayHitCollector collector; 
		hkpCdBody body(&shape,&hkTransform::getIdentity());
		shape.castRayWithCollector(input,body,collector);
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect two hits
		HK_TEST2( results.getSize() == 2, "Only collisions between layers 1,1 and layers 1,2 are enabled. Ray cast returned a different amount of hits"); 
	}
}

void nestedCollectionRayCastTest()
{
	// setup a box
	hkVector4 v; v.set(1,1,1);
	const hkpBoxShape box(v,0.0f);

	// setup a compound shape with a single box inside
	hkpStaticCompoundShape shape1; int boxid;
	{
		hkQsTransform transform; transform.setIdentity(); 
		hkVector4 translation; translation.set(5.0f,0.0f,0.0f); transform.setTranslation(translation);
		boxid = shape1.addInstance(&box, transform);
		shape1.bake();
	}

	// setup another compound shape containing the first compound shape
	hkpStaticCompoundShape shape2; int compoundid;
	{
		hkQsTransform transform; transform.setIdentity(); 
		hkVector4 translation; translation.set(5.0f,0.0f,0.0f); transform.setTranslation(translation);
		compoundid = shape2.addInstance(&shape1, transform);
		shape2.bake();
	}

	// box should now have its center at 10,0,0
	// setup the input for the ray casts
	hkpShapeRayCastInput input;
	{
		input.m_from.set(5.0f,0,0);
		input.m_to.set(10.0f,0,0);
	}

	// expect a hit at (9,0,0), hit fraction 4/5
	hkpShapeRayCastOutput out;
	{
		shape2.castRay(input,out);
		HK_TEST2( hkMath::abs(out.m_hitFraction - 4.0f/5.0f) < 1e-4, "Hit fraction not as expected" );
	}

	// do the same test using a collector
	{
		hkpAllRayHitCollector collector; 
		hkpCdBody body(&shape2,&hkTransform::getIdentity());
		shape2.castRayWithCollector(input,body,collector);
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect one hit
		HK_TEST2( results.getSize() == 1, "Collector failed to return a single hit"); 

		if (results.getSize() == 1)
		{
			// expect a hit at (9,0,0), hit fraction 4/5
			HK_TEST2( hkMath::abs(results[0].m_hitFraction - 4.0f/5.0f) < 1e-4, "Hit fraction not as expected" );
		}
	}
}

void nestedCollectionRayCastFilteringTest()
{
	// setup a box
	hkVector4 v; v.set(1,1,1);
	const hkpBoxShape box(v,0.0f);

	// setup a compound shape with a single box inside
	hkpStaticCompoundShape shape1; int box1id;
	{
		hkQsTransform transform; transform.setIdentity(); 
		hkVector4 translation; translation.set(5.0f,0.0f,0.0f); transform.setTranslation(translation);
		box1id = shape1.addInstance(&box, transform);
		shape1.setInstanceFilterInfo(box1id, hkpGroupFilter::calcFilterInfo(1));
		shape1.bake();
	}

	// setup another compound shape containing the first compound shape and another box
	hkpStaticCompoundShape shape2; int compoundid; int box2id;
	{
		hkQsTransform transform; transform.setIdentity(); 
		hkVector4 translation; translation.set(5.0f,0.0f,0.0f); transform.setTranslation(translation);
		compoundid = shape2.addInstance(&shape1, transform);

		// add box at (0,0,0)
		box2id = shape2.addInstance(&box, hkQsTransform::getIdentity());
		shape2.setInstanceFilterInfo(box2id, hkpGroupFilter::calcFilterInfo(1));


		shape2.bake();
	}


	// setup a group filter
	hkpGroupFilter filter;
	{
		// disable all layers
		filter.disableCollisionsUsingBitfield(0xfffffffe, 0xfffffffe);
		// enable collisions for layers (1,1)
		filter.enableCollisionsBetween(1,1);
	}


	// looking at shape 2, there is now a box at (0,0,0) and at (10,0,0) and both of them
	// have group layer 1 as filter info. Setup a ray the goes through them both
	hkpShapeRayCastInput input;
	{
		input.m_from.set(-2.0f,0,0);
		input.m_to.set(10.0f,0,0);
		input.m_rayShapeCollectionFilter = &filter; 
		input.m_filterInfo = hkpGroupFilter::calcFilterInfo(1);
	}


	// check that we hit both boxes using the collector
	{
		hkpAllRayHitCollector collector; 
		hkpCdBody body(&shape2,&hkTransform::getIdentity());
		shape2.castRayWithCollector(input,body,collector);
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect two hits
		HK_TEST2( results.getSize() == 2, "Excpected two hits, but collector returned somehing else"); 
	}

	// change the filter info for the nested box
	shape1.setInstanceFilterInfo(box1id, hkpGroupFilter::calcFilterInfo(2));


	// check that we now hit one box only 
	{
		hkpAllRayHitCollector collector; 
		hkpCdBody body(&shape2,&hkTransform::getIdentity());
		shape2.castRayWithCollector(input,body,collector);
		const hkArray<hkpWorldRayCastOutput>& results = collector.getHits();

		// expect two hits
		HK_TEST2( results.getSize() == 1, "Expected to hit one box, but got wrong result"); 
	}

}


// Tests the use of the initial hit fraction in the hkpShapeRayCastOuput structure
void testRayCastInitialFraction()
{
	hkVector4 halfExtents; halfExtents.set(1, 1, 1);
	hkpBoxShape box(halfExtents, 0);
	hkpStaticCompoundShape compound;		
	compound.addInstance(&box, hkQsTransform::getIdentity());
	compound.bake();			

	hkpShapeRayCastInput input;
	input.m_from.set(-2, 0, 0);
	input.m_to.set(2, 0, 0);	

	// We will run the tests twice, with and without filter, to check both code paths in 
	// hkpStaticCompoundShape_Internals::RayCastQuery::castLeaf
	hkpGroupFilter filter;
	for (int useFilter = 0; useFilter < 2; ++useFilter)
	{
		input.m_rayShapeCollectionFilter = (useFilter ? &filter : HK_NULL);

		// Miss
		{																	
			hkpShapeRayCastOutput output;
			output.m_hitFraction = 0.1f;				
			const hkBool hasHit = compound.castRay(input, output);

			HK_TEST(!hasHit);
			HK_TEST(output.m_hitFraction == 0.1f);
		}

		// Hit
		{								
			hkpShapeRayCastOutput output;
			output.m_hitFraction = 0.5f;
			const hkBool hasHit = compound.castRay(input, output);

			HK_TEST(hasHit);
			HK_TEST(hkMath::equal(output.m_hitFraction, 0.25f));
		}

		// Border case
		{																
			hkpShapeRayCastOutput output;
			output.m_hitFraction = 0.25f;
			const hkBool hasHit = compound.castRay(input, output);

			HK_TEST(!hasHit);
			HK_TEST(output.m_hitFraction == 0.25f);
		}	
	}
}


// Tests the shape key hierarchies returned by castRay
void testRayCastShapeKeyHierarchy()
{
	hkVector4 halfExtents; halfExtents.set(1, 1, 1);
	hkpBoxShape box(halfExtents, 0);

	// We will run the tests twice, with and without filter, to check both code paths in 
	// hkpStaticCompoundShape_Internals::RayCastQuery::castLeaf
	hkpGroupFilter filter;
	for (int useFilter = 0; useFilter < 2; ++useFilter)
	{

		// SCS with an instance of a non-container shape
		{			
			hkpStaticCompoundShape compound;		
			compound.addInstance(&box, hkQsTransform::getIdentity());
			compound.bake();	

			hkpShapeRayCastInput input;
			input.m_rayShapeCollectionFilter = (useFilter ? &filter : HK_NULL);

			// Hit
			{									
				input.m_from.set(-2, 0, 0);
				input.m_to.set(2, 0, 0);				
				hkpShapeRayCastOutput output;
				compound.castRay(input, output);

				HK_TEST(output.getLevel() == 0);
				HK_TEST(output.m_shapeKeys[0] == 0);
				HK_TEST(output.m_shapeKeys[1] == HK_INVALID_SHAPE_KEY);			
			}

			// Hit with shape keys in the initial output
			{									
				input.m_from.set(-2, 0, 0);
				input.m_to.set(2, 0, 0);				
				hkpShapeRayCastOutput output;
				output.setKey(0xFFFF);
				output.changeLevel(1);
				compound.castRay(input, output);

				HK_TEST(output.getLevel() == 1);
				HK_TEST(output.m_shapeKeys[0] == 0xFFFF);
				HK_TEST(output.m_shapeKeys[1] == 0);
				HK_TEST(output.m_shapeKeys[2] == HK_INVALID_SHAPE_KEY);
			}

			// Miss
			{						
				input.m_from.set(-2, 2, 0);
				input.m_to.set(2, 2, 0);				
				hkpShapeRayCastOutput output;			
				compound.castRay(input, output);

				HK_TEST(output.getLevel() == 0);
				HK_TEST(output.m_shapeKeys[0] == HK_INVALID_SHAPE_KEY);
			}		
		}

		// SCS with an instance of a container shape
		{			
			hkTransform transform; transform.setIdentity();
			hkVector4 translation; translation.set(0, -2, 0);
			transform.setTranslation(translation);
			hkpTransformShape transformA(&box, transform);
			hkpTransformShape transformB(&box, hkTransform::getIdentity());
			hkArray<hkpShape*> shapes;
			shapes.pushBack(&transformA);
			shapes.pushBack(&transformB);
			hkpListShape list(shapes.begin(), shapes.getSize());
			hkpStaticCompoundShape compound;		
			compound.addInstance(&list, hkQsTransform::getIdentity());
			compound.bake();

			hkpShapeRayCastInput input;
			input.m_rayShapeCollectionFilter = (useFilter ? &filter : HK_NULL);

			// Hit
			{						
				input.m_from.set(-2, 0, 0);
				input.m_to.set(2, 0, 0);
				hkpShapeRayCastOutput output;
				compound.castRay(input, output);

				HK_TEST(output.getLevel() == 0);
				HK_TEST(output.m_shapeKeys[0] == 1);
				HK_TEST(output.m_shapeKeys[1] == 0);
				HK_TEST(output.m_shapeKeys[2] == HK_INVALID_SHAPE_KEY);
			}

			// Hit with shape keys in the initial output
			{									
				input.m_from.set(-2, 0, 0);
				input.m_to.set(2, 0, 0);				
				hkpShapeRayCastOutput output;
				output.setKey(0xFFFF);
				output.changeLevel(1);
				compound.castRay(input, output);

				HK_TEST(output.getLevel() == 1);
				HK_TEST(output.m_shapeKeys[0] == 0xFFFF);
				HK_TEST(output.m_shapeKeys[1] == 1);
				HK_TEST(output.m_shapeKeys[2] == 0);
				HK_TEST(output.m_shapeKeys[3] == HK_INVALID_SHAPE_KEY);
			}

			// Miss
			{				
				input.m_from.set(-2, 2, 0);
				input.m_to.set(2, 2, 0);				
				hkpShapeRayCastOutput output;			
				compound.castRay(input, output);

				HK_TEST(output.getLevel() == 0);
				HK_TEST(output.m_shapeKeys[0] == HK_INVALID_SHAPE_KEY);
			}
		}
	}
}


// runs all unit tests
int hkpStaticCompoundShapeRayCastUnitTest()
{	
	checkRayCastNormalTransformation();
	simpleRayCastDisableEnableTest();	
	simpleRayCastWithCollectorDisableEnableTest();
	verySimpleRayCastFilteringTest();
	simpleRayCastFilteringTest();	
	simpleRayCastWithCollectorFilteringTest();
	gridRayCastWithCollectorFilteringTest();
	nestedCollectionRayCastTest();
	nestedCollectionRayCastFilteringTest();	
	testRayCastInitialFraction();
	testRayCastShapeKeyHierarchy();

	return 0;
}


#if defined( HK_COMPILER_MWERKS )
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER( hkpStaticCompoundShapeRayCastUnitTest , "Fast", "Physics2012/Test/UnitTest/Internal/", __FILE__ );

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
