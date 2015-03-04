/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/hkBase.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Common/Base/Types/hkBaseTypes.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>


// check that the support points a and b are equally extremal in the direction of support 
hkBool compareSupportPoints( const hkVector4& a, const hkVector4& b, const hkVector4& support, hkReal epsilon) 
{
	// scale by the largest vector
	hkReal list[3]; list[0] = a.length<3>().getReal(); list[1] = b.length<3>().getReal(); list[2] = support.length<3>().getReal();
	hkAlgorithm::quickSort<hkReal>(list,3);
	hkReal scale = list[2];

	// if scale is zero, the the lengths of all input vectors are 
	// zero and support points are trivially correct
	if (scale<epsilon)
		return true;

	// copy data to avoid corrupting stuff outside
	hkVector4 p1 = a;
	hkVector4 p2 = b;
	hkVector4 direction = support;

    // scale input vectors
	hkSimdReal invScale = hkSimdReal::fromFloat(hkReal(1)/scale);
	p1.mul(invScale);
	p2.mul(invScale);
	direction.mul(invScale);

	// both if a and b is extremal in the direction of support, they
	// must have the same dot product, within the error tolerance 
	hkReal s1 = p1.dot<3>(direction).getReal();
	hkReal s2 = p2.dot<3>(direction).getReal();
	return hkMath::fabs(s1-s2)<epsilon;  
}

hkBool compareHitFractions( hkReal a, hkReal b, hkReal epsilon )
{
	// compare hit fractions 			
	hkReal test = hkMath::fabs(a-b);
	return test < epsilon;
}

hkBool compareNormals( const hkVector4& normal1, const hkVector4& normal2, hkReal epsilon )
{
	// normals are unit length so no need to 
	// scale them for comparison

	// compare hit normals 			
	hkReal test = normal1.distanceTo(normal2).getReal(); 
	return test < epsilon;
}



void setupDodecahedron( hkpConvexVerticesShape& shape, const hkQsTransform& transform ) 
{
	// Data specific to this shape.
	int numVertices = 20;

	// 16 = 4 (size of "each float group", 3 for x,y,z, 1 for padding) * 4 (size of float)
	int stride = hkSizeOf(hkVector4);

	// dodecahedron vertices
	const hkReal phi = hkMath::sqrt(5.0f)/2.0f + 0.5f;
	HK_ALIGN_REAL(hkReal vertices[]) = {
		1.0f,  1.0f,  1.0f, 0.0f, 
		-1.0f,  1.0f,  1.0f, 0.0f, 
		1.0f, -1.0f,  1.0f, 0.0f, 
		-1.0f, -1.0f,  1.0f, 0.0f, 
		1.0f,  1.0f, -1.0f, 0.0f, 
		-1.0f,  1.0f, -1.0f, 0.0f, 
		1.0f, -1.0f, -1.0f, 0.0f, 
		-1.0f, -1.0f, -1.0f, 0.0f,  

		0.0f,  1.0f/phi,  phi, 0.0f, 
		0.0f, -1.0f/phi,  phi, 0.0f, 
		0.0f,  1.0f/phi, -phi, 0.0f, 
		0.0f, -1.0f/phi, -phi, 0.0f,

		1.0f/phi,  phi, 0.0f, 0.0f, 
		-1.0f/phi,  phi, 0.0f, 0.0f, 
		1.0f/phi, -phi, 0.0f, 0.0f, 
		-1.0f/phi, -phi, 0.0f, 0.0f, 

		phi, 0.0f,  1.0f/phi, 0.0f, 
		-phi, 0.0f,  1.0f/phi, 0.0f, 
		phi, 0.0f, -1.0f/phi, 0.0f, 
		-phi, 0.0f, -1.0f/phi, 0.0f 
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

void testCenters(const hkpConvexShape& shape1, const hkpConvexShape& shape2, hkReal epsilon)
{
	hkVector4 c1; shape1.getCentre(c1);
	hkVector4 c2; shape2.getCentre(c2);	
	hkBool condition = c1.distanceTo(c2).getReal() < epsilon;
	HK_TEST(condition);
	HK_ASSERT(0x71e3f50, condition);
}

void testAABB(const hkpConvexShape& shape1, const hkpConvexShape& shape2, hkReal epsilon)
{
	// assuming that shape1 has a tight aabb, it must be
	// contained in or coinciding with the aabb of shape2. 

	// query the shapes for their aabb's
	hkAabb aabb1;
	{
		hkTransform identity; identity.setIdentity();
		shape1.getAabb(identity, 0.0f, aabb1);
	}

	hkAabb aabb2;
	{
		hkTransform identity; identity.setIdentity();
		shape2.getAabb(identity, 0.0f, aabb2);
	}

	// increase the size of aabb2 by epsilon to combat numerical inaccuracy
	hkVector4 eps; eps.set(epsilon,epsilon,epsilon);
	aabb2.m_max.add(eps);
	aabb2.m_min.sub(eps);

	hkBool test1 = aabb2.contains(aabb1);

	HK_TEST( test1 );
	HK_ASSERT(0x530a0291, test1 );
}

void testSupportPoints( const hkpConvexShape& shape1, const hkpConvexShape& shape2, const hkVector4& support, hkReal epsilon)
{
	// query support points from shapes
	hkcdVertex p1; shape1.getSupportingVertex(support,p1);
	hkcdVertex p2; shape2.getSupportingVertex(support,p2);

	// compare
	hkBool condition = compareSupportPoints(p1,p2,support,epsilon);

	HK_TEST(condition);
	HK_ASSERT(0x3f83a6ee,condition);

	// repeat query to help debugging
	if (!condition)
	{
		compareSupportPoints(p1,p2,support,epsilon);
	}
}

void testRayCastAgainstCentre( const hkpConvexShape& shape1, const hkpConvexShape& shape2, const hkVector4& direction, hkReal epsilon) 
{
	// shoot a ray against the shapes' center of mass points, along the direction
	hkpShapeRayCastInput input;
	{
		hkVector4 centre; shape1.getCentre(centre);
		input.m_to = centre;
		hkVector4 from = direction; from.normalize<3>(); from.mul(hkSimdReal_15); from.setAdd(centre, from);
		input.m_from = from;
	}

	// do ray-casts 
	hkpShapeRayCastOutput output1;  shape1.castRay(input,output1);
	hkpShapeRayCastOutput output2;  shape2.castRay(input,output2);

	// compare hit fractions
	{
		hkBool test = compareHitFractions(output1.m_hitFraction,output2.m_hitFraction,epsilon);
		{
			HK_TEST(test); 
			HK_ASSERT(0x29db8a20,test);
			// redo compare to ease debugging
			if (!test) compareHitFractions(output1.m_hitFraction,output2.m_hitFraction,epsilon);
		}
	}

	// compare normals
	// if the ray directly hits an edge or a vertex of a shape, 
	// hkConvexvertices will choose a face to define the normal. This means
	// that such cases can result in different normals for the same points. This case is 
	// rare in practice, but tend to show up in unit tests like this. If we get a wrong normal, 
	// we permute the direction of the ray a tiny bit, and redo the test. 
	if (!compareNormals(output1.m_normal, output2.m_normal, epsilon)) 
	{
		hkVector4 noise; noise.set(0.0001f, -0.0003f, -0.0005f);
		input.m_to.setAdd(input.m_to,noise);
		output1.reset(); output2.reset();

		// redo ray-casts 
		shape1.castRay(input,output1);
		shape2.castRay(input,output2);

		// compare
		hkBool test = compareNormals(output1.m_normal, output2.m_normal, epsilon);
		{
			HK_TEST(test);
			HK_ASSERT(0x3ddf65ee,test);

			// redo the compare, to ease debugging
			if (!test)
			{ 
				output1.reset(); shape1.castRay(input,output1);
				output2.reset(); shape2.castRay(input,output2);
				compareNormals(output1.m_normal, output2.m_normal, epsilon);
			}
		}


	} else {
		HK_TEST(1);
	}

}

void testCollisionSpheres( const hkpConvexShape& shape1, const hkpConvexShape& shape2, hkReal epsilon)
{
	hkInt32 n1 = shape1.getNumCollisionSpheres();
	hkInt32 n2 = shape2.getNumCollisionSpheres();

	HK_TEST( n1 == n2 );
	HK_ASSERT(0x3e160512, n1==n2 );

	// stop now if there are no spheres at all
	if (n1==0)
		return;

	// allocate sphere buffers
	hkArray<hkSphere> spheres1(n1);
	hkArray<hkSphere> spheres2(n2);

	shape1.getCollisionSpheres(spheres1.begin());
	shape2.getCollisionSpheres(spheres2.begin());

	// for each sphere in shape1
	int i=0; while (i<n1)
	{
		const hkSphere s1 = spheres1[i];

		// match s1 in shape2's spheres
		hkReal closest = 9e9f; //Inf?
		hkBool found = false;
		// for each sphere in shape2
		int j=0; while (j<n2)
		{
			const hkSphere s2 = spheres2[j];

			// normalize to the largest vector size
			hkReal l1 = s1.getPosition().length<3>().getReal();
			hkReal l2 = s2.getPosition().length<3>().getReal();
			hkReal scale = l1>l2?l1:l2;

			// both positions are zero vectors
			if ( scale < 1e-6)
			{
				found = true;
				break;
			}

			hkSimdReal invScale = hkSimdReal::fromFloat(hkReal(1)/scale);
			hkVector4 p1 = s1.getPosition(); p1.mul(invScale);
			hkVector4 p2 = s2.getPosition(); p2.mul(invScale);

			hkReal dist = p1.distanceTo(p2).getReal();
			hkReal radiusDist = hkMath::fabs( s1.getRadius() - s2.getRadius() );

			if (dist<closest)
				closest = dist;

			// match position and radius
			if ( dist < epsilon && 
				 radiusDist < epsilon )
			{
				found = true;
				break;
			}

			++j;
		}

		HK_TEST(found);
		HK_ASSERT(0x257552e,found);

		++i;
	}
}

void testMaxProjection( const hkpConvexShape& shape1, const hkpConvexShape& shape2, const hkVector4& direction, hkReal epsilon) 
{
	hkReal max1 = shape1.getMaximumProjection(direction);
	hkReal max2 = shape2.getMaximumProjection(direction);
	hkReal scale = ( ( hkMath::fabs(max1) > hkMath::fabs(max2) ) ? hkMath::fabs(max1) : hkMath::fabs(max2) ); // scale by largest value

	// check for zero size max projections
    if ( hkMath::fabs(scale)<1e-6f )
	{
		HK_TEST(true);
		return;
	}
	
	hkReal diff = hkMath::fabs(max1-max2) / scale;

	hkBool condition = diff < epsilon;
	HK_TEST(condition);
	HK_ASSERT(0x104c4193,condition);

	// redo querys to help debugging
	if (!condition)
	{
		 max1 = shape1.getMaximumProjection(direction);
		 max2 = shape2.getMaximumProjection(direction);		
	}
}

void testConvertVertexIds( const hkpConvexShape& shape1, const hkpConvexShape& shape2, const hkVector4& direction, hkReal epsilon)
{
	// obtain a vertex using the support mapping of shape1	
	hkcdVertex vertex1; hkpVertexId vertex1Id;
	{
		shape1.getSupportingVertex(direction, vertex1);
		vertex1Id = (hkpVertexId) vertex1.getId();
	}

	// and for shape2
	hkcdVertex vertex2; hkpVertexId vertex2Id;
	{
		shape2.getSupportingVertex(direction, vertex2);
		vertex2Id = (hkpVertexId) vertex2.getId();
	}

	// make sure that we get the same vertex back when asking for the ID we got
	// from the support functions. IDs and vertices from shape1 and shape2 are
	// tested independently of each other
	hkcdVertex newVertex1; shape1.convertVertexIdsToVertices( &vertex1Id, 1, &newVertex1);
	hkcdVertex newVertex2; shape2.convertVertexIdsToVertices( &vertex2Id, 1, &newVertex2);

	hkBool condition1 = newVertex1.distanceTo(vertex1).getReal() < epsilon;
	hkBool condition2 = newVertex2.distanceTo(vertex2).getReal() < epsilon;

	HK_TEST(condition1 && condition2);
	HK_ASSERT(0xace246d,condition1 && condition2);
}


int hkpConvexTransformShapeUnitTest_test1() 
{
	//////////////////////////////////////////////////////////////////////////
	// simple test that compares support points of two shapes
	//////////////////////////////////////////////////////////////////////////

	// setup some transform
	hkQsTransform transform;
	{
		transform.setIdentity();
		//hkVector4 scale; scale.set(1.2f,0.2f,1.3f); transform.setScale(scale);
		hkVector4 translation; translation.set(4.0f,-2.3f,5.7f); transform.setTranslation(translation);
		hkVector4 axis; axis.set(5.3f,7.2f,-5.1f); axis.normalize<3>();
		hkQuaternion rotation; rotation.setAxisAngle(axis,1.2353f); transform.setRotation(rotation);
	}

	// transformed dodecahedron shape
	hkpConvexVerticesShape shape1;
	{
		setupDodecahedron(shape1,transform);
	}

	// unscaled shape to be implicitly transformed by 
	hkpConvexVerticesShape dodecahedron;
	{
		hkQsTransform identity; identity.setIdentity();		
		setupDodecahedron(dodecahedron, identity);
	}
	hkpConvexTransformShape shape2( &dodecahedron, transform);
	

	// some test support direction
	hkVector4 support; support.set(-1,7,0.1f);

	// test that the support points returned by shape1
	// and shape2 are consistent with each other 
	testSupportPoints( shape1, shape2, support, 1e-4f);

	// test aabb's
	testAABB(shape1, shape2, 1e-4f);

	// test centre points
	testCenters(shape1, shape2, 1e-4f);

	// test raycast
	testRayCastAgainstCentre(shape1, shape2, support, 1e-4f);

	// check maximum projection
	testMaxProjection(shape1, shape2, support, 1e-3f);

	// inspect that vertex IDs obtained with support functions
	// are the consistent with the vertices obtained using the
	// vertex IDs ( the support direction is just used to get 
	// some vertex from the support mappings) 
	testConvertVertexIds(shape1, shape2, support, 1e-4f);

	// check if the collision spheres match
	testCollisionSpheres(shape1, shape2, 1e-4f);

	return 0;
}

int hkpConvexTransformShapeUnitTest_test2() 
{
	//////////////////////////////////////////////////////////////////////////
	// this test builds a set of transforms and a set of directions. For each 
	// combination of those, both support points and raycasting against the
	// shapes are performed
	//////////////////////////////////////////////////////////////////////////

	const hkReal epsilon = 1e-4f;

	// setup an array of transform
    hkArray<hkQsTransform> transforms;
	{
		// Mixed transform
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(1.2f,0.2f,1.3f); transform.setScale(scale);
			hkVector4 translation; translation.set(4.0f,-2.3f,5.7f); transform.setTranslation(translation);
			hkVector4 axis; axis.set(5.3f,7.2f,-5.1f); axis.normalize<3>();
			hkQuaternion rotation; rotation.setAxisAngle(axis,1.2353f); transform.setRotation(rotation);
			transforms.pushBack(transform);
		}

		// Mixed transform
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(-8.2f,0.2f,1.3f); transform.setScale(scale);
			hkVector4 translation; translation.set(6.0f,2.f,-5.7f); transform.setTranslation(translation);
			hkVector4 axis; axis.set(1.1f,0.2f,-5.1f); axis.normalize<3>();
			hkQuaternion rotation; rotation.setAxisAngle(axis,-0.973f); transform.setRotation(rotation);
			transforms.pushBack(transform);
		}

		// mirror transform
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(-1.0f,-1.0f,-1.0f); transform.setScale(scale);
			transforms.pushBack(transform);
		}

		//  more mirror transform
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set( 1.0f,-1.0f, 1.0f); transform.setScale(scale);
			transforms.pushBack(transform);
		}

		//  more mirror transform
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set( 1.0f, 1.0f, -1.0f); transform.setScale(scale);
			transforms.pushBack(transform);
		}

		//  more mirror transform with some scaling
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set( 1.453f, -0.34f, -1.9862f); transform.setScale(scale);
			transforms.pushBack(transform);
		}


		// translating transform
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 translation; translation.set(-6.0f,0.02f,1.7f); transform.setTranslation(translation);
			transforms.pushBack(transform);
		}

		// mirror and scaling mixed transform
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(-1.3f,-1.2f,3.123f); transform.setScale(scale);
			transforms.pushBack(transform);
		}

	}

	// setup an array of support directions
	hkArray<hkVector4> directions;
	{
		// just some direction
		{
			hkVector4 direction; direction.set(2,-3,7);
			directions.pushBack(direction);
		}

		// ... and another
		{
			hkVector4 direction; direction.set(-4,-2,-9);
			directions.pushBack(direction);
		}


		// upwards
		{
			hkVector4 direction; direction.set(0.0f,1.0f,0.0f);
			directions.pushBack(direction);
		}

		// small direction
		{
			hkVector4 direction; direction.set(0.03f,0.00445f,-0.000895f);
			directions.pushBack(direction);
		}

		// down direction
		{
			hkVector4 direction; direction.set(0.0f,-1.0f,0.0f);
			directions.pushBack(direction);
		}

		// right direction
		{
			hkVector4 direction; direction.set(1.0f,0.0f,0.0f);
			directions.pushBack(direction);
		}

		// left direction
		{
			hkVector4 direction; direction.set(-1.0f,0.0f,0.0f);
			directions.pushBack(direction);
		}

		// a large direction vector
		{
			hkVector4 direction; direction.set(-45.243f,234.0f,4522.5f);
			directions.pushBack(direction);
		}

		// a small direction vector
		{
			hkVector4 direction; direction.set(-0.0045243f,0.0002340f,0.0045225f);
			directions.pushBack(direction);
		}
	}


	// for each transform
	for (hkArray<hkQsTransform>::iterator i=transforms.begin(); i<transforms.end(); ++i) 
	{
		hkQsTransform transform = *i;

		// transformed shape
		hkpConvexVerticesShape shape1;
		{
			setupDodecahedron(shape1,transform);
		}

		// unscaled shape to be implicitly transformed by 
		hkpConvexVerticesShape shape2;
		{
			hkQsTransform identity; identity.setIdentity();
			setupDodecahedron(shape2, identity);
		}

		// setup the transformed version of shape2
		hkpConvexTransformShape shape2t(&shape2,transform);

		// for each support direction
		for (hkArray<hkVector4>::iterator j=directions.begin(); j<directions.end(); ++j)
		{
			hkVector4 direction = *j;

			// test that the support points returned by shape1
			// and shape2 are consistent with each other 
			testSupportPoints( shape1, shape2t, direction, epsilon);

			// test aabb's
			testAABB(shape1, shape2t, epsilon);

			// test centre points
			testCenters(shape1, shape2t, epsilon);

			// test raycast
			testRayCastAgainstCentre(shape1, shape2t, direction, epsilon);

			// test maximum projection
			testMaxProjection(shape1, shape2t, direction, 1e-3f);

			// test that convert id-to-vertex functions are consistent
			testConvertVertexIds(shape1, shape2t, direction, 1e-4f);

			// check if the collision spheres match
			testCollisionSpheres(shape1, shape2t, 1e-3f);

		} // for each direction
	} // for each transform


	return 0;
}

int hkpConvexTransformShapeUnitTest_test3() 
{
	//////////////////////////////////////////////////////////////////////////
	// this test involves flattening of objects. It does not test apply the 
	// ray casting tests because raycasting is currently not supported with
	// flattenig transforms
	//////////////////////////////////////////////////////////////////////////


	const hkReal epsilon = 1e-4f;

	// setup an array of transform
	hkArray<hkQsTransform> transforms;
	{
		// flatten transforms
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(1.0f,0.0f,0.0f); transform.setScale(scale);
			transforms.pushBack(transform);
		}

		// flatten transforms
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(0.0f,1.0f,0.0f); transform.setScale(scale);
			transforms.pushBack(transform);
		}

		// flatten transforms
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(0.0f,0.0f,1.0f); transform.setScale(scale);
			transforms.pushBack(transform);
		}

		// flatten transforms
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(1.0f,1.0f,0.0f); transform.setScale(scale);
			transforms.pushBack(transform);
		}

		// flatten transforms
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(1.0f,0.0f,1.0f); transform.setScale(scale);
			transforms.pushBack(transform);
		}

		// flatten transforms
		{
			hkQsTransform transform; transform.setIdentity();
			hkVector4 scale; scale.set(0.0f,1.0f,1.0f); transform.setScale(scale);
			transforms.pushBack(transform);
		}
	}

	// setup an array of support directions
	hkArray<hkVector4> directions;
	{
		// just some direction
		{
			hkVector4 direction; direction.set(2,-3,7);
			directions.pushBack(direction);
		}

		// ... and another
		{
			hkVector4 direction; direction.set(-4,-2,-9);
			directions.pushBack(direction);
		}


		// upwards
		{
			hkVector4 direction; direction.set(0.0f,1.0f,0.0f);
			directions.pushBack(direction);
		}

		// small direction
		{
			hkVector4 direction; direction.set(0.03f,0.00445f,-0.000895f);
			directions.pushBack(direction);
		}

		// down direction
		{
			hkVector4 direction; direction.set(0.0f,-1.0f,0.0f);
			directions.pushBack(direction);
		}

		// right direction
		{
			hkVector4 direction; direction.set(1.0f,0.0f,0.0f);
			directions.pushBack(direction);
		}

		// left direction
		{
			hkVector4 direction; direction.set(-1.0f,0.0f,0.0f);
			directions.pushBack(direction);
		}

		// a large direction vector
		{
			hkVector4 direction; direction.set(-45.243f,234.0f,4522.5f);
			directions.pushBack(direction);
		}

		// a small direction vector
		{
			hkVector4 direction; direction.set(-0.0045243f,0.0002340f,0.0045225f);
			directions.pushBack(direction);
		}
	}


	// for each transform
	for (hkArray<hkQsTransform>::iterator i=transforms.begin(); i<transforms.end(); ++i) 
	{
		hkQsTransform transform = *i;

		// transformed shape
		hkpConvexVerticesShape shape1;
		{
			setupDodecahedron(shape1,transform);
		}

		// unscaled shape to be implicitly transformed by 
		hkpConvexVerticesShape shape2;
		{
			hkQsTransform identity; identity.setIdentity();
			setupDodecahedron(shape2, identity);
		}

		// setup the transformed version of shape2
		hkpConvexTransformShape shape2t(&shape2,transform);


		// for each support direction
		for (hkArray<hkVector4>::iterator j=directions.begin(); j<directions.end(); ++j)
		{
			hkVector4 direction = *j;

			// test that the support points returned by shape1
			// and shape2 are consistent with each other 
			testSupportPoints( shape1, shape2t, direction, epsilon);

			// test aabb's
			testAABB(shape1, shape2t, epsilon);

			// test centre points
			testCenters(shape1, shape2t, epsilon);

			// test maximum projection
			testMaxProjection(shape1, shape2t, direction, 1e-3f);

			// test that convert id-to-vertex functions are consistent
			testConvertVertexIds(shape1, shape2t, direction, 1e-4f);

			// check if the collision spheres match
			testCollisionSpheres(shape1, shape2t, 1e-4f);

		} // for each direction
	} // for each transform


	return 0;
}



int hkpConvexTransformShapeUnitTest() 
{
	hkpConvexTransformShapeUnitTest_test1();
	hkpConvexTransformShapeUnitTest_test2();

	// don't do the test with flattening, since flattening
	// are not yet fully supported
	//hkpConvexTransformShapeUnitTest_test3();

	return 0;
}


#if defined( HK_COMPILER_MWERKS )
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER( hkpConvexTransformShapeUnitTest , "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__ );

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
