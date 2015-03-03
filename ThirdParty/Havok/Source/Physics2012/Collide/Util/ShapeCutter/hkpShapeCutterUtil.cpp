/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Algorithm/UnionFind/hkUnionFind.h>
#include <Common/GeometryUtilities/Misc/hkGeometryUtils.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Common/Internal/ConvexHull/Deprecated/hkGeomHull.h>
#include <Common/Internal/ConvexHull/Deprecated/hkGeomConvexHullBuilder.h>

#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#include <Physics2012/Collide/Filter/Null/hkpNullCollisionFilter.h>

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/ExtendedMeshShape/hkpExtendedMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>

#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivityUtil.h>

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#include <Physics2012/Collide/Filter/Null/hkpNullCollisionFilter.h>
#include <Physics2012/Collide/Util/ShapeCutter/hkpShapeCutterUtil.h>

enum { INSIDE = 1, OUTSIDE = 2 };


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//                            hkpShapeConnectedCalculator
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

hkpShapeConnectedCalculator::hkpShapeConnectedCalculator(hkpCollisionDispatcher* collisionDispatcher, hkReal maxDistanceForConnection)
:	m_dispatcher(collisionDispatcher)
,	m_maxDistanceForConnection(maxDistanceForConnection)
{
}

hkBool hkpShapeConnectedCalculator::isConnected(const hkpShape* a,const hkTransform& transA,const hkpShape* b,const hkTransform& transB)
{
	hkReal dist;
	if (calculateClosestDistance(m_dispatcher, a, transA, b, transB, m_maxDistanceForConnection, dist))
	{
		if (dist < m_maxDistanceForConnection)
		{
			return true;
		}
	}

	return false;
}

hkBool hkpShapeConnectedCalculator::calculateClosestDistance(hkpCollisionDispatcher* dispatcher, const hkpShape* a, const hkTransform& transA, const hkpShape* b, const hkTransform& transB, hkReal maxDistance, hkReal& distanceOut)
{
	// To find the closest points between our shape pairs we'll make use of the getClosestPoints() method of the agent for those shapes.
	// There is, however, a catch and that is the getClosestPoints() method will only return results that are inside tolerance and since
	// the closest points between our shape pairs are likely to lie beyond the default for this distance we must increase the tolerance.

	hkpNullCollisionFilter filter;
	hkpCollisionInput input;
	input.m_dispatcher = dispatcher;
	input.m_filter = &filter;

	hkReal epsilon = ( maxDistance < 0.1f ) ? maxDistance : 0.1f;
	if ( epsilon < 0.0f ) epsilon = 1e-3f;
	input.setTolerance( maxDistance + epsilon );

	// Get the appropriate function
	hkpCollisionDispatcher::GetClosestPointsFunc getClosestPoints = dispatcher->getGetClosestPointsFunc(a->getType(), b->getType());
	HK_ASSERT2(0x3432432b, getClosestPoints, "No getClosestPoints() function has been registered for the given shape type combination.");
	if ( !getClosestPoints )
	{
		return false;
	}

	// Set up the bodies with the transform
	hkpCdBody bodyA(a, &transA);
	hkpCdBody bodyB(b, &transB);

	hkpClosestCdPointCollector closest;
	getClosestPoints(bodyA, bodyB, input, closest);

	// Find the closest distance
	if ( !closest.hasHit() )
	{
		return false;
	}
	distanceOut = closest.getHitContact().getDistance();
	return true;
}


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//                            hkpShapeCutterUtil
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/// Transform plane, by the inverse of transIn
inline static void HK_CALL hkpShapeCutterUtil_inverseTransformPlane(const hkVector4& plane,const hkTransform& transIn,hkVector4& planeOut)
{
	hkTransform inverse;
	inverse.setInverse(transIn);
	hkVector4Util::transformPlaneEquations(inverse, &plane,1,&planeOut);
}


hkpConvexVerticesShape* HK_CALL hkpShapeCutterUtil::createAabbConvexVerticesShape(const hkAabb& aabb, hkReal convexRadius )
{
	hkInplaceArray<hkVector4,6> planes;
	planes.setSize(6);

	planes[0].set( 1, 0, 0,-aabb.m_max(0) -convexRadius);
	planes[1].set(-1, 0, 0, aabb.m_min(0) -convexRadius);
	planes[2].set( 0, 1, 0,-aabb.m_max(1) -convexRadius);
	planes[3].set( 0,-1, 0, aabb.m_min(1) -convexRadius);
	planes[4].set( 0, 0, 1,-aabb.m_max(2) -convexRadius);
	planes[5].set( 0, 0,-1, aabb.m_min(2) -convexRadius);

	// Set up the vertices

	hkVector4 vertices[8];
	// Work out all 8 combinations
	for (int i = 0; i < 8; i++)
	{
		vertices[i].set((i&1)?aabb.m_max(0):aabb.m_min(0),(i&2)?aabb.m_max(1):aabb.m_min(1),(i&4)?aabb.m_max(2):aabb.m_min(2));
	}

	// Set up the connectivity -> must match up with the planes
	hkpConvexVerticesConnectivity* conn = new hkpConvexVerticesConnectivity;

	hkUint16 indices[] =
	{
		1,3,7,5,
		0,4,6,2,
		2,6,7,3,
		0,1,5,4,
		4,5,7,6,
		0,2,3,1
	};

	conn->m_vertexIndices.insertAt(0,indices,sizeof(indices)/sizeof(indices[0]));

	const hkUint8 vertsPerFace = 4;
	conn->m_numVerticesPerFace.setSize(6,vertsPerFace);

	// Create the shape

	hkStridedVertices stridedVertices; stridedVertices.set(vertices,8);

	hkpConvexVerticesShape* shapeOut = new hkpConvexVerticesShape( stridedVertices, planes, convexRadius );
	shapeOut->setConnectivity(conn, false);
	conn->removeReference();

	return shapeOut;
}


static void HK_CALL hkpShapeCutterUtil_calculatePlane(int* indices,const hkArray<hkVector4>& vertices,hkVector4& planeOut)
{
	const hkVector4& v0 = vertices[indices[0]];
	const hkVector4& v1 = vertices[indices[1]];
	const hkVector4& v2 = vertices[indices[2]];

	// Work out the normal

	hkVector4 e10;
	e10.setSub(v1,v0);

	hkVector4 e20;
	e20.setSub(v2,v0);

	hkVector4 plane;
	plane.setCross(e10,e20);
	plane.normalize<3>();

	// Work out the plane distance
	plane.setComponent<3>( -plane.dot<3>(v0) );

	planeOut = plane;
}

static void HK_CALL hkpShapeCutterUtil_addTriangle(int* indices,hkpConvexVerticesConnectivity* conn,const hkArray<hkVector4>& vertices,hkArray<hkVector4>& planes)
{
	hkVector4 plane;
	hkpShapeCutterUtil_calculatePlane(indices,vertices,plane);

	// Add the plane
	planes.pushBack(plane);

	// Add the indices

	conn->m_vertexIndices.pushBack(hkUint16(indices[0]));
	conn->m_vertexIndices.pushBack(hkUint16(indices[1]));
	conn->m_vertexIndices.pushBack(hkUint16(indices[2]));

	// Num vertices
	conn->m_numVerticesPerFace.pushBack(3);
}

static void HK_CALL hkpShapeCutterUtil_addQuad(int* indices,hkpConvexVerticesConnectivity* conn,const hkArray<hkVector4>& vertices,hkArray<hkVector4>& planes)
{
	hkVector4 plane;
	hkpShapeCutterUtil_calculatePlane(indices,vertices,plane);

	// Add the plane
	planes.pushBack(plane);

	// Add the indices
	conn->m_vertexIndices.pushBack(hkUint16(indices[0]));
	conn->m_vertexIndices.pushBack(hkUint16(indices[1]));
	conn->m_vertexIndices.pushBack(hkUint16(indices[2]));
	conn->m_vertexIndices.pushBack(hkUint16(indices[3]));

	// Num vertices
	conn->m_numVerticesPerFace.pushBack(4);
}

int HK_CALL hkpShapeCutterUtil::approxSphereRows(hkReal edgeSize,int maxFaces,hkReal radius)
{
	hkReal halfCircum = HK_REAL_PI*radius;
	int numRows = int(halfCircum/edgeSize + 0.5f);
	if (numRows < 2) { return 2; }

	if (numRows * numRows > maxFaces)
	{
		numRows = int(hkMath::sqrt(hkReal(maxFaces)));
		if (numRows <2) { return 2; }
		return numRows;
	}

	return numRows;
}

hkpConvexVerticesShape* HK_CALL hkpShapeCutterUtil::createSphereConvexVerticesShape(hkReal radius, hkReal extraConvexRadius, int numRowsIn)
{
	hkArray<hkVector4> vertices;
	hkArray<hkVector4> planes;

	hkVector4 center;
	center.setZero();

	const int numMajor = numRowsIn;
	const int numMinor = numRowsIn+1;

	int rows = numMajor + 1;
	int cols = numMinor;

	//int numVerts = rows * cols;
	//int numIndices = (2*numMajor) * cols;

	hkReal majorStep = (HK_REAL_PI / (hkReal)numMajor); // 180 degrees
	hkReal minorStep = (2.0f*HK_REAL_PI / (hkReal)(numMinor)); // 360 degrees

	// first and last vertex are the two apexs
	{
		for (int i = 0; i < rows; ++i)
		{
			const hkReal a = i * majorStep;

			const hkReal sa = hkMath::sin(a);
			const hkReal r0 = radius * sa;

			const hkReal ca = hkMath::cos(a);
			const hkReal z0 = radius * ca;

			for (int j = 0; j < cols; ++j)
			{
				const hkReal c = j * minorStep;
				const hkReal x = hkMath::cos(c);
				const hkReal y = hkMath::sin(c);

				hkVector4 pos;
				pos.set(x * r0, y * r0, z0);
				pos.add(center);

				vertices.pushBack(pos);

				// Only output one top and one bottom vertex
				if (i == 0 || i == rows - 1)
				{
					break;
				}
			}
		}
	}

	hkpConvexVerticesConnectivity* conn = new hkpConvexVerticesConnectivity;

	int indices[4];
	{
		const int bottom = vertices.getSize() - 1;
		const int bottomStart = 1 + (numMajor - 2) * cols;

		HK_ASSERT(0x3213213,bottom - cols == bottomStart);

		const int top = 0;
		const int topStart = 1;

		for (int j = 0; j < cols - 1; j++)
		{
			// top
			indices[0] = top;
			indices[1] = j + topStart;
			indices[2] = 1 + j + topStart;

			hkpShapeCutterUtil_addTriangle(indices,conn,vertices,planes);

			// bottom
			indices[0] = j + bottomStart;
			indices[1] = bottom;
			indices[2] = j + 1 + bottomStart;
			hkpShapeCutterUtil_addTriangle(indices,conn,vertices,planes);
		}
		// Last top
		indices[0] = top;
		indices[1] = (cols - 1) + topStart;
		indices[2] = topStart;
		hkpShapeCutterUtil_addTriangle(indices,conn,vertices,planes);

		// Last bottom
		indices[0] = (cols - 1) + bottomStart;
		indices[1] = bottom;
		indices[2] = bottomStart;
		hkpShapeCutterUtil_addTriangle(indices,conn,vertices,planes);
	}

	// middle
	for (int i = 1; i < numMajor-1; i++)
	{
		int start0 = cols*(i-1) + 1;
		int start1 = cols*(i -1 + 1) + 1;

		for (int j = 0; j < cols - 1; j++)
		{
			indices[0] = start0 + j;
			indices[1] = start1 + j;
			indices[2] = start1 + j + 1;
			indices[3] = start0 + j + 1;

			hkpShapeCutterUtil_addQuad(indices,conn,vertices,planes);
		}

		indices[0] = start0 + cols - 1;
		indices[1] = start1 + cols - 1;
		indices[2] = start1;
		indices[3] = start0;
		hkpShapeCutterUtil_addQuad(indices,conn,vertices,planes);
	}
	// bottom

	hkStridedVertices stridedVertices;
	stridedVertices.m_vertices = reinterpret_cast<const hkReal*>(&vertices[0]);
	stridedVertices.m_numVertices = vertices.getSize();
	stridedVertices.m_striding = sizeof(hkVector4);

	hkpConvexVerticesShape* shapeOut = new hkpConvexVerticesShape(stridedVertices, planes, extraConvexRadius);
	shapeOut->setConnectivity(conn);
	conn->removeReference();

	return shapeOut;
}

hkpConvexVerticesShape* HK_CALL hkpShapeCutterUtil::createCylinderConvexVerticesShape(hkReal radius, hkReal extraConvexRadius, hkReal height,int numSegs,const hkTransform& trans)
{
	hkArray<hkVector4> vertices;
	hkArray<hkVector4> planes;

	//
	vertices.setSize(numSegs*2);

	hkReal step = (2.0f*HK_REAL_PI)/numSegs;
	for (int i = 0; i < numSegs; i++)
	{
		const hkReal a = i * step;

		const hkReal x = hkMath::sin(a) * radius;
		const hkReal y = hkMath::cos(a) * radius;

		hkVector4 top;
		hkVector4 bottom;

		top.set(x,y,height);
		bottom.set(x,y,0.0f);

		top.setTransformedPos(trans,top);
		bottom.setTransformedPos(trans,bottom);

		vertices[i] = top;
		vertices[i + numSegs] = bottom;
	}

	hkpConvexVerticesConnectivity* conn = new hkpConvexVerticesConnectivity;

	int indices[4];
	for (int i = 0; i < numSegs; i++)
	{
		indices[3] = i;
		indices[2] = numSegs + i;
		indices[1] = numSegs + ((i + 1)%numSegs);
		indices[0] = (i + 1)%numSegs;

		hkpShapeCutterUtil_addQuad(indices,conn,vertices,planes);
	}

	// Cap the top
	{
		for (int i = 0; i < numSegs; i++)
		{
			int index = (numSegs - 1) - i;
			//int index = i;
			conn->m_vertexIndices.pushBack(hkUint16( index));
		}
		conn->m_numVerticesPerFace.pushBack(hkUint8(numSegs));
        hkVector4 plane; plane.set(0, 0, 1, -height);
        planes.pushBack(plane);
	}

	// cap the bottom
	{
		for (int i = 0; i < numSegs; i++)
		{
			int index = i + numSegs;
			//int index = (numSegs - 1) - i + numSegs;
			conn->m_vertexIndices.pushBack(hkUint16(index));
		}
		conn->m_numVerticesPerFace.pushBack(hkUint8(numSegs));
        hkVector4 plane; plane.set(0 ,0, -1, 0);
        planes.pushBack(plane);
	}

    // Transform the cap plane equations
    hkVector4Util::transformPlaneEquations(trans, planes.end() - 2, 2, planes.end() - 2);

	// Create the shape
	hkStridedVertices stridedVertices;
	stridedVertices.m_vertices = reinterpret_cast<const hkReal*>(&vertices[0]);
	stridedVertices.m_numVertices = vertices.getSize();
	stridedVertices.m_striding = sizeof(hkVector4);

	hkpConvexVerticesShape* shapeOut = new hkpConvexVerticesShape(stridedVertices, planes, extraConvexRadius);
	shapeOut->setConnectivity(conn);
	conn->removeReference();

	return shapeOut;
}

hkpConvexVerticesShape* HK_CALL hkpShapeCutterUtil::createCylinderConvexVerticesShape(hkReal radius, hkReal extraConvexRadius, const hkVector4& v0,const hkVector4& v1,int numSegs)
{
	hkReal epsilon = 1e-4f;

	// We need to create a transform
	hkVector4 zAxis;
	zAxis.setSub(v1,v0);

	hkReal height = zAxis.length<3>().getReal();
	if (height < epsilon) { return HK_NULL; }

	zAxis.normalize<3>();
	hkVector4 yAxis;
	hkVector4Util::calculatePerpendicularVector(zAxis,yAxis);

	hkVector4 xAxis;
	xAxis.setCross(yAxis,zAxis);

	hkRotation rot;
	rot.setCols(xAxis,yAxis,zAxis);

	hkTransform trans; trans.set(rot,v0);

	return createCylinderConvexVerticesShape(radius, extraConvexRadius, height, numSegs, trans);
}

hkpConvexVerticesShape* HK_CALL hkpShapeCutterUtil::createCapsuleConvexVerticesShape(const hkVector4& top, const hkVector4& bottom, hkReal radius, hkReal extraConvexRadius, int numSides, int numHeightSegments)
{
	hkTransform identity; identity.setIdentity();
	hkGeometry capsuleGeometry;
	hkGeometryUtils::createCapsuleGeometry(top, bottom, radius, numSides, numHeightSegments, identity, capsuleGeometry);

	hkArray<hkVector4>& vertices = capsuleGeometry.m_vertices;
	hkArray<hkGeometry::Triangle>& triangles = capsuleGeometry.m_triangles;

	hkpConvexVerticesConnectivity* conn = new hkpConvexVerticesConnectivity;

	hkArray<hkVector4> planes;
	planes.reserve(triangles.getSize());

	int indices[3];
	for (int i = 0; i < triangles.getSize(); i++)
	{
		indices[0] = triangles[i].m_a;
		indices[1] = triangles[i].m_b;
		indices[2] = triangles[i].m_c;

		hkpShapeCutterUtil_addTriangle(indices, conn, vertices, planes);
	}

	// Create the shape
	hkStridedVertices stridedVertices;
	stridedVertices.m_vertices = reinterpret_cast<const hkReal*>(&(vertices[0]));
	stridedVertices.m_numVertices = vertices.getSize();
	stridedVertices.m_striding = sizeof(hkVector4);

	hkpConvexVerticesShape* shapeOut = new hkpConvexVerticesShape(stridedVertices, planes, extraConvexRadius);
	shapeOut->setConnectivity(conn);
	conn->removeReference();

	return shapeOut;
}

//
//	Cuts a static compound shape

static const hkcdShape* HK_CALL cutStaticCompound(const hkpShapeCutterUtil::StaticCompoundShape* compoundShapeIn, const hkVector4& plane, hkReal extraConvexRadiusForImplicitShapes)
{
	hkArray<const hkcdShape*> cutShapes;
	hkArray<hkQsTransform> cutTransforms;

	// Cut each instance and return the result
	// Track shape changes to decide whether we can reuse the same shape
	bool changed = false;
	for (int ii = 0; ii < compoundShapeIn->getInstances().getSize(); ii++)
	{
		// Get instance
		const hkpShapeCutterUtil::StaticCompoundShape::Instance& instance = compoundShapeIn->getInstances()[ii];

		// Get transform
		const hkQsTransform instanceTm = instance.getTransform();
		hkTransform tm;
		tm.set(instanceTm.getRotation(), instanceTm.getTranslation());

		// Put the plane into local space
		hkVector4 localPlane;
		hkpShapeCutterUtil_inverseTransformPlane(plane, tm, localPlane);

		// Cut the shape instance
		const hkcdShape* cutShape = hkpShapeCutterUtil::cut(instance.getShape(), localPlane, extraConvexRadiusForImplicitShapes);
		if ( !cutShape )
		{
			changed = true;	// Cut shape is null, we can't reuse the original compound shape!
		}
		else
		{
			cutShapes.pushBack(cutShape);
			cutTransforms.pushBack(instanceTm);
			if ( cutShape != instance.getShape() )
			{
				changed = true;	// Cut shape is different from the child shape, we can't reuse the original compound shape!
			}
		}
	}

	// Build output shape
	const hkcdShape* outShape	= HK_NULL;
	const int numCutShapes		= cutShapes.getSize();
	if ( numCutShapes > 0 )
	{
		if ( numCutShapes == 1 )
		{
			// If there is only one entry -> we don't need a compound shape
			outShape = cutShapes[0];
			outShape->addReference();
		}
		else
		{
			// If the cut shapes are different, we need to produce a new compound
			if ( changed )
			{
				// Need to create a new compound shape
				hkpShapeCutterUtil::StaticCompoundShape* newCp = new hkpShapeCutterUtil::StaticCompoundShape();
				
				for (int k = 0; k < numCutShapes; k++)
				{
					newCp->addInstance(reinterpret_cast<const hkpShape*>(cutShapes[k]), cutTransforms[k]);
				}
				
				newCp->bake();
				outShape = newCp;
			}
			else
			{
				// We can just use the original
				outShape = compoundShapeIn;
				outShape->addReference();
			}
		}

		// Release whats in the array
		hkReferencedObject::removeReferences(cutShapes.begin(), numCutShapes);
	}

	// Return the cut shape
	return outShape;
}

static const hkcdShape* HK_CALL cutExtendedMeshShape(const hkpExtendedMeshShape* emsIn, const hkVector4& plane, hkReal extraConvexRadiusForImplicitShapes)
{
	hkArray<const hkcdShape*> cutShapes;
	hkArray<hkQsTransform> cutTransforms;

	// Cut each instance and return the result
	// Track shape changes to decide whether we can reuse the same shape
	bool changed = false;
	hkpShapeKey shapeKey = emsIn->getFirstKey();
	hkpShapeBuffer shapeBuffer;
	for (int ii = 0; ii < emsIn->getNumChildShapes(); ii++, shapeKey = emsIn->getNextKey(shapeKey))
	{
		// Get child shape
		const hkpShape* childShape = emsIn->getChildShape(shapeKey, shapeBuffer);
		HK_ASSERT(0x10afa3a8, childShape != (hkpShape*)shapeBuffer);

		// Cut the shape instance
		const hkcdShape* cutShape = hkpShapeCutterUtil::cut(childShape, plane, extraConvexRadiusForImplicitShapes);
		if ( !cutShape )
		{
			changed = true;	// Cut shape is null, we can't reuse the original compound shape!
		}
		else
		{
			cutShapes.pushBack(cutShape);
			if ( cutShape != childShape )
			{
				changed = true;	// Cut shape is different from the child shape, we can't reuse the original compound shape!
			}
		}
	}

	// Build output shape
	const hkcdShape* outShape	= HK_NULL;
	const int numCutShapes		= cutShapes.getSize();
	if ( numCutShapes > 0 )
	{
		if ( numCutShapes == 1 )
		{
			// If there is only one entry -> we don't need a compound shape
			outShape = cutShapes[0];
			outShape->addReference();
		}
		else
		{
			// If the cut shapes are different, we need to produce a new compound
			if ( changed )
			{
				// Need to create a new compound shape
				hkpListShape* newEms = new hkpListShape((const hkpShape* const*)cutShapes.begin(), cutShapes.getSize());
				outShape = newEms;
			}
			else
			{
				// We can just use the original
				outShape = emsIn;
				outShape->addReference();
			}
		}

		// Release whats in the array
		hkReferencedObject::removeReferences(cutShapes.begin(), numCutShapes);
	}

	// Return the cut shape
	return outShape;
}

const hkcdShape* HK_CALL hkpShapeCutterUtil::cut(const hkcdShape* shapeIn, const hkVector4& plane, hkReal extraConvexRadiusForImplicitShapes)
{
	// NOTE - other tricks could be done here to approximate shape with other faster/simpler shapes
	// such as boxes or spheres
	const hkReal minVolume = 1e-2f * 1e-2f * 1e-2f; // 1 cm3

	switch (shapeIn->getType())
	{
	case hkcdShapeType::CONVEX_VERTICES:
		{
			const hkpConvexVerticesShape* shape = static_cast<const hkpConvexVerticesShape*>(shapeIn);

			// Ref now on shape -> and has connectivity
			hkResult result = hkpConvexVerticesConnectivityUtil::ensureConnectivity(shape);
			if (result != HK_SUCCESS )
			{
				// Couldn't create the shape with connectivity
				return HK_NULL;
			}

#if defined(HK_DEBUG)
			if ( shape->getRadius() >= (0.05f-HK_REAL_EPSILON)    )
			{
				HK_WARN(0xaf12e110, "hkpConvexVerticesShape's extra radius is pretty big (" << shape->getRadius() << "> 0.5 meter) for a destructable object. This might cause suboptimal visual results.");
			}
			//else if ( radius == 0.0f                     ) HK_WARN(0xaf12e112, "");
#endif

			// Do the cut
			const hkpConvexVerticesShape* cutShape = hkpConvexVerticesConnectivityUtil::cut(shape, plane, shape->getRadius(), minVolume);

			return cutShape;
		}
	case hkcdShapeType::CYLINDER:
		{
			const hkpCylinderShape* cylinderShape = static_cast<const hkpCylinderShape*>(shapeIn);

			hkpConvexVerticesShape* cylinderCvShape = createCylinderConvexVerticesShape(cylinderShape->getCylinderRadius(), extraConvexRadiusForImplicitShapes,
																						cylinderShape->getVertex(0), cylinderShape->getVertex(1), 20);

			// Do the cut
			const hkpShape* newShape = hkpConvexVerticesConnectivityUtil::cut( cylinderCvShape, plane, cylinderCvShape->getRadius(), minVolume );
			cylinderCvShape->removeReference();

			if (newShape == cylinderCvShape)	// pointer compare allowed, even if cylinder shape does not exist any more
			{
 				newShape->removeReference();	
 				cylinderShape->addReference();
				return cylinderShape;
			}
			return newShape;
		}

	case hkcdShapeType::SPHERE:
		{
			const hkpSphereShape* sphereShape = static_cast<const hkpSphereShape*>(shapeIn);

			hkReal radius = sphereShape->getRadius();

			// The sphere is centered around the origin
			// Thus distance is the distance from the origin -which is the w component

			hkReal distance = plane(3);

			if (distance < -radius)
			{
				// Its totally inside
				sphereShape->addReference();
				return sphereShape;
			}
			if (distance > radius)
			{
				// Its totally outside
				return HK_NULL;
			}

			// We need to tesselate the sphere and do a cut
			// Work out the amount of rows
			int numRows = approxSphereRows(1.0f,256,radius);

			if (numRows < 4)
			{
				numRows = 4;
			}

			hkpConvexVerticesShape* sphereCvShape = createSphereConvexVerticesShape(radius, extraConvexRadiusForImplicitShapes, numRows);

			// Do the cut
			const hkpShape* newShape = hkpConvexVerticesConnectivityUtil::cut(sphereCvShape, plane, sphereCvShape->getRadius(), minVolume);

			sphereCvShape->removeReference();
			if (newShape == sphereCvShape)
			{
				newShape->removeReference();
				sphereShape->addReference();
				return sphereShape;
			}

			return newShape;
		}
	case hkcdShapeType::TRANSFORM:
		{
			const hkpTransformShape* transShape = static_cast<const hkpTransformShape*>(shapeIn);
			// Recurse and find if degenerates to a convex hull somewhere
			const hkpShape* child = transShape->getChildShape();
			if (child)
			{
				/// Put the plane into local space
				hkVector4 localPlane;
				hkpShapeCutterUtil_inverseTransformPlane(plane,transShape->getTransform(),localPlane);

				//
				const hkcdShape* newShape = cut(child,localPlane);
				if (newShape == child)
				{
					newShape->removeReference();
					transShape->addReference();
					return transShape;
				}
				if (newShape == HK_NULL) { return newShape; }

				if ( newShape->getType() == hkcdShapeType::CONVEX_VERTICES )
				{
					hkpConvexVerticesShape* cvxVertShape = const_cast<hkpConvexVerticesShape*>( static_cast<const hkpConvexVerticesShape*>(newShape ));
					cvxVertShape->transformVerticesAndPlaneEquations( transShape->getTransform() );
					return newShape;
				}

				hkpTransformShape* newTransShape = new hkpTransformShape(reinterpret_cast<const hkpShape*>(newShape),transShape->getTransform());
				newShape->removeReference();
				return newTransShape;
			}
			return HK_NULL;
		}

	case hkcdShapeType::CONVEX_TRANSLATE:
		{
			const hkpConvexTranslateShape* transShape = static_cast<const hkpConvexTranslateShape*>(shapeIn);

			hkTransform trans = hkTransform::getIdentity();
			trans.setTranslation(transShape->getTranslation());

			const hkpShape* child = transShape->getChildShape();
			if (child)
			{
				hkVector4 localPlane;
				hkpShapeCutterUtil_inverseTransformPlane(plane,trans,localPlane);
				// I need to do an upcast here. It's okay as the child of the original shape
				// must have derived from hkpConvexShape and the new shape must also be convex
				const hkpConvexShape* newShape = static_cast<const hkpConvexShape*>(cut(child,localPlane));

				if (newShape == child)
				{
					newShape->removeReference();
					transShape->addReference();
					return transShape;
				}
				if (newShape == HK_NULL) { return HK_NULL; }

				if ( newShape->getType() == hkcdShapeType::CONVEX_VERTICES )
				{
					hkpConvexVerticesShape* cvxVertShape = const_cast<hkpConvexVerticesShape*>( static_cast<const hkpConvexVerticesShape*>(newShape ));
					cvxVertShape->transformVerticesAndPlaneEquations( trans );
					return newShape;
				}

				hkpConvexTranslateShape* newTransShape = new hkpConvexTranslateShape(newShape,transShape->getTranslation());
				newShape->removeReference();
				return newTransShape;

			}
			return HK_NULL;
		}
	case hkcdShapeType::CONVEX_TRANSFORM:
		{
			const hkpConvexTransformShape* transShape = static_cast<const hkpConvexTransformShape*>(shapeIn);

			const hkpShape* child = transShape->getChildShape();
			if (child)
			{
				hkTransform localTransform;
				transShape->getTransform( &localTransform );

				hkVector4 localPlane;
				hkpShapeCutterUtil_inverseTransformPlane(plane, localTransform, localPlane);
				// Need to do an upcast here -> its okay because the shape must be derived from hkpConvexShape
				// to have been in the original hkpConvexTransformShape
				const hkpConvexShape* newShape = static_cast<const hkpConvexShape*>(cut(child,localPlane));

				if (newShape == child)
				{
					newShape->removeReference();
					transShape->addReference();
					return transShape;
				}
				if (newShape == HK_NULL) { return HK_NULL; }

				if ( newShape->getType() == hkcdShapeType::CONVEX_VERTICES )
				{
					hkpConvexVerticesShape* cvxVertShape = const_cast<hkpConvexVerticesShape*>( static_cast<const hkpConvexVerticesShape*>(newShape) );
					cvxVertShape->transformVerticesAndPlaneEquations( localTransform );
					return newShape;
				}

				hkpConvexTransformShape* newTransShape = new hkpConvexTransformShape(newShape,localTransform);

				newShape->removeReference();
				return newTransShape;
			}
			return HK_NULL;
		}
	case hkcdShapeType::BOX:
		{
			const hkpBoxShape* boxShape = static_cast<const hkpBoxShape*>(shapeIn);

			// If it does we turn the box into a hkpConvexVerticesShape
			const hkVector4& he = boxShape->getHalfExtents();
			hkAabb aabb;
			aabb.m_min.setNeg<4>(he);
			aabb.m_max = he;

			// Construct it
			hkpConvexVerticesShape* boxCvShape = createAabbConvexVerticesShape(aabb, boxShape->getRadius());

			const hkpConvexVerticesShape* newShape = hkpConvexVerticesConnectivityUtil::cut( boxCvShape, plane, boxShape->getRadius(), minVolume );
			boxCvShape->removeReference();
			if (newShape == boxCvShape)
			{
				newShape->removeReference();

				boxShape->addReference();
				return boxShape;
			}
			return newShape;
		}
	case hkcdShapeType::CAPSULE:
		{
			const hkpCapsuleShape* capShape = static_cast<const hkpCapsuleShape*>(shapeIn);

			// Turn it into a hkpConvexVerticesShape
			hkpConvexVerticesShape* capCvShape = createCapsuleConvexVerticesShape(capShape->getVertex(0), capShape->getVertex(1), capShape->getRadius(), extraConvexRadiusForImplicitShapes);

			const hkpConvexVerticesShape* newShape = hkpConvexVerticesConnectivityUtil::cut( capCvShape, plane, capShape->getRadius(), minVolume );
			capCvShape->removeReference();
			if (newShape == capCvShape)
			{
				newShape->removeReference();
				capShape->addReference();
				return capShape;
			}
			return newShape;
		}

	case hkcdShapeType::MOPP:
		{
			const hkpMoppBvTreeShape* cs = static_cast<const hkpMoppBvTreeShape*>(shapeIn);
			return cut( cs->getShapeCollection(), plane, extraConvexRadiusForImplicitShapes );
		}

	case hkcdShapeType::LIST:
		{
			const hkpListShape* listShape = static_cast<const hkpListShape*>(shapeIn);
			hkArray<const hkpShape*> shapeArray;
			hkBool changed = false;

			int numChildShapes = listShape->getNumChildShapes();
			if ( numChildShapes > 256 )
			{
				HK_WARN_ALWAYS(0xabba14ef, "Performance warning: Cutting a list shape with more than 256 children.");
			}

			for (int i = 0; i < numChildShapes; i++)
			{
				const hkpShape* childShape = listShape->getChildShapeInl(i);

				// Lets try cutting it
				const hkpShape* newShape = reinterpret_cast<const hkpShape*>(cut(childShape,plane));

				if (newShape == HK_NULL)
				{
					changed = true;
				}
				else
				{
					shapeArray.pushBack(newShape);
					if (newShape != childShape)
					{
						changed = true;
					}
				}
			}

			const hkcdShape* outShape = HK_NULL;
			if (shapeArray.getSize() > 0)
			{
				if (shapeArray.getSize() == 1)
				{
					// If there is only one entry -> we don't need a list shape
					outShape = shapeArray[0];
					outShape->addReference();
				}
				else
				{
					// If the list shape is different we need to produce a new list
					if (changed)
					{
						// Need to create a new list shape
						outShape = new hkpListShape(&shapeArray[0],shapeArray.getSize());
					}
					else
					{
						// We can just use the original
						outShape = shapeIn;
						outShape->addReference();
					}
				}
				// Release whats in the array
				hkReferencedObject::removeReferences( shapeArray.begin(), shapeArray.getSize() );
			}

			// We are done
			return outShape;
		}

	case hkcdShapeType::STATIC_COMPOUND:
		{
			const StaticCompoundShape* scs = static_cast<const StaticCompoundShape*>(shapeIn);
			return cutStaticCompound(scs, plane, extraConvexRadiusForImplicitShapes);
		}
		break;

	case hkcdShapeType::EXTENDED_MESH:
		{
			const hkpExtendedMeshShape* ems = static_cast<const hkpExtendedMeshShape*>(shapeIn);
			cutExtendedMeshShape(ems, plane, extraConvexRadiusForImplicitShapes);
		}
		break;

	default:
		{
			HK_WARN(0xbd838dd2,"Ignoring unsupported shape type : " << hkGetShapeTypeName(shapeIn->getType()));
			break;
		}
	}

	return HK_NULL;
}

const hkcdShape* HK_CALL hkpShapeCutterUtil::subtractConvexShape( const hkcdShape* shape, const SubtractShapeInput& input )
{
	int numPlanes = input.m_subtractShape->getPlaneEquations().getSize();

	hkInplaceArray<const hkcdShape*,16> cutShapes;
	hkInplaceArrayAligned16<hkTransform,16> cutTransforms;
	{
		// do a distance collision query
		hkpCollidable subtractCollidable( input.m_subtractShape, &input.m_transform );

		hkpCollisionInput collisionInput;
		hkpNullCollisionFilter filter;

		collisionInput.m_tolerance = 0.0f;
		collisionInput.m_filter = &filter;
		collisionInput.m_dispatcher = input.m_dispatcher;

		{
			hkInplaceArray<const hkpConvexShape*,16> childShapes;
			hkInplaceArrayAligned16<hkTransform,16> transforms;
			flattenIntoConvexShapes( shape, hkTransform::getIdentity(), childShapes, transforms  );

			for( int i = 0; i < childShapes.getSize(); i++)
			{
				const hkpShape* childShape = childShapes[i];
				const hkTransform& childTransform = transforms[i];
				hkpCollidable childCollidable( childShape, &childTransform);

				// only cut shapes which intersect, take the uncut shape if it does not collide
				hkpCollisionDispatcher::GetClosestPointsFunc clpFunc = input.m_dispatcher->getGetClosestPointsFunc( childShape->getType(), input.m_subtractShape->getType() );

				hkpClosestCdPointCollector collector;
				clpFunc( childCollidable, subtractCollidable, collisionInput, collector );
				if ( collector.hasHit() && collector.getHit().m_contact.getDistance() < - input.m_allowedPenetration )
				{ 
					hkLocalBuffer<hkVector4> planes(numPlanes);
					hkTransform combined; combined.setMulInverseMul( childTransform, input.m_transform);
					hkVector4Util::transformPlaneEquations( combined, input.m_subtractShape->getPlaneEquations().begin(), numPlanes, planes.begin() );

					const hkcdShape* cutShape = _subtractConvexShape( childShape, planes.begin(), numPlanes, input.m_extraConvexRadiusForImplicitShapes );
					if ( !cutShape )
					{
						continue;
					}
					cutShapes.pushBack( cutShape );
				}
				else
				{
					childShape->addReference();
					cutShapes.pushBack( childShape );
				}
				cutTransforms.pushBack( childTransform );
			}
		}
	}
	if ( cutShapes.isEmpty() )
	{
		shape->addReference();
		return shape;
	}

	// flatten results
	const hkcdShape* newPhysicsShape;
	{
		hkLocalArray<const hkpConvexShape*> resultShapes(cutShapes.getSize()*2);
		hkLocalArray<hkTransform> resultTransforms(cutShapes.getSize()*2);
		for (int i=0; i < cutShapes.getSize();i++)
		{
			hkpShapeCutterUtil::flattenIntoConvexShapes( cutShapes[i], cutTransforms[i], resultShapes, resultTransforms );
		}


		//
		//	Filter physics shapes
		//
		if ( input.m_smallestSphereInside > 0 )
		{
			for ( int i = resultShapes.getSize()-1; i>=0; i-- )
			{
				const hkpConvexShape* cs = resultShapes[i];
				if ( cs->getType() != hkcdShapeType::CONVEX_VERTICES )
				{
					continue;
				}
				// get the planes and reduce them by our test sphere radius
				const hkpConvexVerticesShape* cvs = (const hkpConvexVerticesShape*) cs;
				hkArray<hkVector4> planes; planes = cvs->getPlaneEquations();
				for (int pi = 0; pi < planes.getSize(); pi++){ planes[pi](3) += input.m_smallestSphereInside; }
				hkgpConvexHull::BuildConfig config;
				hkgpConvexHull util;
				int dim = util.buildFromPlanes(planes.begin(), planes.getSize(), config);
				if ( dim < 3 )
				{
					hkArray<hkVector4> verts;		cvs->getOriginalVertices( verts );
					// cross check with a new set of equations
					hkgpConvexHull util2;
					hkgpConvexHull::BuildConfig config2;
					config2.m_buildMassProperties = true;
					int dim2 = util2.build( verts.begin(), verts.getSize(), config2 );
					hkReal volume2 = 0.0f;
					bool shouldBeRemoved = true;
					if ( dim2 == 3 )
					{
						volume2 = util2.getVolume().getReal();

						hkArray<hkVector4> planes2; 
						planes2.setSize( util2.getNumPlanes() );
						for (int n=0; n < planes2.getSize(); n++)
						{
							hkVector4 plane = util2.getPlane( n );
							plane(3) += input.m_smallestSphereInside;
							planes2[n] = plane;
						}
						hkgpConvexHull util3;
						int dim3 = util3.buildFromPlanes(planes2.begin(), planes2.getSize(), config);
						if ( dim3 == 3 )
						{
							shouldBeRemoved = false;
						}
					}

					if ( shouldBeRemoved )
					{
						HK_REPORT( "Dropped too small physics shape in cutout old volume " << volume2  );
						resultShapes.removeAt( i );
						resultTransforms.removeAt( i );
					}
				}
			}
		}

		newPhysicsShape = hkpShapeCutterUtil::createCompound(resultShapes.begin(), resultTransforms.begin(), resultShapes.getSize());
	}
	hkReferencedObject::removeReferences( cutShapes.begin(), cutShapes.getSize() );
	return newPhysicsShape;
}


static const hkpConvexVerticesShape* HK_CALL _intersectWithConvexShape( const hkpConvexShape* shape, const hkVector4* planeEquations, int numPlaneEquations, hkBool* planesUsed, hkReal expandPlaneEquations, hkReal extraConvexRadiusForImplicitShapes  )
{
	const hkpConvexVerticesShape* cvs = hkpShapeCutterUtil::ensureConvexVerticesShape( shape, extraConvexRadiusForImplicitShapes );
	{
		for (int i = 0; i < numPlaneEquations; i++ )
		{
			hkVector4 plane = planeEquations[i];
			plane(3) -= expandPlaneEquations;
			const hkpConvexVerticesShape* cutShape = hkpConvexVerticesConnectivityUtil::cut( cvs, plane, cvs->getRadius(), HK_REAL_EPSILON );
			if ( cutShape == HK_NULL )
			{
				if ( planesUsed )
				{
					planesUsed[i] = true;
				}
				cvs->removeReference();
				return HK_NULL;
			}
			if ( cvs == cutShape )
			{
				cutShape->removeReference();
			}
			else
			{
				if ( planesUsed )
				{
					planesUsed[i] = true;
				}
				cvs->removeReference();
				cvs = cutShape;
			}
		}
	}
	return cvs;
}

const hkcdShape* HK_CALL hkpShapeCutterUtil::intersectWithConvexShape( const hkcdShape* shape, const IntersectShapeInput& input )
{
	hkInplaceArray<const hkpConvexShape*,16> cutShapes;
	hkInplaceArrayAligned16<hkTransform,16> cutTransforms;
	{
		hkInplaceArray<const hkpConvexShape*,16> childShapes;
		hkInplaceArrayAligned16<hkTransform,16> transforms;
		flattenIntoConvexShapes( shape, hkTransform::getIdentity(), childShapes, transforms );

		for( int i = 0; i < childShapes.getSize(); i++)
		{
			const hkpConvexShape* childShape = childShapes[i];
			const hkTransform& childTransform = transforms[i];

			hkLocalBuffer<hkVector4> planes(input.m_numPlaneEquations);
			hkTransform combined; combined.setMulInverseMul( childTransform, input.m_transform );
			hkVector4Util::transformPlaneEquations( combined, input.m_planeEquations, input.m_numPlaneEquations, planes.begin() );

			const hkpConvexShape* cutShape = _intersectWithConvexShape( childShape, planes.begin(), input.m_numPlaneEquations, HK_NULL, input.m_expandPlaneEquations, input.m_extraConvexRadiusForImplicitShapes );
			if ( !cutShape )
			{
				continue;
			}
			cutShapes.pushBack( cutShape );
			cutTransforms.pushBack( childTransform );
		}
	}
	if ( cutShapes.isEmpty() )
	{
		return HK_NULL;
	}

	// combine results
	const hkcdShape* newPhysicsShape = hkpShapeCutterUtil::createCompound(cutShapes.begin(), cutTransforms.begin(), cutShapes.getSize());
	hkReferencedObject::removeReferences( cutShapes.begin(), cutShapes.getSize() );
	return newPhysicsShape;
}

const hkcdShape* HK_CALL hkpShapeCutterUtil::_subtractConvexShape( const hkcdShape* shape, const hkVector4* planes, int numPlanes, hkReal extraConvexRadiusForImplicitShapes )
{
	hkLocalArray<hkVector4> sortedPlanes( numPlanes );
	{
		sortedPlanes.insertAt( 0, planes, numPlanes );
		hkAabb aabb; reinterpret_cast<const hkpShape*>(shape)->getAabb( hkTransform::getIdentity(), HK_REAL_EPSILON, aabb );
		hkVector4 extent; extent.setSub( aabb.m_max, aabb.m_min );
		int indizes[3] = { 0,1,2 };

		// bubble sort extents
		{
			if ( extent(0) < extent(1)) { hkAlgorithm::swap(extent(0), extent(1)), hkAlgorithm::swap( indizes[0], indizes[1]); }
			if ( extent(1) < extent(2)) { hkAlgorithm::swap(extent(1), extent(2)), hkAlgorithm::swap( indizes[1], indizes[2]); }
			if ( extent(0) < extent(1)) { hkAlgorithm::swap(extent(0), extent(1)), hkAlgorithm::swap( indizes[0], indizes[1]); }
		}

		// sort planes, use axis aligned planes first, scale the value for the extents
		for (int i = 0; i < numPlanes-1; i++)
		{
			int bestIndex = i;
			hkSimdReal bestCost = hkSimdReal_Max;
			for ( int j = i; j < numPlanes; j++)
			{
// 				int dirIndex = indizes[ (i%3) ];
// 				hkReal cost = -hkMath::fabs(sortedPlanes[j](dirIndex));


				hkVector4 p = sortedPlanes[j];
				p.setAbs(p);
				int axis = p.getIndexOfMaxAbsComponent<3>();
				hkSimdReal cost = hkSimdReal::fromFloat(hkReal(1.3f)) - p.getComponent(axis);
				cost.div( extent.getComponent( axis ) );

				if ( cost < bestCost )
				{
					bestCost = cost;
					bestIndex = j;
				}
			}
			hkAlgorithm::swap( sortedPlanes[i], sortedPlanes[bestIndex]);
		}
	}

	// find all shapes on the outside
	hkLocalArray<const hkcdShape*> tempShapes( numPlanes );
	hkLocalArray<const hkpConvexShape*> outsideShapes( numPlanes );
	hkLocalArray<hkTransform> transforms( numPlanes );
	const hkcdShape* insideShape = shape;
	{
		insideShape->addReference();

		for (int i =0; i < sortedPlanes.getSize(); i++)
		{
			hkVector4 plane;
			{
				// search the plane which cuts the inside shape best
				hkReal bestCost = HK_REAL_MAX;
				int bestIndex = i;
				for (int s = i; s < sortedPlanes.getSize(); s++)
				{
					hkVector4 p = sortedPlanes[s];
					hkReal pos = reinterpret_cast<const hkpShape*>(insideShape)->getMaximumProjection( p ) + p(3);
					hkVector4 np; np.setNeg<4>(p);
					hkReal neg = reinterpret_cast<const hkpShape*>(insideShape)->getMaximumProjection( np ) + np(3);

					// we found one plane which completely hides the object, stop cutting
					if ( neg <= HK_REAL_EPSILON )
					{
						flattenIntoConvexShapes(insideShape, hkTransform::getIdentity(), outsideShapes, transforms );
						sortedPlanes.setSize(i);
						break;
					}

					if ( pos <= HK_REAL_EPSILON )	// delete plane, does not give any real value
					{
						sortedPlanes.removeAt( s );
						continue;
					}
					int axis = p.getIndexOfMaxAbsComponent<3>();
					hkReal cost = -pos * p(axis) * p(axis);	// favor axis aligned splits
					if ( cost < bestCost )
					{
						bestCost = cost;
						bestIndex = s;
					}
				}
				if ( i >= sortedPlanes.getSize() )
				{
					break;
				}
				hkAlgorithm::swap( sortedPlanes[i], sortedPlanes[bestIndex]);
				plane = sortedPlanes[i];
			}

			const hkcdShape* newInsideShape;
			const hkcdShape* outsideShape;
			cut( insideShape, plane, extraConvexRadiusForImplicitShapes, &newInsideShape, &outsideShape );
			if ( outsideShape )
			{
				tempShapes.pushBack( outsideShape );
				/*hkResult result = */
				flattenIntoConvexShapes(outsideShape, hkTransform::getIdentity(), outsideShapes, transforms );
			}
			if ( newInsideShape )
			{
				insideShape->removeReference();
				insideShape = newInsideShape;
			}
		}
	}

	// lets build a list shape
	const hkpShape* newPhysicsShape = hkpShapeCutterUtil::createCompound(outsideShapes.begin(), transforms.begin(), outsideShapes.getSize());
	hkReferencedObject::removeReferences( tempShapes.begin(), tempShapes.getSize() );
	insideShape->removeReference();
	return newPhysicsShape;
}


/* static */const hkpConvexVerticesShape* HK_CALL hkpShapeCutterUtil::ensureConvexVerticesShape(const hkpConvexShape* shapeIn, hkReal extraConvexRadius)
{
	switch (shapeIn->getType())
	{
	case hkcdShapeType::CONVEX_VERTICES:
		{
			const hkpConvexVerticesShape* shape = static_cast<const hkpConvexVerticesShape*>(shapeIn);
			shape->addReference();
			return shape;
		}
	case hkcdShapeType::BOX:
		{
			const hkpBoxShape* boxShape = static_cast<const hkpBoxShape*>(shapeIn);

			// If it does we turn the box into a hkpConvexVerticesShape
			const hkVector4& he = boxShape->getHalfExtents();
			hkAabb aabb;
			aabb.m_min.setNeg<4>(he);
			aabb.m_max = he;

			// Construct it
			hkpConvexVerticesShape* cvshape = createAabbConvexVerticesShape(aabb, boxShape->getRadius());
			return cvshape;
		}
	case hkcdShapeType::SPHERE:
		{
			const hkpSphereShape* sphereShape = static_cast<const hkpSphereShape*>(shapeIn);

			hkReal radius = sphereShape->getRadius();

			// We need to tesselate the sphere and do a cut
			// Work out the amount of rows
			int numRows = approxSphereRows(1.0f, 256, radius);
			hkpConvexVerticesShape* cvshape = createSphereConvexVerticesShape(radius, extraConvexRadius, numRows);
			return cvshape;
		}
	case hkcdShapeType::CYLINDER:
		{
			const hkpCylinderShape* cylinderShape = static_cast<const hkpCylinderShape*>(shapeIn);

			hkpConvexVerticesShape* cvshape = createCylinderConvexVerticesShape(cylinderShape->getRadius(), extraConvexRadius, cylinderShape->getVertex(0), cylinderShape->getVertex(1), 20);
			return cvshape;
		}
	case hkcdShapeType::CAPSULE:
		{
			const hkpCapsuleShape* capsuleShape = static_cast<const hkpCapsuleShape*>(shapeIn);

			hkpConvexVerticesShape* cvshape = createCapsuleConvexVerticesShape(capsuleShape->getVertex(0), capsuleShape->getVertex(1), capsuleShape->getRadius(), extraConvexRadius);
			return cvshape;
		}
	default:
		{
			return HK_NULL;
		}
	}
}

void HK_CALL hkpShapeCutterUtil::cut(const hkcdShape* shapeIn, const hkVector4& planeIn, hkReal extraConvexRadius, const hkcdShape** shapeAOut, const hkcdShape** shapeBOut)
{
	const hkcdShape* a = cut(shapeIn, planeIn, extraConvexRadius);
	*shapeAOut = a;

	if (a == HK_NULL)
	{
		shapeIn->addReference();
		*shapeBOut = shapeIn;
		return;
	}
	if (a == shapeIn)
	{
		*shapeBOut = HK_NULL;
		return;
	}

	hkVector4 plane;
	plane.setNeg<4>(planeIn);

	*shapeBOut  = cut(shapeIn, plane, extraConvexRadius);
}

const hkpShape* HK_CALL hkpShapeCutterUtil::transformConvexShape(const hkpConvexShape* shape,const hkTransform& parentToWorld)
{
	const hkTransform& identity = hkTransform::getIdentity();
	const hkReal rotationEpsilon = 1e-4f;
	const hkReal translateEpsilon = 1e-4f;

	if (parentToWorld.getRotation().isApproximatelyEqual(identity.getRotation(),rotationEpsilon))
	{
		/// See if the translation is zero
		if (parentToWorld.getTranslation().allEqual<3>(identity.getTranslation(),hkSimdReal::fromFloat(translateEpsilon)))
		{
			shape->addReference();
			return shape;
		}

		return  new hkpConvexTranslateShape(shape,parentToWorld.getTranslation());
	}
	return new hkpConvexTransformShape(shape,parentToWorld);
}

const hkpShape* HK_CALL hkpShapeCutterUtil::transformNonConvexShape(const hkpShape* shape,const hkTransform& transform)
{
	const hkTransform& identity = hkTransform::getIdentity();

	const hkReal epsilon = 1e-4f;

	if (transform.isApproximatelyEqual(identity,epsilon))
	{
		shape->addReference();
		return shape;
	}
	return new hkpTransformShape(shape,transform);
}

const hkpShape* HK_CALL hkpShapeCutterUtil::transformShape(const hkpShape* shape,const hkTransform& transform)
{
	if (shape == HK_NULL) return shape;

	// Holds the current transform
	hkTransform parentToWorld = transform;
	while (true)
	{
		switch (shape->getType())
		{
		case hkcdShapeType::CONVEX_TRANSLATE:
			{
				const hkpConvexTranslateShape* ts = static_cast<const hkpConvexTranslateShape*>( shape );
				hkTransform localToParent = hkTransform::getIdentity();
				localToParent.setTranslation(ts->getTranslation());

				hkTransform localToWorld;
				localToWorld.setMul(parentToWorld,localToParent);

				// Tail recurse (without the call)
				parentToWorld = localToWorld;
				shape = ts->getChildShape();
				break;
			}
		case hkcdShapeType::CONVEX_TRANSFORM:
			{
				const hkpConvexTransformShape* ts = static_cast<const hkpConvexTransformShape*>( shape );
				hkTransform localTransform; ts->getTransform( &localTransform );
				hkTransform localToWorld; localToWorld.setMul( parentToWorld, localTransform );

				// Tail recurse (without the call)
				parentToWorld = localToWorld;
				shape = ts->getChildShape();
				break;
			}
		case hkcdShapeType::TRANSFORM:
			{
				const hkpTransformShape* ts = static_cast<const hkpTransformShape*>( shape );
				hkTransform localToWorld;
				localToWorld.setMul(parentToWorld,ts->getTransform());

				// Tail recurse (without the call)
				parentToWorld = localToWorld;
				shape = ts->getChildShape();
				break;
			}
		case hkcdShapeType::SPHERE:
		case hkcdShapeType::TRIANGLE:
		case hkcdShapeType::BOX:
		case hkcdShapeType::CAPSULE:
		case hkcdShapeType::CYLINDER:
		case hkcdShapeType::CONVEX_VERTICES:
		case hkcdShapeType::CONVEX_PIECE:
			{
				const hkpConvexShape* convexShape = static_cast<const hkpConvexShape*>(shape);
				return transformConvexShape(convexShape,parentToWorld);
			}
		default:
			{
				return transformNonConvexShape(shape,parentToWorld);
			}
		}
	}
}

hkResult HK_CALL hkpShapeCutterUtil::flattenIntoConvexShapes(const hkcdShape* shape, const hkTransform& parentToReference, hkArray<const hkpConvexShape*>& shapes, hkArray<hkTransform>& localToReferenceTransforms )
{
	switch (shape->getType())
	{
	case hkcdShapeType::CONVEX_TRANSLATE:
		{
			const hkpConvexTranslateShape* ts = static_cast<const hkpConvexTranslateShape*>( shape );
			hkTransform localToParent = hkTransform::getIdentity();
			localToParent.setTranslation(ts->getTranslation());

			hkTransform localToReference;
			localToReference.setMul(parentToReference,localToParent);

			return flattenIntoConvexShapes(ts->getChildShape(), localToReference, shapes,  localToReferenceTransforms );
		}
	case hkcdShapeType::CONVEX_TRANSFORM:
		{
			const hkpConvexTransformShape* ts = static_cast<const hkpConvexTransformShape*>( shape );
			hkTransform localTransform; ts->getTransform( &localTransform );
			hkTransform localToReference; localToReference.setMul( parentToReference, localTransform );

			return flattenIntoConvexShapes(ts->getChildShape(), localToReference, shapes, localToReferenceTransforms );
		}
	case hkcdShapeType::TRANSFORM:
		{
			const hkpTransformShape* ts = static_cast<const hkpTransformShape*>( shape );
			hkTransform localToReference;
			localToReference.setMul(parentToReference, ts->getTransform());

			return flattenIntoConvexShapes(ts->getChildShape(), localToReference, shapes, localToReferenceTransforms);
		}
	case hkcdShapeType::BOX:
	case hkcdShapeType::CAPSULE:
	case hkcdShapeType::CYLINDER:
	case hkcdShapeType::SPHERE:
	case hkcdShapeType::TRIANGLE:
	case hkcdShapeType::CONVEX_VERTICES:
		{
			const hkpConvexShape* cs = static_cast<const hkpConvexShape*>(shape);

			localToReferenceTransforms.pushBack(parentToReference);
			shapes.pushBack(cs);

			return HK_SUCCESS;
		}

	case hkcdShapeType::MOPP:
		{
			const hkpMoppBvTreeShape* cs = static_cast<const hkpMoppBvTreeShape*>(shape);
			return flattenIntoConvexShapes( cs->getShapeCollection(), parentToReference, shapes, localToReferenceTransforms );
		}

	case hkcdShapeType::CONVEX_LIST:
	case hkcdShapeType::LIST:
	case hkcdShapeType::COLLECTION:
	case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
	//case hkcdShapeType::TRIANGLE_COLLECTION: does not work because of 0x2a23423
		{
			const hkpShapeContainer* container = reinterpret_cast<const hkpShape*>(shape)->getContainer();
			hkpShapeBuffer buffer;

			for (hkpShapeKey key = container->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = container->getNextKey( key ) )
			{
				const hkpShape* child = container->getChildShape(key, buffer );

				// Check if storage is inside of the buffer
				if (child == reinterpret_cast<const hkpShape*>(&buffer))
				{
					HK_WARN_ALWAYS(0x2a23423,"Cannot flatten with containers that store shapes in a hkpShapeBuffer");
					// Cannot handle this case
					return HK_FAILURE;
				}

				hkResult res = flattenIntoConvexShapes(child, parentToReference, shapes, localToReferenceTransforms);
				if (res == HK_FAILURE) { return res; }
			}
			return HK_SUCCESS;
		}

	case hkcdShapeType::STATIC_COMPOUND:
		{
			const StaticCompoundShape* scs = reinterpret_cast<const StaticCompoundShape*>(shape);

			for (int ii = 0; ii < scs->getInstances().getSize(); ii++)
			{
				const StaticCompoundShape::Instance& instance = scs->getInstances()[ii];

				// Get transform
				const hkQsTransform instanceTm = instance.getTransform();
				hkTransform tm;
				tm.set(instanceTm.getRotation(), instanceTm.getTranslation());

				hkTransform childToReference;
				childToReference.setMul(parentToReference, tm);
				hkResult res = flattenIntoConvexShapes(instance.getShape(), childToReference, shapes, localToReferenceTransforms);
				if ( res == HK_FAILURE )
				{
					return res;
				}
			}
		}
		return HK_SUCCESS;

	case hkcdShapeType::EXTENDED_MESH:
		{
			const hkpExtendedMeshShape* ems = reinterpret_cast<const hkpExtendedMeshShape*>(shape);

			hkpShapeBuffer shapeBuffer;
			for (hkpShapeKey shapeKey = ems->getFirstKey(); shapeKey != HK_INVALID_SHAPE_KEY; shapeKey = ems->getNextKey( shapeKey ) )
			{
				const hkpShape* child = ems->getChildShape(shapeKey, shapeBuffer );

				// Check if storage is inside of the buffer
				if ( child == reinterpret_cast<const hkpShape*>(&shapeBuffer) )
				{
					HK_WARN_ALWAYS(0x2a23423,"Cannot flatten with containers that store shapes in a hkpShapeBuffer");
					// Cannot handle this case
					return HK_FAILURE;
				}

				hkResult res = flattenIntoConvexShapes(child, parentToReference, shapes, localToReferenceTransforms);
				if (res == HK_FAILURE) { return res; }
			}
		}
		return HK_SUCCESS;

	default:
		HK_WARN_ALWAYS(0xf0434565,"Cannot flatten shape type " << hkGetShapeTypeName(shape->getType()) << "." );
		return HK_FAILURE;
	}
}

void HK_CALL hkpShapeCutterUtil::findConnectedIslands(hkpShapeConnectedCalculator* connected, const hkArray<const hkpConvexShape*>& shapes, const hkArray<hkTransform>& transforms, hkArray<int>& partitionSizes, hkArray<int>& partitions)
{
	HK_ASSERT(0x342ab443,transforms.getSize() == shapes.getSize());

	const int numShapes = shapes.getSize();
	if (numShapes == 0) { return; }
	if (numShapes == 1)
	{
		partitionSizes.pushBack(1);
		partitions.pushBack(0);
		return;
	}

	hkLocalBuffer<int> parents(numShapes);
	hkUnionFind unionFind(parents,numShapes);

	unionFind.beginAddEdges();
	for (int i = 0; i < numShapes; i++)
	{
		const hkTransform& curTrans = transforms[i];
		const hkpConvexShape* curShape = shapes[i];

		for (int j = i + 1; j < numShapes; j++)
		{
			const hkpConvexShape* otherShape = shapes[j];
			const hkTransform& otherTrans = transforms[j];

			if (connected->isConnected(curShape,curTrans,otherShape,otherTrans))
			{
				unionFind.addEdge(i,j);
			}
		}
	}
	unionFind.endAddEdges();

	// If one group
	if (unionFind.isOneGroup())
	{
		partitions.setSize(numShapes);
		for (int i = 0; i < numShapes; i++) partitions[i] = i;
		partitionSizes.setSize(1);
		partitionSizes[0] = numShapes;
		return;
	}
	unionFind.assignGroups(partitionSizes);
	unionFind.sortByGroupId(partitionSizes, partitions);
}

const hkpShape* HK_CALL hkpShapeCutterUtil::createCompound(const hkpConvexShape*const * shapes,const hkTransform* transforms,int size)
{
	if (size <= 0)
	{
		return HK_NULL;
	}
	if (size == 1)
	{
		return transformConvexShape(*shapes,*transforms);
	}

	hkLocalArray<const hkpShape*> shapesOut(size);
	for (int i = 0; i < size; i++)
	{
		shapesOut.pushBackUnchecked(transformConvexShape(shapes[i],transforms[i]));
	}

	hkpListShape* result = new hkpListShape(shapesOut.begin(),shapesOut.getSize());
	hkReferencedObject::removeReferences( shapesOut.begin(),shapesOut.getSize() );
	return result;
}

const hkpShape* HK_CALL hkpShapeCutterUtil::createCompoundExtendedMesh(const hkpConvexShape*const * shapes, const hkTransform* transforms, int size)
{
	const hkpListShape* list = reinterpret_cast<const hkpListShape*>(hkpShapeCutterUtil::createCompound(shapes, transforms, size));

	int numChildShapes = list->getNumChildShapes();
	int numBitsForSubpartIndex = 1;
	while ( (1<<(numBitsForSubpartIndex-1)) < numChildShapes ) { numBitsForSubpartIndex++; }

	// Convert list shapes transformed children into an extended mesh shape
	hkpExtendedMeshShape* extendedMesh = new hkpExtendedMeshShape(hkConvexShapeDefaultRadius, numBitsForSubpartIndex);

	for (int childIdx=0; childIdx < numChildShapes; childIdx++)
	{
		const hkpShape* child = list->getChildShapeInl( childIdx );

		switch (child->getType())
		{
			case hkcdShapeType::MOPP:
			{
				const hkpMoppBvTreeShape* moppShape = reinterpret_cast<const hkpMoppBvTreeShape*>(child);
				const hkpShapeCollection* shapeCollection = moppShape->getShapeCollection();

				if (shapeCollection->getType() == hkcdShapeType::TRIANGLE_COLLECTION )
				{
					// Assume a simple mesh shape - the tools always produce this
					const hkpSimpleMeshShape* meshShape = reinterpret_cast<const hkpSimpleMeshShape*>(moppShape->getShapeCollection());

					//Put triangles in a single subpart
					hkpExtendedMeshShape::TrianglesSubpart triSubpart;
					triSubpart.m_numTriangleShapes = meshShape->m_triangles.getSize();
					triSubpart.m_vertexBase = &meshShape->m_vertices.begin()[0](0);
					triSubpart.m_vertexStriding = hkSizeOf(hkVector4);

					triSubpart.m_numVertices = meshShape->m_vertices.getSize();
					triSubpart.m_indexBase = meshShape->m_triangles.begin();
					triSubpart.m_stridingType = hkpExtendedMeshShape::INDICES_INT32;
					triSubpart.m_indexStriding = hkSizeOf(hkpSimpleMeshShape::Triangle);

					extendedMesh->addTrianglesSubpart( triSubpart );
				}
			}
			break;

		default:
			if (child->isConvex())
			{
				const hkpConvexShape* convexChild =  reinterpret_cast<const hkpConvexShape*>(child);
				hkpExtendedMeshShape::ShapesSubpart shapesSub( &convexChild, 1, hkTransform::getIdentity() );
				
				extendedMesh->addShapesSubpart( shapesSub );
			}
			break;
		}
	}

	list->removeReference();
	return extendedMesh;
}


int HK_CALL hkpShapeCutterUtil::findShapeSideOfPlane(const hkpConvexShape* shapeIn,const hkVector4& plane)
{
	switch (shapeIn->getType())
	{
	default:
	case hkcdShapeType::BOX:
		{
			const hkpBoxShape* box = static_cast<const hkpBoxShape*>(shapeIn);

			HK_ASSERT2(0xad87633a, 8 == box->getNumCollisionSpheres(), "Box shape does not have 8 collision spheres.");
			hkSphere sphereBuffer[8];
			const hkSphere* spheres = box->getCollisionSpheres(sphereBuffer);

			// copy from 'case hkcdShapeType::CONVEX_VERTICES' below..
			int sides = 0;
			for (int i = 0; i < 8; i++)
			{
				hkSimdReal side = plane.dot4xyz1(spheres[i].getPosition());

				sides |= side.isLessZero() ? INSIDE : OUTSIDE;

				if (sides == (INSIDE|OUTSIDE)) return 0;
			}
			switch (sides)
			{
			case INSIDE: return -1;
			case OUTSIDE: return 1;
			default: break;
			}
			return 0;
		}
	case hkcdShapeType::CAPSULE:
		{
			const hkpCapsuleShape* cap = static_cast<const hkpCapsuleShape*>(shapeIn);

			const hkSimdReal radius = hkSimdReal::fromFloat(cap->getRadius());
			hkSimdReal side0 = plane.dot4xyz1(cap->getVertex<0>());
			hkSimdReal side1 = plane.dot4xyz1(cap->getVertex<1>());

			// points on opposite side
			if ((side0 * side1).isLessZero()) return -1;

			// intersection found if either endpoint is within radius distance of plane
			side0.setAbs(side0);
			side1.setAbs(side1);
			if (side0 < radius || side1 < radius) return -1;

			// tangent
			const hkSimdReal eps = hkSimdReal_Eps;
			if (side0 < eps || side1 < eps) return 0;

			return 1;
		}
	case hkcdShapeType::CYLINDER:
		{
			HK_ASSERT2(0x7bba5466, false, "Unhandled shape type");
			break;
		}
	case hkcdShapeType::SPHERE:
		{
			const hkpSphereShape* sphere = static_cast<const hkpSphereShape*>(shapeIn);

			const hkSimdReal dist = plane.getW();
			const hkSimdReal radius = hkSimdReal::fromFloat(sphere->getRadius());
			if ((dist - radius).isGreaterZero()) return  1;
			if ((dist + radius).isLessZero())    return -1;
			return 0;
		}
	case hkcdShapeType::CONVEX_VERTICES:
		{
			int sides = 0;

			const hkpConvexVerticesShape* shape = static_cast<const hkpConvexVerticesShape*>(shapeIn);
			hkArray<hkVector4> vertices;

			shape->getOriginalVertices( vertices);
			const int numVertices = vertices.getSize();

			for (int i = 0; i < numVertices; i++)
			{
				hkSimdReal side = plane.dot4xyz1(vertices[i]);

				sides |= side.isLessZero() ? INSIDE : OUTSIDE;

				if (sides == (INSIDE|OUTSIDE)) return 0;
			}
			switch (sides)
			{
			case INSIDE: return -1;
			case OUTSIDE: return 1;
			default: break;
			}
			return 0;
		}
	}
	return 0;
}

int HK_CALL hkpShapeCutterUtil::findShapeSideOfPlane(const hkpShape* shapeIn, const hkVector4& plane)
{
	// get all of the leaves
	hkArray<hkTransform> transforms;
	hkArray<const hkpConvexShape*> shapes;

	hkResult res = hkpShapeCutterUtil::flattenIntoConvexShapes(shapeIn, hkTransform::getIdentity(), shapes, transforms);

	// Don't know the result
	if (res != HK_SUCCESS) return 0;

	const int numShapes = shapes.getSize();

	int side = 0;
	for (int i = 0; i < numShapes; i++)
	{
		const hkpConvexShape* convexShape = shapes[i];
		const hkTransform& localToWorld = transforms[i];

		hkTransform worldToLocal;
		worldToLocal.setInverse(localToWorld);

		hkVector4 localPlane;
		hkVector4Util::transformPlaneEquation(worldToLocal, plane, localPlane);

		// Work out which side the shape is on

		int shapeSide = findShapeSideOfPlane(convexShape,localPlane);
		if (shapeSide == 0) return 0;

		side |= (shapeSide < 0) ? INSIDE : OUTSIDE;

		if (side == (INSIDE|OUTSIDE)) return 0;
	}

	switch (side)
	{
	case INSIDE: return -1;
	case OUTSIDE: return 1;
	default: break;
	}
	return 0;
}


struct ConvexShapeCollector
{
	void collect( const hkpConvexShape* shape, const hkTransform& transform);
};

void HK_CALL hkpShapeCutterUtil::iterateConvexLeafShapes( const hkcdShape* shape, const hkTransform& transform, ConvexShapeCollector& collector )
{
	switch (shape->getType())
	{
	case hkcdShapeType::CONVEX_TRANSLATE:
		{
			const hkpConvexTranslateShape* ts = static_cast<const hkpConvexTranslateShape*>( shape );
			hkTransform localToParent = hkTransform::getIdentity();
			localToParent.setTranslation(ts->getTranslation());

			hkTransform localToReference;
			localToReference.setMul(transform,localToParent);
			return iterateConvexLeafShapes(ts->getChildShape(), localToReference, collector );
		}
	case hkcdShapeType::CONVEX_TRANSFORM:
		{
			const hkpConvexTransformShape* ts = static_cast<const hkpConvexTransformShape*>( shape );
			hkTransform localTransform; ts->getTransform( &localTransform );
			hkTransform localToReference; localToReference.setMul( transform, localTransform );

			return iterateConvexLeafShapes(ts->getChildShape(), localToReference, collector );
		}
	case hkcdShapeType::TRANSFORM:
		{
			const hkpTransformShape* ts = static_cast<const hkpTransformShape*>( shape );
			hkTransform localToReference;
			localToReference.setMul(transform, ts->getTransform());

			return iterateConvexLeafShapes(ts->getChildShape(), localToReference, collector );
		}



	case hkcdShapeType::MOPP:
		{
			const hkpMoppBvTreeShape* cs = static_cast<const hkpMoppBvTreeShape*>(shape);
			return iterateConvexLeafShapes( cs->getShapeCollection(), transform, collector );
		}

	case hkcdShapeType::CONVEX_LIST:
	case hkcdShapeType::LIST:
	case hkcdShapeType::COLLECTION:
	case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
		//case hkcdShapeType::TRIANGLE_COLLECTION: does not work because of 0x2a23423
		{
			const hkpShapeContainer* container = reinterpret_cast<const hkpShape*>(shape)->getContainer();
			hkpShapeBuffer buffer;

			for (hkpShapeKey key = container->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = container->getNextKey( key ) )
			{
				const hkpShape* child = container->getChildShape(key, buffer );

				// Check if storage is inside of the buffer
				if (child == reinterpret_cast<const hkpShape*>(&buffer))
				{
					HK_WARN_ALWAYS(0x2a23423,"Cannot flatten with containers that store shapes in a ShapeBuffer");
					// Cannot handle this case
					return;
				}
				iterateConvexLeafShapes(child, transform, collector );
			}
			return;
		}

	case hkcdShapeType::CONVEX_VERTICES:
	case hkcdShapeType::BOX:
	case hkcdShapeType::CAPSULE:
	case hkcdShapeType::CYLINDER:
	case hkcdShapeType::SPHERE:
	case hkcdShapeType::TRIANGLE:	
		{
			collector.collect( (const hkpConvexShape*)shape, transform );
			break;
		}
default:
		HK_WARN_ALWAYS(0xf0434565,"Unknown type in iterateConvexLeaveShapes" << hkGetShapeTypeName(shape->getType()) << "." );
	}
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
