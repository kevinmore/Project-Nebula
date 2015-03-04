/*
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

// PCH
#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Common/Internal/ConvexHull/hkGeometryUtility.h>
#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>
#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshSystem.h>
#include <Common/GeometryUtilities/Mesh/Converters/SceneDataToMesh/hkSceneDataToMeshConverter.h>

#include <Physics2012/Collide/Util/ShapeShrinker/hkpShapeShrinker.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>

#include <Physics2012/Utilities/Collide/ShapeUtils/CreateShape/hkpCreateShapeUtility.h>


static void _getStridedVertices(const hkArray<hkVector4>& vertices, const hkMatrix4& vertexTransform, hkStridedVertices& stridedVertsOut, hkArray<hkVector4>& tempStorageOut)
{
	stridedVertsOut.m_numVertices	= 0;
	stridedVertsOut.m_striding		= sizeof(hkVector4);

	const hkBool32 vertexTransformIsIdentity = vertexTransform.isApproximatelyIdentity(hkSimdReal::fromFloat(1e-3f));

	// Transform vertices if case
	if (!vertexTransformIsIdentity)
	{
		int numVerts = vertices.getSize();
		tempStorageOut.setSize(numVerts);
		const hkVector4 * pSrc = vertices.begin();
		hkVector4		* pDst = tempStorageOut.begin();

		for(int vi = 0 ; vi < numVerts ; vi++)
		{
			vertexTransform.transformPosition(pSrc[vi], pDst[vi]);
		}

		if (tempStorageOut.getSize() > 0)
		{
			stridedVertsOut.m_numVertices	= tempStorageOut.getSize();
			stridedVertsOut.m_vertices		= &tempStorageOut.begin()[0](0);
		}
	}
	else
	{
		// Can convert directly, no need to copy
		stridedVertsOut.m_numVertices	= vertices.getSize();
		stridedVertsOut.m_vertices		= &vertices.begin()[0](0);
	}
}

//	Computes the AABB from a set of vertices

hkResult HK_CALL hkpCreateShapeUtility::computeAABB(const hkArray<hkVector4>& vertices, const hkMatrix4& vertexTransform, hkReal minBoxExtent, hkVector4& halfExtentsOut, hkTransform& obbTransformOut)
{
	hkArray<hkVector4>	tempStorage;
	hkStridedVertices	stridedVerts;
	_getStridedVertices(vertices, vertexTransform, stridedVerts, tempStorage);

	if (stridedVerts.m_numVertices > 0)
	{
		hkVector4 minBox = hkVector4::getConstant<HK_QUADREAL_MAX>();
		hkVector4 maxBox = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();
		{
			const int floatstriding = stridedVerts.m_striding/hkSizeOf(hkReal);
			for( int i = 0; i < stridedVerts.m_numVertices * 3; i++)
			{
				const int vidx = i / 3; // vertex index (0..numVertices)
				const int component = i % 3; // component (0,1,2 == x,y,z)

				hkReal value = stridedVerts.m_vertices[vidx*floatstriding+component];

				maxBox(component) = hkMath::max2(maxBox(component), value);

				minBox(component) = hkMath::min2(minBox(component), value);

			}
		}

		hkVector4 boxCenter; boxCenter.setInterpolate(minBox, maxBox, hkSimdReal_Inv2);

		obbTransformOut.set(hkQuaternion::getIdentity(), boxCenter);
		halfExtentsOut.setSub(maxBox, minBox); 
		
		if (minBoxExtent > hkReal(0))
		{
			hkVector4 minBoxExtentSr; minBoxExtentSr.setAll(minBoxExtent);
			halfExtentsOut.setMax(minBoxExtentSr, halfExtentsOut);
		}

		halfExtentsOut.mul(hkSimdReal_Inv2);

		if (halfExtentsOut.lengthSquared<3>() > hkSimdReal::fromFloat(0.0001f*0.0001f))
		{
			return HK_SUCCESS;
		}
	}

	return HK_FAILURE;
}

//	Computes the OBB of a set of vertices

hkResult HK_CALL hkpCreateShapeUtility::computeOBB(const hkArray<hkVector4>& vertices, const hkMatrix4& vertexTransform, hkReal minBoxExtent, hkVector4& halfExtentsOut, hkTransform& obbTransformOut)
{
	// Use OBB
	hkArray<hkVector4>	tempStorage;
	hkStridedVertices	stridedVerts;
	_getStridedVertices( vertices, vertexTransform, stridedVerts, tempStorage );

	if (stridedVerts.m_numVertices > 0)
	{
		hkBool wasEnabled = hkError::getInstance().isEnabled( 0x34df5494 );
		hkError::getInstance().setEnabled( 0x34df5494, false );

		hkGeometryUtility::calcObb(stridedVerts, halfExtentsOut, obbTransformOut);

		hkError::getInstance().setEnabled( 0x34df5494, wasEnabled );

		if (minBoxExtent > hkReal(0))
		{
			hkVector4 minBoxHalfExtent; minBoxHalfExtent.setAll(minBoxExtent * 0.5f);
			halfExtentsOut.setMax(minBoxHalfExtent, halfExtentsOut);
		}

		if (halfExtentsOut.lengthSquared<3>() > hkSimdReal::fromFloat(0.0001f*0.0001f) )
		{
			return HK_SUCCESS;
		}
	}

	return HK_FAILURE;
}

//	Creates a box shape from a set of vertices

hkResult HK_CALL hkpCreateShapeUtility::createBoxShape(CreateShapeInput& input, ShapeInfoOutput& output)
{
	// Handle the case where the name is null
	const char * szName = input.m_szMeshName ? input.m_szMeshName : "";

	// Compute OBB / AABB
	hkVector4	halfExtents;
	hkTransform geomFromBox;
	hkResult	obbOk;
	{
		if (input.m_bestFittingShapes)
		{
			obbOk = computeOBB (input.m_vertices, input.m_vertexTM, input.m_minBoxExtent, halfExtents, geomFromBox);
		}
		else
		{
			obbOk = computeAABB(input.m_vertices, input.m_vertexTM, input.m_minBoxExtent, halfExtents, geomFromBox);
		}
	
		if (obbOk == HK_FAILURE)
		{
			HK_WARN_ALWAYS(0x512ba184, "Couldn't calculate OBB / AABB for mesh " << szName);
			return HK_FAILURE;
		}
	}

	// see if we need to add the obb transform to the transform stack
	output.m_extraShapeTransform	= input.m_extraShapeTransform;
	output.m_decomposedWorldT		= input.m_decomposedWorldT;
	if (!geomFromBox.isApproximatelyEqual( hkTransform::getIdentity() ))
	{
		// EXP-1202
		output.m_extraShapeTransform = geomFromBox;

		// add the obbTrans to the decomposed shape trans
		hkTransform res;
		res.setMul(output.m_decomposedWorldT, geomFromBox);
		output.m_decomposedWorldT = res;
	}

	bool automaticShrinkingEnabled	= input.m_enableAutomaticShapeShrinking;
	hkReal extraRadiusOverride		= input.m_extraRadiusOverride; // Override range is between 0 and 1, so we are safe to use this for the time being.
	hkReal extraRadius				= hkReal(0);

	if ( !automaticShrinkingEnabled )
	{
		if ( extraRadiusOverride >= hkReal(0) )
		{
			extraRadius = extraRadiusOverride;
		}
		else
		{
			extraRadius = input.m_defaultConvexRadius;
		}
	}

	hkpBoxShape * boxShape = new hkpBoxShape( halfExtents, extraRadius );

	// Shrink shape if option is enabled (with meaningful values).
	if ( automaticShrinkingEnabled && input.m_relShrinkRadius != hkReal(0) && input.m_maxVertexDisplacement != hkReal(0) )
	{
		hkpShapeShrinker::shrinkBoxShape( boxShape, input.m_relShrinkRadius, input.m_maxVertexDisplacement );
	}

	// Issue some warnings if final convex/extra radius is either 0.0 or rather large.
	{
		hkReal convexRadius = boxShape->getRadius();
		hkVector4 absExtent; absExtent.setAbs(halfExtents); 
		const hkReal meshSize = absExtent.horizontalMax<3>().getReal();

		if ( convexRadius == hkReal(0) )
		{
			HK_WARN_ALWAYS (0xabbabfd2, "'" << szName << "' - Calculated 'extra radius' for box shape is 0. Performance may be affected.");
		}
		else if ( convexRadius > meshSize )
		{
			HK_WARN_ALWAYS (0xabbabad3, "'" << szName << "' - Calculated 'extra radius' for box shape is very large (radius " << convexRadius << " > halfExtent " << meshSize << " )");
		}
	}

	output.m_shape    = boxShape;
	output.m_isConvex = true;

	return HK_SUCCESS;
}

//	Creates a sphere shape from a set of vertices

hkResult HK_CALL hkpCreateShapeUtility::createSphereShape(CreateShapeInput& input, ShapeInfoOutput& output)
{
	// Handle the case where the name is null
	const char * szName = input.m_szMeshName ? input.m_szMeshName : "";

	// Compute OBB / AABB
	hkVector4	halfExtents;
	hkTransform geomFromBox;
	hkResult	obbOk;
	{
		if (input.m_bestFittingShapes)
		{
			obbOk = computeOBB (input.m_vertices, input.m_vertexTM, input.m_minBoxExtent, halfExtents, geomFromBox);
		}
		else
		{
			obbOk = computeAABB(input.m_vertices, input.m_vertexTM, input.m_minBoxExtent, halfExtents, geomFromBox);
		}

		if (obbOk == HK_FAILURE)
		{
			HK_WARN_ALWAYS(0x512ba184, "Couldn't calculate OBB / AABB for mesh " << szName);
			return HK_FAILURE;
		}
	}

	// see if we need to add the obb transform to the transform stack
	output.m_extraShapeTransform	= input.m_extraShapeTransform;
	output.m_decomposedWorldT		= input.m_decomposedWorldT;
	if (!geomFromBox.isApproximatelyEqual( hkTransform::getIdentity() ))
	{
		// EXP-1202
		output.m_extraShapeTransform = geomFromBox;

		// add the obbTrans to the decomposed shape trans
		hkTransform res;
		res.setMul(output.m_decomposedWorldT, geomFromBox);
		output.m_decomposedWorldT = res;
	}

	hkVector4 absExtent; absExtent.setAbs(halfExtents); 
	const hkReal radius = absExtent.horizontalMax<3>().getReal();
	output.m_shape		= new hkpSphereShape (radius);
	output.m_isConvex	= true;

	if ( input.m_enableAutomaticShapeShrinking )
	{
		if ( input.m_relShrinkRadius != hkReal(0) && input.m_maxVertexDisplacement != hkReal(0) )
		{
			HK_REPORT("'" << szName << "' - Shape shrinking not supported for sphere shapes.");
		}
	}
	else
	{
		if ( input.m_defaultConvexRadius != 0.0f )
		{
			HK_REPORT("'" << szName << "' - Convex radius not supported for sphere shapes.");
		}
	}

	return HK_SUCCESS;
}

//	Creates a capsule shape from a set of vertices

hkResult HK_CALL hkpCreateShapeUtility::createCapsuleShape(CreateShapeInput& input, ShapeInfoOutput& output)
{
	// Handle the case where the name is null
	const char * szName = input.m_szMeshName ? input.m_szMeshName : "";

	// Compute OBB / AABB
	hkVector4	halfExtents;
	hkTransform geomFromBox;
	hkResult	obbOk;
	{
		if (input.m_bestFittingShapes)
		{
			obbOk = computeOBB (input.m_vertices, input.m_vertexTM, input.m_minBoxExtent, halfExtents, geomFromBox);
		}
		else
		{
			obbOk = computeAABB(input.m_vertices, input.m_vertexTM, input.m_minBoxExtent, halfExtents, geomFromBox);
		}

		if (obbOk == HK_FAILURE)
		{
			HK_WARN_ALWAYS(0x512ba184, "Couldn't calculate OBB / AABB for mesh " << szName);
			return HK_FAILURE;
		}
	}

	// Just copy the transforms
	output.m_extraShapeTransform	= input.m_extraShapeTransform;
	output.m_decomposedWorldT		= input.m_decomposedWorldT;

	hkVector4 vecA;
	hkVector4 vecB;
	int majorAxis = halfExtents.getIndexOfMaxAbsComponent<3>();
	vecA.setZero();
	vecA(majorAxis) = -halfExtents(majorAxis);

	vecB.setZero();
	vecB(majorAxis) = halfExtents(majorAxis);

	halfExtents.zeroComponent(majorAxis);
	hkReal radius = halfExtents(halfExtents.getIndexOfMaxAbsComponent<3>());

	// Move in capsule end points
	hkVector4 radiusDelta; 
	radiusDelta.setZero();
	radiusDelta(majorAxis) = radius;
	vecA.add(radiusDelta);
	vecB.sub(radiusDelta);

	// Transform end points
	hkVector4 endPointA;
	hkVector4 endPointB;
	endPointA.setTransformedPos(geomFromBox, vecA);
	endPointB.setTransformedPos(geomFromBox, vecB);

	// EXP-386 : Handle degenerate capsules
	if (endPointA.allEqual<3>(endPointB,hkSimdReal::fromFloat(1e-3f)))
	{
		HK_WARN_ALWAYS(0xabba873d, "Degenerate Capsule (" << szName << "): Using Sphere Shape Instead");

		return createSphereShape(input, output);
	}

	output.m_shape  = new hkpCapsuleShape (endPointA, endPointB, radius );
	output.m_isConvex = true;

	if ( input.m_enableAutomaticShapeShrinking )
	{
		if ( input.m_relShrinkRadius != hkReal(0) && input.m_maxVertexDisplacement != hkReal(0) )
		{
			HK_REPORT("'" << szName << "' - Shape shrinking not supported for capsule shapes.");
		}
	}
	else
	{
		if ( input.m_defaultConvexRadius != hkReal(0) )
		{
			HK_REPORT("'" << szName << "' - Convex radius not supported for capsule shapes.");
		}
	}

	return HK_SUCCESS;
}

//	Creates a cylinder shape from a set of vertices
HK_DISABLE_OPTIMIZATION_VS2008_X64
hkResult HK_CALL hkpCreateShapeUtility::createCylinderShape(CreateShapeInput& input, ShapeInfoOutput& output)
{
	// Handle the case where the name is null
	const char * szName = input.m_szMeshName ? input.m_szMeshName : "";

	// Compute OBB / AABB
	hkVector4	halfExtents;
	hkTransform geomFromBox;
	hkResult	obbOk;
	{
		if (input.m_bestFittingShapes)
		{
			obbOk = computeOBB (input.m_vertices, input.m_vertexTM, input.m_minBoxExtent, halfExtents, geomFromBox);
		}
		else
		{
			obbOk = computeAABB(input.m_vertices, input.m_vertexTM, input.m_minBoxExtent, halfExtents, geomFromBox);
		}

		if (obbOk == HK_FAILURE)
		{
			HK_WARN_ALWAYS(0x512ba184, "Couldn't calculate OBB / AABB for mesh " << szName);
			return HK_FAILURE;
		}
	}

	// Just copy the transforms
	output.m_extraShapeTransform	= input.m_extraShapeTransform;
	output.m_decomposedWorldT		= input.m_decomposedWorldT;

	// Work on the vertices in "box space"
	hkTransform boxFromGeom; boxFromGeom.setInverse(geomFromBox);
	hkArray<hkVector4> transformedVerts;
	{
		int numVerts = input.m_vertices.getSize();
		for (int i = 0 ; i < numVerts ; i++)
		{
			hkVector4 newVert;
			newVert.setTransformedPos(boxFromGeom, input.m_vertices[i]);
			transformedVerts.pushBack(newVert);
		}
	}

	int mainAxis = -1;
	{
		// For the cylinder, there should be two axis that are mostly equal
		// The other axis is used as the main axis (the one with the biggest difference)
		hkSimdReal maxDiff = hkSimdReal_Minus1;
		hkSimdReal radius = hkSimdReal_Minus1;
		for (int axis=0; axis<3; axis++)
		{
			const int axis2 = (axis+1) % 3;
			const int axis3 = (axis+2) % 3;
			const hkSimdReal comp  = halfExtents.getComponent(axis);
			const hkSimdReal comp2 = halfExtents.getComponent(axis2);
			const hkSimdReal comp3 = halfExtents.getComponent(axis3);

			hkSimdReal diff1; diff1.setAbs( comp - comp2 );
			hkSimdReal diff2; diff2.setAbs( comp - comp3 );
			hkSimdReal diff; diff.setMin( diff1,diff2 );
			if ( diff > maxDiff )
			{
				mainAxis = axis;
				maxDiff = diff;
				// We take the radius from the biggest remaining axis
				radius.setMax( comp2, comp3 );
			}
		}

		// If the chosen axis is not particularly different (less than 10%) from the rest of axis
		// use a different algorithm
		hkSimdReal absE; absE.setAbs(hkSimdReal_1 - (halfExtents.getComponent(mainAxis) / radius));
		if (absE < hkSimdReal::fromFloat(0.1f))
		{
			// We find the "main axis" by projecting the vertices to the different planes
			// and finding which projection fits a circle/ellipse best
			mainAxis = -1;	

			hkSimdReal minAxisDist = hkSimdReal::fromFloat(100.0f);
			for (int axis=0; axis<3; axis++)
			{
				const int axisA = (axis+1) %3;
				const int axisB = (axis+2) %3;

				// We scale the values
				const hkSimdReal stretchA = halfExtents.getComponent(axisA);
				const hkSimdReal stretchB = halfExtents.getComponent(axisB);

				hkSimdReal thisAxisMaxDist = hkSimdReal_Minus1;

				for (int v=0; v<transformedVerts.getSize(); v++)
				{
					const hkVector4& vertex = transformedVerts[v];
					hkVector4 projectedVertex; projectedVertex.set(vertex.getComponent(axisA) / stretchA, vertex.getComponent(axisB) / stretchB, hkSimdReal_0, hkSimdReal_0);

					// We now have a vertex projected and X Y plane, inside the square of [(-1,-1), (1,1)]
					// See how far it is from the circle (we ignore those inside the circle
					const hkSimdReal distance = projectedVertex.length<2>();
					if (distance > thisAxisMaxDist)
					{
						thisAxisMaxDist = distance;
					}
				}

				if (thisAxisMaxDist < minAxisDist)
				{
					minAxisDist = thisAxisMaxDist;
					mainAxis = axis;
				}
			}

		}
	}

	hkReal radius;
	hkVector4 pointA;
	hkVector4 pointB;

	{
		// We now have the main axis, construct the capsule from it
		// We take the radius from the biggest remaining axis
		const int axis2 = (mainAxis+1) % 3;
		const int axis3 = (mainAxis+2) % 3;
		radius = hkMath::max2( halfExtents(axis2), halfExtents(axis3) );

		hkVector4 pointAbox; pointAbox.setZero( );
		hkVector4 pointBbox; pointBbox.setZero( );

		pointAbox(mainAxis) = halfExtents(mainAxis);
		pointBbox(mainAxis) = -halfExtents(mainAxis);

		// Finally, transform the vertices using the box transform
		pointA.setTransformedPos(geomFromBox, pointAbox);
		pointB.setTransformedPos(geomFromBox, pointBbox);
	}

	bool automaticShrinkingEnabled	= input.m_enableAutomaticShapeShrinking;
	hkReal extraRadiusOverride		= input.m_extraRadiusOverride;

	hkReal extraRadius = 0;
	if ( !automaticShrinkingEnabled )
	{
		if ( extraRadiusOverride >= 0 )
		{
			extraRadius = extraRadiusOverride;
		}
		else
		{
			extraRadius = input.m_defaultConvexRadius;
		}
	}

	hkpCylinderShape* cylinderShape = new hkpCylinderShape(pointA, pointB, radius, extraRadius);

	// Shrink shape if option is enabled (with meaningful values).
	if ( automaticShrinkingEnabled && input.m_relShrinkRadius != 0 && input.m_maxVertexDisplacement != 0 )
	{
		hkpShapeShrinker::shrinkCylinderShape( cylinderShape, input.m_relShrinkRadius, input.m_maxVertexDisplacement);
	}

	// Issue some warnings if calculated convex/extra radius is either 0.0 or rather large.
	{
		hkReal convexRadius = cylinderShape->getRadius();
		const hkReal meshSize = halfExtents (halfExtents.getIndexOfMaxAbsComponent<3>());

		if ( convexRadius == 0 )
		{
			HK_WARN_ALWAYS (0xabbabfd4, "'" << szName << "' - Calculated 'extra radius' for cylinder shape is 0. Performance may be affected.");
		}
		else if ( convexRadius > meshSize )
		{
			HK_WARN_ALWAYS (0xabbabad5, "'" << szName << "' - Calculated 'extra radius' for cylinder shape is very large (radius " << convexRadius << " > halfExtent " << meshSize << " )");
		}
	}

	output.m_shape = cylinderShape;
	output.m_isConvex = true;

	return HK_SUCCESS;
}
HK_RESTORE_OPTIMIZATION_VS2008_X64

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
