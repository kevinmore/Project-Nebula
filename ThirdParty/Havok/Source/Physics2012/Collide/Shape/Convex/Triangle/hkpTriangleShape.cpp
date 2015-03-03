/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastTriangle.h>

#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>

#include <Physics2012/Internal/Collide/Gjk/hkpGskCache.h>

#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayBundleCastInput.h>

#if !defined(HK_PLATFORM_SPU)

#if ((HK_POINTER_SIZE==4) && !defined(HK_REAL_IS_DOUBLE) && (HK_NATIVE_ALIGNMENT==16) && !defined(HK_COMPILER_HAS_INTRINSICS_NEON))
HK_COMPILE_TIME_ASSERT( sizeof ( hkpTriangleShape ) == 6 * 16 );
#endif

//
//	Serialization constructor

hkpTriangleShape::hkpTriangleShape( hkFinishLoadedObjectFlag flag )
:	hkpConvexShape( flag )
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpTriangleShape));
}


void hkpTriangleShape::getFirstVertex(hkVector4& v) const
{
	v = getVertex<0>();
}

#endif

void hkpTriangleShape::getSupportingVertex(hkVector4Parameter direction, hkcdVertex& supportingVertexOut) const
{
	// binary equivalents
	const hkRotation* v3 = reinterpret_cast< const hkRotation* >(getVertices());

	hkVector4 supportDots;		supportDots._setRotatedInverseDir( *v3, direction );
	int vID = supportDots.getIndexOfMaxComponent<3>();

	static_cast<hkVector4&>(supportingVertexOut) = *hkAddByteOffsetConst<hkVector4>( getVertices(), vID * sizeof(hkVector4) );

	if ( direction.dot<3>(m_extrusion).isGreaterZero() )
	{
		vID += 3;
		supportingVertexOut.add( m_extrusion );
	}
	supportingVertexOut.setInt24W( vID );
}

// Removing need for integer division
static hkPadSpu<int> mod3Table[] = { 0, 1, 2, 0, 1, 2 };
static hkReal addExtrusionTable[] = { 0, 0, 0, 1.f, 1.f, 1.f };

void hkpTriangleShape::convertVertexIdsToVertices(const hkpVertexId* ids, int numIds, hkcdVertex* verticesOut) const
{
	for (int i = numIds-1; i>=0; i--)
	{
		static_cast<hkVector4&>(verticesOut[0]) =  *hkAddByteOffsetConst<hkVector4>( getVertices(), mod3Table[ids[0]] * sizeof(hkVector4) );
		static_cast<hkVector4&>(verticesOut[0]).addMul(  hkSimdReal::fromFloat(addExtrusionTable[ids[0]]), m_extrusion);
		verticesOut[0].setInt24W( ids[0] );
		verticesOut++;
		ids++;
	}
}

void hkpTriangleShape::getCentre(hkVector4& centreOut) const
{
	centreOut.setAdd(m_vertexA, m_vertexB);
	centreOut.add(m_vertexC);
	centreOut.mul(hkSimdReal_Inv3);
}


static inline int getSingleEdgeBitcode(hkUint16 triangleEdgesBitcode, int edgeIndex)
{
	int edgeAngleBitcode = (triangleEdgesBitcode >> (edgeIndex*5)) & 0x1f; // filter 5bits
	return edgeAngleBitcode;
}


//
// This static function simply calculates the normal for a triangle defined by its three vertices. The vertices have to be
// supplied in counter-clockwise order.
//
static inline void calcAntiClockwiseTriangleNormal(const hkVector4& vertex0, const hkVector4& vertex1, const hkVector4& vertex2, hkVector4& normal)
{
	hkVector4 edges[2];
	{
		edges[0].setSub(vertex1, vertex0);
		edges[1].setSub(vertex2, vertex1);
	}

	normal.setCross(edges[0], edges[1]);
	normal.normalize<3>();
}

// Welding type can be WELDING_TYPE_ANTICLOCKWISE (0) or WELDING_TYPE_CLOCKWISE (4) when this table is used.
static hkReal flipNormalBasedOnWeldingTypeTable[] = { 1.f, 0.f, 0.f, 0.f, -1.f };


int hkpTriangleShape::weldContactPoint(hkpVertexId* featurePoints, hkUint8& numFeaturePoints, hkVector4& contactPointWs, const hkTransform* thisObjTransform, const hkpConvexShape* collidingShape, const hkTransform* collidingTransform, hkVector4& separatingNormalInOut ) const
{
	hkpWeldingUtility::WeldingType weldingType = m_weldingType;
	if (weldingType == hkpWeldingUtility::WELDING_TYPE_NONE)
	{
		return WELD_RESULT_ACCEPT_CONTACT_POINT_UNMODIFIED;
	}

	// TODO - Handle vertex collisions correctly

	// The following code takes the gskCache, and sets edgeId0 and edgeId1 to be the vertices of the edge of the collision.
	// Vertex ids in the gsk cache can be 0-5. Vertex ids 3-5 are "extruded" vertices, and are projected onto the real vertices,
	// i.e. mapped to vertex ids 0 - 2.  If we have a vertex face collision where the face is an extruded face, we map this to
	// the corresponding projected edge.  If we have a vertex collision we (currently) use the vertex as the start of the edge
	// to weld to.  We make sure the edge is pointing in the anticlockwise direction.

	int vertexId0 = mod3Table[ featurePoints[0] ];	// jump to the front side, don't use extruded vertices
	int vertexId1;

	if (numFeaturePoints > 1)
	{
		vertexId1 = mod3Table[ featurePoints[1] ];

		// Check if vertex face collision
		if ( numFeaturePoints == 3 )
		{
			// If the collision is against the actual (non-extruded) face, just return, no welding needed.
			if (featurePoints[0] + featurePoints[1] + featurePoints[2] == 3 )
			{
				return WELD_RESULT_ACCEPT_CONTACT_POINT_UNMODIFIED;
			}

			// If the first 2 vertices project to the same point, replace vertex 1 with the 3rd vertex from the face collision, to make sure
			// the face projects to the edge.
			if (vertexId1 == vertexId0)
			{
				vertexId1 = mod3Table[ featurePoints[2] ];
			}

			// We have hit an extruded face - change this to an edge so when put into the gsk manifold it will be treated
			// as an edge point, and have its normal set for it. Note: this relies on the triangle being object B in the collision.
			numFeaturePoints = 2;
			featurePoints[0] = hkpVertexId(vertexId0);
			featurePoints[1] = hkpVertexId(vertexId1);
		}

		// Make sure the edge is pointing in the anticlockwise direction
		if (mod3Table[ vertexId0 + 1 ] != vertexId1 )
		{
			vertexId0 = vertexId1;
		}
	}
	vertexId1 = mod3Table[ vertexId0 + 1 ];


	// Rotate the normal into local space
	hkVector4 sepNormal = separatingNormalInOut;
	hkVector4 sepNormalLocal;	sepNormalLocal._setRotatedInverseDir( thisObjTransform->getRotation(), sepNormal );

	hkVector4 triangleNormal; calcAntiClockwiseTriangleNormal(m_vertexA, m_vertexB, m_vertexC, triangleNormal);

	hkSimdReal cosSnapAngle; cosSnapAngle.setAbs( sepNormalLocal.dot<3>(triangleNormal) );
	hkSimdReal penetrationDist = sepNormal.getW();
	penetrationDist.mul(cosSnapAngle);

	hkVector4 edge;  edge.setSub( getVertex( vertexId1 ), getVertex( vertexId0 ) ); 
	edge.normalize<3>();

	int edgeBitcode = getSingleEdgeBitcode(m_weldingInfo, vertexId0);
	if (weldingType != hkpWeldingUtility::WELDING_TYPE_TWO_SIDED)
	{
		HK_ASSERT2( 0xf0233212, weldingType == hkpWeldingUtility::WELDING_TYPE_CLOCKWISE || weldingType == hkpWeldingUtility::WELDING_TYPE_ANTICLOCKWISE, "hkTriangle::m_weldingType is not set, this will crash the engine" );
		if( !hkpWeldingUtility::shouldSnapOneSided( weldingType, triangleNormal, sepNormalLocal, edgeBitcode ) )
		{
			return WELD_RESULT_ACCEPT_CONTACT_POINT_UNMODIFIED;
		}

		/*
		if ( numFeaturePoints == 1  || numFeaturePoints == 2 && mod3Table[featurePoints[1] ] == vertexId0 )
		{
			// do not snap vertex collision
			return WELD_RESULT_REJECT_CONTACT_POINT;
		}
		*/

		hkpWeldingUtility::calcSnapVectorOneSided( triangleNormal, edge, edgeBitcode, weldingType, sepNormal );

		// Rotate the welded normal back into world space
		sepNormal._setRotatedDir(thisObjTransform->getRotation(), sepNormal );

		if (m_isExtruded)
		{
			// Modify distance to be the height under the plane of the triangle
			{
				triangleNormal.mul( hkSimdReal::fromFloat(flipNormalBasedOnWeldingTypeTable[weldingType]) );

				HK_ASSERT2(0x509fea65, triangleNormal.dot<3>(m_extrusion).getReal() < 0, "Inconsistent triangle extrusion / winding found");
				hkVector4 contactPointLocal; contactPointLocal.setTransformedInversePos( *thisObjTransform, contactPointWs);
				hkVector4 depthVec; depthVec.setSub( contactPointLocal, getVertex<0>() );
				const hkSimdReal depth = depthVec.dot<3>( triangleNormal );

				// This min2 may have an effect if the triangle extrusion is at a large angle to its normal
				penetrationDist.setMin(depth, penetrationDist);
			}

RETURN_ACCEPT_CONTACT_POINT_MODIFIED:
#if !defined(HK_PLATFORM_SPU)
			separatingNormalInOut.setXYZ_W(sepNormal, penetrationDist);
#else
			sepNormal.setComponent<3>(penetrationDist);
			separatingNormalInOut = sepNormal;
#endif
			return WELD_RESULT_ACCEPT_CONTACT_POINT_MODIFIED;
		}
		else
		{
			// If the triangle is not extruded, then perform an extra check: Only accept if the welded contact point "supports" the centre of the colliding object.  If it does not
			// then we reject the point.  This is here to prevent welding trying to pull objects through triangles from the wrong side.
			// It prevents objects becoming stuck in thin walls.

			hkVector4 centreOfObjectA; collidingShape->getCentre(centreOfObjectA);
			centreOfObjectA.setTransformedPos( *collidingTransform, centreOfObjectA );
			hkVector4 centreToPlane; centreToPlane.setSub(contactPointWs, centreOfObjectA );

			if ( centreToPlane.dot<3>(sepNormal).isLessZero() )
			{
				goto RETURN_ACCEPT_CONTACT_POINT_MODIFIED;
			}
			return WELD_RESULT_REJECT_CONTACT_POINT;
		}
	}
	else
	{
		hkpWeldingUtility::SectorType sector = hkpWeldingUtility::getSector(triangleNormal, sepNormalLocal , edgeBitcode ); 

		switch ( sector )
		{
		case hkpWeldingUtility::REJECT:
			{
				return WELD_RESULT_REJECT_CONTACT_POINT;
			}

		case hkpWeldingUtility::SNAP_0:
		case hkpWeldingUtility::SNAP_1:
			{
				if ( numFeaturePoints == 1  )
				{	
					// do not snap vertex collision
					return WELD_RESULT_REJECT_CONTACT_POINT;
				}
				hkpWeldingUtility::snapCollisionNormal(triangleNormal, edge, edgeBitcode, sector, sepNormalLocal ); 
				// Rotate the welded normal back into world space
				sepNormal._setRotatedDir(thisObjTransform->getRotation(), sepNormalLocal );
				goto RETURN_ACCEPT_CONTACT_POINT_MODIFIED;
			}

		default:
			break;
		}
	}
	return WELD_RESULT_ACCEPT_CONTACT_POINT_UNMODIFIED;
}


void hkpTriangleShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	hkSimdReal tol4;
	tol4.setFromFloat( tolerance + m_radius );

	hkVector4 v0; v0.setTransformedPos( localToWorld, getVertex<0>() );
	hkVector4 v1; v1.setTransformedPos( localToWorld, getVertex<1>() );
	hkVector4 v2; v2.setTransformedPos( localToWorld, getVertex<2>() );

	out.m_min.setMin( v0, v1 );
	out.m_max.setMax( v0, v1 );
	out.m_min.setMin( out.m_min, v2 );
	out.m_max.setMax( out.m_max, v2 );


	hkVector4 extrudedMin; extrudedMin.setAdd(out.m_min, m_extrusion );
	out.m_min.setMin( out.m_min, extrudedMin );

	hkVector4 extrudedMax; extrudedMax.setAdd(out.m_max, m_extrusion );
	out.m_max.setMax( out.m_max, extrudedMax );

	out.m_min.setSub( out.m_min,tol4 );
	out.m_max.setAdd( out.m_max,tol4 );
}

const hkSphere* hkpTriangleShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	hkSphere* s = sphereBuffer;

	const hkSimdReal myRadius = hkSimdReal::fromFloat(m_radius);

	{
		s->setPositionAndRadius( getVertex<0>(), myRadius );
		s++;
	}
	{
		s->setPositionAndRadius( getVertex<1>(), myRadius );
		s++;
	}
	{
		s->setPositionAndRadius( getVertex<2>(), myRadius );
		s++;
	}
	if( m_isExtruded )
	{
		{
			hkVector4& pos = s->getPositionAndRadius();
			pos.setAdd( getVertex<0>(), m_extrusion );
			pos.setW(myRadius);
			s++;
		}
		{
			hkVector4& pos = s->getPositionAndRadius();
			pos.setAdd( getVertex<1>(), m_extrusion );
			pos.setW(myRadius);
			s++;
		}
		{
			hkVector4& pos = s->getPositionAndRadius();
			pos.setAdd( getVertex<2>(), m_extrusion );
			pos.setW(myRadius);
			s++;
		}
	}

	return sphereBuffer;
}

hkBool hkpTriangleShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcTriangle", HK_NULL);
	
	hkcdRay ray; ray.setEndPoints( input.m_from, input.m_to );

	hkSimdReal fraction = hkSimdReal::fromFloat(results.m_hitFraction);
	const hkSimdReal tolerance = hkSimdReal::fromFloat(0.0001f);
	hkVector4 normal;
	hkBool32 ret = hkcdSegmentTriangleIntersect(ray,
												getVertex<0>(), getVertex<1>(), getVertex<2>(),
												tolerance, normal, fraction);

	if ( ret )
	{
		results.m_normal = normal;
		results.m_hitFraction = fraction.getReal();
		results.setKey(HK_INVALID_SHAPE_KEY);
	}
	HK_TIMER_END();
	return ret ? true : false;
}

// Might be possible to collapse the dot/cross products here. It's pretty fast as it is, though.
static inline void diffCrossDiffDotNormal(hkVector4Parameter p_i, hkVector4Parameter p_j, const hkVector4* hitPoints, hkVector4Parameter normal, hkVector4& dotsOut)
{
	//for (int i=0; i<4; i++)
	//{
	//	hkVector4 hitPoint; hitPoint.set(hitPoints[0](i), hitPoints[1](i), hitPoints[2](i));
	//	hkVector4 diff1, diff2;
	//	diff1.setSub4(p_i, hitPoint);
	//	diff2.setSub4(p_j, hitPoint);
	//	hkVector4 cp; cp.setCross(diff1, diff2);
	//	dots(i) = (hkReal) cp.dot3(normal);
	//}
	hkVector4 cp, A[3], B[3];

	{
		A[0].setBroadcast<0>(p_i);
		A[0].sub(hitPoints[0]);

		A[1].setBroadcast<1>(p_i);
		A[1].sub(hitPoints[1]);

		A[2].setBroadcast<2>(p_i);
		A[2].sub(hitPoints[2]);

		
		B[0].setBroadcast<0>(p_j);
		B[0].sub(hitPoints[0]);

		B[1].setBroadcast<1>(p_j);
		B[1].sub(hitPoints[1]);

		B[2].setBroadcast<2>(p_j);
		B[2].sub(hitPoints[2]);
	}
	
	// cp = A cross B
	hkVector4 temp;
	hkVector4 dots; dots.setZero();
	
	cp.setMul(A[1], B[2]); temp.setMul(A[2], B[1]); cp.sub(temp);
	dots.addMul(normal.getComponent<0>(), cp);

	cp.setMul(A[2], B[0]); temp.setMul(A[0], B[2]); cp.sub(temp);
	dots.addMul(normal.getComponent<1>(), cp);

	cp.setMul(A[0], B[1]); temp.setMul(A[1], B[0]); cp.sub(temp);
	dots.addMul(normal.getComponent<2>(), cp);

	
	dotsOut = dots;
}

hkVector4Comparison hkpTriangleShape::castRayBundle(const hkpShapeRayBundleCastInput& input, hkpShapeRayBundleCastOutput& results, hkVector4ComparisonParameter mask) const
{
	HK_TIME_CODE_BLOCK("rayBundleTriangle", HK_NULL);

	HK_ASSERT2(0x313fbf5a, mask.anyIsSet(), "Calling castRayBundle with no active rays!");

	hkcdRayBundle ray;
	ray.m_activeRays	= mask;
	ray.m_start			= input.m_from;
	ray.m_end			= input.m_to;

	const hkSimdReal eps = hkSimdReal::fromFloat(0.0001f);

	hkFourTransposedPoints transposedNormals;
	hkVector4 hitFractions;
	hitFractions.set(results.m_outputs[0].m_hitFraction, results.m_outputs[1].m_hitFraction, results.m_outputs[2].m_hitFraction, results.m_outputs[3].m_hitFraction);

	
	hkVector4Comparison activeMask = hkcdSegmentBundleTriangleIntersect(ray,
																		getVertex<0>(), getVertex<1>(), getVertex<2>(),
																		eps, transposedNormals, hitFractions);

	

	hkVector4 normals[4];
	transposedNormals.extract(normals[0], normals[1], normals[2], normals[3]);

	for (int i = 0; i < 4; i++)
	{
		if( activeMask.anyIsSet(hkVector4Comparison::getMaskForComponent(i) ) )
		{
			results.m_outputs[i].m_hitFraction = hitFractions(i);
			results.m_outputs[i].m_normal = normals[i];
			results.m_outputs[i].setKey(HK_INVALID_SHAPE_KEY);
		}
	}

	// Return the mask of rays that hit
	return activeMask;
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
