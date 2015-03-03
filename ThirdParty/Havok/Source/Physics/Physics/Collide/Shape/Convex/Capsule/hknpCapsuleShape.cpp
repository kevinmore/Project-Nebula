/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Math/Quaternion/hkQuaternionUtil.h>
#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h>
#include <Common/Base/Algorithm/Collide/LineSegment/hkLineSegmentUtil.h>

#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Geometry/Internal/Algorithms/Distance/hkcdDistancePointLine.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>


#if !defined(HK_PLATFORM_SPU)

hknpCapsuleShape* HK_CALL hknpCapsuleShape::createCapsuleShape( hkVector4Parameter a, hkVector4Parameter b, hkReal radius )
{
	int shapeSize;
	void * buffer = allocateConvexPolytopeShape(8, 6, 24, HKNP_CAPSULE_BASE_SIZE, shapeSize);
	hknpCapsuleShape* capsule = new (buffer) hknpCapsuleShape(a, b, radius);
	capsule->m_memSizeAndFlags = (hkUint16) shapeSize;
	return capsule;
}

hknpCapsuleShape::hknpCapsuleShape( class hkFinishLoadedObjectFlag flag )
	:	hknpConvexPolytopeShape(flag)
{
}

void hknpCapsuleShape::buildMassProperties( const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const
{
	// We ignore massConfig.m_quality here since we can easily compute the "correct" properties
	hkResult result = hknpConvexShapeUtil::buildCapsuleMassProperties( massConfig, m_a, m_b, m_convexRadius, massPropertiesOut );
	if( result == HK_FAILURE )
	{
		// Fall back to AABB approximation.
		hknpShape::buildMassProperties( massConfig, massPropertiesOut );
	}
}

#endif

int hknpCapsuleShape::calcSize() const
{
	return calcConvexPolytopeShapeSize(8, 6, 24, HKNP_CAPSULE_BASE_SIZE);
}

int hknpCapsuleShape::getNumberOfSupportVertices() const
{
	return 8;
}

const hkcdVertex* hknpCapsuleShape::getSupportVertices( hkcdVertex* HK_RESTRICT vertexBuffer, int bufferSize ) const
{
	HK_ASSERT2( 0xaf14e121, bufferSize >= 8, "vertexBuffer is too small." );

	vertexBuffer[0].assign( m_a );
	vertexBuffer[1].setInterpolate( m_a, m_b, hkSimdReal::fromFloat(1.0f/7.0f));
	vertexBuffer[2].setInterpolate( m_a, m_b, hkSimdReal::fromFloat(2.0f/7.0f));
	vertexBuffer[3].setInterpolate( m_a, m_b, hkSimdReal::fromFloat(3.0f/7.0f));
	vertexBuffer[4].setInterpolate( m_a, m_b, hkSimdReal::fromFloat(4.0f/7.0f));
	vertexBuffer[5].setInterpolate( m_a, m_b, hkSimdReal::fromFloat(5.0f/7.0f));
	vertexBuffer[6].setInterpolate( m_a, m_b, hkSimdReal::fromFloat(6.0f/7.0f));
	vertexBuffer[7].assign( m_b );

	vertexBuffer[0].setInt24W( 0 );
	vertexBuffer[1].setInt24W( 1 );
	vertexBuffer[2].setInt24W( 2 );
	vertexBuffer[3].setInt24W( 3 );
	vertexBuffer[4].setInt24W( 4 );
	vertexBuffer[5].setInt24W( 5 );
	vertexBuffer[6].setInt24W( 6 );
	vertexBuffer[7].setInt24W( 7 );

	return vertexBuffer;
}

void hknpCapsuleShape::getSignedDistances( const hknpShape::SdfQuery& query, SdfContactPoint* contactsOut ) const
{
	//hkLineSegmentUtil::capsuleCapsuleManifold( *m_a, )
	for (int i =0; i < query.m_numSpheres; i++)
	{
		hkVector4 start = query.m_sphereCenters[i];
		hknpShape::SdfContactPoint* HK_RESTRICT hit = contactsOut+i;
		hkVector4 position;
		hkSimdReal distanceSquared = hkcdPointSegmentDistanceSquared( start, m_a, m_b, &position );
		hkVector4 dir; dir.setSub( start, position );
		hkSimdReal distance = dir.normalizeWithLength<3, HK_ACC_23_BIT, HK_SQRT_SET_ZERO>();
		hkSimdReal queryRadius; queryRadius.setFromFloat(query.m_spheresRadius);
		hkSimdReal capsRadius; capsRadius.setFromFloat( m_convexRadius );
		distance.sub( queryRadius + capsRadius );
		distance.store<1>(&hit->m_distance);
		hit->m_position.setAddMul( position, dir, capsRadius );
		hit->m_normal = dir;
		hit->m_shapeTag = HKNP_INVALID_SHAPE_TAG;
		hit->m_shapeKey = HKNP_INVALID_SHAPE_KEY;
	}
}

int hknpCapsuleShape::getSignedDistanceContacts(
	const hknpSimulationThreadContext& tl, const hknpShape* queryShape, const hkTransform& sdfFromQueryTransform,
	hkReal maxDistance, int vertexIdOffset, SdfContactPointWriter& contactPointsOut ) const
{
	if ( queryShape->getType() != hknpShapeType::CAPSULE )
	{
		hknpShape::getSignedDistanceContacts( tl, queryShape, sdfFromQueryTransform, maxDistance, vertexIdOffset, contactPointsOut );
	}
	else
	{
		const hknpCapsuleShape* qCapsule = (const hknpCapsuleShape*)queryShape;
		hkVector4 queryPoints[2];
		queryPoints[0]._setTransformedPos( sdfFromQueryTransform, qCapsule->m_a );
		queryPoints[1]._setTransformedPos( sdfFromQueryTransform, qCapsule->m_b );

		hkContactPoint cps[3];
		cps[0].setDistance( maxDistance );
		cps[1].setDistance( maxDistance );
		cps[2].setDistance( maxDistance );

		hkLineSegmentUtil::capsuleCapsuleManifold( queryPoints, qCapsule->m_convexRadius, &m_a, m_convexRadius, cps );
		for (int i = 0; i < 3; i++)
		{
			const hkContactPoint& cp = cps[i];
			if ( cp.getDistance() < maxDistance )
			{
				SdfContactPoint* HK_RESTRICT sdfCp = contactPointsOut.reserve( sizeof(SdfContactPoint));
				sdfCp->m_position = cp.getPosition();
				sdfCp->m_normal   = cp.getNormal();
				sdfCp->m_distance = cp.getDistance();
				sdfCp->m_vertexId = VertexIndex(i + vertexIdOffset);
				sdfCp->m_shapeKey = HKNP_INVALID_SHAPE_KEY;
				sdfCp->m_shapeTag = HKNP_INVALID_SHAPE_TAG;

				contactPointsOut.advance(sizeof(SdfContactPoint));
			}
		}
	}
	return 3;
}

#if !defined(HK_PLATFORM_SPU)

hkResult hknpCapsuleShape::buildSurfaceGeometry( const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const
{
	const int			steps = 16;
	const hkReal		invSteps = 1.0f / (hkReal)(steps-1);
	const hkVector4		centers[2] = {m_a,m_b};
	hkVector4			axis; axis.setSub(m_a,m_b); axis.normalize<3>();
	hkQuaternion		rot; hkQuaternionUtil::_computeShortestRotation(hkVector4::getConstant<HK_QUADREAL_0010>(), axis, rot);
	hkSimdReal			radius; radius.setFromFloat(m_convexRadius);
	hkArray<hkVector4>	vertices; vertices.reserve(steps * steps);
	for(int i=0; i<steps; ++i)
	{
		hkVector4	uv; uv.setZero();
		uv(0) = i * invSteps;
		for(int j=0; j<steps; ++j)
		{
			uv(1) = j * invSteps;
			hkVector4	n; hkGeometryProcessing::octahedronToNormal(uv, n);
			n._setRotatedDir(rot, n);
			n.mul(radius);
			n.add(centers[n.dot<3>(axis).getReal()<0.0f?1:0]);
			vertices.pushBackUnchecked(n);
		}
	}

	hkgpConvexHull	hull;
	if(hull.build(vertices) == 3)
	{
		hull.generateGeometry(hkgpConvexHull::SOURCE_VERTICES, *geometryOut);
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

#endif

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
