/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>

#include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>


HK_COMPILE_TIME_ASSERT( sizeof( hknpSphereShape )== sizeof( hknpConvexShape ) );

#if !defined(HK_PLATFORM_SPU)

hknpSphereShape::hknpSphereShape( class hkFinishLoadedObjectFlag flag )
	:	hknpConvexShape(flag)
{
	if( flag.m_finishing )
	{
		m_flags.orWith(USE_SINGLE_POINT_MANIFOLD);
	}
}

hknpSphereShape* HK_CALL hknpSphereShape::createSphereShape( hkVector4Parameter center, hkReal radius )
{
	const int sizeOfBaseClass = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT,sizeof(hknpSphereShape));
	int shapeSize;
	void * buffer = allocateConvexShape(4, sizeOfBaseClass, shapeSize);
	hknpSphereShape* sphere = new (buffer) hknpSphereShape(center, radius);
	sphere->m_memSizeAndFlags = (hkUint16) shapeSize;
	return sphere;
}

hkReal hknpSphereShape::calcMinAngleBetweenFaces() const
{
	return HK_REAL_PI;
}

void hknpSphereShape::buildMassProperties( const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const
{
	// We ignore massConfig.m_quality here since we can easily compute the "correct" properties
	hkResult result = hknpConvexShapeUtil::buildSphereMassProperties( massConfig, getVertex(0), m_convexRadius, massPropertiesOut );
	if( result == HK_FAILURE )
	{
		// Fall back to AABB approximation.
		hknpShape::buildMassProperties( massConfig, massPropertiesOut );
	}
}

hkResult hknpSphereShape::buildSurfaceGeometry( const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const
{
	const int			steps = 16;
	const hkReal		invSteps = 1.0f / (hkReal)(steps-1);
	const hkVector4		center = getVertex(0);
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
			n.mul(radius);
			n.add(center);
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

#endif // if !defined(HK_PLATFORM_SPU)

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
