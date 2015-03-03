/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Deprecated/MultiSphere/hkpMultiSphereShape.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>

hkpMultiSphereShape::hkpMultiSphereShape(const hkVector4* spheres, int numSpheres)
: hkpSphereRepShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpMultiSphereShape))
{

	HK_WARN(0x758787be,"Use of hkpMultiSphereShape is deprecated. Please use another shape.");


#if defined(HK_COMPILER_MSVC) && (HK_COMPILER_MSVC_VERSION >= 1300) // msvc6 bug
    HK_COMPILE_TIME_ASSERT(sizeof(m_spheres)/sizeof(hkVector4) == MAX_SPHERES);
#endif

	HK_ASSERT2(0x6d7430b4,  numSpheres <= hkpMultiSphereShape::MAX_SPHERES, "the hkpMultiSphereShape does not support so many spheres");
	for (int i = 0; i < numSpheres; i++ )
	{
		m_spheres[i] = spheres[i];
	}
	m_numSpheres = numSpheres;
}

#if !defined(HK_PLATFORM_SPU)

hkpMultiSphereShape::hkpMultiSphereShape( hkFinishLoadedObjectFlag flag )
:	hkpSphereRepShape(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpMultiSphereShape));
}

#endif

void hkpMultiSphereShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	HK_WARN_ONCE(0x758787be,"Use of hkpMultiSphereShape is deprecated. Please use another shape.");


	hkVector4 worldSpheres[MAX_SPHERES];

	hkVector4Util::transformPoints( localToWorld, &m_spheres[0], getNumSpheres(), &worldSpheres[0] );
	hkVector4 absMin; absMin.setXYZ_0(hkVector4::getConstant<HK_QUADREAL_MAX>());
	hkVector4 absMax; absMax.setXYZ_0(hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>());

	for(int i = 0; i < m_numSpheres; ++i)
	{
		hkVector4 r;
		r.setBroadcastXYZ(3, m_spheres[i]);
		hkVector4 min; min.setSub(worldSpheres[i], r);
		hkVector4 max; max.setAdd(worldSpheres[i], r);

		absMin.setMin( absMin, min );
		absMax.setMax( absMax, max );
	}

	hkVector4 tol4; tol4.setZero(); tol4.setXYZ( tolerance );

	out.m_min.setSub(absMin, tol4);
	out.m_max.setAdd(absMax, tol4);
}

int hkpMultiSphereShape::getNumCollisionSpheres() const
{
	return getNumSpheres();
}

const hkSphere* hkpMultiSphereShape::getCollisionSpheres( hkSphere* sphereBuffer ) const
{
	for (int i = 0; i < getNumSpheres(); i++)
	{
		sphereBuffer[i] = reinterpret_cast<const hkSphere&>(m_spheres[i]);
	}
	return sphereBuffer;
}

static int castRayInternal( const hkpShapeRayCastInput& input, const hkVector4* m_spheres, int m_numSpheres, hkReal* distOut, int* indexOut )
{
	//
	//	This functions is a modified version of
	//  http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter1.htm
	//  Modifications include changing the order of if statements to prevent
	//  any division which can produce a number greater than 1
	//

	HK_TIMER_BEGIN("rcMultiSpher", HK_NULL);

	// No hit found yet
	int numHits = 0;
	for ( int i = 0; i<m_numSpheres; i++ )
	{
		hkSimdReal	radius2 = m_spheres[i].getW(); radius2.mul(radius2);

		//
		// solve quadratic function: ax*x + bx + c = 0
		//

		hkVector4 localTo( input.m_to ); localTo.sub( m_spheres[i] );
		hkVector4 localFrom( input.m_from ); localFrom.sub( m_spheres[i] );
		hkVector4 dir; dir.setSub( localTo, localFrom );

		const hkSimdReal C = localFrom.lengthSquared<3>() - radius2;

		hkSimdReal B = dir.dot<3>( localFrom ); B.add(B);
		if ( B.isGreaterEqualZero() )
		{
			// ray points away from sphere center
			continue;
		}

		const hkSimdReal A = dir.lengthSquared<3>();

		const hkSimdReal det = B*B - hkSimdReal_4*A*C;
		if ( det.isLessEqualZero() )
		{
			//
			//	Infinite ray does not hit
			//
			continue;
		}

		const hkSimdReal sqDet = det.sqrt<HK_ACC_23_BIT,HK_SQRT_IGNORE>();

		hkSimdReal t = (-B - sqDet) * hkSimdReal_Inv2;

		if ( t >= A )
		{
			//
			//	hits behind endpoint
			//
			continue;
		}

		if ( t.isLessZero() )
		{
			//
			// start point inside
			//
			continue;
		}

		//  Note: we know that t > 0
		//  Also that A > t
		//  So this division is safe and results in a point between 0 and 1

		t = t / A;

		// Check if this hit is closer than any previous

		distOut[numHits] = t.getReal();
		indexOut[numHits] = i;
		++numHits;
	}
	HK_TIMER_END();
	return numHits;
}

static int getBestHit( const hkVector4* m_spheres, int m_numSpheres,
					  hkReal* dist, int* sphereIndex, int nhit,
					  const hkpShapeRayCastInput& input,
					  hkpShapeRayCastOutput& results)
{
	hkReal bestDist = results.m_hitFraction;
	int bestIndexIndex = -1;
	for( int i = 0; i < nhit; ++i )
	{
		if( dist[i] < bestDist )
		{
			bestDist  = dist[i];
			bestIndexIndex = i;
		}
	}
	if( bestIndexIndex != -1 )
	{
		results.setKey(HK_INVALID_SHAPE_KEY);

		int i = sphereIndex[bestIndexIndex];
		results.m_hitFraction = bestDist;
		hkVector4 localTo; localTo.setSub( input.m_to, m_spheres[i] );
		hkVector4 localFrom; localFrom.setSub( input.m_from, m_spheres[i] );
		results.m_normal.setInterpolate( localFrom, localTo, hkSimdReal::fromFloat(bestDist) );
		hkSimdReal invW; invW.setReciprocal(m_spheres[i].getW());
		results.m_normal.mul( invW );
		results.m_extraInfo = i;

		return bestIndexIndex;
	}
	return -1;
}

hkBool hkpMultiSphereShape::castRay( const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results ) const
{
	hkReal dist[MAX_SPHERES];
	int idx[MAX_SPHERES];
	int nhit = castRayInternal(input, m_spheres, m_numSpheres, dist, idx);
	return getBestHit(m_spheres, m_numSpheres,  dist, idx, nhit,  input, results) != -1;
}

void hkpMultiSphereShape::castRayWithCollector( const hkpShapeRayCastInput& inputLocal, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	HK_ASSERT2(0x4033ce56,  cdBody.getShape() == this, "inconsistent cdBody, shapePointer is wrong" );
	hkReal dist[MAX_SPHERES];
	int idx[MAX_SPHERES];
	int nhit = castRayInternal(inputLocal, m_spheres, m_numSpheres, dist, idx);

	hkpShapeRayCastOutput results;
	while(nhit)
	{
		results.reset();
		int indexIndex = getBestHit(m_spheres, m_numSpheres,  dist, idx, nhit,  inputLocal, results);

		results.m_normal._setRotatedDir( cdBody.getTransform().getRotation(), results.m_normal );
		collector.addRayHit( cdBody, results );

		--nhit;
		dist[indexIndex] = dist[nhit];
		idx[indexIndex] = idx[nhit];
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
