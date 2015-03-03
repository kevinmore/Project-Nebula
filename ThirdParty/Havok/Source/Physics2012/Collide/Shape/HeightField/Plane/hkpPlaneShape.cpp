/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/HeightField/Plane/hkpPlaneShape.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>

#if !defined(HK_PLATFORM_SPU)

hkpPlaneShape::hkpPlaneShape( hkFinishLoadedObjectFlag flag )
:	hkpHeightFieldShape(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpPlaneShape));
}

#endif

hkpPlaneShape::hkpPlaneShape(const hkVector4& plane, const hkAabb& aabb)
:	hkpHeightFieldShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpPlaneShape))
{
	m_plane = plane;
	aabb.getCenter( m_aabbCenter );
	aabb.getHalfExtents( m_aabbHalfExtents );
}

hkpPlaneShape::hkpPlaneShape( const hkVector4& direction, const hkVector4& center, const hkVector4& halfExtents )
:	hkpHeightFieldShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpPlaneShape))
{
	m_plane.setXYZ_W(direction, - direction.dot<3>( center ));
	m_aabbCenter = center;
	m_aabbHalfExtents = halfExtents;
}

void hkpPlaneShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	hkAabbUtil::calcAabb( localToWorld, m_aabbHalfExtents, m_aabbCenter, hkSimdReal::fromFloat(tolerance), out );
}


hkBool hkpPlaneShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIME_CODE_BLOCK("rcPlane", HK_NULL);

	const hkSimdReal f = m_plane.dot4xyz1( input.m_from );
	const hkSimdReal t = m_plane.dot4xyz1( input.m_to );
	if ( f.isGreaterEqualZero() && t.isLessZero() )
	{
		const hkSimdReal hitFraction = f / ( f - t );
		if( hitFraction < hkSimdReal::fromFloat(results.m_hitFraction) )
		{
			// Check if it is inside the AABB
			hkVector4 hitPoint; hitPoint.setInterpolate( input.m_from, input.m_to, hitFraction );
			hitPoint.sub( m_aabbCenter );
			hitPoint.setAbs( hitPoint );

			if ( hitPoint.lessEqual( m_aabbHalfExtents ).allAreSet<hkVector4ComparisonMask::MASK_XYZ>() )
			{
				hitFraction.store<1>((hkReal*)(&results.m_hitFraction));
				results.m_normal = m_plane;
				results.setKey( HK_INVALID_SHAPE_KEY );

				return true;
			}
		}
	}

	return false;
}

void hkpPlaneShape::castRayWithCollector( const hkpShapeRayCastInput& inputLocal, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	HK_ASSERT2(0x7f1d0d08,  cdBody.getShape() == this, "inconsistent cdBody, shapePointer is wrong" );
	hkpShapeRayCastOutput results;
	results.m_hitFraction = collector.m_earlyOutHitFraction;

	if ( castRay( inputLocal, results ) )
	{
		results.m_normal._setRotatedDir( cdBody.getTransform().getRotation(), results.m_normal );
		collector.addRayHit( cdBody, results );
	}
}

void hkpPlaneShape::castSphere( const hkpSphereCastInput& inputLocal, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	const hkSimdReal radius = hkSimdReal::fromFloat(inputLocal.m_radius);
	const hkSimdReal f = m_plane.dot4xyz1( inputLocal.m_from ) - radius;
	const hkSimdReal t = m_plane.dot4xyz1( inputLocal.m_to ) - radius;

	if ( t.isGreaterEqualZero() )
	{
		return;
	}

	if ( (f - t) < hkSimdReal::fromFloat(inputLocal.m_maxExtraPenetration) )
	{
		return;
	}

	hkSimdReal hitFraction = f / (f - t);
	hitFraction.zeroIfTrue( f.lessEqualZero() );
	if( hitFraction < hkSimdReal::fromFloat(collector.m_earlyOutHitFraction) )
	{
		// Check if it is inside the AABB
		hkVector4 hitPoint; hitPoint.setInterpolate( inputLocal.m_from, inputLocal.m_to, hitFraction );
		hitPoint.sub( m_aabbCenter );
		hitPoint.setAbs( hitPoint );

		if ( hitPoint.lessEqual( m_aabbHalfExtents ).allAreSet<hkVector4ComparisonMask::MASK_XYZ>() )
		{
			hkpShapeRayCastOutput output;
			hitFraction.store<1>(&output.m_hitFraction);
			output.m_normal = m_plane;
			output.setKey( HK_INVALID_SHAPE_KEY );
			collector.addRayHit( cdBody, output );
		}
	}
}


void hkpPlaneShape::collideSpheres( const CollideSpheresInput& input, SphereCollisionOutput* outputArray) const
{
    hkVector4* o = outputArray;
    hkSphere* s = input.m_spheres;

    for (int i = input.m_numSpheres-1; i>=0 ; i-- )
    {
		const hkSimdReal d = m_plane.dot4xyz1( s->getPosition() ) - s->getRadiusSimdReal();
        o[0].setXYZ_W(m_plane, d);

        o++;
        s++;
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
