/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Misc/MultiRay/hkpMultiRayShape.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

#if !defined(HK_PLATFORM_SPU)

//
//	Serialization constructor

hkpMultiRayShape::hkpMultiRayShape( hkFinishLoadedObjectFlag flag )
:	hkpShape(flag)
,	m_rays(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpMultiRayShape));
}

#endif

hkpMultiRayShape::hkpMultiRayShape(const Ray* Rays, int nRays, hkReal rayPenetrationDistance)
:	hkpShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpMultiRayShape))
{
    m_rayPenetrationDistance = rayPenetrationDistance;
	Ray* ray = m_rays.expandBy( nRays );
	for (int i = nRays-1; i>=0; i-- )
	{
		*ray = *Rays;
		hkVector4 diff; diff.setSub(ray->m_end , ray->m_start);
		ray->m_start.setComponent<3>(diff.length<3>());

		// Extend it by the tolerance
		diff.normalize<3>();
		diff.mul(hkSimdReal::fromFloat(m_rayPenetrationDistance));
		ray->m_end.add(diff);

		ray++;
		Rays++;
	}
}

void hkpMultiRayShape::castRayWithCollector( const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
}

hkBool hkpMultiRayShape::castRay( const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results ) const
{
	return false;
}


void hkpMultiRayShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	hkInplaceArrayAligned16<Ray,16> worldRays;
	worldRays.setSize( m_rays.getSize() );

	hkVector4Util::transformPoints( localToWorld, &m_rays[0].m_start, m_rays.getSize()*2, &worldRays[0].m_start );

	hkVector4 absMin; absMin.setXYZ_0(hkVector4::getConstant<HK_QUADREAL_MAX>());
	hkVector4 absMax; absMax.setXYZ_0(hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>());

	const Ray* ray =  &worldRays[0];
	for(int i = 0; i < worldRays.getSize(); ++i)
	{
		absMin.setMin( absMin, ray->m_end );
		absMin.setMin( absMin, ray->m_start );
		absMax.setMax( absMax, ray->m_end );
		absMax.setMax( absMax, ray->m_start );
		ray++;
	}
	out.m_min = absMin;
	out.m_max = absMax;

//	DISPLAY_POINT(absMin, 0xffffffff);
//	DISPLAY_POINT(absMax, 0xffffffff);
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
