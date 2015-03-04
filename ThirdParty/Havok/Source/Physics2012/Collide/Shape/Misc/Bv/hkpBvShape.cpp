/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>

hkpBvShape::hkpBvShape( const hkpShape* boundingVolumeShape, const hkpShape* childShape )
:	hkpShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpBvShape))
,	m_boundingVolumeShape(boundingVolumeShape)
,	m_childShape(childShape)
{
	HK_ASSERT2(0x1f5e7ff0, childShape != HK_NULL, "Child shape cannot be NULL");
	HK_ASSERT2(0x2d058dd7, boundingVolumeShape != HK_NULL, "Bounding volume cannot be NULL");

	m_boundingVolumeShape->addReference();
}

#if !defined(HK_PLATFORM_SPU)

//
//	Serialization constructor

hkpBvShape::hkpBvShape( hkFinishLoadedObjectFlag flag )
:	hkpShape(flag)
,	m_childShape(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpBvShape));
}

//
//	Destructor

hkpBvShape::~hkpBvShape()
{
	m_boundingVolumeShape->removeReference();
}

#endif

void hkpBvShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	getBoundingVolumeShape()->getAabb( localToWorld, tolerance, out );
}

hkBool hkpBvShape::castRay( const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results ) const
{
	// Comment in this code if you wish to get a callback from a phantom callback shape only
	// if the ray hits the bv shape. This is commented out, because if the ray starts inside
	// the bv shape, it will not hit the bv shape, so no callback at all will be fired.

	//hkpShapeRayCastOutput testOutput;
	//if ( getBoundingVolumeShape()->castRay( input, testOutput) )
	HK_TIMER_BEGIN("rcBvShape", HK_NULL);
	results.changeLevel(1);
	hkBool result = m_childShape->castRay( input, results );
	results.changeLevel(-1);
	if( result )
	{
		results.setKey(0);
	}
	HK_TIMER_END();
	return result;
}


void hkpBvShape::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	// Comment in this code if you wish to get a callback from a phantom callback shape only
	// if the ray hits the bv shape. This is commented out, because if the ray starts inside
	// the bv shape, it will not hit the bv shape, so no callback at all will be fired.

	//hkpShapeRayCastOutput testOutput;
	//if ( getBoundingVolumeShape()->castRay( input, testOutput) )
	HK_TIMER_BEGIN("rcBvShape", HK_NULL);
	{
		hkpCdBody body( &cdBody );
		const hkpShape* child = getChildShape();
		body.setShape( child, 0 );
		child->castRayWithCollector( input, body, collector );
	}
	HK_TIMER_END();
}

#if !defined(HK_PLATFORM_SPU)

const hkpShapeContainer* hkpBvShape::getContainer() const
{
	return &m_childShape;
}

#endif

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
