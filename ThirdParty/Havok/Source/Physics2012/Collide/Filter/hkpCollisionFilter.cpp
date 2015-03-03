/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>

#if HK_POINTER_SIZE == 4
HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpCollisionFilter,m_type) == 32 );
#endif /* HK_POINTER_SIZE */

hkpCollisionFilter::hkpCollisionFilter() : m_type(HK_FILTER_UNKNOWN)
{
}

int hkpCollisionFilter::numShapeKeyHitsLimitBreached( const hkpCollisionInput& input, 
										 const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
										 const hkpBvTreeShape* bvTreeShapeB, hkAabb& aabb,
										 hkpShapeKey* shapeKeysInOut,
										 int shapeKeysCapacity) const 
{
	HK_WARN(0xad87baca, "Critical peformance warning: hkpBvTreeShape::queryAabb() returned more than 4K hkpShapeKey hits. All shapes above 4K are dropped. Consider implementing a custom handler for this case.");
	// Default: no action.
	return shapeKeysCapacity;
}

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
