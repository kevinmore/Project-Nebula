/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>

int hkpRayHitCollector::shapeKeysFromCdBody( hkpShapeKey* buf, int maxKeys, const hkpCdBody& body )
{
	hkpCdBody const* bodies[hkpShapeRayCastOutputPpu::MAX_HIERARCHY_DEPTH];
	int i = 0;
	for( const hkpCdBody* b = &body; b->getParent() != HK_NULL; ++i )
	{
		HK_ASSERT2(0xad765433, i < hkpShapeRayCastOutputPpu::MAX_HIERARCHY_DEPTH, "Local buffer overflown -- hkpCdBody heirarchy deeper than hkpShapeRayCastOutput::MAX_HIERARCHY_DEPTH.");
		bodies[i] = b;
		b = b->getParent();
	}
	int j = 0;
	for( ; i > 0 && j < maxKeys-1; ++j )
	{
		buf[j] = bodies[--i]->getShapeKey();
	}
	buf[j] = HK_INVALID_SHAPE_KEY;
	return j + 1;
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
