/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Util/ShapeDepth/hkpShapeDepthUtil.h>
#include <Physics2012/Collide/Shape/hkpShape.h>
#include <Physics2012/Collide/Shape/hkpShapeContainer.h>

hkUint8 hkpShapeDepthUtil::s_defaultMinimumChildDepth = 0;

hkUint8 hkpShapeDepthUtil::getShapeDepth( const hkpShape* shape )
{
	const hkpShapeContainer *const container = shape->getContainer();
	if ( container )
	{
		hkpShapeBuffer buffer;
		
		hkUint8 maxChildDepth = 0;
		for ( hkpShapeKey key = container->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = container->getNextKey( key ) )
		{
			const hkpShape *const childShape = container->getChildShape( key, buffer );
			maxChildDepth = hkMath::max2( getShapeDepth( childShape ), maxChildDepth );
		}
		HK_ASSERT2( 0x95bd1a2e, maxChildDepth != 255, "Shape is too deep." );
		return hkMath::max2(hkUint8(maxChildDepth + 1), s_defaultMinimumChildDepth);
	}
	else
	{
		return s_defaultMinimumChildDepth;
	}
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
