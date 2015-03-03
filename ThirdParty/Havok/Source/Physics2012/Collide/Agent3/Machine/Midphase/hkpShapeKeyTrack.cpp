/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpShapeKeyTrack.h>

hkpShapeKey* hkpShapeKeyTrackConsumer::getShapeKeysInBuffer( int numShapeKeys )
{
	HK_TIMER_BEGIN("ReadShapeKeys",HK_NULL);
	setNumElements( numShapeKeys );

	hkpShapeKey* shapeKeys = hkAllocateStack<hkpShapeKey>( numShapeKeys + 2, "Shapekeys" );
	{
		int count = 0;
		int batchSize;
		hkpShapeKey* key;
		while ( ( batchSize = accessBatch( key ) ) > 0 )
		{
			HK_ASSERT2( 0x23a7e88b, count + batchSize <= numShapeKeys, "Shape key track consumer provided too many shape keys." );
			hkString::memCpy4( shapeKeys + count, key, batchSize );
			count += batchSize;
		}
		shapeKeys[numShapeKeys] = HK_INVALID_SHAPE_KEY;
		shapeKeys[numShapeKeys + 1] = HK_INVALID_SHAPE_KEY;
	}
	HK_TIMER_END();
	return shapeKeys;
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
