/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE hkBool32 hknpGroupCollisionFilter::_isCollisionEnabled( hkUint32 infoA, hkUint32 infoB ) const
{
	// If the objects are in the same system group, but not system group 0,
	// then the decision of whether to collide is based exclusively on the
	// objects' SubSystemId and SubSystemDontCollideWith.
	// Otherwise, the decision is based exclusively on the objects' layers.

	const hkUint32 zeroIfSameSystemGroup = (infoA^infoB) & 0xffff0000;
	const hkUint32 f = 0x1f;

	// check for identical system groups
	if ( zeroIfSameSystemGroup == 0)
	{
		// check whether system group was set (nonzero)
		if ( (infoA & 0xffff0000) != 0 )
		{
			// check whether we allow collisions
			int idA = (infoA >> 5) & f;
			int dontCollideB = (infoB >> 10) & f;
			if ( idA == dontCollideB )
			{
				return false;
			}

			int idB = (infoB >> 5) & f;
			int dontCollideA = (infoA >> 10) & f;
			return ( idB - dontCollideA );
		}
	}

	// use the layers to decide
	hkUint32 layerBitsA = m_collisionLookupTable[ infoA & f ];
	hkUint32 layerBitsB = hkUint32(1 << (infoB & f));

	return layerBitsA & layerBitsB;
}


HK_FORCE_INLINE int hknpGroupCollisionFilter::getNewSystemGroup()
{
	return ++m_nextFreeSystemGroup;
}


/*static*/ HK_FORCE_INLINE hkUint32 HK_CALL hknpGroupCollisionFilter::calcFilterInfo( int layer, int systemGroup, int subSystemId, int subSystemDontCollideWith )
{
	HK_ASSERT(0x1902b596,  layer >=0 && layer < 32 );
	HK_ASSERT(0x1902b597,  subSystemId  >=0 && subSystemId  < 32 );
	HK_ASSERT(0x1902b598,  subSystemDontCollideWith  >=0 && subSystemDontCollideWith  < 32 );
	HK_ASSERT(0x5ae6770c,  systemGroup>=0 && systemGroup < 0x10000);

	return hkUint32( (subSystemId<<5) | ( subSystemDontCollideWith<<10) | (systemGroup<<16) | layer);
}


/*static*/ HK_FORCE_INLINE int HK_CALL hknpGroupCollisionFilter::getLayerFromFilterInfo( hkUint32 filterInfo )
{
	return filterInfo & 0x1f;
}


/*static*/ HK_FORCE_INLINE int HK_CALL hknpGroupCollisionFilter::setLayer( hkUint32 filterInfo, int newLayer )
{
	hkUint32 collisionLayerMask = 0xffffffff - 0x1f;
	return newLayer + ( collisionLayerMask & filterInfo);
}


/*static*/ HK_FORCE_INLINE int HK_CALL hknpGroupCollisionFilter::getSystemGroupFromFilterInfo( hkUint32 filterInfo )
{
	return filterInfo>>16;
}


/*static*/ HK_FORCE_INLINE int HK_CALL hknpGroupCollisionFilter::getSubSystemIdFromFilterInfo( hkUint32 filterInfo )
{
	return (filterInfo >> 5) & 0x1f;
}


/*static*/ HK_FORCE_INLINE int HK_CALL hknpGroupCollisionFilter::getSubSystemDontCollideWithFromFilterInfo( hkUint32 filterInfo )
{
	return (filterInfo >> 10) & 0x1f;
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
