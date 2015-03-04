/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE hknpBody::Flags hknpCollisionDispatcher::getKeyframeCacheCreationMask() const
{
	return m_keyframeCacheCreationMask;
}

HK_FORCE_INLINE void hknpCollisionDispatcher::setKeyframeCacheCreationMask( hknpBody::Flags flags )
{
	m_keyframeCacheCreationMask = flags;
}

HK_FORCE_INLINE void hknpCollisionDispatcher::registerCacheCreator(
	hknpCollisionDispatchType::Enum typeA, hknpCollisionDispatchType::Enum typeB,
	hknpCollisionDispatcher::CreateFunc createFn )
{
	HK_ASSERT2( 0x30ff763b, createFn, "Cannot use a null cache creation function" );
	m_dispatchTable[ typeA ][ typeB ] = createFn;
	m_dispatchTable[ typeB ][ typeA ] = createFn;
}

HK_FORCE_INLINE void hknpCollisionDispatcher::createCollisionCache(
	const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
	hknpCollisionDispatchType::Enum typeA, hknpCollisionDispatchType::Enum typeB,
	hknpCdCacheWriter* cacheWriter ) const
{
	// Assume a valid function pointer, since we initialized the table with dummy ones
	CreateFunc createFn = m_dispatchTable[ typeA ][ typeB ];
	return (*createFn)( world, bodyA, bodyB, cacheWriter );
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
