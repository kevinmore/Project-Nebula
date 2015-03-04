/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

inline unsigned hknpDefaultViewerColorScheme::UIdMapOperations::hash( hknpDefaultViewerColorScheme::Uid key, unsigned modulus )
{
	hkUint64 value = key.m_bodyId.value() + ( hkUlong(key.m_world) << 16 );
	return ( unsigned(value) * 2654435761U ) & modulus;
}

inline void hknpDefaultViewerColorScheme::UIdMapOperations::invalidate( hknpDefaultViewerColorScheme::Uid& key )
{
	key.m_world = HK_NULL;
}

inline hkBool32 hknpDefaultViewerColorScheme::UIdMapOperations::isValid( hknpDefaultViewerColorScheme::Uid key )
{
	return ( key.m_world != HK_NULL );
}

inline hkBool32 hknpDefaultViewerColorScheme::UIdMapOperations::equal( hknpDefaultViewerColorScheme::Uid key0, hknpDefaultViewerColorScheme::Uid key1 )
{
	return ( key0.m_world == key1.m_world ) && ( key0.m_bodyId == key1.m_bodyId );
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
