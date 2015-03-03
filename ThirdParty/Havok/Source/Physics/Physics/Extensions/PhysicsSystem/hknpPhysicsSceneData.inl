/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE const hknpWorldCinfo* hknpPhysicsSceneData::getWorldCinfo() const
{
	return m_worldCinfo == HK_NULL ? HK_NULL : &m_worldCinfo->m_info;
}

#ifndef HK_PLATFORM_SPU

HK_FORCE_INLINE void hknpPhysicsSceneData::setWorldCinfo( const hknpWorldCinfo* info )
{
	if (m_worldCinfo != HK_NULL)
	{
		m_worldCinfo->removeReference();
	}
	m_worldCinfo = new hknpRefWorldCinfo;
	m_worldCinfo->m_info = *info;
}

#endif

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
