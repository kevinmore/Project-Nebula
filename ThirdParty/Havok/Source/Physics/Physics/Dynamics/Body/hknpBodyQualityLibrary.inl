/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE const hknpBodyQuality& hknpBodyQualityLibrary::getEntry( hknpBodyQualityId id ) const
{
	HK_ASSERT( 0x6522f5da, id.value() < hknpBodyQualityId::MAX_NUM_QUALITIES );
	return m_qualities[ id.value() ];
}

#ifndef HK_PLATFORM_SPU

HK_FORCE_INLINE void hknpBodyQualityLibrary::updateEntry( hknpBodyQualityId id, const hknpBodyQuality& quality )
{
	HK_ASSERT2( 0x1be81427, id.value() < hknpBodyQualityId::MAX_NUM_QUALITIES, "Invalid quality ID" );
	HK_ASSERT2( 0x1be81428, id.value() >= hknpMaterialId::NUM_PRESETS, "Cannot modify preset qualities" );

	if( !(m_qualities[id.value()] == quality) )
	{
		m_qualities[id.value()] = quality;
		m_qualityModifiedSignal._fire(id);
	}
}

#endif

HK_FORCE_INLINE const hknpBodyQuality* hknpBodyQualityLibrary::getBuffer() const
{
	return &m_qualities[0];
}

HK_FORCE_INLINE int hknpBodyQualityLibrary::getCapacity() const
{
	return hknpBodyQualityId::MAX_NUM_QUALITIES;
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
