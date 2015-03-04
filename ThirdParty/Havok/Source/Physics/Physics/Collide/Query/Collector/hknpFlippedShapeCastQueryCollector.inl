/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hknpFlippedShapeCastQueryCollector::hknpFlippedShapeCastQueryCollector( hkVector4Parameter castDirectionWS, hknpCollisionQueryCollector* childCollector )
{
	m_castDirectionWS		= castDirectionWS;
	m_childCollector		= childCollector;
	m_earlyOutHitFraction	= m_childCollector->getEarlyOutHitFraction();
}

HK_FORCE_INLINE void hknpFlippedShapeCastQueryCollector::reset()
{
	return m_childCollector->reset();
}

HK_FORCE_INLINE bool hknpFlippedShapeCastQueryCollector::hasHit() const
{
	return m_childCollector->hasHit();
}

HK_FORCE_INLINE int hknpFlippedShapeCastQueryCollector::getNumHits() const
{
	return m_childCollector->getNumHits();
}

HK_FORCE_INLINE const hknpCollisionResult*hknpFlippedShapeCastQueryCollector:: getHits() const
{
	return m_childCollector->getHits();
}

HK_FORCE_INLINE void hknpFlippedShapeCastQueryCollector::addHit( const hknpCollisionResult& flippedHit )
{
	hkSimdReal fraction = hkSimdReal::fromFloat( flippedHit.m_fraction );

	hknpCollisionResult hit(flippedHit);
	{
		hit.m_position						. setAddMul( flippedHit.m_position, m_castDirectionWS, fraction );
		hit.m_normal						. setNeg<3>( flippedHit.m_normal );
		hit.m_queryBodyInfo = flippedHit.m_hitBodyInfo;
		hit.m_hitBodyInfo = flippedHit.m_queryBodyInfo;
	}

	m_childCollector->addHit( hit );

	// We need to update our local EOHF every time it might potentially have changed in the child collector
	// as this value is queried by e.g. the compressed mesh shape's code.
	m_earlyOutHitFraction = m_childCollector->getEarlyOutHitFraction();
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
