/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


hknpManifoldStatusEvent::hknpManifoldStatusEvent(
	hknpBodyId idA, hknpShapeKey keyA,
	hknpBodyId idB, hknpShapeKey keyB,
	hknpManifoldCollisionCache* manifoldCache, Status status )
:	hknpBinaryBodyEvent( hknpEventType::MANIFOLD_STATUS, sizeof(*this), idA, idB )
{
	m_status        = status;
	m_manifoldCache = manifoldCache;
	m_shapeKeys[0]  = keyA;
	m_shapeKeys[1]  = keyB;
}

HK_FORCE_INLINE hkBool32 hknpManifoldStatusEvent::involvesTriggerVolume() const
{
	HK_ASSERT( 0x764ddfdf, m_manifoldCache );
	return m_manifoldCache->m_bodyAndMaterialFlags & hknpMaterial::ENABLE_TRIGGER_VOLUME;
}


hknpManifoldProcessedEvent::hknpManifoldProcessedEvent(
	hknpBodyId idA, hknpShapeKey shapeKeyA,
	hknpBodyId idB, hknpShapeKey shapeKeyB )
:	hknpBinaryBodyEvent( hknpEventType::MANIFOLD_PROCESSED, sizeof(*this), idA, idB )
{
	m_shapeKeys[0] = shapeKeyA;
	m_shapeKeys[1] = shapeKeyB;
	m_flipped      = false;
}

HK_FORCE_INLINE hkBool32 hknpManifoldProcessedEvent::involvesTriggerVolume() const
{
	HK_ASSERT( 0x339cf704, m_manifoldCache );
	return m_manifoldCache->m_bodyAndMaterialFlags & hknpMaterial::ENABLE_TRIGGER_VOLUME;
}

HK_FORCE_INLINE void hknpManifoldProcessedEvent::flip()
{
	m_manifold.flip();
	hknpBodyId h = m_bodyIds[0]; m_bodyIds[0] = m_bodyIds[1]; m_bodyIds[1] = h;
	m_flipped = 1^m_flipped;
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
