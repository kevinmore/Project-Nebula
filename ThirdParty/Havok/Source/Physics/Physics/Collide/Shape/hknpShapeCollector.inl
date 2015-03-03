/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hknpShapeCollector::hknpShapeCollector( hknpTriangleShape* triangleShapePrototype )
{
	/* HK_ON_DEBUG( hkString::memSet(this, 0xcd, sizeof(hknpShapeCollector)); ) */
	m_triangleShapePrototype = triangleShapePrototype;

#if defined(HK_PLATFORM_SPU)
	m_shapeBufferPtr = &m_shapeBuffer[0];
	m_shapeBufferSize = HKNP_SHAPE_COLLECTOR_BUFFER_SIZE;
#endif
}


HK_FORCE_INLINE void hknpShapeCollector::reset( const hkTransform& transform )
{
	m_shapeOut = HK_NULL;
	m_parentShape = HK_NULL;
	m_transformOut = transform;
	m_scaleOut = hkVector4::getConstant<HK_QUADREAL_1>();
	m_shapeTagOut = HKNP_INVALID_SHAPE_TAG;
	m_shapeTagPath.clear();
	m_transformModifiedFlag = false;

#if defined(HK_PLATFORM_SPU)
	m_shapeBufferPtr = &m_shapeBuffer[0];
	m_shapeBufferSize = HKNP_SHAPE_COLLECTOR_BUFFER_SIZE;
#endif
}


HK_FORCE_INLINE void hknpShapeCollector::checkForReset()
{
	HK_ASSERT2( 0xf0c5ffd1, m_shapeOut == HK_NULL, "It seems that you are using a hknpShapeCollector without calling reset() prior to calling getLeafShape().");
}


#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE hknpShapeCollectorWithInplaceTriangle::hknpShapeCollectorWithInplaceTriangle()
:	hknpShapeCollector( m_inplaceTriangleShape.getTriangleShape() )
{
}


HK_FORCE_INLINE hknpShapeCollectorWithInplaceTriangle::~hknpShapeCollectorWithInplaceTriangle()
{
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
