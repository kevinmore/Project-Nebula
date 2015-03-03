/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#if !defined( HK_PLATFORM_SPU )

HK_FORCE_INLINE	hknpShapeInstance::hknpShapeInstance()
:	m_transform(hkTransform::getIdentity())
,	m_scale(hkVector4::getConstant<HK_QUADREAL_1>())
,	m_shape(HK_NULL)
,	m_shapeTag(HKNP_INVALID_SHAPE_TAG)
{
	setFlags(DEFAULT_FLAGS);
	hkString::memSet(m_padding, 0, NUM_PADDING_BYTES);
}

HK_FORCE_INLINE hknpShapeInstance::hknpShapeInstance( const hknpShape* shape, const hkTransform& transform )
:	m_transform(hkTransform::getIdentity())
,	m_scale(hkVector4::getConstant<HK_QUADREAL_1>())
,	m_shapeTag(HKNP_INVALID_SHAPE_TAG)
{
	setFlags(DEFAULT_FLAGS);
	hkString::memSet(m_padding, 0, NUM_PADDING_BYTES);

	setShape(shape);
	setTransform(transform);
}

HK_FORCE_INLINE	hknpShapeInstance::hknpShapeInstance( hkFinishLoadedObjectFlag flag )
:	m_shape(flag)
{
}

HK_FORCE_INLINE void hknpShapeInstance::setShapeTag( hknpShapeTag shapeTag )
{
	m_shapeTag = shapeTag;
}

HK_FORCE_INLINE void hknpShapeInstance::setEnabled( bool isEnabledParam )
{
	setFlags( isEnabledParam ? ( getFlags() | IS_ENABLED ) : ( getFlags() & ~IS_ENABLED ) );
}

HK_FORCE_INLINE void hknpShapeInstance::setFlags( int flags )
{
	m_transform.getColumn(0).setInt24W(flags);
}

HK_FORCE_INLINE void hknpShapeInstance::setShapeMemorySize( int size )
{
	m_transform.getColumn(2).setInt24W(size);
}

HK_FORCE_INLINE void hknpShapeInstance::setLeafIndex( int index )
{
	m_transform.getTranslation().setInt24W(index);
}

#endif	// !HK_PLATFORM_SPU


HK_FORCE_INLINE int hknpShapeInstance::getFlags() const
{
	return m_transform.getColumn<0>().getInt24W();
}

HK_FORCE_INLINE int hknpShapeInstance::getShapeMemorySize() const
{
	return m_transform.getColumn<2>().getInt24W();
}

HK_FORCE_INLINE int hknpShapeInstance::getLeafIndex() const
{
	return m_transform.getTranslation().getInt24W();
}

HK_FORCE_INLINE const hknpShape* hknpShapeInstance::getShape() const
{
	return m_shape.val();
}

HK_FORCE_INLINE const hkTransform& hknpShapeInstance::getTransform() const
{
	return m_transform;
}

HK_FORCE_INLINE const hkVector4& hknpShapeInstance::getScale() const
{
	return m_scale;
}

HK_FORCE_INLINE hknpShape::ScaleMode hknpShapeInstance::getScaleMode() const
{
	return ( getFlags() & SCALE_SURFACE ) ? hknpShape::SCALE_SURFACE : hknpShape::SCALE_VERTICES;
}

HK_FORCE_INLINE hknpShapeTag hknpShapeInstance::getShapeTag() const
{
	return m_shapeTag;
}

HK_FORCE_INLINE bool hknpShapeInstance::isEnabled() const
{
	return ( getFlags() & IS_ENABLED );
}

HK_FORCE_INLINE void hknpShapeInstance::setEmpty(hknpShapeInstance& shapeInstance, hkUint32 next)
{
	// Check that there is enough space in m_padding to store the next index and the isEmpty flag
	HK_COMPILE_TIME_ASSERT(sizeof(shapeInstance.m_padding) >= 4);
	HK_COMPILE_TIME_ASSERT(EMPTY_FLAG_PADDING_BYTE <= sizeof(shapeInstance.m_padding));

	// Make sure all members have serializable values
	::new (reinterpret_cast<hkPlacementNewArg*>(&shapeInstance)) hknpShapeInstance();

	// Use an invalid but serializable value to mark the object as empty
	shapeInstance.m_padding[EMPTY_FLAG_PADDING_BYTE] = 0xFF;

	// Store next empty element index in a data member where it will be serialized properly
	
	
	*((hkUint32*)shapeInstance.m_padding) = next;
}

HK_FORCE_INLINE hkUint32 hknpShapeInstance::getNext(const hknpShapeInstance& shapeInstance)
{
	return *((hkUint32*)shapeInstance.m_padding);
}

HK_FORCE_INLINE hkBool32 hknpShapeInstance::isEmpty(const hknpShapeInstance& shapeInstance)
{
	return shapeInstance.m_padding[EMPTY_FLAG_PADDING_BYTE];
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
