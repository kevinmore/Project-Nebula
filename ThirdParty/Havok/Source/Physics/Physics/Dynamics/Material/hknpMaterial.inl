/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE bool hknpMaterial::operator==( const hknpMaterial& other ) const
{
	return
		(m_name == other.m_name) &&
		(0 == hkString::memCmp( (const hkUint32*)&m_isExclusive, (const hkUint32*)&other.m_isExclusive,
								 hkGetByteOffsetInt(&m_isExclusive, &m_isShared) ));
}


HK_FORCE_INLINE hknpMaterialDescriptor::hknpMaterialDescriptor()
:	m_materialId( hknpMaterialId::INVALID )
{

}


HK_FORCE_INLINE void hknpMaterial::FreeListArrayOperations::setEmpty( hknpMaterial& material, hkUint32 next )
{
	// Make sure all data members have serializable values
	::new (reinterpret_cast<hkPlacementNewArg*>(&material)) hknpMaterial();

	// Use an invalid but serializable value to mark the object as empty
	material.m_maxContactImpulse = -1;

	// Store next empty element index in a data member where it will be serialized properly
	(hkUint32&)material.m_isExclusive = next;
}

HK_FORCE_INLINE hkUint32 hknpMaterial::FreeListArrayOperations::getNext( const hknpMaterial& material )
{
	return (const hkUint32&)material.m_isExclusive;
}

HK_FORCE_INLINE hkBool32 hknpMaterial::FreeListArrayOperations::isEmpty( const hknpMaterial& material )
{
	return material.m_maxContactImpulse < 0;
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
