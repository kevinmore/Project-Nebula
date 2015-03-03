/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hkReal hknpCompressedHeightFieldShape::decompress(hkUint16 comp) const
{
	return hkReal(comp) * m_scale + m_offset;
}

HK_FORCE_INLINE hkUint16 hknpCompressedHeightFieldShape::compress(hkReal uncomp) const
{
	return static_cast<hkUint16> ( (uncomp - m_offset) / m_scale );
}

HK_FORCE_INLINE static hkReal hknpCompressedHeightFieldShape_getHeightAt(const hknpCompressedHeightFieldShape* self, int x, int z)
{
	x = hkMath::clamp(x,0,self->m_intSizeX);
	z = hkMath::clamp(z,0,self->m_intSizeZ);
	const int index = z*(self->m_intSizeX+1) + x;
#ifdef HK_PLATFORM_SPU
	const hkUint16 compressedVal = g_SpuCollideUntypedCache->getArrayItem<hkUint16, HK_SPU_UNTYPED_CACHE_LINE_SIZE>(self->m_storage.begin(), index);
#else
	const hkUint16 compressedVal = self->m_storage[index];
#endif

	return self->decompress( compressedVal );
}

HK_FORCE_INLINE static hknpShapeTag hknpCompressedHeightFieldShape_getShapeTagAt(const hknpCompressedHeightFieldShape* self, int x, int z)
{
	if (self->m_shapeTags.isEmpty()) return HKNP_INVALID_SHAPE_TAG;
	x = hkMath::clamp(x,0,self->m_intSizeX);
	z = hkMath::clamp(z,0,self->m_intSizeZ);
	const int index = z*(self->m_intSizeX+1) + x;
#ifdef HK_PLATFORM_SPU
	return g_SpuCollideUntypedCache->getArrayItem<hknpShapeTag, HK_SPU_UNTYPED_CACHE_LINE_SIZE>(self->m_shapeTags.begin(), index);
#else
	return self->m_shapeTags[index];
#endif
}

HK_FORCE_INLINE void hknpCompressedHeightFieldShape::getQuadInfoAt(int x, int z, hkVector4* heightOut, hknpShapeTag* shapeTagOut, hkBool32 *triangleFlipOut) const
{
	*triangleFlipOut = m_triangleFlip;
	*shapeTagOut = hknpCompressedHeightFieldShape_getShapeTagAt(this,x,z);
	heightOut->set(
		hknpCompressedHeightFieldShape_getHeightAt(this,x,z),
		hknpCompressedHeightFieldShape_getHeightAt(this,x+1,z),
		hknpCompressedHeightFieldShape_getHeightAt(this,x,z+1),
		hknpCompressedHeightFieldShape_getHeightAt(this,x+1,z+1)
	);
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
