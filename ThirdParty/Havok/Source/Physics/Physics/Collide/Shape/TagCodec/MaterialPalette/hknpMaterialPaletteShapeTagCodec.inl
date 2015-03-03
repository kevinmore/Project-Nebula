/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE hknpMaterialPaletteShapeTagCodec::hknpMaterialPaletteShapeTagCodec(hknpMaterialLibrary* materialLibrary)
	: hknpShapeTagCodec(MATERIAL_PALETTE_CODEC), m_materialLibrary(materialLibrary)
{
	HK_ON_DEBUG(HK_SUBSCRIBE_TO_SIGNAL(m_materialLibrary->m_materialRemovedSignal, this, hknpMaterialPaletteShapeTagCodec));
}

HK_FORCE_INLINE hknpMaterialPaletteShapeTagCodec::~hknpMaterialPaletteShapeTagCodec()
{
	HK_ON_DEBUG(m_materialLibrary->m_materialRemovedSignal.unsubscribe(this, &hknpMaterialPaletteShapeTagCodec::onMaterialRemovedSignal));
}

HK_FORCE_INLINE hknpShapeTag HK_CALL encode(hknpMaterialPaletteEntryId paletteEntryId)
{
	return hknpShapeTag(paletteEntryId.value());
}

HK_FORCE_INLINE void hknpMaterialPaletteShapeTagCodec::getPaletteMaterials(const hknpMaterialPalette* palette,
	const hknpMaterialId** materials, int* numMaterials) const
{
	PaletteMap::Iterator iterator = m_paletteMap.findKey(palette);
	if (m_paletteMap.isValid(iterator))
	{
		PaletteInfo info = m_paletteMap.getValue(iterator);
		*materials = &m_paletteMaterials[info.m_firstMaterialIndex];
		*numMaterials = info.m_numEntries;
	}
	else
	{
		*materials = HK_NULL;
		*numMaterials = 0;
	}
}

#endif

HK_FORCE_INLINE void hknpMaterialPaletteShapeTagCodec::decode( hknpShapeTag shapeTag, const Context* context,
															  hkUint32* collisionFilterInfo, hknpMaterialId* materialId, hkUint64* userData ) const
{
	// If parent shape has a non-default shape tag codec info use it to decode the shape tag material and override
	// with it the input one
	HK_ASSERT(0x7df98ce6, context && context->m_parentShape);
	const hknpCompositeShape* compositeParent = context->m_parentShape->asCompositeShape();
	HK_ASSERT(0x45d74d69, compositeParent);
	if (compositeParent->m_shapeTagCodecInfo != HKNP_INVALID_SHAPE_TAG_CODEC_INFO)
	{
		hknpMaterialPaletteEntryId entryId(shapeTag);
		*materialId = m_paletteMaterials[compositeParent->m_shapeTagCodecInfo + entryId.value()];
	}
}

HK_FORCE_INLINE hkArray<hknpMaterialId>& hknpMaterialPaletteShapeTagCodec::getPaletteMaterialsStorage()
{
	return m_paletteMaterials;
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
