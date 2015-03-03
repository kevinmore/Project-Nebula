/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/TagCodec/MaterialPalette/hknpMaterialPaletteShapeTagCodec.h>

// Force explicit template instantiation
#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase<const hknpMaterialPalette*, hknpMaterialPaletteShapeTagCodec::PaletteInfo>;


void hknpMaterialPaletteShapeTagCodec::registerShape(hknpCompositeShape* shape, const hknpMaterialPalette* palette)
{
	PaletteInfo info;
	PaletteMap::Iterator iterator = m_paletteMap.findKey(palette);

	// If we have not seen the palette before add it to the palette map and its descriptors to the library
	if (!m_paletteMap.isValid(iterator))
	{
		hkUint32 materialIndex = m_paletteMaterials.getSize();
		HK_ASSERT2(0x436922ef, materialIndex < HKNP_INVALID_SHAPE_TAG_CODEC_INFO,  "Too many palette materials, try reducing the number of palettes and sharing them between shapes to avoid duplication");
		const hkArray<hknpMaterialDescriptor>& descriptors = palette->getEntries();
		int numEntries = descriptors.getSize();
		HK_ASSERT2(0x4288e4a7, numEntries <= 0xFFFF, "Too many entries in material palette");

		// Insert palette info in the map
		info.m_firstMaterialIndex = (hknpShapeTagCodecInfo) materialIndex;
		info.m_numEntries = (hknpMaterialPaletteEntryId::Type) numEntries;
		m_paletteMap.insert(palette, info);

		// Realize material descriptors
		m_paletteMaterials.expandBy(numEntries);
		for (int i = 0; i < numEntries; ++i)
		{
			m_paletteMaterials[materialIndex + i] = m_materialLibrary->addEntry(descriptors[i]);
		}
	}

	// We have seen the palette before, obtain its info
	else
	{
		info = m_paletteMap.getValue(iterator);
	}


	ShapeMap::Iterator shapeIterator = m_shapeMap.findKey(shape);

	// If the shape is already registered check if it has changed its palette
	if (m_shapeMap.isValid(shapeIterator))
	{
		const hknpMaterialPalette* oldPalette = m_shapeMap.getValue(shapeIterator);

		// If the palette has not changed update the shape tag codec info and return
		if (oldPalette == palette)
		{
			shape->m_shapeTagCodecInfo = info.m_firstMaterialIndex;
			return;
		}

		unregisterShape(shape);
	}

	// Insert shape in map with the palette and set its shape tag codec info
	m_shapeMap.insert(shape, palette);
	shape->m_shapeTagCodecInfo = hknpShapeTagCodecInfo(info.m_firstMaterialIndex);
}


void hknpMaterialPaletteShapeTagCodec::unregisterShape(hknpCompositeShape* shape)
{
	// If the shape is registered remove it from the map and remove its palette if no other shape is using it
	ShapeMap::Iterator shapeIterator = m_shapeMap.findKey(shape);
	if (m_shapeMap.isValid(shapeIterator))
	{
		const hknpMaterialPalette* palette = m_shapeMap.getValue(shapeIterator);
		m_shapeMap.remove(shapeIterator);

		// Check if any other shape is using the same palette
		for (shapeIterator = m_shapeMap.getIterator(); m_shapeMap.isValid(shapeIterator); shapeIterator = m_shapeMap.getNext(shapeIterator))
		{
			if (m_shapeMap.getValue(shapeIterator) == palette)
			{
				break;
			}
		}

		// If no other shape is using it invalidate its materials and remove it from the palette map
		if (!m_shapeMap.isValid(shapeIterator))
		{
			PaletteMap::Iterator paletteIterator = m_paletteMap.findKey(palette);
			HK_ASSERT2(0x143951f7, m_paletteMap.isValid(paletteIterator), "Could not find the palette registered with this shape");

			// Mark all palette materials as invalid
			PaletteInfo info = m_paletteMap.getValue(paletteIterator);
			for (int entryIndex = 0;  entryIndex < info.m_numEntries; ++entryIndex)
			{
				m_paletteMaterials[info.m_firstMaterialIndex + entryIndex] = hknpMaterialId::invalid();
			}

			m_paletteMap.remove(paletteIterator);
		}
	}

	shape->m_shapeTagCodecInfo = HKNP_INVALID_SHAPE_TAG_CODEC_INFO;
}


void hknpMaterialPaletteShapeTagCodec::onMaterialRemovedSignal(hknpMaterialId materialId)
{
	for (int i = 0 ; i < m_paletteMaterials.getSize(); ++i)
	{
		HK_ASSERT2(0x6111911f, materialId != m_paletteMaterials[i], "A material in use by a palette has been removed from the material library");
	}
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
