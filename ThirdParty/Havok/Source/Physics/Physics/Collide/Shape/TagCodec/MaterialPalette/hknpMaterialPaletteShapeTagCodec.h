/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MATERIAL_PALETTE_SHAPE_TAG_CODEC_H
#define HKNP_MATERIAL_PALETTE_SHAPE_TAG_CODEC_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Collide/Shape/TagCodec/hknpShapeTagCodec.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterialLibrary.h>
#include <Common/Base/Container/PointerMap/hkPointerMap.h>
#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterialPalette.h>


/// This codec uses material palettes to reduce the number of bits required to store material information in shape tags.
/// The palette entry ID is encoded in the shape tag instead of the material ID and no collision filter info or user data
/// information is stored or decoded, see hknpUFMShapeTagCodec for this.
///
/// In order to use a shape with this codec it must be first registered via a registerShape() call, together with the
/// palette that was used to generate its shape tags. This method will add the material descriptors in the palette
/// to the material library provided on construction of the codec and store in the shape's shape tag codec info an index
/// to access the materials obtained (see the registerShape() description for more details).
///
/// A shape that has not been registered in the codec can only be decoded if its shape tag codec info is
/// HKNP_INVALID_SHAPE_TAG_CODEC_INFO, in which case only collision filter info and user data will be decoded from the
/// shape tag.
///
/// When a registered shape is not going to be used anymore it must be unregistered in order to mark its palette
/// materials as unused, otherwise trying to remove them from the library will raise an assert.
class hknpMaterialPaletteShapeTagCodec : public hknpShapeTagCodec
{
	public:

		/// Encode a palette entry ID into a shape tag.
		static HK_FORCE_INLINE hknpShapeTag HK_CALL encode( hknpMaterialPaletteEntryId paletteEntryId );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

#if !defined(HK_PLATFORM_SPU)

		/// Constructor. Receives the material library where palette material descriptors will be added to.
		HK_FORCE_INLINE hknpMaterialPaletteShapeTagCodec( hknpMaterialLibrary* materialLibrary );

		HK_FORCE_INLINE ~hknpMaterialPaletteShapeTagCodec();

#else

		/// Constructs an empty codec on SPU. In order to decode using this codec m_paletteMaterials must be filled in
		/// with the data from PPU first.
		hknpMaterialPaletteShapeTagCodec() : hknpShapeTagCodec(MATERIAL_PALETTE_CODEC) {}

#endif

		/// Register a composite shape together with the palette that will be used to decode its shape tags. If the
		/// palette has not been registered before with any other shape its material descriptors will be added
		/// to m_materialLibrary, otherwise the previously obtained materials will be used. An index to the first
		/// material of the palette will be stored in the shape's m_shapeTagCodecInfo. It is not necessary to keep the
		/// palette alive after registration.
		void registerShape( hknpCompositeShape* shape, const hknpMaterialPalette* materialPalette );

		/// Clear the shape tag codec info in the shape setting it to HKNP_INVALID_SHAPE_TAG_CODEC_INFO. The shape's
		/// palette will be marked as disabled if there are no other shapes using it.
		void unregisterShape( hknpCompositeShape* shape );

		/// Decode a shape tag using the given context. The shape tag codec info of context.m_parentShape will be used
		/// to translate from the palette entry encoded in the shape tag to a material ID. If no parent shape is provided
		/// or its shape tag codec info is HKNP_INVALID_SHAPE_TAG_CODEC_INFO the input material will not be overwritten.
		/// Input collision filter info and user data are not modified.
		HK_FORCE_INLINE virtual void decode(
			hknpShapeTag shapeTag, const Context* context,
			hkUint32* collisionFilterInfo, hknpMaterialId* materialId, hkUint64* userData ) const;

		/// Get read only access to the IDs of the library materials used by a palette.
		HK_FORCE_INLINE void getPaletteMaterials(
			const hknpMaterialPalette* palette, const hknpMaterialId** materials, int* numMaterials ) const;

		/// Get read-write access to the IDs of the library materials used by all palettes.
		HK_FORCE_INLINE hkArray<hknpMaterialId>& getPaletteMaterialsStorage();

		/// Internal. Listens for removals from the material library.
		/// Used in debug to assert that the material is not in use by any palette.
		void onMaterialRemovedSignal( hknpMaterialId materialId );

	protected:

		/// Information about a registered material palette
		struct PaletteInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, PaletteInfo );

			/// Index of the first material ID belonging to the palette in m_paletteMaterials.
			hknpShapeTagCodecInfo m_firstMaterialIndex;

			/// Number of materials in the palette.
			hknpMaterialPaletteEntryId::Type m_numEntries;
		};

		/// Map from palette address to information about the palette materials
		typedef hkMap<const hknpMaterialPalette*, PaletteInfo> PaletteMap;

		/// Map from shape address to material palette address
		typedef hkPointerMap<const hknpCompositeShape*, const hknpMaterialPalette*> ShapeMap;

	protected:

		/// Array with the materials of all registered palettes. This is the only member required for decoding.
		/// Aligned for SPU access.
		HK_ALIGN16(hkArray<hknpMaterialId> m_paletteMaterials);

#if !defined(HK_PLATFORM_SPU)

		/// Material library where palette descriptors will be added to. The codec will listen to material removals
		/// in this library and assert if the material is in use by any palette.
		hkRefPtr<class hknpMaterialLibrary> m_materialLibrary;

		/// Array with information about each registered palette.
		PaletteMap m_paletteMap;

		/// Maps the address of a registered shape to the information of its palette.
		ShapeMap m_shapeMap;

#endif
};

#include <Physics/Physics/Collide/Shape/TagCodec/MaterialPalette/hknpMaterialPaletteShapeTagCodec.inl>

#endif	// HKNP_MATERIAL_PALETTE_SHAPE_TAG_CODEC_H

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
