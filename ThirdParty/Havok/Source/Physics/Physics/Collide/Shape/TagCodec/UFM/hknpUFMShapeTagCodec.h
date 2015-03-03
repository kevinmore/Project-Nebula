/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_UFM_SHAPE_TAG_CODEC_H
#define HKNP_UFM_SHAPE_TAG_CODEC_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Collide/Shape/TagCodec/MaterialPalette/hknpMaterialPaletteShapeTagCodec.h>


/// This codec encodes user data, collision filter info and material palette entry ID into a single shape tag using the
/// number of bits specified as template parameters for each piece of data. Refer to hknpMaterialPaletteShapeTagCodec
/// for information on how to setup the codec appropriately to decode shapes using material palettes.
template <int USER_DATA_SIZE, int COLLISION_FILTER_INFO_SIZE, int MATERIAL_PALETTE_ENTRY_ID_SIZE>
class hknpUFMShapeTagCodec : public hknpMaterialPaletteShapeTagCodec
{
	public:

		/// Encodes into a hknpShapeTag all child shape information required for decoding.
		static hknpShapeTag HK_CALL encode(
			hknpMaterialPaletteEntryId paletteEntryId, hkUint32 collisionFilterInfo, hkUint64 userData );

	public:

		HK_FORCE_INLINE hknpUFMShapeTagCodec( hknpMaterialLibrary* materialLibrary );

#if defined(HK_PLATFORM_SPU)

		/// Creates an empty codec on SPU. See hknpMaterialPaletteShapeTagCodec constructor for details on use.
		hknpUFMShapeTagCodec() : hknpMaterialPaletteShapeTagCodec() {}

#endif

		/// Decode the material palette entry id from a \a shapeTag.
		HK_FORCE_INLINE static hknpMaterialPaletteEntryId decodeMaterialPaletteEntryId( hknpShapeTag shapeTag );

		/// Decode the collision filter information from a \a shapeTag.
		HK_FORCE_INLINE static hkUint32 decodeCollisionFilterInfo( hknpShapeTag shapeTag );

		/// Decode the user data from a \a shapeTag.
		HK_FORCE_INLINE static hkUint64 decodeUserData( hknpShapeTag shapeTag );

		/// Decodes a shape tag using the given context.
		/// See hknpMaterialPaletteShapeTagCodec for details about how the context is used to decode the material ID.
		HK_FORCE_INLINE virtual void decode(
			hknpShapeTag shapeTag, const Context* context,
			hkUint32* collisionFilterInfo, hknpMaterialId* materialId, hkUint64* userData ) const;
};


/// UFM codec with 3 bits for User data, 5, for collision Filter info and 8 for Material palette entry ID
typedef hknpUFMShapeTagCodec<3, 5, 8> hknpUFM358ShapeTagCodec;


#include <Physics/Physics/Collide/Shape/TagCodec/UFM/hknpUFMShapeTagCodec.inl>

#endif	// HKNP_UFM_SHAPE_TAG_CODEC_H

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
