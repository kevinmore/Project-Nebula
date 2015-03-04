/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


template <int U_SIZE, int F_SIZE, int M_SIZE>
HK_FORCE_INLINE hknpUFMShapeTagCodec<U_SIZE, F_SIZE, M_SIZE>::hknpUFMShapeTagCodec(hknpMaterialLibrary* materialLibrary)
	: hknpMaterialPaletteShapeTagCodec(materialLibrary)
{
	m_type = UFM_CODEC;
}

template <int U_SIZE, int F_SIZE, int M_SIZE>
hknpShapeTag HK_CALL hknpUFMShapeTagCodec<U_SIZE, F_SIZE, M_SIZE>::encode(
	hknpMaterialPaletteEntryId paletteEntryId, hkUint32 collisionFilterInfo, hkUint64 userData)
{
	// Make sure the hknpShapeTag is wide enough to store all bits for this codec.
	HK_COMPILE_TIME_ASSERT(sizeof(hknpShapeTag) >= ((U_SIZE + F_SIZE + M_SIZE) >> 3));

	int paletteEntryIdInt = paletteEntryId.value();

	HK_ASSERT2(0xafeeeaa1, paletteEntryIdInt	< (1 << M_SIZE), "The material palette entry ID provided is too large.");
	HK_ASSERT2(0xafeeeaa2, collisionFilterInfo	< (1 << F_SIZE), "The collision filter info provided is too large.");
	HK_ASSERT2(0xafeeeaa3, userData				< (1 << U_SIZE), "The user data provided is too large.");

	paletteEntryIdInt	&= ((1 << M_SIZE) - 1);
	collisionFilterInfo	&= ((1 << F_SIZE) - 1);
	userData            &= ((1 << U_SIZE) -1 );

	const int userDataShifted   = (int)userData       << (M_SIZE + F_SIZE);
	const int filterInfoShifted = collisionFilterInfo << (M_SIZE);

	const hknpShapeTag shapeTag = hknpShapeTag(paletteEntryIdInt | filterInfoShifted | userDataShifted);

	return shapeTag;
}

template <int U_SIZE, int F_SIZE, int M_SIZE>
HK_FORCE_INLINE hknpMaterialPaletteEntryId hknpUFMShapeTagCodec<U_SIZE, F_SIZE, M_SIZE>::decodeMaterialPaletteEntryId(hknpShapeTag shapeTag)
{
	HK_ASSERT2( 0xaf1fd2ad, shapeTag != HKNP_INVALID_SHAPE_TAG, "You cannot decode an invalid shape tag." );
	const int materialPaletteEntryIdMask = ((1<<M_SIZE)-1);
	int materialPaletteEntryId =  shapeTag & materialPaletteEntryIdMask;
	return hknpMaterialPaletteEntryId(materialPaletteEntryId);
}

template <int U_SIZE, int F_SIZE, int M_SIZE>
HK_FORCE_INLINE hkUint32 hknpUFMShapeTagCodec<U_SIZE, F_SIZE, M_SIZE>::decodeCollisionFilterInfo(hknpShapeTag shapeTag)
{
	HK_ASSERT2( 0xaf1fd2ae, shapeTag != HKNP_INVALID_SHAPE_TAG, "You cannot decode an invalid shape tag." );
	const int collisionFilterInfoMask = ((1<<F_SIZE)-1) <<  M_SIZE;
	int collisionFilterInfo = (shapeTag & collisionFilterInfoMask) >> M_SIZE;
	return collisionFilterInfo;
}

template <int U_SIZE, int F_SIZE, int M_SIZE>
HK_FORCE_INLINE hkUint64 hknpUFMShapeTagCodec<U_SIZE, F_SIZE, M_SIZE>::decodeUserData(hknpShapeTag shapeTag)
{
	HK_ASSERT2( 0xaf1fd2af, shapeTag != HKNP_INVALID_SHAPE_TAG, "You cannot decode an invalid shape tag." );
	const int userDataMask = ((1<<U_SIZE)-1) << (M_SIZE+F_SIZE);
	int userData = (shapeTag & userDataMask) >> (M_SIZE+F_SIZE);
	return userData;
}

template <int U_SIZE, int F_SIZE, int M_SIZE>
HK_FORCE_INLINE	void hknpUFMShapeTagCodec<U_SIZE, F_SIZE, M_SIZE>::decode(
	hknpShapeTag shapeTag, const Context* context,
	hkUint32* collisionFilterInfo, hknpMaterialId* materialId, hkUint64* userData ) const
{
	if (shapeTag != HKNP_INVALID_SHAPE_TAG)
	{
		*collisionFilterInfo = decodeCollisionFilterInfo(shapeTag);

		// Decode the palette entry Id using the material palette codec
		hknpShapeTag paletteEntryId = shapeTag & ((1 << M_SIZE) - 1);
		hknpMaterialPaletteShapeTagCodec::decode(paletteEntryId, context, collisionFilterInfo, materialId, userData);

		*userData = decodeUserData(shapeTag);
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
