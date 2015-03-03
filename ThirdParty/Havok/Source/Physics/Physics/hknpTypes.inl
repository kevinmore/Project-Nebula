/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


extern const hkUint32 hknpShapeKeyPath_usedBitsMaskTable[33];

HK_FORCE_INLINE	void hknpShapeKeyPath::reset()
{
	m_key  = HKNP_INVALID_SHAPE_KEY;
	m_size = 0;
}

HK_FORCE_INLINE hknpShapeKeyPath::hknpShapeKeyPath()
{
	reset();
}

HK_FORCE_INLINE hknpShapeKeyPath::hknpShapeKeyPath(hknpShapeKey sourceKey, int bitSize)
{
	setFromKey(sourceKey, bitSize);
}

HK_FORCE_INLINE	hknpShapeKey hknpShapeKeyPath::makeKey( hknpShapeKey subkey, int sizeInBits ) const
{
	const int size = m_size + sizeInBits;
	HK_ASSERT2(0xAEEDCB81, size <= (hkSizeOf(hknpShapeKey) * 8), "Shape key overflow");

	hknpShapeKey maskedOldKey = m_key & hknpShapeKeyPath_usedBitsMaskTable[m_size];

	// +1 before shift, -1 afterwards sets all unused bits to 1.
	hknpShapeKey shiftedSubkey = ((subkey+1) << HKNP_NUM_UNUSED_SHAPE_KEY_BITS(size)) - 1;

	return maskedOldKey | shiftedSubkey;
}

HK_FORCE_INLINE	void hknpShapeKeyPath::appendSubKey( hknpShapeKey subkey, int sizeInBits )
{
	m_key = makeKey( subkey, sizeInBits );
	m_size += sizeInBits;
}

HK_FORCE_INLINE	hknpShapeKey hknpShapeKeyPath::getKey() const
{
	return m_key;
}

HK_FORCE_INLINE	void hknpShapeKeyPath::setFromKey( hknpShapeKey sourceKey, int bitSize )
{
	HK_ASSERT2( 0xaf1151e1, bitSize <= (hkSizeOf(hknpShapeKey) * 8), "Shape key's size exceeds maximum." );
	HK_ASSERT2( 0xaf1151ef, (sourceKey != HKNP_INVALID_SHAPE_KEY) || (bitSize == 0), "You need to set the key length to 0 when passing in HKNP_INVALID_SHAPE_KEY." );

	m_key  = sourceKey;
	m_size = bitSize;
}

HK_FORCE_INLINE	void hknpShapeKeyPath::setUnusedBits()
{
	hkUint32 unusedBits = ~hknpShapeKeyPath_usedBitsMaskTable[m_size];
	m_key |= unusedBits;
}

HK_FORCE_INLINE	int hknpShapeKeyPath::getKeySize() const
{
	return m_size;
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
