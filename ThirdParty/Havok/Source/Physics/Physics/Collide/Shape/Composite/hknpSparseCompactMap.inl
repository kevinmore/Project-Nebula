/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

template <typename StoreT>
hknpSparseCompactMap<StoreT>::hknpSparseCompactMap(class hkFinishLoadedObjectFlag flag)
 : m_primaryKeyToIndex(flag)
 , m_valueAndSecondaryKeys(flag)
{
}

template <typename StoreT>
hknpSparseCompactMap<StoreT>::hknpSparseCompactMap()
{
	m_secondaryKeyMask = 0xffffffff;
	m_sencondaryKeyBits = 0;
}

template <typename StoreT>
void hknpSparseCompactMap<StoreT>::buildMap(int keyBits, int primaryKeyBits, int valueBits, hknpSparseCompactMapUtil::Entry* entries, int numEntries)
{
	if (keyBits - primaryKeyBits + valueBits > int(8 * sizeof(StoreT)))
	{
		hkStringBuf errorMsg;
		errorMsg.printf( "SparseCompactMap cannot hold all the bits required, keyBits(%i)-primaryKeyBits(%i)+valueBits(%i) must be less than %i", keyBits, primaryKeyBits, valueBits, 8*sizeof(StoreT));
		HK_ERROR( 0xaf0ee222, errorMsg.cString() );
	}
	m_secondaryKeyMask = (1<<(keyBits-primaryKeyBits))-1;
	m_sencondaryKeyBits = keyBits-primaryKeyBits;

	hkUint32 numPrimaryKeys = 1<<primaryKeyBits;

	m_primaryKeyToIndex.clear();
	m_primaryKeyToIndex.reserve(numPrimaryKeys+1);
	m_valueAndSecondaryKeys.clear();
	m_valueAndSecondaryKeys.reserve(numEntries);

	hknpSparseCompactMapUtil::sort( entries, numEntries );
	//hkSort( entries, numEntries );
	hkUint16 currentIndex = 0;
	m_primaryKeyToIndex.pushBackUnchecked(0);
	for (hkUint32 i=0; i<numPrimaryKeys; i++)
	{
		while (1)
		{
			if (currentIndex>=numEntries)
			{
				m_primaryKeyToIndex.pushBackUnchecked(currentIndex);
				break;
			}
			hkUint32 primaryKey = entries[currentIndex].m_key >> m_sencondaryKeyBits;
			if (primaryKey>i)
			{
				m_primaryKeyToIndex.pushBackUnchecked(currentIndex);
				break;
			}
			HK_ASSERT(0x52184852, i==primaryKey);
			StoreT val = (StoreT)((entries[currentIndex].m_key&m_secondaryKeyMask) | (entries[currentIndex].m_value<<m_sencondaryKeyBits));
			m_valueAndSecondaryKeys.pushBackUnchecked(val);
			currentIndex++;
		}
	}
}

// Lookup a value. Returns 0xffffffff if not found.
template <typename StoreT>
hkUint32 HK_FORCE_INLINE hknpSparseCompactMap<StoreT>::lookup(hkUint32 key) const
{
	if (m_secondaryKeyMask==0xffffffff) return 0xffffffff;

	hkUint32 secondaryKey = key & m_secondaryKeyMask;
	hkUint32 primaryKey = key >> m_sencondaryKeyBits;

	// Look up the bounds using the primary key.
	// Then do a binary search on the secondary key
	hkUint32 lowerBound = m_primaryKeyToIndex[primaryKey];
	hkUint32 upperBound = m_primaryKeyToIndex[primaryKey+1];
	while (lowerBound<upperBound)
	{
		hkUint32 mid = (lowerBound+upperBound)>>1;
		hkUint32 midValue = m_valueAndSecondaryKeys[mid];
		hkUint32 midSecondaryKey = midValue&m_secondaryKeyMask;
		if (midSecondaryKey>secondaryKey)
		{
			upperBound = mid;
		}
		else if (midSecondaryKey<secondaryKey)
		{
			lowerBound = mid+1;
		}
		else //secondaryKey == midSecondaryKey
		{
			return midValue>>m_sencondaryKeyBits;
		}
	}
	return 0xffffffff;
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
