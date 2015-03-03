/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */



HK_FORCE_INLINE hknpMinMaxQuadTree::MinMaxLevel::MinMaxLevel()
{
}


HK_FORCE_INLINE hknpMinMaxQuadTree::MinMaxLevel::MinMaxLevel( const hknpMinMaxQuadTree::MinMaxLevel& l )
:	m_xRes(l.m_xRes)
,	m_zRes(l.m_zRes)
{
	m_minMaxData = l.m_minMaxData;
}


HK_FORCE_INLINE hknpMinMaxQuadTree::MinMaxLevel::MinMaxLevel( hkFinishLoadedObjectFlag f )
:	m_minMaxData(f)
{
}


HK_FORCE_INLINE void hknpMinMaxQuadTree::getMinMax(int level, int x, int z, hkVector4* HK_RESTRICT minOut, hkVector4* HK_RESTRICT maxOut) const
{
	HK_ASSERT(0x42893575, !m_coarseTreeData.isEmpty());
	HK_ASSERT(0x42893572, level<m_coarseTreeData.getSize());
#if !defined( HK_PLATFORM_SPU )
	const MinMaxLevel& coarseLevel = m_coarseTreeData[level];
#else
	const MinMaxLevel& coarseLevel = g_SpuCollideUntypedCache->getArrayItem<MinMaxLevel, HK_SPU_UNTYPED_CACHE_LINE_SIZE>(m_coarseTreeData.begin(), level);
#endif
	if (x>=coarseLevel.m_xRes || z>=coarseLevel.m_zRes) return;	// Can happen for non power-of-two height fields.
	int index = 4*(x*coarseLevel.m_zRes+z);

#if !defined( HK_PLATFORM_SPU )
	hkIntVector intMinMax; intMinMax.loadNotAligned<4>(&coarseLevel.m_minMaxData[index]);
#else
	int min1 = g_SpuCollideUntypedCache->getArrayItem<hkUint32, HK_SPU_UNTYPED_CACHE_LINE_SIZE>(coarseLevel.m_minMaxData.begin(), index);
	int min2 = g_SpuCollideUntypedCache->getArrayItem<hkUint32, HK_SPU_UNTYPED_CACHE_LINE_SIZE>(coarseLevel.m_minMaxData.begin(), index+1);
	int max1 = g_SpuCollideUntypedCache->getArrayItem<hkUint32, HK_SPU_UNTYPED_CACHE_LINE_SIZE>(coarseLevel.m_minMaxData.begin(), index+2);
	int max2 = g_SpuCollideUntypedCache->getArrayItem<hkUint32, HK_SPU_UNTYPED_CACHE_LINE_SIZE>(coarseLevel.m_minMaxData.begin(), index+3);
	hkIntVector intMinMax; intMinMax.set(min1, min2, max1, max2);
#endif
	hkSimdReal multiplier = hkSimdReal::fromFloat(m_multiplier);
	{
		hkIntVector minIntVector;
		minIntVector.setConvertLowerU16ToU32(intMinMax);
#if HK_ENDIAN_BIG
		// Need to shuffle it due to Endianness because we loaded in Int32 arrays
		minIntVector.setPermutation<hkVectorPermutation::YXWZ>(minIntVector);
#endif
		hkVector4 minFloatVector; minIntVector.convertU32ToF32(minFloatVector);
		minOut->setAddMul(m_offset, minFloatVector, multiplier);
	}
	{
		hkIntVector maxIntVector;
		maxIntVector.setConvertUpperU16ToU32(intMinMax);
#if HK_ENDIAN_BIG
		// Need to shuffle it due to Endianness because we loaded in Int32 arrays
		maxIntVector.setPermutation<hkVectorPermutation::YXWZ>(maxIntVector);
#endif
		hkVector4 maxFloatVector; maxIntVector.convertU32ToF32(maxFloatVector);
		maxOut->setAddMul(m_offset, maxFloatVector, multiplier);
	}
}


HK_FORCE_INLINE void hknpMinMaxQuadTree::setMinMax(int x, int z, hkSimdRealParameter min, hkSimdRealParameter max)
{
	int ix = x/2; int iz = z/2;
	MinMaxLevel& coarseLevel = m_coarseTreeData[0];
	HK_ASSERT(0x448A3576, ix>=0 && ix<coarseLevel.m_xRes && iz>=0 && iz<coarseLevel.m_zRes);
	int index = 4*(ix*coarseLevel.m_zRes+iz);

	hkSimdReal invMultiplier = hkSimdReal::fromFloat(m_invMultiplier);

	hkUint16 intMin;
	hkSimdReal floatMin = (min-m_offset.getComponent<0>())*invMultiplier - hkSimdReal_1;
	floatMin.storeSaturateUint16(&intMin);

	hkUint16 intMax;
	hkSimdReal floatMax = (max-m_offset.getComponent<0>())*invMultiplier + hkSimdReal_1;
	floatMax.storeSaturateUint16(&intMax);

	switch((x&1)+2*(z&1)) {
	case 0:
		coarseLevel.m_minMaxData[index] = (coarseLevel.m_minMaxData[index]&0xffff0000) | intMin;
		coarseLevel.m_minMaxData[index+2] = (coarseLevel.m_minMaxData[index+2]&0xffff0000) | intMax;
		break;
	case 1:
		coarseLevel.m_minMaxData[index] = (coarseLevel.m_minMaxData[index]&0x0000ffff) | (intMin<<16);
		coarseLevel.m_minMaxData[index+2] = (coarseLevel.m_minMaxData[index+2]&0x0000ffff) | (intMax<<16);
		break;
	case 2:
		coarseLevel.m_minMaxData[index+1] = (coarseLevel.m_minMaxData[index+1]&0x0000ffff) | (intMin<<16);
		coarseLevel.m_minMaxData[index+3] = (coarseLevel.m_minMaxData[index+3]&0x0000ffff) | (intMax<<16);
		break;
	case 3:
		coarseLevel.m_minMaxData[index+1] = (coarseLevel.m_minMaxData[index+1]&0xffff0000) | intMin;
		coarseLevel.m_minMaxData[index+3] = (coarseLevel.m_minMaxData[index+3]&0xffff0000) | intMax;
		break;
	}
}


HK_FORCE_INLINE void hknpMinMaxQuadTree::buildTree(hkReal* newMinOut, hkReal* newMaxOut)
{
	updateRegion(0, 0, m_coarseTreeData[0].m_xRes-1, m_coarseTreeData[0].m_zRes-1, newMinOut, newMaxOut);
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
