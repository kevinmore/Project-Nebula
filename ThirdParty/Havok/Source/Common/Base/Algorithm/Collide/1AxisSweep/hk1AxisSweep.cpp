/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h> // Precompiled Header

#include <Common/Base/Algorithm/Sort/hkRadixSort.h>
#include <Common/Base/Algorithm/Collide/1AxisSweep/hk1AxisSweep.h>

template<int flip>
HK_FORCE_INLINE hkKeyPair* hk1AxisSweep_appendPair(const hk1AxisSweep::AabbInt& aabb0, const hk1AxisSweep::AabbInt& aabb1, hkKeyPair* HK_RESTRICT pairOut, const hkKeyPair* HK_RESTRICT end, hkPadSpu<int>& numPairsSkipped)
{
	if ( pairOut < end)
	{
		if ( !flip )
		{
			pairOut->m_keyA = aabb0.getKey();
			pairOut->m_keyB = aabb1.getKey();
		}
		else
		{
			pairOut->m_keyA = aabb1.getKey();
			pairOut->m_keyB = aabb0.getKey();
		}
		pairOut++;
	}
	else
	{
		numPairsSkipped = numPairsSkipped + 1;
	}
	return pairOut;
}

template<int flip>
HK_FORCE_INLINE void hk1AxisSweep_arrayAppendPair(const hk1AxisSweep::AabbInt& aabb0, const hk1AxisSweep::AabbInt& aabb1, hkArray<hkKeyPair>& pairsOut)
{
	hkKeyPair& pairOut = pairsOut.expandOne();
	
	if ( !flip )
	{
		pairOut.m_keyA = aabb0.getKey();
		pairOut.m_keyB = aabb1.getKey();
	}
	else
	{
		pairOut.m_keyA = aabb1.getKey();
		pairOut.m_keyB = aabb0.getKey();
	}
}

template<int flipKeys>
HK_FORCE_INLINE void hk1AxisSweep_arrayScanList(const hk1AxisSweep::AabbInt& query,  const hk1AxisSweep::AabbInt* HK_RESTRICT sxyz, hkArray<hkKeyPair>& pairsOut)
{
#if !defined(HK_PLATFORM_SPU)
	hkUint32 maxX = query.m_max[0];
	while( sxyz->m_min[0] <= maxX )
	{
		int ov0 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[0] );
		int ov1 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[1] );
		int ov2 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[2] );
		int ov3 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[3] );

		if ( !((ov0&ov1) & (ov2&ov3)) )
		{
			if ( !ov0 )
			{
				hk1AxisSweep_arrayAppendPair<flipKeys>(query, sxyz[0], pairsOut);
			}
			if ( !ov1 && (sxyz[1].m_min[0] <= maxX) )
			{
				hk1AxisSweep_arrayAppendPair<flipKeys>(query, sxyz[1], pairsOut);
			}

			if ( !ov2 && (sxyz[2].m_min[0] <= maxX) )
			{
				hk1AxisSweep_arrayAppendPair<flipKeys>(query, sxyz[2], pairsOut);
			}
			if ( !ov3 && (sxyz[3].m_min[0] <= maxX) )
			{
				hk1AxisSweep_arrayAppendPair<flipKeys>(query, sxyz[3], pairsOut);
			}
		}

		sxyz += 4;
	}
#else
	hkUint32 maxX = query.m_max[0];
	while( sxyz->m_min[0] < maxX )
	{
		int ov0 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[0] );
		if ( !ov0 )
		{
			hk1AxisSweep_arrayAppendPair<flipKeys>(query, sxyz[0], pairsOut);
		}
		sxyz++;
	}
#endif
}



template<int flipKeys>
HK_FORCE_INLINE hkKeyPair* hk1AxisSweep_scanList(	const hk1AxisSweep::AabbInt& query,  const hk1AxisSweep::AabbInt* HK_RESTRICT sxyz, hkKeyPair* HK_RESTRICT pairsOut, const hkKeyPair* HK_RESTRICT end, hkPadSpu<int>& numPairsSkipped )
{
#if !defined(HK_PLATFORM_SPU)
	hkUint32 maxX = query.m_max[0];
	while( sxyz->m_min[0] <= maxX )
	{
		int ov0 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[0] );
		int ov1 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[1] );
		int ov2 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[2] );
		int ov3 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[3] );

		if ( !((ov0&ov1) & (ov2&ov3)) )
		{
			{
				if ( !ov0 )
				{
					pairsOut = hk1AxisSweep_appendPair<flipKeys>( query, sxyz[0], pairsOut, end, numPairsSkipped  );
				}
				if (!ov1 )
				{
					if ( sxyz[1].m_min[0] <= maxX )
					{
						pairsOut = hk1AxisSweep_appendPair<flipKeys>( query, sxyz[1], pairsOut, end, numPairsSkipped  );
					}
				}
			}

			{
				if ( !ov2 ) 
				{
					if ( sxyz[2].m_min[0] <= maxX )
					{
						pairsOut = hk1AxisSweep_appendPair<flipKeys>( query, sxyz[2], pairsOut, end, numPairsSkipped  );
					}
				}
				if ( !ov3 ) 
				{
					if ( sxyz[3].m_min[0] <= maxX )
					{
						pairsOut = hk1AxisSweep_appendPair<flipKeys>( query, sxyz[3], pairsOut, end, numPairsSkipped  );
					}
				}
			}
		}
		sxyz+=4;
	}
#else
	hkUint32 maxX = query.m_max[0];
	while( sxyz->m_min[0] < maxX )
	{
		int ov0 = hk1AxisSweep::AabbInt::yzDisjoint( query, sxyz[0] );
		if ( !ov0 )
		{
			pairsOut = hk1AxisSweep_appendPair<flipKeys>( query, sxyz[0], pairsOut, end, numPairsSkipped  );
		}
		sxyz++;
	}
#endif

	return pairsOut;
}


int HK_CALL hk1AxisSweep::collide( const hk1AxisSweep::AabbInt* pa, int numA, const hk1AxisSweep::AabbInt* pb, int numB, hkKeyPair* HK_RESTRICT pairsOut, int maxNumPairs, hkPadSpu<int>& numPairsSkipped)
{
#if defined(HK_DEBUG)
	HK_ASSERT2(0xad8750aa, numA == 0 || pa[numA-1].m_min[0] != hkUint32(-1), "numA should not include the padding elements at the end.");
	HK_ASSERT2(0xad8756aa, numB == 0 || pb[numB-1].m_min[0] != hkUint32(-1), "numB should not include the padding elements at the end.");
	// assert that the input lists are sorted
	{	for (int i =0 ; i < numA-1; i++){ HK_ASSERT( 0xf0341232, pa[i].m_min[0] <= pa[i+1].m_min[0]); }	}
	{	for (int i =0 ; i < numB-1; i++){ HK_ASSERT( 0xf0341233, pb[i].m_min[0] <= pb[i+1].m_min[0]); }	}
	{	for (int q =0; q < 4; q++ ){ 	HK_ASSERT2(0xad8757ab, pa[numA+q].m_min[0] == hkUint32(-1), "Four max-value padding elements are required at the end.");}	}
	{	for (int q =0; q < 4; q++ ){ 	HK_ASSERT2(0xad8757ab, pb[numB+q].m_min[0] == hkUint32(-1), "Four max-value padding elements are required at the end.");}	}
#endif

	hkKeyPair* end = pairsOut + maxNumPairs;
	hkKeyPair* HK_RESTRICT pairs = pairsOut;
	numPairsSkipped = 0;

	while ( true )
	{
		if ( pa->m_min[0] > pb->m_min[0] )
		{
			if ( numB-- <= 0 ) { break; }
			const bool flipKeys = true;
			pairs = hk1AxisSweep_scanList<flipKeys>( *pb, pa, pairs, end, numPairsSkipped );
			pb++;
		}
		else
		{
			if ( numA-- <= 0 ) { break; }
			const bool dontflipKeys = false;
			pairs = hk1AxisSweep_scanList<dontflipKeys>( *pa, pb, pairs, end, numPairsSkipped );
			pa++;
		}
	}
	return int(pairs - pairsOut);
}

void HK_CALL hk1AxisSweep::collide( const AabbInt* pa, int numA, const AabbInt* pb, int numB, hkArray<hkKeyPair>& pairsOut)
{
#if defined(HK_DEBUG)
	HK_ASSERT2(0xad8750aa, numA == 0 || pa[numA-1].m_min[0] != hkUint32(-1), "numA should not include the padding elements at the end.");
	HK_ASSERT2(0xad8756aa, numB == 0 || pb[numB-1].m_min[0] != hkUint32(-1), "numB should not include the padding elements at the end.");
	// assert that the input lists are sorted
	{	for (int i =0 ; i < numA-1; i++){ HK_ASSERT( 0xf0341232, pa[i].m_min[0] <= pa[i+1].m_min[0]); }	}
	{	for (int i =0 ; i < numB-1; i++){ HK_ASSERT( 0xf0341233, pb[i].m_min[0] <= pb[i+1].m_min[0]); }	}
	{	for (int q =0; q < 4; q++ ){ 	HK_ASSERT2(0xad8757ab, pa[numA+q].m_min[0] == hkUint32(-1), "Four max-value padding elements are required at the end.");}	}
	{	for (int q =0; q < 4; q++ ){ 	HK_ASSERT2(0xad8757ab, pb[numB+q].m_min[0] == hkUint32(-1), "Four max-value padding elements are required at the end.");}	}
#endif

	while ( true )
	{
		if ( pa->m_min[0] > pb->m_min[0] )
		{
			if ( numB-- <= 0 ) { break; }
			hk1AxisSweep_arrayScanList<true>(*pb, pa, pairsOut);
			pb++;
		}
		else
		{
			if ( numA-- <= 0 ) { break; }
			hk1AxisSweep_arrayScanList<false>(*pa, pb, pairsOut);
			pa++;
		}
	}
}

int HK_CALL hk1AxisSweep::collide( const hk1AxisSweep::AabbInt* pa, int numA, hkKeyPair* HK_RESTRICT pairsOut, int maxNumPairs, hkPadSpu<int>& numPairsSkipped)
{


#if defined(HK_DEBUG)
	HK_ASSERT2(0xad8751aa, numA == 0 || pa[numA-1].m_min[0] != hkUint32(-1), "numA should not include the padding elements at the end.");
	// assert that the input lists are sorted
	{	for (int i =0 ; i < numA-1; i++){ HK_ASSERT( 0xf0341234, pa[i].m_min[0] <= pa[i+1].m_min[0]); }	}
	{	for (int q =0; q < 4; q++ ){ 	HK_ASSERT2(0xad8757ab, pa[numA+q].m_min[0] == hkUint32(-1), "Four max-value padding elements are required at the end.");}	}
#endif

	hkKeyPair* end = pairsOut + maxNumPairs;
	hkKeyPair* HK_RESTRICT pairs = pairsOut;
	numPairsSkipped = 0;

	while ( --numA > 0 )	// this iterates numA-1
	{
		const bool dontflipKeys = false;
		pairs = hk1AxisSweep_scanList<dontflipKeys>( *pa, pa+1, pairs, end, numPairsSkipped );
		pa++;
	}
	return int(pairs - pairsOut);
}

void HK_CALL hk1AxisSweep::collide( const hk1AxisSweep::AabbInt* pa, int numA, hkArray<hkKeyPair>& pairsOut)
{
#if defined(HK_DEBUG)
	HK_ASSERT2(0xad8751aa, numA == 0 || pa[numA-1].m_min[0] != hkUint32(-1), "numA should not include the padding elements at the end.");
	// assert that the input lists are sorted
	{	for (int i =0 ; i < numA-1; i++){ HK_ASSERT( 0xf0341234, pa[i].m_min[0] <= pa[i+1].m_min[0]); }	}
	{	for (int q =0; q < 4; q++ ){ 	HK_ASSERT2(0xad8757ab, pa[numA+q].m_min[0] == hkUint32(-1), "Four max-value padding elements are required at the end.");}	}
#endif

	while ( --numA > 0 )	// this iterates numA-1
	{
		hk1AxisSweep_arrayScanList<false>(*pa, pa + 1, pairsOut);
		pa++;
	}
}

void HK_CALL hk1AxisSweep::sortAabbs(hk1AxisSweep::AabbInt* aabbs, int size)
{
	{
		// Make it multiple of 4
		// This is okay cos we know the AABBs array is padded with 4 extra entries
		int fixedSize = (size + 3) & ~3;

		hkArray<hkRadixSort::SortData32>::Temp sortArray(fixedSize);

		for (int i = 0; i < fixedSize; i++)
		{
			hkRadixSort::SortData32& entry = sortArray[i];

			entry.m_key = aabbs[i].m_min[0];
			entry.m_userData = i;
		}

		{
			hkArray<hkRadixSort::SortData32>::Temp buffer( fixedSize ) ;
			hkRadixSort::sort32(sortArray.begin(), fixedSize, buffer.begin());
		}

		hkArray<hk1AxisSweep::AabbInt>::Temp sortedAabbs(size);

		for (int i = 0; i < size; i++)
		{
			sortedAabbs[i] = aabbs[sortArray[i].m_userData];
		}

		// Copy back
		hkString::memCpy16(aabbs, sortedAabbs.begin(), size * sizeof(hk1AxisSweep::AabbInt)/16);
	}
}

HK_COMPILE_TIME_ASSERT(sizeof(hk1AxisSweep::AabbInt) >= sizeof(hkRadixSort::SortData32) );

void HK_CALL hk1AxisSweep::sortAabbs( AabbInt* aabbs, int size, hkArrayBase<hkRadixSort::SortData32>& sortArray, hkArrayBase<AabbInt>& sortedAabbs )
{
	{
		// Make it multiple of 4
		// This is okay cos we know the AABBs array is padded with 4 extra entries
		int fixedSize = HK_NEXT_MULTIPLE_OF(4,size);

		HK_ASSERT(0x3bb9cae0, sortArray.getSize() >= fixedSize );
		HK_ASSERT(0x3bb9cae0, sortedAabbs.getSize() >= fixedSize );
		
		for (int i = 0; i < fixedSize; i++)
		{
			hkRadixSort::SortData32& entry = sortArray[i];

			entry.m_key = aabbs[i].m_min[0];
			entry.m_userData = i;
		}

		{
			// We need a buffer of fixedSize hkRadixSort::SortData32's
			// The sortedAabbs is at least that big
			hkRadixSort::SortData32* buffer = reinterpret_cast<hkRadixSort::SortData32*> (sortedAabbs.begin());
			hkRadixSort::sort32(sortArray.begin(), fixedSize, buffer );
		}

		for (int i = 0; i < size; i++)
		{
			sortedAabbs[i] = aabbs[sortArray[i].m_userData];
		}

		// Copy back
		hkString::memCpy16(aabbs, sortedAabbs.begin(), size * sizeof(hk1AxisSweep::AabbInt)/16);
	}
}


HK_FORCE_INLINE hkUint32 hkRealToOrderedUint(const hkReal& in)
{
#if defined(HK_REAL_IS_DOUBLE) && HK_ENDIAN_LITTLE
	hkInt32 i = ((hkInt32*)&in)[1];
#else
	hkInt32 i = ((hkInt32*)&in)[0];
#endif
	return (hkUint32(i >> 31) | hkUint32(0x80000000)) ^ hkUint32(i);
}



#if ((HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && defined(HK_COMPILER_HAS_INTRINSICS_IA32))

// SSE2 integer code!
#include <emmintrin.h>

HK_ALIGN16( static hkUint32 simdSignBit[4]) = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
HK_ALIGN16( static hkUint32 simdOne[4]) =     { 0x00000001, 0x00000001, 0x00000001, 0x00000001 };

void hk1AxisSweep::AabbInt::set( const hkAabb& aabbIn, int key )
{
	const __m128i signBit = _mm_load_si128( (const __m128i*)simdSignBit);
	const __m128i one = _mm_load_si128( (const __m128i*)simdOne);

#if defined(HK_REAL_IS_DOUBLE)
#if HK_SSE_VERSION >= 0x50
	__m128 minXYZW = _mm256_cvtpd_ps(aabbIn.m_min.m_quad);
	__m128i min = _mm_castps_si128(minXYZW);
#else
	__m128 minXY = _mm_cvtpd_ps(aabbIn.m_min.m_quad.xy);
	__m128 minZW = _mm_cvtpd_ps(aabbIn.m_min.m_quad.zw);
	__m128 minXYZW = _mm_shuffle_ps(minXY,minZW,_MM_SHUFFLE(1,0,1,0));
	__m128i min = _mm_castps_si128(minXYZW);
#endif
#else
	__m128i min = _mm_load_si128( (const __m128i*)&aabbIn.m_min);
#endif
	min = _mm_xor_si128(_mm_or_si128(_mm_srai_epi32(min, 31), signBit), min);

#if defined(HK_REAL_IS_DOUBLE)
#if HK_SSE_VERSION >= 0x50
	__m128 maxXYZW = _mm256_cvtpd_ps(aabbIn.m_max.m_quad);
	__m128i max = _mm_castps_si128(maxXYZW);
#else
	__m128 maxXY = _mm_cvtpd_ps(aabbIn.m_max.m_quad.xy);
	__m128 maxZW = _mm_cvtpd_ps(aabbIn.m_max.m_quad.zw);
	__m128 maxXYZW = _mm_shuffle_ps(maxXY,maxZW,_MM_SHUFFLE(1,0,1,0));
	__m128i max = _mm_castps_si128(maxXYZW);
#endif
#else
	__m128i max = _mm_load_si128( (const __m128i*)&aabbIn.m_max);
#endif
	max = _mm_xor_si128(_mm_or_si128(_mm_srai_epi32(max, 31), signBit), max);

	// Shift down
	min = _mm_srli_epi32(min, 1);
	max = _mm_add_epi32(_mm_srli_epi32(max, 1), one);

	// Set the key
	min = _mm_insert_epi16(min, key, 6);
	min = _mm_insert_epi16(min, hkUint32(key) >> 16, 7);

	// Store the result
	_mm_store_si128((__m128i*) &m_min, min);
	_mm_store_si128((__m128i*) &m_max, max);
}

#else

void hk1AxisSweep::AabbInt::set( const hkAabb& aabbIn, int key )
{
	// I need the shift because the max Uint allowed is 0x7fffffff
	m_min[0] = hkRealToOrderedUint(aabbIn.m_min(0)) >> 1;
	m_min[1] = hkRealToOrderedUint(aabbIn.m_min(1)) >> 1;
	m_min[2] = hkRealToOrderedUint(aabbIn.m_min(2)) >> 1;
	getKey() = key;

	// I add one to make sure all have volume.
	m_max[0] = (hkRealToOrderedUint(aabbIn.m_max(0)) >> 1) + 1;
	m_max[1] = (hkRealToOrderedUint(aabbIn.m_max(1)) >> 1) + 1;
	m_max[2] = (hkRealToOrderedUint(aabbIn.m_max(2)) >> 1) + 1;
}

#endif

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
