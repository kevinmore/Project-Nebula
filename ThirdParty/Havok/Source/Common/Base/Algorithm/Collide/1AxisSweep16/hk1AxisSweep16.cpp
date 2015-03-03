/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Collide/1AxisSweep16/hk1AxisSweep16.h>
#include <Common/Base/Algorithm/Sort/hkRadixSort.h>
#include <Common/Base/Math/Vector/hkIntVector.h>

template<int flip, typename T>
HK_FORCE_INLINE typename hk1AxisSweep16<T>::KeyPair* hk1AxisSweep_appendPair(const hkAabb16& aabb0, const hkAabb16& aabb1, 
													 typename hk1AxisSweep16<T>::KeyPair* HK_RESTRICT pairOut, 
													 const typename hk1AxisSweep16<T>::KeyPair* HK_RESTRICT end, hkPadSpu<int>& numPairsSkipped)
{
	{
		if ( pairOut < end)
		{
			if ( !flip )
			{
				aabb0.getKey(pairOut->m_keyA);
				aabb1.getKey(pairOut->m_keyB);
			}
			else
			{
				aabb1.getKey(pairOut->m_keyA);
				aabb0.getKey(pairOut->m_keyB);
			}
			pairOut++;
		}
		else
		{
			numPairsSkipped = numPairsSkipped + 1;
		}
	}

	return pairOut;
}

template<int flipKeys, typename T>
HK_FORCE_INLINE typename hk1AxisSweep16<T>::KeyPair* hk1AxisSweep_scanList(	const hkAabb16& query,  const hkAabb16* HK_RESTRICT sxyz, 
																			typename hk1AxisSweep16<T>::KeyPair* HK_RESTRICT pairsOut, 
																			const typename hk1AxisSweep16<T>::KeyPair* HK_RESTRICT end, hkPadSpu<int>& numPairsSkipped )
{
	hkUint32 maxX = query.m_max[0];
	while( sxyz->m_min[0] <= maxX )
	{
		int ov0 = hkAabb16::yzDisjoint( query, sxyz[0] );
		int ov1 = hkAabb16::yzDisjoint( query, sxyz[1] );
		int ov2 = hkAabb16::yzDisjoint( query, sxyz[2] );
		int ov3 = hkAabb16::yzDisjoint( query, sxyz[3] );

		if ( !((ov0&ov1) & (ov2&ov3)) )
		{
			{
				if ( !ov0 )
				{
					pairsOut = hk1AxisSweep_appendPair<flipKeys, T>( query, sxyz[0], pairsOut, end, numPairsSkipped  );
				}
				if (!ov1 )
				{
					if ( sxyz[1].m_min[0] <= maxX )
					{
						pairsOut = hk1AxisSweep_appendPair<flipKeys, T>( query, sxyz[1], pairsOut, end, numPairsSkipped  );
					}
				}
			}

			{
				if ( !ov2 ) 
				{
					if ( sxyz[2].m_min[0] <= maxX )
					{
						pairsOut = hk1AxisSweep_appendPair<flipKeys, T>( query, sxyz[2], pairsOut, end, numPairsSkipped  );
					}
				}
				if ( !ov3 ) 
				{
					if ( sxyz[3].m_min[0] <= maxX )
					{
						pairsOut = hk1AxisSweep_appendPair<flipKeys, T>( query, sxyz[3], pairsOut, end, numPairsSkipped  );
					}
				}
			}
		}
		sxyz+=4;
	}
	return pairsOut;
}


template<int flipKeys, typename T>
HK_FORCE_INLINE typename hk1AxisSweep16<T>::KeyPair* hk1AxisSweep_scanListPadding2(	const hkAabb16& query,  const hkAabb16* HK_RESTRICT sxyz, 
																					typename hk1AxisSweep16<T>::KeyPair* HK_RESTRICT pairsOut, 
																					const typename hk1AxisSweep16<T>::KeyPair* HK_RESTRICT end, hkPadSpu<int>& numPairsSkipped )
{
	hkUint32 maxX = query.m_max[0];
	while( sxyz->m_min[0] <= maxX )
	{
		int ov0 = hkAabb16::yzDisjoint( query, sxyz[0] );
		int ov1 = hkAabb16::yzDisjoint( query, sxyz[1] );

		if ( ! (ov0&ov1)  )
		{
			if ( !ov0 )
			{
				pairsOut = hk1AxisSweep_appendPair<flipKeys, T>( query, sxyz[0], pairsOut, end, numPairsSkipped  );
			}
			if (!ov1 )
			{
				if ( sxyz[1].m_min[0] <= maxX )
				{
					pairsOut = hk1AxisSweep_appendPair<flipKeys, T>( query, sxyz[1], pairsOut, end, numPairsSkipped  );
				}
			}
		}
		sxyz+=2;
	}
	return pairsOut;
}

template<int flipKeys, typename T>
HK_FORCE_INLINE typename hk1AxisSweep16<T>::KeyPair* hk1AxisSweep_scanListSIMD(
	const hkAabb16& query,
	const hkAabb16* HK_RESTRICT sxyz, const hkUint16* yzMin, const hkUint16* yzMax,
	typename hk1AxisSweep16<T>::KeyPair* HK_RESTRICT pairsOut,
	const typename hk1AxisSweep16<T>::KeyPair* end, hkPadSpu<int>& numPairsSkipped )
{
	hkIntVector qMi;
	hkIntVector qMa; 

	qMi.load<4>( (const hkUint32*)&query.m_min[0] );	// load min(x,y,z,key)  : max(x,y,z,key)
	qMi.setShiftLeft128<2>(qMi);						// -> miny, minz, key, maxx,     maxy, maxz, maxw, 0
	qMa.setBroadcast<2>( qMi );
	qMi.setBroadcast<0>( qMi );

	static HK_ALIGN16( hkUint32 ) andBits[4] = { 0x80008000, 0x80008000,0x80008000 ,0x80008000 };
	hkUint32 maxX = query.m_max[0];
	hkIntVector zero; zero.setZero();
	while( sxyz->m_min[0] <= maxX )
	{
		hkIntVector hMi; hMi.loadNotAligned<4>( (const hkUint32*)yzMin );
		hkIntVector hMa; hMa.loadNotAligned<4>( (const hkUint32*)yzMax );
		hMa.setSubU32( hMa, qMi );
		hMi.setSubU32( qMa, hMi );
		hMa.setOr( hMa, hMi );
		hMa.setAnd( hMa, (const hkIntVector&)andBits );
		hkVector4Comparison overlaps = hMa.compareEqualS32( zero );

		if ( overlaps.anyIsSet() )
		{
			hkVector4Comparison::Mask mask = overlaps.getMask();
			if ( mask & hkVector4ComparisonMask::MASK_X )
			{
				pairsOut = hk1AxisSweep_appendPair<flipKeys,T>( query, sxyz[0], pairsOut, end, numPairsSkipped  );
			}
			if ( mask & hkVector4ComparisonMask::MASK_Y )
			{
				if ( sxyz[1].m_min[0] <= maxX )
				{
					pairsOut = hk1AxisSweep_appendPair<flipKeys,T>( query, sxyz[1], pairsOut, end, numPairsSkipped  );
				}
			}
			if ( mask & hkVector4ComparisonMask::MASK_Z )
			{
				if ( sxyz[2].m_min[0] <= maxX )
				{
					pairsOut = hk1AxisSweep_appendPair<flipKeys,T>( query, sxyz[2], pairsOut, end, numPairsSkipped  );
				}
			}
			if ( mask & hkVector4ComparisonMask::MASK_W )
			{
				if ( sxyz[3].m_min[0] <= maxX )
				{
					pairsOut = hk1AxisSweep_appendPair<flipKeys,T>( query, sxyz[3], pairsOut, end, numPairsSkipped  );
				}
			}
		}
		yzMin += 4*2;
		yzMax += 4*2;
		sxyz+=4;
	}

	return pairsOut;
}

template<typename T>
int HK_CALL hk1AxisSweep16<T>::collide( const hkAabb16* pa, int numA, 
									KeyPair* HK_RESTRICT pairsOut, int maxNumPairs, hkPadSpu<int>& numPairsSkippedOut )
{
#ifdef HK_DEBUG
	HK_ASSERT2(0x5afbfde0, numA == 0 || pa[numA-1].m_min[0] != hkUint16(-1), "numA must not include the padding elements at the end.");
	for( int i=0; i<numA-1; i++){	HK_ASSERT(0x16bac266, pa[i].m_min[0] <= pa[i+1].m_min[0]);	}
	for (int q=0;    q < 4; q++ ) { HK_ASSERT2(0x159ffbac, pa[numA+q].m_min[0] == hkUint16(-1), "Four max-value padding elements are required at the end."); }
#endif


	KeyPair* end = pairsOut + maxNumPairs;
	KeyPair* HK_RESTRICT pairs = pairsOut;
	numPairsSkippedOut = 0;

	while ( --numA > 0 )	// this iterates numA-1
	{
		const bool dontflipKeys = false;
		pairs = hk1AxisSweep_scanList<dontflipKeys,T>( *pa, pa+1, pairs, end, numPairsSkippedOut );
		pa++;
	}
	return int(pairs - pairsOut);
}


template<typename T>
int HK_CALL hk1AxisSweep16<T>::collidePadding2( const hkAabb16* pa, int numA, 
									   KeyPair* HK_RESTRICT pairsOut, int maxNumPairs, hkPadSpu<int>& numPairsSkippedOut )
{
#ifdef HK_DEBUG
	HK_ASSERT2(0x5afbfde0, numA == 0 || pa[numA-1].m_min[0] != hkUint16(-1), "numA must not include the padding elements at the end.");
	for( int i=0; i<numA-1; i++){	HK_ASSERT(0x16bac266, pa[i].m_min[0] <= pa[i+1].m_min[0]);	}
	for (int q=0;    q < 2; q++ ) { HK_ASSERT2(0x159ffbac, pa[numA+q].m_min[0] == hkUint16(-1), "Two max-value padding elements are required at the end."); }
#endif


	KeyPair* end = pairsOut + maxNumPairs;
	KeyPair* HK_RESTRICT pairs = pairsOut;
	numPairsSkippedOut = 0;

	while ( --numA > 0 )	// this iterates numA-1
	{
		const bool dontflipKeys = false;
		pairs = hk1AxisSweep_scanListPadding2<dontflipKeys,T>( *pa, pa+1, pairs, end, numPairsSkippedOut );
		pa++;
	}
	return int(pairs - pairsOut);
}

template<typename T>
int HK_CALL hk1AxisSweep16<T>::collideSIMD( const hkAabb16* pa, int numA, void* buffer, int bufferSizeInBytes,
									   KeyPair* HK_RESTRICT pairsOut, int maxNumPairs, hkPadSpu<int>& numPairsSkippedOut )
{
#ifdef HK_DEBUG
	HK_ASSERT2(0x5afbfde0, numA == 0 || pa[numA-1].m_min[0] != hkUint16(-1), "numA must not include the padding elements at the end.");
	for( int i=0; i<numA-1; i++){	HK_ASSERT(0x16bac266, pa[i].m_min[0] <= pa[i+1].m_min[0]);	}
	for (int q=0;    q < 4; q++ ) { HK_ASSERT2(0x159ffbac, pa[numA+q].m_min[0] == hkUint16(-1), "Four max-value padding elements are required at the end."); }
#endif

#if !defined(HK_1AXIS_SWEEP_USE_SIMD)
	return collide( pa, numA, pairsOut, maxNumPairs, numPairsSkippedOut );
#else
	hkUint16* yzMin = (hkUint16*) buffer;
	hkUint16* yzMax = yzMin + HK_NEXT_MULTIPLE_OF(4,numA)*2 + 16;
	HK_ASSERT2( 0xf0ede454, hkAddByteOffset(buffer, bufferSizeInBytes) >= yzMax + HK_NEXT_MULTIPLE_OF(4,numA)*2 + 16, "Supplied buffer size is too small ");
	{
		int i=0;
		for (; i< numA; i+=4)
		{
			hkIntVector a; a.load<4>( (const hkUint32*)&pa[i] );
			hkIntVector b; b.load<4>( (const hkUint32*)&pa[i+1] );
			hkIntVector c; c.load<4>( (const hkUint32*)&pa[i+2] );
			hkIntVector d; d.load<4>( (const hkUint32*)&pa[i+3] );
			a.setShiftLeft128<2>(a);
			b.setShiftLeft128<2>(b);
			c.setShiftLeft128<2>(c);
			d.setShiftLeft128<2>(d);
			a.setPermutation<hkVectorPermutation::XZYW>( a );	// now we have min[xy] max[xy]  min[z] max[z]
			b.setPermutation<hkVectorPermutation::XZYW>( b );
			c.setPermutation<hkVectorPermutation::XZYW>( c );
			d.setPermutation<hkVectorPermutation::XZYW>( d );
			hkIntVector e; e.setMergeHead32( a, c );			// amin[xy], cmin[xy], amax[xy], cmax[xy]
			hkIntVector f; f.setMergeHead32( b, d );			// bmin[xy], dmin[xy], bmax[xy], dmax[xy]
			hkIntVector yzMi; yzMi.setMergeHead32( e, f );		// amin[xy], bmin[xy], cmin[xy], dmin[xy]
			hkIntVector yzMa; yzMa.setMergeTail32( e, f );	
			yzMi.store<4>( (hkUint32*) &yzMin[i*2] );
			yzMa.store<4>( (hkUint32*) &yzMax[i*2] );
		}
		static HK_ALIGN16(hkUint16) maxU16[8] = { 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff };
		hkIntVector maxI; maxI.load<4>( (const hkUint32*) maxU16 );
		hkIntVector zero; zero.setZero();
		zero.store<4>( (hkUint32*) &yzMax[i*2] );
		maxI.store<4>( (hkUint32*) &yzMin[i*2] );
	}

	KeyPair* end = pairsOut + maxNumPairs;
	KeyPair* HK_RESTRICT pairs = pairsOut;
	numPairsSkippedOut = 0;

	while ( --numA > 0 )	// this iterates numA-1
	{
		yzMin+=2;
		yzMax+=2;

		const bool dontflipKeys = false;
		pairs = hk1AxisSweep_scanListSIMD<dontflipKeys,T>( *pa, pa+1, yzMin, yzMax, pairs, end, numPairsSkippedOut );
		pa++;
	}
	return int(pairs - pairsOut);
#endif
}


template<typename T>
int HK_CALL hk1AxisSweep16<T>::collide( const hkAabb16* pa, int numA, const hkAabb16* pb, int numB, 
									KeyPair* HK_RESTRICT pairsOut, int maxNumPairs, hkPadSpu<int>& numPairsSkipped )
{
#if defined(HK_DEBUG)
	HK_ASSERT2(0x2df15ed4, numA == 0 || pa[numA-1].m_min[0] != hkUint16(-1), "numA should not include the padding elements at the end.");
	HK_ASSERT2(0x40bbd0f7, numB == 0 || pb[numB-1].m_min[0] != hkUint16(-1), "numA should not include the padding elements at the end.");

	for (int q=0; q < 4; q++ )
	{
		HK_ASSERT2(0x2739daed, pa[numA+q].m_min[0] == hkUint16(-1), "Four max-value padding elements are required at the end.");
		HK_ASSERT2(0x220f8ab,  pb[numB+q].m_min[0] == hkUint16(-1), "Four max-value padding elements are required at the end.");
	}
	// assert that the input lists are sorted
	{	for (int i =0 ; i < numA-1; i++){ HK_ASSERT(0x31c5128c, pa[i].m_min[0] <= pa[i+1].m_min[0]); }	}
	{	for (int i =0 ; i < numB-1; i++){ HK_ASSERT(0x5cbc1bc0, pb[i].m_min[0] <= pb[i+1].m_min[0]); }	}
#endif


	KeyPair* end = pairsOut + maxNumPairs;
	KeyPair* HK_RESTRICT pairs = pairsOut;
	numPairsSkipped = 0;

	while ( true )
	{
		if ( pa->m_min[0] > pb->m_min[0] )
		{
			const bool flipKeys = true;
			pairs = hk1AxisSweep_scanList<flipKeys,T>( *pb, pa, pairs, end, numPairsSkipped );
			pb++;
			if ( --numB <= 0 ) { break; }
		}
		else
		{
			const bool dontflipKeys = false;
			pairs = hk1AxisSweep_scanList<dontflipKeys,T>( *pa, pb, pairs, end, numPairsSkipped );
			pa++;
			if ( --numA <= 0 ) { break; }
		}
	}
	return int(pairs - pairsOut);
}


template<typename T>
int HK_CALL hk1AxisSweep16<T>::collidePadding2( const hkAabb16* pa, int numA, const hkAabb16* pb, int numB, 
									   KeyPair* HK_RESTRICT pairsOut, int maxNumPairs, hkPadSpu<int>& numPairsSkipped )
{
#if defined(HK_DEBUG)
	HK_ASSERT2(0x2df15ed4, numA == 0 || pa[numA-1].m_min[0] != hkUint16(-1), "numA should not include the padding elements at the end.");
	HK_ASSERT2(0x40bbd0f7, numB == 0 || pb[numB-1].m_min[0] != hkUint16(-1), "numA should not include the padding elements at the end.");
	HK_ASSERT(0x40bbd0f8, (numA + numB) != 0 || (pb[0].m_min[0] > pa[0].m_max[0]));	// Necessary to avoid correct behavior during the first call to hk1AxisSweep_scanListPadding2 

	for (int q=0; q < 2; q++ )
	{
		HK_ASSERT2(0x2739daed, pa[numA+q].m_min[0] == hkUint16(-1), "Four max-value padding elements are required at the end.");
		HK_ASSERT2(0x220f8ab,  pb[numB+q].m_min[0] == hkUint16(-1), "Four max-value padding elements are required at the end.");
	}
	// assert that the input lists are sorted
	{	for (int i =0 ; i < numA-1; i++){ HK_ASSERT(0x31c5128c, pa[i].m_min[0] <= pa[i+1].m_min[0]); }	}
	{	for (int i =0 ; i < numB-1; i++){ HK_ASSERT(0x5cbc1bc0, pb[i].m_min[0] <= pb[i+1].m_min[0]); }	}
#endif


	KeyPair* end = pairsOut + maxNumPairs;
	KeyPair* HK_RESTRICT pairs = pairsOut;
	numPairsSkipped = 0;

	while ( true )
	{
		if ( pa->m_min[0] > pb->m_min[0] )
		{
			const bool flipKeys = true;
			pairs = hk1AxisSweep_scanListPadding2<flipKeys,T>( *pb, pa, pairs, end, numPairsSkipped );
			pb++;
			if ( --numB <= 0 ) { break; }
		}
		else
		{
			const bool dontflipKeys = false;
			pairs = hk1AxisSweep_scanListPadding2<dontflipKeys,T>( *pa, pb, pairs, end, numPairsSkipped );
			pa++;
			if ( --numA <= 0 ) { break; }
		}
	}
	return int(pairs - pairsOut);
}

template<typename T>
int HK_CALL hk1AxisSweep16<T>::collideSIMD( const hkAabb16* pa, int numA, const hkAabb16* pb, int numB, void* buffer, int bufferSizeInBytes,
									   KeyPair* HK_RESTRICT pairsOut, int maxNumPairs, hkPadSpu<int>& numPairsSkipped )
{
#if defined(HK_DEBUG)
	HK_ASSERT2(0x2df15ed4, numA == 0 || pa[numA-1].m_min[0] != hkUint16(-1), "numA should not include the padding elements at the end.");
	HK_ASSERT2(0x40bbd0f7, numB == 0 || pb[numB-1].m_min[0] != hkUint16(-1), "numA should not include the padding elements at the end.");

	for (int q=0; q < 4; q++ )
	{
		HK_ASSERT2(0x2739daed, pa[numA+q].m_min[0] == hkUint16(-1), "Four max-value padding elements are required at the end.");
		HK_ASSERT2(0x220f8ab,  pb[numB+q].m_min[0] == hkUint16(-1), "Four max-value padding elements are required at the end.");
	}
	// assert that the input lists are sorted
	{	for (int i =0 ; i < numA-1; i++){ HK_ASSERT(0x31c5128c, pa[i].m_min[0] <= pa[i+1].m_min[0]); }	}
	{	for (int i =0 ; i < numB-1; i++){ HK_ASSERT(0x5cbc1bc0, pb[i].m_min[0] <= pb[i+1].m_min[0]); }	}
#endif

#if !defined(HK_1AXIS_SWEEP_USE_SIMD)
	return collide( pa, numA, pb, numB, pairsOut, maxNumPairs, numPairsSkipped );
#else
	KeyPair* end = pairsOut + maxNumPairs;
	KeyPair* HK_RESTRICT pairs = pairsOut;
	numPairsSkipped = 0;

	hkUint16* yzMinA = (hkUint16*) buffer;
	hkUint16* yzMaxA = yzMinA + numB+8;
	hkUint16* yzMinB = (hkUint16*) buffer;
	hkUint16* yzMaxB = yzMinA + numB+8;
	{
		for (int i=0; i< numA; i+=4)
		{
			hkIntVector a; a.load<4>( (const hkUint32*)&pa[i] );
			hkIntVector b; b.load<4>( (const hkUint32*)&pa[i+1] );
			hkIntVector c; c.load<4>( (const hkUint32*)&pa[i+2] );
			hkIntVector d; d.load<4>( (const hkUint32*)&pa[i+3] );
			a.setShiftLeft128<2>(a);
			b.setShiftLeft128<2>(b);
			c.setShiftLeft128<2>(c);
			d.setShiftLeft128<2>(d);
			a.setPermutation<hkVectorPermutation::XZYW>( a );	// now we have min[xy] max[xy]  min[z] max[z]
			b.setPermutation<hkVectorPermutation::XZYW>( b );
			c.setPermutation<hkVectorPermutation::XZYW>( c );
			d.setPermutation<hkVectorPermutation::XZYW>( d );
			hkIntVector e; e.setMergeHead32( a, c );			// amin[xy], cmin[xy], amax[xy], cmax[xy]
			hkIntVector f; f.setMergeHead32( b, d );			// bmin[xy], dmin[xy], bmax[xy], dmax[xy]
			hkIntVector yzMi; yzMi.setMergeHead32( e, f );		// amin[xy], bmin[xy], cmin[xy], dmin[xy]
			hkIntVector yzMa; yzMa.setMergeTail32( e, f );	
			yzMi.store<4>( (hkUint32*) &yzMinA[i*2] );
			yzMa.store<4>( (hkUint32*) &yzMaxA[i*2] );
		}
	}

	while ( true )
	{
		if ( pa->m_min[0] > pb->m_min[0] )
		{
			const bool flipKeys = true;
			pairs = hk1AxisSweep_scanListSIMD<flipKeys,T>( *pb, pa, yzMinA, yzMaxA, pairs, end, numPairsSkipped );
			pb++;
			yzMinA += 2;
			yzMaxA += 2;
			if ( --numB <= 0 ) { break; }
		}
		else
		{
			const bool dontflipKeys = false;
			pairs = hk1AxisSweep_scanListSIMD<dontflipKeys,T>( *pa, pb, yzMinB, yzMaxB, pairs, end, numPairsSkipped );
			pa++;
			yzMinB += 2;
			yzMaxB += 2;
			if ( --numA <= 0 ) { break; }
		}
	}
	return int(pairs - pairsOut);
#endif
}

template<typename T>
void HK_CALL hk1AxisSweep16<T>::sortAabbs(hkAabb16* aabbs, int size)
{
	int fixedSize = (size + 3) & ~3;
	hkArray<hkRadixSort::SortData16> sortBuffer(fixedSize << 1);
	hkArray<hkAabb16> tempAabbs(size);

	sortAabbs(aabbs, size, sortBuffer, tempAabbs);
}

template<typename T>
void HK_CALL hk1AxisSweep16<T>::sortAabbs(hkAabb16* aabbs, int size, hkArray<hkRadixSort::SortData16>& sortBuffer, hkArray<hkAabb16>& tempAabbs)
{
	// Make it multiple of 4
	// This is okay cos we know the aabbs array is padded with 4 extra entries
	int fixedSize = (size + 3) & ~3;
	HK_ASSERT(0x48d3f24e, size < 0xffff);
	HK_ASSERT(0x48d3f24e, sortBuffer.getSize() >= (fixedSize << 1));
	HK_ASSERT(0x48d3f24e, tempAabbs.getSize() >= size);

	hkRadixSort::SortData16* sortArray = &sortBuffer[0];

	for (int i = 0; i < fixedSize; i++)
	{
		hkRadixSort::SortData16& entry = sortArray[i];

		entry.m_key = (hkUint16)aabbs[i].m_min[0];
		entry.m_userData = (hkUint16)i;
	}
	for( int i=size; i<fixedSize; i++ )
	{
		hkRadixSort::SortData16& entry = sortArray[i];
		entry.m_key = 0xffff;
	}

	{
		hkRadixSort::SortData16* buffer = &sortBuffer[fixedSize];
		hkRadixSort::sort16(sortArray, fixedSize, buffer);
	}

	for (int i = 0; i < size; i++)
	{
		tempAabbs[i] = aabbs[sortArray[i].m_userData];
	}

	// Copy back
	hkString::memCpy16(aabbs, tempAabbs.begin(), size * (sizeof(hkAabb16)/16) );
}

template class hk1AxisSweep16< hkUint16 >;
template class hk1AxisSweep16< hkUint32 >;

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
