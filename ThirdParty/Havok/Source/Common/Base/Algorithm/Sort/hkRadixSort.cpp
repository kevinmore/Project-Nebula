/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Sort/hkRadixSort.h>

#if HK_ENDIAN_BIG
	enum { HK_RADIX_SORT_KEY16_0 = 1, HK_RADIX_SORT_KEY16_1 = 0};
	enum { HK_RADIX_SORT_KEY32_0 = 3, HK_RADIX_SORT_KEY32_1 = 2, HK_RADIX_SORT_KEY32_2 = 1, HK_RADIX_SORT_KEY32_3 = 0};
#else
	enum { HK_RADIX_SORT_KEY16_0 = 0, HK_RADIX_SORT_KEY16_1 = 1};
	enum { HK_RADIX_SORT_KEY32_0 = 0, HK_RADIX_SORT_KEY32_1 = 1, HK_RADIX_SORT_KEY32_2 = 2, HK_RADIX_SORT_KEY32_3 = 3};
#endif

enum {
	HK_RADIX_SORT_NUM_LD_BITS = 8,
	HK_RADIX_SORT_NUM_TABLES = 256
};


void HK_CALL hkRadixSort::sort16( SortData16* data, int numObjects, SortData16* buffer )
{
	const int increment = 4;

	HK_ASSERT2( 0xf0e591df, (numObjects & (increment-1)) == 0, "You can only sort an array with a multiple of 4 size" );
	HK_ALIGN16(int table0  [HK_RADIX_SORT_NUM_TABLES]);
	HK_ALIGN16(int table1 [HK_RADIX_SORT_NUM_TABLES]);	
	{
		hkString::memClear16(table0,  HK_RADIX_SORT_NUM_TABLES * sizeof(int)/16 );
		hkString::memClear16(table1,  HK_RADIX_SORT_NUM_TABLES * sizeof(int)/16 );
	}


#define UPDATE_TABLE16( source, offset )	{	\
	int t0 = source[offset].m_keys[HK_RADIX_SORT_KEY16_0];		\
	int t1 = source[offset].m_keys[HK_RADIX_SORT_KEY16_1];		\
	int v0 = table0 [ t0 ];										\
	int v1 = table1[ t1 ];										\
	v0 += 1;	v1 += 1;										\
	table0 [t0]  = v0;											\
	table1 [t1]  = v1;											\
	}

	//
	// calculate the bucket size for each run
	//
	{
		const SortData16* HK_RESTRICT source = data;
		const SortData16* HK_RESTRICT dest  = buffer;

		// count num objects per table entry and prefetch data 
		for (int i =0; i < numObjects; i += increment )
		{
			if ( increment > 0 ) { UPDATE_TABLE16( source, 0 ); }		hkMath::prefetch128( hkAddByteOffsetConst(source,1024) );
			if ( increment > 1 ) { UPDATE_TABLE16( source, 1 );	}	hkMath::prefetch128( hkAddByteOffsetConst(dest,  1024) );
			if ( increment > 2 ) { UPDATE_TABLE16( source, 2 ); }
			if ( increment > 3 ) { UPDATE_TABLE16( source, 3 ); }
			source += increment;
			dest += increment;
		}
	}

	// distribute objects
	SortData16* c0[HK_RADIX_SORT_NUM_TABLES];
	SortData16* c1[HK_RADIX_SORT_NUM_TABLES];
	{
		SortData16* HK_RESTRICT source = data;
		SortData16* HK_RESTRICT dest   = buffer;
		c0[0] = dest;	c1[0] = source;
		for (int i = 1; i < HK_RADIX_SORT_NUM_TABLES; ++i)
		{
			c0[i] = c0[i-1] + table0[i-1];
			c1[i] = c1[i-1] + table1[i-1];
		}
	}

	// sort using lower bits
	{
		const SortData16* HK_RESTRICT source = data;
		for (int i =0; i < numObjects; i+=increment )
		{
			if ( increment > 0 ){ const int ti = (source[i+0].m_keys[HK_RADIX_SORT_KEY16_0]);	*(c0[ ti ]++) = source[i]; }
			if ( increment > 1 ){ const int ti = (source[i+1].m_keys[HK_RADIX_SORT_KEY16_0]);	*(c0[ ti ]++) = source[i+1]; }
			if ( increment > 2 ){ const int ti = (source[i+2].m_keys[HK_RADIX_SORT_KEY16_0]);	*(c0[ ti ]++) = source[i+2]; }
			if ( increment > 3 ){ const int ti = (source[i+3].m_keys[HK_RADIX_SORT_KEY16_0]);	*(c0[ ti ]++) = source[i+3]; }
		}
	}

	// sort using higher bits
	{
		const SortData16* HK_RESTRICT source = buffer;
		for (int i =0; i < numObjects; i+=increment )
		{
			if ( increment > 0 ){ const int ti = (source[i  ].m_keys[HK_RADIX_SORT_KEY16_1]);	*(c1[ ti ]++) = source[i]; }
			if ( increment > 1 ){ const int ti = (source[i+1].m_keys[HK_RADIX_SORT_KEY16_1]);	*(c1[ ti ]++) = source[i+1]; }
			if ( increment > 2 ){ const int ti = (source[i+2].m_keys[HK_RADIX_SORT_KEY16_1]);	*(c1[ ti ]++) = source[i+2]; }
			if ( increment > 3 ){ const int ti = (source[i+3].m_keys[HK_RADIX_SORT_KEY16_1]);	*(c1[ ti ]++) = source[i+3]; }
		}
	}

#ifdef HK_DEBUG
	// check
	for ( int i =0; i < numObjects-1; i++)
	{
		HK_ASSERT(0x23502af3, data[i].m_key <= data[i+1].m_key );
	}
#endif
}


void HK_CALL hkRadixSort::sort32( SortData32* data, int numObjects, SortData32* buffer )
{
	const int increment = 4;

	HK_ASSERT2( 0xf0e591de, (numObjects & (increment-1)) == 0, "You can only sort an array with a multiple of 4 size" );

	HK_ALIGN16(int table0 [HK_RADIX_SORT_NUM_TABLES]);
	HK_ALIGN16(int table1 [HK_RADIX_SORT_NUM_TABLES]);	
	HK_ALIGN16(int table2 [HK_RADIX_SORT_NUM_TABLES]);	
	HK_ALIGN16(int table3 [HK_RADIX_SORT_NUM_TABLES]);	
	{
		hkString::memClear16(table0,   HK_RADIX_SORT_NUM_TABLES * sizeof(int)/16 );
		hkString::memClear16(table1,  HK_RADIX_SORT_NUM_TABLES * sizeof(int)/16 );
		hkString::memClear16(table2,  HK_RADIX_SORT_NUM_TABLES * sizeof(int)/16 );
		hkString::memClear16(table3,  HK_RADIX_SORT_NUM_TABLES * sizeof(int)/16 );
	}


#define UPDATE_TABLE32( source, offset )	{	\
	int t0 = source[offset].m_keys[HK_RADIX_SORT_KEY32_0];		\
	int t1 = source[offset].m_keys[HK_RADIX_SORT_KEY32_1];		\
	int t2 = source[offset].m_keys[HK_RADIX_SORT_KEY32_2];		\
	int t3 = source[offset].m_keys[HK_RADIX_SORT_KEY32_3];		\
	int v0 = table0 [ t0 ];										\
	int v1 = table1 [ t1 ];										\
	int v2 = table2 [ t2 ];										\
	int v3 = table3 [ t3 ];										\
	v0 += 1;	v1 += 1; v2 += 1; v3 += 1;						\
	table0 [t0]  = v0;											\
	table1 [t1]  = v1;											\
	table2 [t2]  = v2;											\
	table3 [t3]  = v3;											\
	}

	//
	// calculate the bucket size for each run
	//
	{
		const SortData32* HK_RESTRICT source = data;
		const SortData32* HK_RESTRICT dest  = buffer;

		// count num objects per table entry and prefetch data 
		for (int i =0; i < numObjects; i += increment )
		{
			if ( increment > 0 ) { UPDATE_TABLE32( source, 0 );	}		hkMath::prefetch128( hkAddByteOffsetConst(source,1024) );
			if ( increment > 1 ) { UPDATE_TABLE32( source, 1 ); }		hkMath::prefetch128( hkAddByteOffsetConst(dest,  1024) );
			if ( increment > 2 ) { UPDATE_TABLE32( source, 2 ); }
			if ( increment > 3 ) { UPDATE_TABLE32( source, 3 ); }
			source += increment;
			dest += increment;
		}
	}

	// distribute objects
	SortData32* c0[HK_RADIX_SORT_NUM_TABLES];
	SortData32* c1[HK_RADIX_SORT_NUM_TABLES];
	SortData32* c2[HK_RADIX_SORT_NUM_TABLES];
	SortData32* c3[HK_RADIX_SORT_NUM_TABLES];
	{
		SortData32* HK_RESTRICT source = data;
		SortData32* HK_RESTRICT dest   = buffer;
		c0[0] = dest;	c1[0] = source; c2[0] = dest;	c3[0] = source;
		for (int i = 1; i < HK_RADIX_SORT_NUM_TABLES; ++i)
		{
			c0[i] = c0[i-1] + table0[i-1];
			c1[i] = c1[i-1] + table1[i-1];
			c2[i] = c2[i-1] + table2[i-1];
			c3[i] = c3[i-1] + table3[i-1];
		}
	}

	// sort using 0 bits
	{
		const SortData32* HK_RESTRICT source = data;
		for (int i =0; i < numObjects; i+=increment )
		{
			if ( increment > 0 ){ const int ti = (source[i+0].m_keys[HK_RADIX_SORT_KEY32_0]);	*(c0[ ti ]++) = source[i]; }
			if ( increment > 1 ){ const int ti = (source[i+1].m_keys[HK_RADIX_SORT_KEY32_0]);	*(c0[ ti ]++) = source[i+1]; }
			if ( increment > 2 ){ const int ti = (source[i+2].m_keys[HK_RADIX_SORT_KEY32_0]);	*(c0[ ti ]++) = source[i+2]; }
			if ( increment > 3 ){ const int ti = (source[i+3].m_keys[HK_RADIX_SORT_KEY32_0]);	*(c0[ ti ]++) = source[i+3]; }
		}
	}

	// sort using 1 bits
	{
		const SortData32* HK_RESTRICT source = buffer;
		for (int i =0; i < numObjects; i+=increment )
		{
			if ( increment > 0 ){ const int ti = (source[i  ].m_keys[HK_RADIX_SORT_KEY32_1]);	*(c1[ ti ]++) = source[i]; }
			if ( increment > 1 ){ const int ti = (source[i+1].m_keys[HK_RADIX_SORT_KEY32_1]);	*(c1[ ti ]++) = source[i+1]; }
			if ( increment > 2 ){ const int ti = (source[i+2].m_keys[HK_RADIX_SORT_KEY32_1]);	*(c1[ ti ]++) = source[i+2]; }
			if ( increment > 3 ){ const int ti = (source[i+3].m_keys[HK_RADIX_SORT_KEY32_1]);	*(c1[ ti ]++) = source[i+3]; }
		}
	}

	// sort using 2 bits
	{
		const SortData32* HK_RESTRICT source = data;
		for (int i =0; i < numObjects; i+=increment )
		{
			if ( increment > 0 ){ const int ti = (source[i  ].m_keys[HK_RADIX_SORT_KEY32_2]);	*(c2[ ti ]++) = source[i]; }
			if ( increment > 1 ){ const int ti = (source[i+1].m_keys[HK_RADIX_SORT_KEY32_2]);	*(c2[ ti ]++) = source[i+1]; }
			if ( increment > 2 ){ const int ti = (source[i+2].m_keys[HK_RADIX_SORT_KEY32_2]);	*(c2[ ti ]++) = source[i+2]; }
			if ( increment > 3 ){ const int ti = (source[i+3].m_keys[HK_RADIX_SORT_KEY32_2]);	*(c2[ ti ]++) = source[i+3]; }
		}
	}

	// sort using 3 bits
	{
		const SortData32* HK_RESTRICT source = buffer;
		for (int i =0; i < numObjects; i+=increment )
		{
			if ( increment > 0 ){ const int ti = (source[i  ].m_keys[HK_RADIX_SORT_KEY32_3]);	*(c3[ ti ]++) = source[i]; }
			if ( increment > 1 ){ const int ti = (source[i+1].m_keys[HK_RADIX_SORT_KEY32_3]);	*(c3[ ti ]++) = source[i+1]; }
			if ( increment > 2 ){ const int ti = (source[i+2].m_keys[HK_RADIX_SORT_KEY32_3]);	*(c3[ ti ]++) = source[i+2]; }
			if ( increment > 3 ){ const int ti = (source[i+3].m_keys[HK_RADIX_SORT_KEY32_3]);	*(c3[ ti ]++) = source[i+3]; }
		}
	}

#ifdef HK_DEBUG
	// check
	for ( int i =0; i < numObjects-1; i++)
	{
		HK_ASSERT(0x23502af3, data[i].m_key <= data[i+1].m_key );
	}
#endif
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
