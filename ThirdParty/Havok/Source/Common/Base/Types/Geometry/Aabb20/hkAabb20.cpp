/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Geometry/Aabb20/hkAabb20.h>
#include <Common/Base/Math/Vector/hkIntVector.h>


#if 0
void hkAabb20::setExtents( const hkAabb20* aabbsIn, int numAabbsIn )
{
#if defined(HK_USING_GENERIC_INT_VECTOR_IMPLEMENTATION)
	{
		hkAabb20 aabbOut;	aabbOut.setEmpty();
		for(int i=0; i<numAabbsIn; i++)
		{
			aabbOut.includeAabb( aabbsIn[i] );
		}
		this[0] = aabbOut;
	}
#else
	hkIntVector vmin; vmin.load<4>( (const hkUint32*)aabbsIn );
	hkIntVector vmax = vmin;
	for (int i = 1; i < numAabbsIn; i++ )
	{
		hkIntVector a; a.load<4>( (const hkUint32*)&aabbsIn[i]  );
		vmin.setMinS16( vmin, a );
		vmax.setMaxS16( vmax, a );
	}
	vmax.store<4>( (hkUint32*)this);
	vmin.store<2>( (hkUint32*)this);
#endif
}


void hkAabb24_16_24::setIntersection( const hkAabb24_16_24& aabb0, const hkAabb24_16_24& aabb1  )
{
#if defined(HK_USING_GENERIC_INT_VECTOR_IMPLEMENTATION)
	{
		m_max[0] = hkMath::min2( aabb0.m_max[0], aabb1.m_max[0] );
		m_max[1] = hkMath::min2( aabb0.m_max[1], aabb1.m_max[1] );
		m_max[2] = hkMath::min2( aabb0.m_max[2], aabb1.m_max[2] );

		m_min[0] = hkMath::max2( aabb0.m_min[0], aabb1.m_min[0] );
		m_min[1] = hkMath::max2( aabb0.m_min[1], aabb1.m_min[1] );
		m_min[2] = hkMath::max2( aabb0.m_min[2], aabb1.m_min[2] );

	}
#else
	hkIntVector vmin; vmin.load<4>( (const hkUint32*)&aabb0 );
	hkIntVector vmax = vmin;
	hkIntVector a; a.load<4>( (const hkUint32*)&aabb1  );
	vmin.setMaxS16( vmin, a );
	vmax.setMinS16( vmax, a );
	vmax.store<4>( (hkUint32*)this);
	vmin.store<2>( (hkUint32*)this);
#endif
}


void hkAabb24_16_24::setExtentsOfCenters( const hkAabb24_16_24* aabbsIn, int numAabbsIn )
{
#if defined(HK_USING_GENERIC_INT_VECTOR_IMPLEMENTATION)
	{
		hkUint32 leafMax2[3], leafMin2[3];
		hkAabb24_16_24 aabbOut;	aabbOut.setEmpty();
		for(int ie=0; ie<numAabbs; ie++)
		{
			hkUint32 center2[3];
			aabbsIn[ie].getCenter2( center2 );
			aabbOut.includePoint( leafMax2, leafMin2, center2 );
		}
		this[0] = aabbOut;
	}
#else
	hkIntVector vmin; 
	{
		hkIntVector mi; mi.load<4>( (const hkUint32*)&aabbsIn[0]  );
		hkIntVector ma; ma.setPermutation<hkVectorPermutation::ZWWW>(mi);
		hkIntVector center2; center2.setAddSaturateU16( mi, ma );
		vmin.setShiftRight16<1>( center2);
	}
	hkIntVector vmax = vmin;
	for (int i = 1; i < numAabbsIn; i++ )
	{
		hkIntVector mi; mi.load<4>( (const hkUint32*)&aabbsIn[i]  );
		hkIntVector ma; ma.setPermutation<hkVectorPermutation::ZWWW>(mi);
		hkIntVector center2; center2.setAddSaturateU16( mi, ma );
		hkIntVector center; center.setShiftRight16<1>( center2);

		vmin.setMinS16( vmin, center );
		vmax.setMaxS16( vmax, center );
	}
	vmin.store<2>( ((hkUint32*)this) + 0 );
	vmax.store<2>( ((hkUint32*)this) + 2);
#endif
}

#endif

void hkAabb24_16_24_Codec::set( const hkAabb& aabb )
{
	hkVector4 extent; extent.setSub( aabb.m_max, aabb.m_min );
	int minComp = extent.getIndexOfMinComponent<3>();
	if ( minComp == 2 )
	{
		m_yzIsReversed.set<hkVector4ComparisonMask::MASK_XYZW>();
		m_aabb24_16_24_Max.set(	hkReal(0x7ffff0LL), hkReal(0x7ffff0LL), hkReal(0x7ff0LL), 1.0f ); 
	}
	else
	{
		m_aabb24_16_24_Max.set(	hkReal(0x7ffff0LL), hkReal(0x7ff0LL),	hkReal(0x7ffff0LL), 1.0f ); 
		m_yzIsReversed.set<hkVector4ComparisonMask::MASK_NONE>();
	}

	hkVector4 span;		span.setSub( aabb.m_max, aabb.m_min);
	hkVector4 spanInv;	spanInv.setReciprocal(span); 

	spanInv.setComponent<3>(hkSimdReal_1);
	hkVector4 s; s.setReciprocal(m_aabb24_16_24_Max);
	hkVector4 rounding; rounding.setMul( s, span );

	m_bitScale.setMul( m_aabb24_16_24_Max, spanInv );
	m_bitOffsetLow.setNeg<4>( aabb.m_min );
	m_bitOffsetHigh.setAdd( m_bitOffsetLow, rounding );

	m_bitScale	   .zeroComponent<3>();
	m_bitOffsetLow .zeroComponent<3>();
	m_bitOffsetHigh.zeroComponent<3>();

	m_bitScaleInv.setReciprocal( m_bitScale );
	m_bitScaleInv.zeroComponent<3>();
}


bool test( hkAabb24_16_24& a, hkAabb24_16_24& b)
{
	return a.disjoint(b);
}

void test2( const hkAabb24_16_24_Codec& codec, const hkAabb& aabbF, hkAabb24_16_24* aabbOut )
{
	codec.packAabb( aabbF, aabbOut );
}

void test3( const hkAabb24_16_24_Codec& codec, const hkAabb24_16_24& aabb, hkAabb* aabbF )
{
	codec.unpackAabbUnscaled( aabb, aabbF );
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
