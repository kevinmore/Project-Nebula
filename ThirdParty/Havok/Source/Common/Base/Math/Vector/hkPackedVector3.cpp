/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkPackedVector3.h>

void hkPackedVector3::pack( hkVector4fParameter vIn )
{

	//
	//	We set the .w component to a very tiny number, so that our horizontalMax4
	//  always returns a 'normal' number, even if vIn is zero
	//
	hkVector4f v; v.setXYZ_W( vIn, hkVector4f::getConstant( HK_QUADREAL_EPS_SQRD) );

	union { const hkUint32* i; hkIntVector* v; } i2v;
	union { const hkUint32* i; hkVector4f* v; } i2f;
	union { hkVector4f* f; hkIntVector* i; } f2i;
	union { hkVector4f* f; hkIntVector* i; } f2i2;

	//
	// we need to increase the max by the rounding done later to avoid an overflow
	//
	static HK_ALIGN16( const hkUint32 rounding[4] )					= { 0x3F800080, 0x3F800080, 0x3F800080, 0x3F800080 };
	i2f.i = rounding;
	hkVector4f vRounded; vRounded.setMul(v, *i2f.v );

	//
	// get the maximum absolute value3 and remove any mantissa bits
	//
	static HK_ALIGN16( const hkUint32 mask[4] )					= { 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000 };
	i2v.i = mask;
	hkVector4f ma;	
	f2i.f = &ma;
	f2i2.f = &vRounded;
	f2i.i->setAnd( *f2i2.i, *i2v.v );
	ma.setHorizontalMax<4>( ma );

	//
	// divide by maximum exponent
	//
	f2i.f = &ma;
	hkIntVector iMa = *f2i.i;

	static HK_ALIGN16( const hkUint32 offset[4] ) = { 0x4e800000, 0x4e800000, 0x4e800000, 0x4e800000 };
	i2v.i = offset;
	iMa.setSubU32( *i2v.v, iMa);	// calc the exponent

	hkVector4f correctedV; 
	f2i.f = &correctedV;
	f2i2.f = &v;
	f2i.i->setAddU32( *f2i2.i, iMa );	// divide by subtracting the exponent

	//
	// Convert to integer
	//
	hkIntVector result; result.setConvertF32toS32( correctedV );


	//
	//	Rounding correction
	//
	static HK_ALIGN16( const hkUint32 roundingCorrection[4] )	= { 0x8000, 0x8000, 0x8000, 0 };
	i2v.i = roundingCorrection;
	result.setAddU32( result, *i2v.v);

#if HK_ENDIAN_LITTLE == 1
	const int endianOffset = 1;
#else
	const int endianOffset = 0;
#endif
	m_values[0] = result.getU16<0+endianOffset>();
	m_values[1] = result.getU16<2+endianOffset>();
	m_values[2] = result.getU16<4+endianOffset>();
	m_values[3] = 0x3f80 - iMa.getU16<0+endianOffset>();	// 1.0f / correction 
}


void hkPackedVector3::pack( hkVector4dParameter vIn )
{
	//
	//	We set the .w component to a very tiny number, so that our horizontalMax4
	//  always returns a 'normal' number, even if vIn is zero
	//
	HK_ALIGN16(hkFloat32 v[4]);
	v[0] = hkFloat32(vIn(0));
	v[1] = hkFloat32(vIn(1));
	v[2] = hkFloat32(vIn(2));
	v[3] = hkFloat32(HK_FLOAT_EPSILON * HK_FLOAT_EPSILON);

	union { const hkUint32* i; hkIntVector* v; } i2v;
	union { const hkUint32* i; hkFloat32* v; } i2f;
	union { hkFloat32* f; hkIntVector* i; } f2i;
	union { hkFloat32* f; hkIntVector* i; } f2i2;

	//
	// we need to increase the max by the rounding done later to avoid an overflow
	//
	static HK_ALIGN16( const hkUint32 rounding[4] )					= { 0x3F800080, 0x3F800080, 0x3F800080, 0x3F800080 };
	i2f.i = rounding;
	HK_ALIGN16(hkFloat32 vRounded[4]); 
	vRounded[0] = v[0] * i2f.v[0];
	vRounded[1] = v[1] * i2f.v[1];
	vRounded[2] = v[2] * i2f.v[2];
	vRounded[3] = v[3] * i2f.v[3];

	//
	// get the maximum absolute value3 and remove any mantissa bits
	//
	static HK_ALIGN16( const hkUint32 mask[4] )					= { 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000 };
	i2v.i = mask;
	HK_ALIGN16(hkFloat32 ma[4]);	
	f2i.f = ma;
	f2i2.f = vRounded;
	f2i.i->setAnd( *f2i2.i, *i2v.v );
	{
		hkFloat32 h_ma = (ma[1] > ma[0]) ? ma[1] : ma[0];
		h_ma = (ma[2] > h_ma) ? ma[2] : h_ma;
		h_ma = (ma[3] > h_ma) ? ma[3] : h_ma;
		ma[0] = ma[1] = ma[2] = ma[3] = h_ma;
	}

	//
	// divide by maximum exponent
	//
	f2i.f = ma;
	hkIntVector iMa = *f2i.i;

	static HK_ALIGN16( const hkUint32 offset[4] ) = { 0x4e800000, 0x4e800000, 0x4e800000, 0x4e800000 };
	i2v.i = offset;
	iMa.setSubU32( *i2v.v, iMa);	// calc the exponent

	HK_ALIGN16(hkFloat32 correctedV[4]); 
	f2i.f = correctedV;
	f2i2.f = v;
	f2i.i->setAddU32( *f2i2.i, iMa );	// divide by subtracting the exponent

	//
	// Convert to integer
	//
	hkIntVector result;
#if defined(HK_COMPILER_HAS_INTRINSICS_IA32) && !(defined(HK_PLATFORM_LINUX)) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	{
		static HK_ALIGN16( const hkUint32 two31[4] )  = { 0x4F000000, 0x4F000000, 0x4F000000, 0x4F000000 }; // 2^31 as float

		__m128 cv = _mm_load_ps(correctedV);
		// Convert float to signed int, with AltiVec style overflow
		// (i.e. large float -> 0x7fffffff instead of 0x80000000)
		__m128 overflow = _mm_cmpge_ps( cv, *(__m128*)&two31);
		__m128i r = _mm_cvttps_epi32( cv ); // force round to zero like AltiVec
		result.m_quad = _mm_xor_si128( r, _mm_castps_si128(overflow) );
	}
#else
	result.set(hkMath::hkFloatToInt(correctedV[0]),hkMath::hkFloatToInt(correctedV[1]),hkMath::hkFloatToInt(correctedV[2]),hkMath::hkFloatToInt(correctedV[3]));
#endif

	//
	//	Rounding correction
	//
	static HK_ALIGN16( const hkUint32 roundingCorrection[4] )	= { 0x8000, 0x8000, 0x8000, 0 };
	i2v.i = roundingCorrection;
	result.setAddU32( result, *i2v.v);

#if HK_ENDIAN_LITTLE == 1
	const int endianOffset = 1;
#else
	const int endianOffset = 0;
#endif
	m_values[0] = result.getU16<0+endianOffset>();
	m_values[1] = result.getU16<2+endianOffset>();
	m_values[2] = result.getU16<4+endianOffset>();
	m_values[3] = 0x3f80 - iMa.getU16<0+endianOffset>();	// 1.0f / correction 
}


void hkPackedVector8_3::pack( hkVector4fParameter vIn )
{
	union { const hkUint32* i; hkIntVector* v; } i2v;

	//
	//	We set the .w component to a very tiny number, so that our horizontalMax4
	//  always returns a 'normal' number, even if vIn is zero
	//
	hkVector4f v; v.setXYZ_W( vIn, hkVector4f::getConstant<HK_QUADREAL_EPS_SQRD>() );

	union { const hkUint32* i; hkVector4f* v; } i2f;
	union { hkVector4f* f; hkIntVector* i; } f2i;
	union { hkVector4f* f; hkIntVector* i; } f2i2;

	//
	// we need to increase the max by the rounding done later to avoid an overflow
	//
	static HK_ALIGN16( const hkUint32 rounding[4] )					= { 0x3F808000, 0x3F808000, 0x3F808000, 0x3F808000 };
	i2f.i = rounding;
	hkVector4f vRounded; vRounded.setMul(v, *i2f.v );

	//
	// get the maximum absolute value3 and remove any mantissa bits
	//
	static HK_ALIGN16( const hkUint32 mask[4] )					= { 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000 };
	i2v.i = mask;
	hkVector4f ma;	
	f2i.f = &ma;
	f2i2.f = &vRounded;
	f2i.i->setAnd( *f2i2.i, *i2v.v );
	ma.setHorizontalMax<4>( ma );

	//
	// divide by maximum exponent
	//
	f2i.f = &ma;
	hkIntVector iMa = *f2i.i;

	static HK_ALIGN16( const hkUint32 offset[4] ) = { 0x4e800000, 0x4e800000, 0x4e800000, 0x4e800000 };
	i2v.i = offset;
	iMa.setSubU32( *i2v.v, iMa);	// calc the exponent

	hkVector4f correctedV; 
	f2i.f = &correctedV;
	f2i2.f = &v;
	f2i.i->setAddU32( *f2i2.i, iMa );	// divide by subtracting the exponent

	//
	// Convert to integer
	//
	hkIntVector result; result.setConvertF32toS32( correctedV );

	//
	//	Rounding correction
	//
	static HK_ALIGN16( const hkUint32 roundingCorrection[4] )	= { 0x800000, 0x800000, 0x800000, 0 };
	i2v.i = roundingCorrection;
	result.setAddU32( result, *i2v.v);

#if HK_ENDIAN_LITTLE == 1
	const int endianOffset8  = 3;
	const int endianOffset16 = 1;
#else
	const int endianOffset8 = 0;
	const int endianOffset16 = 0;
#endif
	m_values[0] = result.getU8<0+endianOffset8>();
	m_values[1] = result.getU8<4+endianOffset8>();
	m_values[2] = result.getU8<8+endianOffset8>();
	m_values[3] = (hkUint8)((0x3f80 - iMa.getU16<0+endianOffset16>())>>7);	// 1.0f / correction 
}



void hkPackedVector8_3::pack( hkVector4dParameter vIn )
{
	union { const hkUint32* i; hkIntVector* v; } i2v;

	HK_ALIGN_DOUBLE(hkFloat32 fIn[4]);
	hkUint32* iIn = (hkUint32*)fIn;
	HK_ALIGN_DOUBLE(hkFloat32 fRounded[3]);
	hkUint32* iRounded = (hkUint32*)fRounded;
	HK_ALIGN_DOUBLE(hkFloat32 fCorrected[3]);
	hkUint32* iCorrected = (hkUint32*)fCorrected;

	vIn.store<4>(&fIn[0]);

	const hkUint32 rounding = 0x3F808000;
	const hkUint32 mask     = 0x7f800000;
	const hkUint32 offset   = 0x4e800000;
	const hkFloat32 eps2    = 1.192092896e-07F * 1.192092896e-07F;

	fRounded[0] = *((hkFloat32*)&rounding) * fIn[0] ;
	fRounded[1] = *((hkFloat32*)&rounding) * fIn[1] ;
	fRounded[2] = *((hkFloat32*)&rounding) * fIn[2] ;

	iRounded[0] &= mask;
	iRounded[1] &= mask;
	iRounded[2] &= mask;

	hkFloat32 fMax = hkMath::max2(fRounded[0], fRounded[1]); fMax = hkMath::max2(fRounded[2], fMax); fMax = hkMath::max2(eps2, fMax);

	hkUint32 iMax = offset - *((hkUint32*)&fMax);
	hkIntVector iMa; iMa.setAll(iMax);

	iCorrected[0] = iIn[0] + iMax;
	iCorrected[1] = iIn[1] + iMax;
	iCorrected[2] = iIn[2] + iMax;

	hkIntVector result;
	result.set(	hkMath::hkToIntFast(fCorrected[0]),
				hkMath::hkToIntFast(fCorrected[1]),
				hkMath::hkToIntFast(fCorrected[2]),
				iMax);



	//
	//	Rounding correction
	//
	static HK_ALIGN16( const hkUint32 roundingCorrection[4] )	= { 0x800000, 0x800000, 0x800000, 0 };
	i2v.i = roundingCorrection;
	result.setAddU32( result, *i2v.v);

#if HK_ENDIAN_LITTLE == 1
	const int endianOffset8  = 3;
	const int endianOffset16 = 1;
#else
	const int endianOffset8 = 0;
	const int endianOffset16 = 0;
#endif
	m_values[0] = result.getU8<0+endianOffset8>();
	m_values[1] = result.getU8<4+endianOffset8>();
	m_values[2] = result.getU8<8+endianOffset8>();
	m_values[3] = (hkUint8)((0x3f80 - iMa.getU16<0+endianOffset16>())>>7);	// 1.0f / correction 
}


HK_ALIGN16( const hkUint32 hkPackedUnitVector_m_offset[4] ) = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };

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
