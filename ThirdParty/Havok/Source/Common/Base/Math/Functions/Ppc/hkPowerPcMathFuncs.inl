/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

namespace hkMath
{
#define HK_MATH_sqrt
	HK_FORCE_INLINE static hkFloat32 HK_CALL sqrt(const hkFloat32 r) 
	{ 
		return __fsqrts(r);
	}
	HK_FORCE_INLINE static hkDouble64 HK_CALL sqrt(const hkDouble64 r) 
	{ 
#ifdef HK_COMPILER_SNC
		return ::sqrt(r);
#else
		return __fsqrt(r);
#endif
	}

#if defined(HK_PLATFORM_PS3_PPU)
#define HK_MATH_fabs
	HK_FORCE_INLINE static hkFloat32 HK_CALL fabs(const hkFloat32 r) 
	{ 
		return __fabsf(r);
	}
	HK_FORCE_INLINE static hkDouble64 HK_CALL fabs(const hkDouble64 r) 
	{ 
		return __fabs(r);
	}
	// on x360 the normal fabs is the fast one (taken care of by ppcintrinsics.h)
#endif


#ifdef HK_PLATFORM_XBOX360
#	define __hkFselFloat  __fself
#else
#	define __hkFselFloat  __fsels
#endif
#define __hkFselDouble __fsel

#define HK_MATH_min2
	template <typename T1, typename T2>
	HK_FORCE_INLINE static T1 HK_CALL min2( T1 x, T2 y)
	{
		return x < T1(y) ? x : T1(y);
	}
	template <>
	HK_FORCE_INLINE static hkFloat32 HK_CALL min2<hkFloat32, hkFloat32>( hkFloat32 x, hkFloat32 y)
	{
		return __hkFselFloat( x - y , y , x);
	}
	template <>
	HK_FORCE_INLINE static hkDouble64 HK_CALL min2<hkDouble64, hkDouble64>( hkDouble64 x, hkDouble64 y)
	{
		return __hkFselDouble( x - y , y , x);
	}

#define HK_MATH_max2
	template <typename T1, typename T2>
	HK_FORCE_INLINE static T1 HK_CALL max2( T1 x, T2 y)
	{
		return x > T1(y) ? x : T1(y);
	}
	template <>
	HK_FORCE_INLINE static hkFloat32 HK_CALL max2<hkFloat32, hkFloat32>( hkFloat32 x, hkFloat32 y)
	{
		return __hkFselFloat( x - y , x , y);
	}
	template <>
	HK_FORCE_INLINE static hkDouble64 HK_CALL max2<hkDouble64, hkDouble64>( hkDouble64 x, hkDouble64 y)
	{
		return __hkFselDouble( x - y , x , y);
	}

#define HK_MATH_clamp
	template <typename T1, typename T2, typename T3>
	HK_FORCE_INLINE static T1 HK_CALL clamp( T1 x, T2 mi, T3 ma)
	{
		if ( x < mi ) return (T1) mi;
		if ( x > ma ) return (T1) ma;
		return x;
	}
	template <>
	HK_FORCE_INLINE static hkFloat32 HK_CALL clamp<hkFloat32, hkFloat32, hkFloat32>( hkFloat32 x, hkFloat32 mi, hkFloat32 ma)
	{
		x = max2<hkFloat32>(x, mi);
		x = min2<hkFloat32>(x, ma);
		return x;
	}
	template <>
	HK_FORCE_INLINE static hkDouble64 HK_CALL clamp<hkDouble64, hkDouble64, hkDouble64>( hkDouble64 x, hkDouble64 mi, hkDouble64 ma)
	{
		x = max2<hkDouble64>(x, mi);
		x = min2<hkDouble64>(x, ma);
		return x;
	}

#define HK_MATH_prefetch128
	HK_FORCE_INLINE static void HK_CALL prefetch128( const void* p)
	{
#		if defined(HK_PLATFORM_PS3_SPU)
#		elif defined(HK_PLATFORM_PS3_PPU)
		__dcbt(p);
#		else			// BOX360
		__dcbt(0, p);
#		endif
	}

#define HK_MATH_forcePrefetch
	template<int SIZE>
	HK_FORCE_INLINE static void HK_CALL forcePrefetch( const void* p )
	{
#		if defined(HK_PLATFORM_PS3_SPU)
#		elif defined(HK_PLATFORM_PS3_PPU)
		__dcbt(p);
		if ( SIZE > 128){ __dcbt(hkAddByteOffsetConst(p,128)); }
		if ( SIZE > 256){ __dcbt(hkAddByteOffsetConst(p,256)); }
		if ( SIZE > 384){ __dcbt(hkAddByteOffsetConst(p,384)); }
#		else	// XBOX360
		__dcbt(0, p);
		if ( SIZE > 128){ __dcbt(128, p); }
		if ( SIZE > 256){ __dcbt(256, p); }
		if ( SIZE > 384){ __dcbt(384, p); }
#		endif
	}

#if defined(HK_PLATFORM_XBOX360)

#define HK_MATH_fselectGreaterEqualZero
	HK_FORCE_INLINE static hkFloat32 HK_CALL fselectGreaterEqualZero( hkFloat32 testVar, hkFloat32 ifTrue, hkFloat32 ifFalse)
	{
		return __hkFselFloat(testVar, ifTrue, ifFalse);
	}
	HK_FORCE_INLINE static hkDouble64 HK_CALL fselectGreaterEqualZero( hkDouble64 testVar, hkDouble64 ifTrue, hkDouble64 ifFalse)
	{
		return __hkFselDouble(testVar, ifTrue, ifFalse);
	}

#define HK_MATH_fselectGreaterZero
	HK_FORCE_INLINE static hkFloat32 HK_CALL fselectGreaterZero( hkFloat32 testVar, hkFloat32 ifTrue, hkFloat32 ifFalse)
	{ 
		return __hkFselFloat(-testVar, ifFalse, ifTrue);
	}
	HK_FORCE_INLINE static hkDouble64 HK_CALL fselectGreaterZero( hkDouble64 testVar, hkDouble64 ifTrue, hkDouble64 ifFalse)
	{ 
		return __hkFselDouble(-testVar, ifFalse, ifTrue);
	}

#define HK_MATH_fselectLessEqualZero
	HK_FORCE_INLINE static hkFloat32 HK_CALL fselectLessEqualZero( hkFloat32 testVar, hkFloat32 ifTrue, hkFloat32 ifFalse)
	{
		return __hkFselFloat(-testVar, ifTrue, ifFalse);
	}
	HK_FORCE_INLINE static hkDouble64 HK_CALL fselectLessEqualZero( hkDouble64 testVar, hkDouble64 ifTrue, hkDouble64 ifFalse)
	{
		return __hkFselDouble(-testVar, ifTrue, ifFalse);
	}

#define HK_MATH_fselectLessZero
	HK_FORCE_INLINE static hkFloat32 HK_CALL fselectLessZero( hkFloat32 testVar, hkFloat32 ifTrue, hkFloat32 ifFalse)
	{
		return __hkFselFloat(testVar, ifFalse, ifTrue);
	}
	HK_FORCE_INLINE static hkDouble64 HK_CALL fselectLessZero( hkDouble64 testVar, hkDouble64 ifTrue, hkDouble64 ifFalse)
	{
		return __hkFselDouble(testVar, ifFalse, ifTrue);
	}

#define HK_MATH_fselectEqualZero
	HK_FORCE_INLINE static hkFloat32 HK_CALL fselectEqualZero( hkFloat32 testVar, hkFloat32 ifTrue, hkFloat32 ifFalse)
	{ 
		return __hkFselFloat( -testVar, __hkFselFloat(testVar, ifTrue, ifFalse), ifFalse );
	}
	HK_FORCE_INLINE static hkDouble64 HK_CALL fselectEqualZero( hkDouble64 testVar, hkDouble64 ifTrue, hkDouble64 ifFalse)
	{ 
		return __hkFselDouble( -testVar, __hkFselDouble(testVar, ifTrue, ifFalse), ifFalse );
	}

#define HK_MATH_fselectNotEqualZero
	HK_FORCE_INLINE static hkFloat32 HK_CALL fselectNotEqualZero( hkFloat32 testVar, hkFloat32 ifTrue, hkFloat32 ifFalse)
	{ 
		return __hkFselFloat( -testVar, __hkFselFloat(testVar, ifFalse, ifTrue), ifTrue );
	}
	HK_FORCE_INLINE static hkDouble64 HK_CALL fselectNotEqualZero( hkDouble64 testVar, hkDouble64 ifTrue, hkDouble64 ifFalse)
	{ 
		return __hkFselDouble( -testVar, __hkFselDouble(testVar, ifFalse, ifTrue), ifTrue );
	}

#endif

#define HK_MATH_countLeadingZeros
	template <typename T>
	HK_FORCE_INLINE static int HK_CALL countLeadingZeros(T x);

	template <>
	HK_FORCE_INLINE static int HK_CALL countLeadingZeros<hkUint32>(hkUint32 x)
	{
#		if defined (HK_PLATFORM_PS3_PPU)
		return  __cntlzw(x);
#		else	// XBOX360
		return _CountLeadingZeros(x);
#		endif
	}

	template <>
	HK_FORCE_INLINE static int HK_CALL countLeadingZeros<int>(int x)	{	return countLeadingZeros<hkUint32>((hkUint32)x);	}

	template <>
	HK_FORCE_INLINE static int HK_CALL countLeadingZeros<hkUint64>(hkUint64 x)
	{
#		if defined (HK_PLATFORM_PS3_PPU)
		const int loBits = countLeadingZeros<hkUint32>(x & 0xFFFFFFFF);
		const int hiBits = countLeadingZeros<hkUint32>((x >> 32L) & 0xFFFFFFFF);
		return (hiBits < 32) ? hiBits : (32 + loBits);
#	else	// XBOX360
		return _CountLeadingZeros64(x);
#	endif
	}


#undef __hkFselFloat
#undef __hkFselDouble
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
