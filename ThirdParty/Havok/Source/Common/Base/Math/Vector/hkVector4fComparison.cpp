/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

// needed for getMaskForComponent
HK_COMPILE_TIME_ASSERT( hkVector4ComparisonMask::MASK_X < hkVector4ComparisonMask::MASK_Y );
HK_COMPILE_TIME_ASSERT( hkVector4ComparisonMask::MASK_Y < hkVector4ComparisonMask::MASK_Z );
HK_COMPILE_TIME_ASSERT( hkVector4ComparisonMask::MASK_Z < hkVector4ComparisonMask::MASK_W );

//                                                                       0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
HK_ALIGN16(const hkUint8 hkVector4Comparison_maskToFirstIndex[16]) = {   0,   0,   1,   0,   2,   0,   1,   0,   3,   0,   1,   0,   2,   0,   1,   0 };
HK_ALIGN16(const hkUint8 hkVector4Comparison_maskToLastIndex[16]) = {    0,   0,   1,   1,   2,   2,   2,   2,   3,   3,   3,   3,   3,   3,   3,   3 };
//                                                                          0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15

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
