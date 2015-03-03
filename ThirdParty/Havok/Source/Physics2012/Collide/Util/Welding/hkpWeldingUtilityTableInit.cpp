/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Util/Welding/hkpWeldingUtility.h>

//
// hkpWeldingUtility::getSector assumes the following:

HK_COMPILE_TIME_ASSERT(hkpWeldingUtility::ACCEPT_0	== 1);
HK_COMPILE_TIME_ASSERT(hkpWeldingUtility::SNAP_0	== 0);
HK_COMPILE_TIME_ASSERT(hkpWeldingUtility::REJECT	== 2);
HK_COMPILE_TIME_ASSERT(hkpWeldingUtility::SNAP_1	== 4);
HK_COMPILE_TIME_ASSERT(hkpWeldingUtility::ACCEPT_1	== 3);

HK_COMPILE_TIME_ASSERT(hkVector4ComparisonMask::MASK_X	== 1);
HK_COMPILE_TIME_ASSERT(hkVector4ComparisonMask::MASK_Y	== 2);
HK_COMPILE_TIME_ASSERT(hkVector4ComparisonMask::MASK_Z	== 4);
HK_COMPILE_TIME_ASSERT(hkVector4ComparisonMask::MASK_W	== 8);

HK_ALIGN16(hkpWeldingUtility::SinCosTableEntry hkpWeldingUtility::m_sinCosTable[hkpWeldingUtility::NUM_ANGLES+1]);

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
