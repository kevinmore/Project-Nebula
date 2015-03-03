/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Geometry/Aabb16/hkAabb16.h>
#include <Common/Base/Types/Geometry/IntSpaceUtil/hkIntSpaceUtil.h>

const hkQuadReal hkIntSpaceUtil::s_aabb16Max = HK_QUADREAL_CONSTANT(hkIntSpaceUtil::AABB_UINT16_MAX_VALUE,
																			 hkIntSpaceUtil::AABB_UINT16_MAX_VALUE,
																			 hkIntSpaceUtil::AABB_UINT16_MAX_VALUE,
																			 hkIntSpaceUtil::AABB_UINT16_MAX_VALUE);


void hkIntSpaceUtil::set( const hkAabb& aabb )
{
	m_aabb = aabb;
	hkVector4 span;		span.setSub( aabb.m_max, aabb.m_min);
	hkVector4 spanInv;
	spanInv.setReciprocal(span); 
	spanInv.setComponent<3>(hkSimdReal_1);
	hkSimdReal s; s.setFromFloat(1.0f/hkIntSpaceUtil::AABB_UINT16_MAX_VALUE);
	hkVector4 rounding; rounding.setMul( s, span);

	hkSimdReal v; v.setFromFloat(hkFloat32(hkIntSpaceUtil::AABB_UINT16_MAX_VALUE));
	m_bitScale.setMul( v, spanInv );
	m_bitOffsetLow.setNeg<4>( aabb.m_min );
	m_bitOffsetHigh.setAdd( m_bitOffsetLow, rounding );

	m_bitScale	   .zeroComponent<3>();
	m_bitOffsetLow .zeroComponent<3>();
	m_bitOffsetHigh.zeroComponent<3>();

	m_bitScaleInv.setReciprocal( m_bitScale );
	m_bitScaleInv.zeroComponent<3>();
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
