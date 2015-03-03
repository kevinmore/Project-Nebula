/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>
#include <Common/Base/Config/hkOptionalComponent.h>

void HK_CALL hkpSampledHeightField_registerAllRayCastFunctions()
{
	// Use coarse min max tree algorithm when available. Fall back to DDA algorithm otherwise.
	hkpSampledHeightFieldShape::s_rayCastFunc = &hkpSampledHeightFieldShape::castRayDefault; 
#ifndef HK_PLATFORM_SPU
	hkpSampledHeightFieldShape::s_sphereCastFunc = &hkpSampledHeightFieldShape::castSphereDefault; 
#endif
}
HK_OPTIONAL_COMPONENT_DEFINE_MANUAL(hkpSampledHeightField_AllCasts, hkpSampledHeightField_registerAllRayCastFunctions);

//
// Ray and sphere cast internal function pointer interface
//

void hkpSampledHeightFieldShape::castRayDefault( const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkpSampledHeightField_AllCasts);
	if (m_coarseness > 0 )
	{
		castRayCoarseTree( input, cdBody, collector );
	}
	else
	{
		castRayDda( input, cdBody, collector );
	}
}
#ifndef HK_PLATFORM_SPU
void hkpSampledHeightFieldShape::castSphereDefault( const hkpSphereCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkpSampledHeightField_AllCasts);
	if (m_coarseness > 0) 
	{
		castSphereCoarseTree( input, cdBody, collector );
	}
	else
	{
		castSphereDda( input, cdBody, collector );
	}
}
#endif

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
