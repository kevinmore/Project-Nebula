/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/GeometryUtilities/Misc/hkGeometryUtils.h>
#include <Common/Visualize/Shape/hkDisplayCapsule.h>

hkDisplayCapsule::hkDisplayCapsule( const hkVector4& top, const hkVector4& bottom, hkReal radius, int numSides , int numHeightSegments )
:  hkDisplayGeometry(HK_DISPLAY_CAPSULE), 
   m_numSides(numSides), m_numHeightSegments(numHeightSegments)
{
	m_top = top;
	m_bottom = bottom;
	m_radius = radius; 
}

void hkDisplayCapsule::buildGeometry()
{
	HK_ASSERT2(0x612ccd44, m_geometry==HK_NULL, "Already built");
	m_geometry = new hkGeometry();
	hkGeometryUtils::createCapsuleGeometry(m_top, m_bottom, m_radius, m_numHeightSegments, m_numSides, hkTransform::getIdentity(), *m_geometry);
}

void hkDisplayCapsule::getWireframeGeometry( hkArrayBase<hkVector4>& lines, hkMemoryAllocator& a )
{
	//TODO
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
