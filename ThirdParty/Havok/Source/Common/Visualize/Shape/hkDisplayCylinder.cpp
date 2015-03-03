/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Visualize/hkVisualize.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Visualize/Shape/hkDisplayCylinder.h>
#include <Common/GeometryUtilities/Misc/hkGeometryUtils.h>

hkDisplayCylinder::hkDisplayCylinder( const hkVector4& top, const hkVector4& bottom, hkReal radius, int numSides , int numHeightSegments )
:  hkDisplayGeometry(HK_DISPLAY_CYLINDER), 
   m_numSides(numSides), m_numHeightSegments(numHeightSegments)
{
	m_top = top;
	m_bottom = bottom;
	m_radius = radius; 
}

/*static hkBool isOutward( hkGeometry::Triangle& t, hkArray<hkVector4>& verts )
{
	hkVector4 e0; e0.setSub( verts[t.m_b], verts[t.m_a] );
	hkVector4 e1; e1.setSub( verts[t.m_c], verts[t.m_a] );
	hkVector4 c; c.setCross( e0, e1 );
	return c.dot<3>(verts[t.m_a]) > 0;	
}*/


void hkDisplayCylinder::buildGeometry()
{
	m_geometry = new hkGeometry;
	hkGeometryUtils::createCylinderGeometry(m_top, m_bottom, m_radius, m_numSides, m_numHeightSegments, *m_geometry);
}

void hkDisplayCylinder::getWireframeGeometry( hkArrayBase<hkVector4>& lines, hkMemoryAllocator& a )
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
