/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/Shape/hkDisplayPlane.h>
#include <Common/GeometryUtilities/Misc/hkGeometryUtils.h>

hkDisplayPlane::hkDisplayPlane(const hkVector4& normal, const hkVector4& perpToNormal, 
							   const hkVector4& center, const hkVector4& extent)
							   :	hkDisplayGeometry(HK_DISPLAY_PLANE),
							   m_normal(normal),
							   m_center(center),
							   m_perpToNormal(perpToNormal),
							   m_extent(extent)
{

}

hkDisplayPlane::hkDisplayPlane()
:	hkDisplayGeometry(HK_DISPLAY_PLANE) 
{
	m_extent.setZero();
	m_normal.setZero();
	m_center.setZero();
	m_perpToNormal.setZero();
}

hkVector4& hkDisplayPlane::getNormal()
{
	return m_normal;
}

hkVector4& hkDisplayPlane::getCenter()
{
	return m_center;
}

hkVector4& hkDisplayPlane::getPerpToNormal()
{
	return m_perpToNormal;
}

hkVector4& hkDisplayPlane::getExtents()
{
	return m_extent;
}


void hkDisplayPlane::setParameters(const hkVector4& normal, const hkVector4& perpToNormal, 
								   const hkVector4& center, const hkVector4& extent)
{
	m_normal = normal;
	m_center = center;
	m_perpToNormal = perpToNormal;
	m_extent = extent;
}

void hkDisplayPlane::buildGeometry()
{
	// build triangle hkGeometry
	m_geometry = new hkGeometry;
	hkGeometryUtils::createPlaneGeometry( m_normal, m_perpToNormal, m_center, m_extent, *m_geometry);
}


void hkDisplayPlane::getWireframeGeometry( hkArrayBase<hkVector4>& lines, hkMemoryAllocator& a )
{
	lines._setSize( a, 12 );

	hkGeometry geometry;
	hkGeometryUtils::createPlaneGeometry( m_normal, m_perpToNormal, m_center, m_extent, geometry);
	
	lines[0] = geometry.m_vertices[0];
	lines[1] = geometry.m_vertices[1];

	lines[2] = geometry.m_vertices[1];
	lines[3] = geometry.m_vertices[2];

	lines[4] = geometry.m_vertices[2];
	lines[5] = geometry.m_vertices[3];

	lines[6] = geometry.m_vertices[3];
	lines[7] = geometry.m_vertices[0];

	lines[8] = geometry.m_vertices[0];
	lines[9] = geometry.m_vertices[2];

	lines[10] = geometry.m_vertices[1];
	lines[11] = geometry.m_vertices[3];
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
