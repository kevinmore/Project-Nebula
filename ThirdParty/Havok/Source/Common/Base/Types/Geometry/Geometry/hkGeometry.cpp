/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>


/// Copy constructor. Required since hkArray's copy constructor is not public.
hkGeometry::hkGeometry (const hkGeometry& other)
: hkReferencedObject(other)
{
	m_vertices = other.m_vertices;
	m_triangles = other.m_triangles;
}

/// Finish constructor (for internal use).
hkGeometry::hkGeometry (hkFinishLoadedObjectFlag f)
: hkReferencedObject(f),
m_vertices(f),
m_triangles(f)
{

}

/// Clear content.
void hkGeometry::clear()
{
	m_triangles.clear();
	m_vertices.clear();
}

/// Check if data is numerically valid
hkBool hkGeometry::isValid() const
{
	// Vertices		
	const unsigned int numVertices = m_vertices.getSize();
	for (unsigned int i = 0; i < numVertices; ++i)
	{
		if (!m_vertices[i].isOk<3>())
		{
			return false;
		}
	}

	// Check that the triangle indices are in the range [0, numVertices - 1]
	for (int i = 0; i < m_triangles.getSize(); ++i)
	{
		const Triangle& triangle = m_triangles[i];
		if (static_cast<unsigned int>(triangle.m_a) >= numVertices ||
			static_cast<unsigned int>(triangle.m_b) >= numVertices ||
			static_cast<unsigned int>(triangle.m_c) >= numVertices)
		{
			return false;
		}
	}
	return true;
}

hkResult hkGeometry::appendGeometry(const hkGeometry& geometry, const hkMatrix4* transform)
{
	const int baseVertex = m_vertices.getSize(); 
	const int baseTriangle = m_triangles.getSize();	
	
	hkResult vertRes = m_vertices.reserve( baseVertex + geometry.m_vertices.getSize() );
	if (vertRes != HK_SUCCESS)
		return HK_FAILURE;
	hkResult triRes = m_triangles.reserve( baseTriangle + geometry.m_triangles.getSize() );
	if (triRes!= HK_SUCCESS)
		return HK_FAILURE;

	m_vertices.append(geometry.m_vertices);
	m_triangles.append(geometry.m_triangles);

	if(transform)
	{
		for (int i = baseVertex; i < m_vertices.getSize(); ++i)
		{
			hkVector4	v; transform->transformPosition(m_vertices[i], v);
			m_vertices[i].setXYZ(v);
		}
	}

	if(baseVertex > 0)
	{
		for (int i = baseTriangle; i < m_triangles.getSize(); ++i)
		{
			m_triangles[i].m_a	+=	baseVertex;
			m_triangles[i].m_b	+=	baseVertex;
			m_triangles[i].m_c	+=	baseVertex;
		}
	}

	return HK_SUCCESS;
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
