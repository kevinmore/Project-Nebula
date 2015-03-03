/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_UTIL2_GEOM_EDGE_H
#define HK_UTIL2_GEOM_EDGE_H

class hkGeomTriangle;

#define HK_GEOM_EDGE_ALIGNMENT (4*sizeof(hkGeomEdge))
#define HK_GEOM_EDGE_MASK (4*sizeof(hkGeomEdge)-1)


/// This is an internal structure used by the convex hull algorithm.
class hkGeomEdge
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_GEOMETRY, hkGeomEdge );

		hkGeomEdge():m_info(0) 
		{
		}

		const hkVector4* getVertex( const hkVector4* vertexBase ) const
		{			
			return vertexBase + m_vertex;
		}

		hkGeomEdge* getMirror( hkGeomEdge* edgeBase ) const
		{
			return edgeBase + m_mirror;
		}

		hkGeomEdge* getNext( hkGeomEdge* edgeBase ) const
		{
			return edgeBase + m_next;
		}


	public:
		
		hkUint16	m_vertex;
		hkUint16	m_mirror;
		hkUint16	m_next;
		hkUint16    m_info:16;
};

#endif // HK_UTIL2_GEOM_EDGE_H

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
