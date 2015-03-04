/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#ifndef HK_CONVEX_HULL_H
#define HK_CONVEX_HULL_H

#include <Common/Internal/ConvexHull/Deprecated/hkGeomEdge.h>
enum neighbourDirection
{
	CLOCKWISE = -1,
	COUNTERCLOCKWISE = 1,
	NEIGHBOURDIRECTION_MAX
};


typedef hkUint16 hkGeomEdgeIndex;


/// This is an internal structure used by the convex hull algorithm.
class hkGeomHull
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_GEOMETRY, hkGeomHull );

		hkGeomHull();

		void initializeWithVertex( int vertexIndex );
		void initializeWithEdge  ( int v1, int v2 );
		void initializeWithTriangle( int v1, int v2, int v3 );


		hkBool isValidTopology();
		void visitAllNextAndMirrorEdges( hkGeomEdge* edge );

		//void simplifyTopology();

	public:
		// done implicitely by iterating over the connectivity, void getNeighbours( const hkVector4* vertex, hkArray<neighbourVertex>& neighbours );
		struct WrappingEdge
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_GEOMETRY, hkGeomHull::WrappingEdge );

			hkGeomEdgeIndex m_index[2];
		};

		// The convex hull is represented as a collection of vertices and edges; the
		// edges form a doubly-connected-edge-list.

		hkVector4* m_vertexBase;

		hkInplaceArray<hkGeomEdge,128> m_edges;
};

#endif //HK_CONVEX_HULL_H

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
