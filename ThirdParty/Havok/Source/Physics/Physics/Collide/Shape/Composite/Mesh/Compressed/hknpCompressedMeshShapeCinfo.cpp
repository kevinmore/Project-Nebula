/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShapeCinfo.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>


hknpCompressedMeshShapeCinfo::hknpCompressedMeshShapeCinfo()
	: m_convexRadius( 0.0f ),
	  m_mergeCoplanarTriangles( true ),
	  m_mergeCoplanarTrianglesTolerance( 0.01f ),
	  m_preserveVertexOrder( false ),
	  m_flagConcaveTriangles( true ),
	  m_maxConvexShapeError( HK_REAL_MAX ),
	  m_optimizeForSpeed( false ),
	  m_triangleIndexToShapeKeyMap( HK_NULL ),
	  m_triangleIndexToVertexOrderMap( HK_NULL )
{

}

int	hknpDefaultCompressedMeshShapeCinfo::getNumVertices() const
{
	return m_geometry ? m_geometry->m_vertices.getSize() : 0;
}

int	hknpDefaultCompressedMeshShapeCinfo::getNumTriangles() const
{
	return m_geometry ? m_geometry->m_triangles.getSize() : 0;
}

void hknpDefaultCompressedMeshShapeCinfo::getVertex( int vi, hkVector4& vertexOut ) const
{
	vertexOut = m_geometry->m_vertices[vi];
}

void hknpDefaultCompressedMeshShapeCinfo::getIndices( int ti, int* indices ) const
{
	const hkGeometry::Triangle& t = m_geometry->m_triangles[ti];
	indices[0] = t.m_a; indices[1] = t.m_b; indices[2] = t.m_c;
}

hknpShapeTag hknpDefaultCompressedMeshShapeCinfo::getTriangleShapeTag(int triangleIndex) const
{
	return static_cast<hknpShapeTag>(m_geometry->m_triangles[triangleIndex].m_material);
}

int hknpDefaultCompressedMeshShapeCinfo::getNumConvexShapes() const
{
	return m_numConvexShapes;
}

const hknpShapeInstance& hknpDefaultCompressedMeshShapeCinfo::getConvexShape(int convexIndex) const
{
	HK_ASSERT2(0x742c3efc, convexIndex < m_numConvexShapes, "Convex index out of range");
	return m_convexShapes[convexIndex];
}

hknpShapeTag hknpDefaultCompressedMeshShapeCinfo::getConvexShapeTag(int convexIndex) const
{
	HK_ASSERT2(0x742c3efc, convexIndex < m_numConvexShapes, "Convex index out of range");
	return m_convexShapes[convexIndex].getShapeTag();
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
