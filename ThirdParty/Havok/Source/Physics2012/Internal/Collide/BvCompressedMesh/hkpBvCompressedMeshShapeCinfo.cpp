/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Internal/hkpInternal.h>
#include <Physics2012/Internal/Collide/BvCompressedMesh/hkpBvCompressedMeshShapeCinfo.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>


hkpBvCompressedMeshShapeCinfo::hkpBvCompressedMeshShapeCinfo()
:	m_collisionFilterInfoMode( hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_NONE )
,	m_userDataMode( hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_NONE )
,	m_convexRadius( 0.0f )
,	m_vertexWeldingTolerance( 0.0f )
,	m_maxConvexShapeError( HK_REAL_MAX )
,	m_maxVerticesError( HK_REAL_MAX )
,	m_weldingType( hkpWeldingUtility::WELDING_TYPE_NONE )
,	m_weldOpenEdges( true )
,	m_preserveVertexOrder( false )
,	m_triangleIndexToShapeKeyMap( HK_NULL )
,	m_convexShapeIndexToShapeKeyMap( HK_NULL )
{}


hkpDefaultBvCompressedMeshShapeCinfo::~hkpDefaultBvCompressedMeshShapeCinfo()
{
	// Remove convex shape references
	for( int i=0; i<m_shapes.getSize(); ++i )
	{
		if( m_shapes[i].m_shape )
		{
			m_shapes[i].m_shape->removeReference();
			m_shapes[i].m_shape = HK_NULL;
		}
	}
}

int	hkpDefaultBvCompressedMeshShapeCinfo::getNumVertices() const
{
	return m_geometry ? m_geometry->m_vertices.getSize() : 0;
}

int	hkpDefaultBvCompressedMeshShapeCinfo::getNumTriangles() const
{
	return m_geometry ? m_geometry->m_triangles.getSize() : 0;
}

void hkpDefaultBvCompressedMeshShapeCinfo::getVertex( int vi, hkVector4& vertexOut ) const
{
	vertexOut = m_geometry->m_vertices[vi];
}

void hkpDefaultBvCompressedMeshShapeCinfo::getIndices( int ti, int* indices ) const
{
	const hkGeometry::Triangle& t = m_geometry->m_triangles[ti];
	indices[0] = t.m_a; indices[1] = t.m_b; indices[2] = t.m_c;
}

hkUint32 hkpDefaultBvCompressedMeshShapeCinfo::getTriangleUserData(int triangleIndex) const
{
	return m_geometry->m_triangles[triangleIndex].m_material;
}

int	hkpDefaultBvCompressedMeshShapeCinfo::getNumConvexShapes() const
{
	return m_shapes.getSize();
}

void hkpDefaultBvCompressedMeshShapeCinfo::getConvexShape( int chi, const hkpConvexShape*& convexShapeOut, hkQsTransform& transformOut ) const
{
	convexShapeOut = m_shapes[chi].m_shape;
	transformOut = m_shapes[chi].m_transform;
}

hkUint32 hkpDefaultBvCompressedMeshShapeCinfo::getConvexShapeUserData(int convexIndex) const 
{ 
	return (hkUint32)(m_shapes[convexIndex].m_shape->getUserData());
}

void hkpDefaultBvCompressedMeshShapeCinfo::addConvexShape(const hkpConvexShape* shape, const hkQsTransform& transform)
{
	HK_ON_DEBUG( hkVector4 scaleX; scaleX.setAll(transform.getScale().getComponent<0>()); const hkcdShape::ShapeType shapeType = shape->getType(); );
	HK_ASSERT2( 0x17a594f6, shapeType == hkcdShapeType::BOX || shapeType == hkcdShapeType::CONVEX_VERTICES || transform.getScale().allExactlyEqual<3>(scaleX), "This shape type does not support non-uniform scale" );
	HK_ASSERT2( 0x406721bf, transform.m_scale.notEqualZero().allAreSet<hkVector4ComparisonMask::MASK_XYZ>(), "Flattening scales (scales with any component set to zero) are not supported" );

	ConvexShapeInfo& csi = m_shapes.expandOne();
	csi.m_shape		= shape;
	csi.m_transform	= transform;

	shape->addReference();
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
