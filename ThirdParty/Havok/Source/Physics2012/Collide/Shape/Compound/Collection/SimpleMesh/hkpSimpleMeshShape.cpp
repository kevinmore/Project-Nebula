/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>


hkpSimpleMeshShape::hkpSimpleMeshShape( hkReal radius )
:	hkpShapeCollection(HKCD_SHAPE_TYPE_FROM_CLASS(hkpSimpleMeshShape), COLLECTION_SIMPLE_MESH)
,	m_radius(radius)
{
	m_weldingType = hkpWeldingUtility::WELDING_TYPE_NONE;
}

#if !defined(HK_PLATFORM_SPU)

hkpSimpleMeshShape::hkpSimpleMeshShape( hkFinishLoadedObjectFlag flag )
:	hkpShapeCollection(flag)
,	m_vertices(flag)
,	m_triangles(flag)
,	m_materialIndices(flag)
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpSimpleMeshShape));
		m_collectionType = COLLECTION_SIMPLE_MESH;
	}
}

#endif

void hkpSimpleMeshShape::setWeldingInfo( hkpShapeKey key, hkInt16 weldingInfo )
{
	int index = key;
	HK_ASSERT3(0x3b082fa1, index >= 0 && index < m_triangles.getSize(), "hkpSimpleMeshShape does not have a triangle at index" << index);
	m_triangles[index].m_weldingInfo = weldingInfo;
}


void hkpSimpleMeshShape::initWeldingInfo( hkpWeldingUtility::WeldingType weldingType )
{
	m_weldingType = weldingType;
	for(int triangleIndex = 0; triangleIndex < m_triangles.getSize(); ++triangleIndex)
	{
		m_triangles[triangleIndex].m_weldingInfo = 0;
	}
}


void hkpSimpleMeshShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	// warning: not all vertices might be used, so it is not enough to go through vertexarray !
	// potential optimization (same for hkpMeshShape): speedup by lazy evaluation and storing the cached version, having a modified flag

	out.m_min = hkVector4::getConstant<HK_QUADREAL_MAX>();
	out.m_max = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();

	for (int i=0;i<m_vertices.getSize();i++)
	{
		const hkVector4& vLocal = m_vertices[i];
		hkVector4 vWorld; vWorld._setTransformedPos( localToWorld, vLocal );

		out.m_min.setMin( out.m_min, vWorld );
		out.m_max.setMax( out.m_max, vWorld );
	}

	hkSimdReal tol4; tol4.setFromFloat( tolerance + m_radius);
	out.m_min.setSub( out.m_min,tol4 );
	out.m_max.setAdd( out.m_max,tol4 );
}


const hkpShape* hkpSimpleMeshShape::getChildShape( hkpShapeKey key, hkpShapeBuffer& buffer) const
{
	int index = key;
	HK_ASSERT2(0x593f7a2f,  index >= 0 && index < m_triangles.getSize(), "hkpShapeKey invalid");

	hkpTriangleShape *ts = new( buffer ) hkpTriangleShape( m_radius, m_triangles[index].m_weldingInfo, m_weldingType );

	if ( 1 ||  index & 1 )
	{
		ts->setVertex<0>( m_vertices[m_triangles[index].m_a] );
		ts->setVertex<1>( m_vertices[m_triangles[index].m_b] );
		ts->setVertex<2>( m_vertices[m_triangles[index].m_c] );
	}
	else
	{
		ts->setVertex<2>( m_vertices[m_triangles[index].m_a] );
		ts->setVertex<1>( m_vertices[m_triangles[index].m_b] );
		ts->setVertex<0>( m_vertices[m_triangles[index].m_c] );
	}

	return ts;
}


hkpShapeKey hkpSimpleMeshShape::getFirstKey() const
{
	for( int key = 0; key < m_triangles.getSize(); ++key )
	{
		if ( hkpTriangleUtil::isDegenerate(
			m_vertices[m_triangles[key].m_a],
			m_vertices[m_triangles[key].m_b],
			m_vertices[m_triangles[key].m_c],
			hkDefaultTriangleDegeneracyTolerance ) == false )
		{
			return hkpShapeKey(key);
		}
	}
	return HK_INVALID_SHAPE_KEY;
}

hkpShapeKey hkpSimpleMeshShape::getNextKey( hkpShapeKey oldKey ) const
{
	for( int key = int(oldKey)+1; key < m_triangles.getSize(); ++key )
	{
		if ( hkpTriangleUtil::isDegenerate(
			m_vertices[m_triangles[key].m_a],
			m_vertices[m_triangles[key].m_b],
			m_vertices[m_triangles[key].m_c],
			hkDefaultTriangleDegeneracyTolerance ) == false )
		{
			return hkpShapeKey(key);
		}
	}
	return HK_INVALID_SHAPE_KEY;
}

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
