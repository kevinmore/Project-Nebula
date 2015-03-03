/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Deprecated/CompressedMesh/hkTjunctionDetector.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppUtility.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppCompilerInput.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>


// tells if the vertices are equal or almost equal within a given tolerance
bool equalVertices( hkVector4Parameter a, hkVector4Parameter b, hkSimdRealParameter tolerance )
{
	if ( a.equal( b ).allAreSet() )
		return true;

	hkVector4 delta;
	delta.setSub( a, b );
	return delta.lengthSquared<3>() <= (tolerance*tolerance);
}

hkTjunctionDetector::ProximityInfoEnum HK_CALL hkTjunctionDetector::vertexCloseToTriangle( const hkVector4& vertex, const hkVector4& triA, 
																				 const hkVector4& triB, const hkVector4& triC, hkReal tolerance )
{
	hkVector4 edge0;
	edge0.setSub( triB, triA );

	hkVector4 edge1;
	edge1.setSub( triC, triB );

	hkVector4 edge2;
	edge2.setSub( triA, triC );

	hkVector4 delta0;
	delta0.setSub( vertex, triA );

	hkVector4 delta1;
	delta1.setSub( vertex, triB );

	hkVector4 delta2;
	delta2.setSub( vertex, triC );

	const hkSimdReal toleranceSr = hkSimdReal::fromFloat(tolerance);
	const hkSimdReal deltaLen0 = delta0.length<3>();
	const hkSimdReal deltaLen1 = delta1.length<3>();
	const hkSimdReal deltaLen2 = delta2.length<3>();

	const hkSimdReal len0 = edge0.length<3>();	
	if ( ( deltaLen0 < len0 ) && ( deltaLen1 < len0 ) && ( deltaLen0 + deltaLen1 - len0 < toleranceSr ) )
	{
		return NEAR_EDGE0;
	}

	const hkSimdReal len1 = edge1.length<3>();
	if ( ( deltaLen1 < len1 ) && ( deltaLen2 < len1 ) && ( deltaLen1 + deltaLen2 - len1 < toleranceSr ) )
	{
		return NEAR_EDGE1;
	}

	const hkSimdReal len2 = edge2.length<3>();
	if ( ( deltaLen2 < len2 ) && ( deltaLen0 < len2 ) && ( deltaLen2 + deltaLen0 - len2 < toleranceSr ) )
	{
		return NEAR_EDGE2;
	}

	// Compute distance to plane
	hkVector4 normal;
	normal.setCross( edge0, edge1 );
	normal.normalize<3>();

	hkSimdReal distH; distH.setAbs( delta0.dot<3>( normal ) );

	// If the vertex is far from the plane stop here
	if ( distH.isGreater(toleranceSr) )
		return NEAR_NONE;

	hkRotation simultaneousDots;

	hkVector4 cross0;
	cross0.setCross( delta0, edge0 );
	hkVector4 cross1;
	cross1.setCross( delta1, edge1 );
	hkVector4 cross2;
	cross2.setCross( delta2, edge2 );

	simultaneousDots.setCols( cross0, cross1, cross2 );
	hkVector4 dots;
	dots._setRotatedInverseDir( simultaneousDots, normal );

	hkVector4 zero; zero.setAll( toleranceSr );
	if ( dots.allLess<3>( zero ) )
	{
		return NEAR_FACE;
	}

	return NEAR_NONE;
}

bool findTriangle( hkpSimpleMeshShape* mesh, const hkArray<hkpShapeKey>& triangles, int a, int b, int c )
{
	for ( int i = 0; i < triangles.getSize(); ++i )
	{
		hkpShapeKey key = triangles[i];

		const hkpSimpleMeshShape::Triangle& triangle = mesh->m_triangles[key];

		if ( ( triangle.m_a == a || triangle.m_a == b || triangle.m_a == c ) && 
			( triangle.m_b == a || triangle.m_b == b || triangle.m_b == c ) &&
			( triangle.m_c == a || triangle.m_c == b || triangle.m_c == c ) )
		{
			return true;
		}
	}

	return false;
}


void HK_CALL hkTjunctionDetector::detect( hkpSimpleMeshShape* mesh, const hkpBvTreeShape* bvTree, hkArray<ProximityInfo>& Tjunctions, 
										 hkArray<hkVector4>& weldedVertices, hkReal junctionTolerance, hkReal weldTolerance )
{
	// Iterate through all vertices
	for ( int index = 0; index < mesh->m_vertices.getSize(); ++index )
	{
		const hkVector4& vertex = mesh->m_vertices[index];

		hkAabb aabb;
		aabb.setEmpty();
		aabb.includePoint( vertex );
		aabb.expandBy( hkSimdReal::fromFloat(junctionTolerance) );

		
		hkInplaceArray< hkpShapeKey, 128 > collectedKeysAdjacent;
		bvTree->queryAabb( aabb, collectedKeysAdjacent );

		bool welded = false;

		for ( int k = 0; k < collectedKeysAdjacent.getSize(); ++k )
		{
			hkpShapeKey adjacentTriangleShapeKey = collectedKeysAdjacent[k];

			const hkpSimpleMeshShape::Triangle& adjacentTriangle = mesh->m_triangles[adjacentTriangleShapeKey];

			// Check if the vertex index is referenced by this triangle
			if ( adjacentTriangle.m_a == index || adjacentTriangle.m_b == index || adjacentTriangle.m_c == index )
			{
				continue;
			}

			const hkVector4& v0 = mesh->m_vertices[adjacentTriangle.m_a];
			const hkVector4& v1 = mesh->m_vertices[adjacentTriangle.m_b];
			const hkVector4& v2 = mesh->m_vertices[adjacentTriangle.m_c];

			// Check if the vertex is welded to any other one from the triangle
			const hkSimdReal weldToleranceSr = hkSimdReal::fromFloat(weldTolerance);
			if ( equalVertices( vertex, v0, weldToleranceSr ) || 
				equalVertices( vertex, v1, weldToleranceSr ) || 
				equalVertices( vertex, v2, weldToleranceSr ) )
			{
				welded = true;
				continue;
			}

			ProximityInfoEnum result = vertexCloseToTriangle( vertex, v0, v1, v2, junctionTolerance );
			if ( result != NEAR_NONE )
			{
				// check if the vertex is connected to the edge forming a triangle
				if ( ( ( result == NEAR_EDGE0 ) && findTriangle( mesh, collectedKeysAdjacent, index, adjacentTriangle.m_a, adjacentTriangle.m_b ) ) ||
					( ( result == NEAR_EDGE1 ) && findTriangle( mesh, collectedKeysAdjacent, index, adjacentTriangle.m_b, adjacentTriangle.m_c ) ) ||
					( ( result == NEAR_EDGE2 ) && findTriangle( mesh, collectedKeysAdjacent, index, adjacentTriangle.m_a, adjacentTriangle.m_c ) ) )
				{
					continue;
				}

				ProximityInfo info;
				info.m_index = index;
				info.m_key = adjacentTriangleShapeKey;
				info.m_type = result;
				info.m_vertex = vertex;
				info.m_v0 = v0;
				info.m_v1 = v1;
				info.m_v2 = v2;

				Tjunctions.pushBack( info );
			}
		}

		if ( welded )
		{
			weldedVertices.pushBack( vertex );
		}
	}
}

hkpSimpleMeshShape* HK_CALL hkTjunctionDetector::createSimpleMeshFromGeometry( const hkGeometry& geometry )
{
	hkpSimpleMeshShape* meshShape = new hkpSimpleMeshShape();
	meshShape->m_vertices = geometry.m_vertices;
	meshShape->m_triangles.setSize( geometry.m_triangles.getSize() );
	for ( int i = 0; i < geometry.m_triangles.getSize(); ++i )
	{
		meshShape->m_triangles[i].m_a = geometry.m_triangles[i].m_a;
		meshShape->m_triangles[i].m_b = geometry.m_triangles[i].m_b;
		meshShape->m_triangles[i].m_c = geometry.m_triangles[i].m_c;
	}

	return meshShape;
}

void HK_CALL hkTjunctionDetector::detect( const hkGeometry& geometry, hkArray<hkTjunctionDetector::ProximityInfo>& Tjunctions, 
										 hkArray<hkVector4>& weldedVertices, hkReal tolerance, hkReal weldTolerance )
{
	// check to see if not all triangles are degenerate (in which case we cannot build the MOPP)
	bool allDegenerate = true;
	for ( int i = 0; i < geometry.m_triangles.getSize(); ++i )
	{
		const hkVector4& v0 = geometry.m_vertices[ geometry.m_triangles[i].m_a ];
		const hkVector4& v1 = geometry.m_vertices[ geometry.m_triangles[i].m_b ];
		const hkVector4& v2 = geometry.m_vertices[ geometry.m_triangles[i].m_c ];
		if ( !hkpTriangleUtil::isDegenerate( v0, v1, v2 ) )
		{
			allDegenerate = false;
			break;
		}
	}
	if ( allDegenerate )
	{
		return;
	}

	// create the simple mesh shape from the geometry
	hkpSimpleMeshShape* meshShape = createSimpleMeshFromGeometry( geometry );

	// build the MOPP tree around the shape
	hkpMoppCompilerInput moppInput;
	moppInput.m_enableChunkSubdivision = true;

	// Build the code at runtime
	hkpMoppCode* code = hkpMoppUtility::buildCode( meshShape, moppInput );	

	hkpMoppBvTreeShape* moppShape = HK_NULL;
	moppShape = new hkpMoppBvTreeShape(meshShape, code);
	
	code->removeReference();
	meshShape->removeReference();

	detect( meshShape, moppShape, Tjunctions, weldedVertices, tolerance, weldTolerance );

	moppShape->removeReference();
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
