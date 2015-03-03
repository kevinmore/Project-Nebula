/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Deprecated/CompressedMesh/hkpCompressedMeshShapeBuilder.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>
#include <Common/GeometryUtilities/Misc/hkGeometryUtils.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Internal/GeometryProcessing/IndexedMesh/hkgpIndexedMesh.h>
#include <Common/Internal/ConvexHull/hkGeometryUtility.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>

#define MAX_UINT16 ( (1 << 16) - 1 )
#define MAX_UINT8 ( (1 << 8) - 1 )
#define MAX_TRIANGLES_PER_CHUNK MAX_UINT8

hkpCompressedMeshShape* hkpCompressedMeshShapeBuilder::createMeshShape( hkReal quantizationError, hkpCompressedMeshShape::MaterialType materialType, int bitsPerIndex )
{
	// create the mesh
	hkpCompressedMeshShape* meshShape = new hkpCompressedMeshShape( bitsPerIndex );	
	meshShape->m_error = quantizationError;
	meshShape->m_materialType = materialType;
	meshShape->m_bounds.setEmpty();
	
	
	m_shapeKeys.clear();	
	m_mesh = meshShape;
	m_error = quantizationError;

	return meshShape;
}

void hkpCompressedMeshShapeBuilder::gatherStatistics( hkpCompressedMeshShape* meshShape )
{
	// estimate the memory size
	int numVertices = 0;
	int size = sizeof( hkpCompressedMeshShape );
	int strips = 0;
	for ( int i = 0; i < meshShape->m_chunks.getSize(); ++i )
	{
		numVertices += meshShape->m_chunks[i].m_vertices.getSize() / 3;
		strips += meshShape->m_chunks[i].m_stripLengths.getSize();
		size += sizeof( hkpCompressedMeshShape::Chunk );
		size += meshShape->m_chunks[i].m_vertices.getSize() * sizeof( hkUint16 );
		size += meshShape->m_chunks[i].m_indices.getSize() * sizeof ( hkUint16 );
		size += meshShape->m_chunks[i].m_stripLengths.getSize() * sizeof( hkUint16 );
		size += meshShape->m_chunks[i].m_weldingInfo.getSize() * sizeof( hkUint16 );
	}
	int marker = size;
	m_statistics.m_chunksSize = marker;
	for ( int i = 0; i < meshShape->m_convexPieces.getSize(); ++i )
	{
		size += sizeof( hkpCompressedMeshShape::ConvexPiece );
		size += meshShape->m_convexPieces[i].m_vertices.getSize() * sizeof( hkUint16 );
	}
	m_statistics.m_convexPiecesSize = size - marker;
	marker = size;

	numVertices += meshShape->m_bigVertices.getSize();	
	size += meshShape->m_bigTriangles.getSize() * sizeof( hkpCompressedMeshShape::BigTriangle );
	size += meshShape->m_bigVertices.getSize() * sizeof( hkVector4 );
	m_statistics.m_bigTrianglesSize = size - marker;
	marker = size;

	size += meshShape->m_materials.getSize() * sizeof( hkUint32 );
	size += meshShape->m_materials8.getSize() * sizeof( hkUint8 );
	size += meshShape->m_materials16.getSize() * sizeof( hkUint16 );
	size += meshShape->m_transforms.getSize() * sizeof( hkTransform );

	// use hkGeometry as reference (no instances)
	int refSize = sizeof( hkGeometry );
	refSize += m_statistics.m_numVertices * sizeof( hkVector4 );
	refSize += m_statistics.m_numTriangles * sizeof( hkGeometry::Triangle );

	// build statistics
	m_statistics.m_maxExtent = MAX_UINT16 * m_error;
	m_statistics.m_maxIndex = MAX_UINT16;
	m_statistics.m_error = m_error;
	m_statistics.m_numExcessVertices = numVertices - m_statistics.m_numVertices;
	m_statistics.m_numChunks = meshShape->m_chunks.getSize();
	m_statistics.m_size = size;
	m_statistics.m_compressRatio = (hkReal)size / (hkReal)refSize;
	m_statistics.m_numStrips = strips;
}

void hkpCompressedMeshShapeBuilder::addGeometry( const hkGeometry& geometry, const hkMatrix4& transform, hkpCompressedMeshShape* meshShape )
{
	m_mesh = meshShape;
	m_statistics.m_numVertices += geometry.m_vertices.getSize();
	m_statistics.m_numTriangles += geometry.m_triangles.getSize();

	// copy the input geometry to a local one so it can be modified
	m_geometry = geometry;
	hkQsTransform rts;
	rts.set( transform);

	for ( int v = 0; v < m_geometry.m_vertices.getSize(); ++v )
	{
		m_geometry.m_vertices[v].setTransformedPos( rts, m_geometry.m_vertices[v] );
	}
	HK_ON_DEBUG( m_originalGeometry = m_geometry );

	// create the mapping root node
	MappingTree* mapping = HK_NULL;
	int keysBase = 0;
	if ( m_createMapping )
	{
		mapping = new MappingTree();
		mapping->m_triangles.setSize( geometry.m_triangles.getSize() );
		for ( int i = 0; i < mapping->m_triangles.getSize(); ++i )
		{
			mapping->m_triangles[i].m_originalIndex = i;
			mapping->m_triangles[i].m_winding = 0;
		}

		m_bigMapping.clear();
		
		// initialize the keys to invalid
		keysBase = m_shapeKeys.getSize();
		m_shapeKeys.expandBy( geometry.m_triangles.getSize() );
		for ( int i = keysBase; i < m_shapeKeys.getSize(); ++i )
		{
			m_shapeKeys[i] = HK_INVALID_SHAPE_KEY;
		}
	}

	// snap the geometry to grid while preserving T-junctions (and welding vertices)
	if ( m_preserveTjunctions )
	{
		snapGeometry( mapping );
	}
	else
	{
		snapGeometry( m_geometry, m_error );
	}

	// remove the triangles that can't be quantized
	filterGeometry( mapping );
	if ( m_geometry.m_triangles.getSize() == 0 && m_leftOver.m_triangles.getSize() == 0 )
	{
		return;
	}

	// build the chunk list
	addChunk( m_geometry, meshShape->m_chunks, mapping );
	
	//meshShape->m_chunks.optimizeCapacity( 0, true );

	// build the global list
	const int bigTrianglesBase = addBigTriangles( meshShape );

	// build the shape key array
	if ( m_createMapping )
	{		
		// get the keys from the mapping tree
		mapping->getKeys( m_shapeKeys, keysBase );
		delete mapping;
		
		// get the keys from the big triangles mapping		
		for ( int i = bigTrianglesBase; i < meshShape->m_bigTriangles.getSize(); ++i )
		{
			TriangleMapping& triMap = m_bigMapping[i - bigTrianglesBase];
			triMap.m_triangleIndex = i;
			triMap.m_key = triMap.m_triangleIndex;
			m_shapeKeys[keysBase + triMap.m_originalIndex] = triMap.m_key;
		}
	}
}

int hkpCompressedMeshShapeBuilder::beginSubpart( hkpCompressedMeshShape* compressedMesh )
{
	Subpart sub;
	sub.m_numInstances = 0;
	sub.m_numConvexPieces = 0;
	sub.m_numChunks = 0;
	sub.m_numBigTriangles = 0;
	sub.m_numTriangles  = 0;
	sub.m_chunkOffset = compressedMesh->m_chunks.getSize();
	sub.m_bigOffset = compressedMesh->m_bigTriangles.getSize();
	sub.m_convexOffset = compressedMesh->m_convexPieces.getSize();
	sub.m_geomOffset = m_shapeKeys.getSize();
	
	subparts.pushBack( sub );
	return subparts.getSize() - 1;
}

void hkpCompressedMeshShapeBuilder::endSubpart( hkpCompressedMeshShape* compressedMesh )
{
	Subpart& sub = subparts[subparts.getSize() - 1];
	sub.m_numChunks = compressedMesh->m_chunks.getSize() - sub.m_chunkOffset;
	sub.m_numBigTriangles = compressedMesh->m_bigTriangles.getSize() - sub.m_bigOffset;
	sub.m_numConvexPieces = compressedMesh->m_convexPieces.getSize() - sub.m_convexOffset;
	if ( m_createMapping )
	{
		sub.m_numTriangles = m_shapeKeys.getSize() - sub.m_geomOffset;
	}
}

void hkpCompressedMeshShapeBuilder::addInstance( int subpartID, const hkMatrix4& transform, hkpCompressedMeshShape* compressedMesh, 
												hkArray<hkpShapeKey>* shapeKeyMap )
{
	

	// get the transform data and add it to the mesh
	hkQsTransform rts;
	rts.set(transform);

	int transformIndex = compressedMesh->m_transforms.getSize();
	HK_ASSERT2(0x3dac21c0, transformIndex <= MAX_UINT16, "Exceeded the number of supported transforms in the Compressed Mesh Shape");
	compressedMesh->m_transforms.pushBack( rts );

	Subpart& subpart = subparts[subpartID];

	// clone chunks
	int start = subpart.m_chunkOffset;
	int stop = start + subpart.m_numChunks;
	int chunksBase = compressedMesh->m_chunks.getSize();
	for ( int i = start; i < stop; ++i )
	{
		int index = i;
		if ( subpart.m_numInstances == 0 )
		{
			compressedMesh->m_chunks[i].m_transformIndex = (hkUint16)transformIndex;
		}
		else
		{
			// create a new instance chunk for each chunk of the subpart
			hkpCompressedMeshShape::Chunk chunk;
			HK_ASSERT2(0x3dac21c0, i <= MAX_UINT16, "Exceeded the number of supported chunks in the Compressed Mesh Shape");
			chunk.m_reference = (hkUint16)i;
			chunk.m_transformIndex = (hkUint16)transformIndex;
			index = compressedMesh->m_chunks.getSize();
			compressedMesh->m_chunks.pushBack( chunk );
			
			m_statistics.m_numChunkClones++;
		}

		// include the transformed AABB
		hkAabb bounds; 
		compressedMesh->getChunkAabb( compressedMesh->m_chunks[index], bounds );
		compressedMesh->m_bounds.includeAabb( bounds );
	}

	// clone big triangles
	start = subparts[subpartID].m_bigOffset;
	stop = start + subparts[subpartID].m_numBigTriangles;
	int bigBase = compressedMesh->m_bigTriangles.getSize();
	for ( int i = start; i < stop; ++i )
	{
		int index = i;
		if ( subpart.m_numInstances == 0 )
		{
			compressedMesh->m_bigTriangles[i].m_transformIndex = (hkUint16)transformIndex;
		}
		else
		{
			hkpCompressedMeshShape::BigTriangle triangle = compressedMesh->m_bigTriangles[i];
			triangle.m_transformIndex = (hkUint16)transformIndex;
			index = compressedMesh->m_bigTriangles.getSize();
			compressedMesh->m_bigTriangles.pushBack( triangle );
		}

		// include AABB
		hkVector4 v0, v1, v2;
		compressedMesh->getBigTriangle( index, v0, v1, v2 );

		hkAabb bounds;
		bounds.m_min.setMin( v0, v1 );
		bounds.m_min.setMin( bounds.m_min, v2 );
		bounds.m_max.setMax( v0, v1 );
		bounds.m_max.setMax( bounds.m_max, v2 );
		compressedMesh->m_bounds.includeAabb( bounds );
	}

	// copy and fix the shape keys into the mapping table
	if ( m_createMapping && shapeKeyMap != HK_NULL )
	{
		start = subpart.m_geomOffset;
		stop = start + subpart.m_numTriangles;
		for ( int i = start; i < stop; ++i )
		{
			hkpShapeKey key = m_shapeKeys[i];

			// fix the key
			if ( key != HK_INVALID_SHAPE_KEY && subpart.m_numInstances != 0 )
			{
				const int chunkId = key >> compressedMesh->m_bitsPerWIndex;
				const int wIndex = key & compressedMesh->m_wIndexMask;			

				if ( chunkId == 0 )
				{
					key = bigBase + wIndex - subpart.m_bigOffset;
				}
				else if ( chunkId != ( ( 1 << (32 - compressedMesh->m_bitsPerWIndex) ) - 1 ) )
				{
					const int newId = chunksBase + chunkId - subpart.m_chunkOffset;
					key = ( newId << compressedMesh->m_bitsPerWIndex ) | wIndex;
				}
			}

			shapeKeyMap->pushBack( key );
		}
	}

	// clone convex pieces
	{
		start = subparts[subpartID].m_convexOffset;
		stop = start + subparts[subpartID].m_numConvexPieces;
		for ( int i = start; i < stop; ++i )
		{
			int index = i;
			if ( subparts[subpartID].m_numInstances != 0 )
			{
				hkpCompressedMeshShape::ConvexPiece convexPiece;
				HK_ASSERT2(0x3dac21c0, i <= MAX_UINT16, "Exceeded the index range of convex pieces in the Compressed Mesh Shape");
				convexPiece.m_reference = (hkUint16)i;
				convexPiece.m_transformIndex = (hkUint16)transformIndex;
				index = compressedMesh->m_convexPieces.getSize();
				compressedMesh->m_convexPieces.pushBack( convexPiece );
			}
			else
			{
				compressedMesh->m_convexPieces[i].m_transformIndex = (hkUint16)transformIndex;
			}
			
			// include the transformed AABB
			hkAabb bounds; 
			compressedMesh->getConvexPieceAabb( compressedMesh->m_convexPieces[index], bounds );
			compressedMesh->m_bounds.includeAabb( bounds );			
		}
	}

	++subpart.m_numInstances;
}

hkpCompressedMeshShape* HK_CALL hkpCompressedMeshShapeBuilder::createMeshFromGeometry( const hkGeometry& geometry, 
																			   hkReal quantizationError,
																			   hkpCompressedMeshShape::MaterialType materialType )
{
	// instantiate a compressed mesh shape builder
	hkpCompressedMeshShapeBuilder builder;
	
	hkpCompressedMeshShape* meshShape = builder.createMeshShape( quantizationError, materialType );
	int subpart = builder.beginSubpart( meshShape );
	builder.addGeometry( geometry, hkMatrix4::getIdentity(), meshShape );
	builder.endSubpart( meshShape );
	builder.addInstance( subpart, hkMatrix4::getIdentity(), meshShape );
	return meshShape;
}

// appends a given geometry to a destination one
void hkpCompressedMeshShapeBuilder::appendGeometry( hkGeometry& dest, const hkGeometry& geom )
{
	int numTri = dest.m_triangles.getSize();
	int numVert = dest.m_vertices.getSize();
	dest.m_triangles.append( geom.m_triangles.begin(), geom.m_triangles.getSize() );
	for ( int i = numTri; i < dest.m_triangles.getSize(); ++i )
	{
		dest.m_triangles[i].m_a += numVert;
		dest.m_triangles[i].m_b += numVert;
		dest.m_triangles[i].m_c += numVert;
	}
	dest.m_vertices.append( geom.m_vertices.begin(), geom.m_vertices.getSize() );
}

bool hasDifferentMaterials( const hkGeometry& geometry )
{
	const int firstMaterial = geometry.m_triangles[0].m_material;
	for ( int i = 1; i < geometry.m_triangles.getSize(); ++i )
	{
		if ( geometry.m_triangles[i].m_material != firstMaterial )
		{
			return true;
		}
	}
	return false;
}

void hkpCompressedMeshShapeBuilder::addChunk( const hkGeometry& geometry, hkArray<hkpCompressedMeshShape::Chunk>& chunks, 
											 MappingTree* mapping )
{
	// calculate the AABB of the geometry
	hkAabb bounds;
	hkAabbUtil::calcAabb( geometry.m_vertices.begin(), geometry.m_vertices.getSize(), bounds );

	// find the longest extent
	hkVector4 extents;
	extents.setSub( bounds.m_max, bounds.m_min );
	int dir = extents.getIndexOfMaxAbsComponent<3>();
	hkReal maxExtent = extents(dir);

	// split the geometry if needed
	if ( geometry.m_vertices.getSize() > MAX_TRIANGLES_PER_CHUNK || maxExtent > ( MAX_UINT16 - 1 ) * m_error ||
		( m_mesh->m_materialType == hkpCompressedMeshShape::MATERIAL_SINGLE_VALUE_PER_CHUNK && hasDifferentMaterials( geometry ) ) )
	{
		// find the half plane (passing through the center of the AABB)
		hkVector4 center;
		bounds.getCenter( center );
		hkReal half = center(dir);

		hkGeometry geom1;
		hkGeometry geom2;
		splitGeometry( geometry, geom1, geom2, dir, half, mapping );		
		// if one of the outputs is empty it means that the geometry cannot be further split
		if ( geom1.m_triangles.getSize() != 0 && geom2.m_triangles.getSize() != 0 )
		{
			if ( mapping != HK_NULL )
			{
				addChunk( geom1, chunks, mapping->m_left );
				addChunk( geom2, chunks, mapping->m_right );
			}
			else
			{
				addChunk( geom1, chunks, HK_NULL );
				addChunk( geom2, chunks, HK_NULL );
			}
		}
		else
		{
			if ( geom1.m_triangles.getSize() != 0 )
			{
				appendGeometry( m_leftOver, geom1 );
				if ( mapping != HK_NULL )
				{
					m_bigMapping.append( mapping->m_left->m_triangles.begin(), mapping->m_left->m_triangles.getSize() );
				}
			}
			else if ( geom2.m_triangles.getSize() != 0 )
			{
				appendGeometry( m_leftOver, geom2 );
				if ( mapping != HK_NULL )
				{
					m_bigMapping.append( mapping->m_right->m_triangles.begin(), mapping->m_right->m_triangles.getSize() );
				}
			}
			if ( mapping != HK_NULL )
			{
				delete mapping->m_left;
				mapping->m_left = HK_NULL;
				delete mapping->m_right;
				mapping->m_right = HK_NULL;
			}
		}
	}
	else
	// or else add it to the chunk list
	{
		hkpCompressedMeshShape::Chunk chunk;
		createChunk( geometry, bounds.m_min, chunk, mapping );
		chunks.pushBack( chunk );
		HK_ASSERT2( 0x5fd1ab23, chunks.getSize() <= (1 << (32 - m_mesh->m_bitsPerWIndex)) - 1, "You have exceeded the maximum number of chunks."
			"Try reducing the number of bits per index in the shape key." );
		// assign the chunk to the triangle mapping and compute the key
		if ( mapping != HK_NULL )
		{
			for ( int i = 0; i < mapping->m_triangles.getSize(); ++i )
			{
				const int chunkIndex = chunks.getSize() - 1;
				const int winding = mapping->m_triangles[i].m_winding;
				const int triangleIndex = mapping->m_triangles[i].m_triangleIndex;
				mapping->m_triangles[i].m_chunkIndex = chunkIndex;				
				mapping->m_triangles[i].m_key = hkpShapeKey( ( ( chunkIndex + 1 ) << m_mesh->m_bitsPerWIndex ) | 
					( ( winding & 1 ) << m_mesh->m_bitsPerIndex ) | ( triangleIndex & m_mesh->m_wIndexMask ) );
			}
		}
	}
}

void HK_CALL hkpCompressedMeshShapeBuilder::snapToGrid( hkVector4& v, hkReal error )
{
	const hkSimdReal errorS = hkSimdReal::fromFloat(error);
	hkSimdReal invError; invError.setReciprocal(errorS);
	v.mul( invError );
	hkVector4 vp; vp.setAdd(v,hkVector4::getConstant<HK_QUADREAL_INV_2>());
	hkIntVector vi; vi.setConvertF32toS32(vp);
	hkVector4 vv; vi.convertS32ToF32(vv); vv.mul(errorS);
	v.setXYZ(vv);
}

#ifdef HK_DEBUG

void hkpCompressedMeshShapeBuilder::testMapping( const hkGeometry& testGeom, const hkArray<TriangleMapping>& triMap )
{
	for ( int i = 0; i < triMap.getSize(); ++i )
	{
		int j = triMap[i].m_originalIndex;
		hkGeometry::Triangle tri1 = testGeom.m_triangles[i];
		
		testVertices( j, testGeom.m_vertices, tri1.m_a, tri1.m_b, tri1.m_c );
	}
}

void hkpCompressedMeshShapeBuilder::testVertices( int originalIndex, const hkArray<hkVector4>& vertices, int a, int b, int c)
{
	const hkVector4& v1 = vertices[a];
	const hkVector4& v2 = vertices[b];
	const hkVector4& v3 = vertices[c];
	hkVector4 c1;
	hkpTriangleUtil::calcCentroid( c1, v1, v2, v3 );

	hkGeometry::Triangle tri2 = m_originalGeometry.m_triangles[originalIndex];
	const hkVector4& w1 = m_originalGeometry.m_vertices[tri2.m_a];
	const hkVector4& w2 = m_originalGeometry.m_vertices[tri2.m_b];
	const hkVector4& w3 = m_originalGeometry.m_vertices[tri2.m_c];
	hkVector4 c2;
	hkpTriangleUtil::calcCentroid( c2, w1, w2, w3 );

	hkVector4 c12;
	c12.setSub( c1, c2 );
	HK_ON_DEBUG( hkReal len = c12.length<3>().getReal(); )

	const hkReal threshold = 0.5f; //hkMath::sqrt( 3 * m_error );
	HK_ASSERT2(0x7da0c48d, len < threshold, " triangles not equal");
}

#endif // HK_DEBUG

void hkpCompressedMeshShapeBuilder::addMaterial( hkpCompressedMeshShape::Chunk& chunk, const int material )
{
	if ( m_mesh->m_materialType == hkpCompressedMeshShape::MATERIAL_FOUR_BYTES_PER_TRIANGLE )
	{
		m_mesh->m_materials.pushBack( material );
	}
	else if ( m_mesh->m_materialType == hkpCompressedMeshShape::MATERIAL_TWO_BYTES_PER_TRIANGLE )
	{
		HK_ASSERT2(0x59aaecdb, material <= MAX_UINT16, " materials exceed two bytes per triangle");
		m_mesh->m_materials16.pushBack( (hkUint16)material );
	}
	else if ( m_mesh->m_materialType == hkpCompressedMeshShape::MATERIAL_ONE_BYTE_PER_TRIANGLE )
	{
		HK_ASSERT2(0x59aaecdb, material <= MAX_UINT8, " materials exceed one byte per triangle");
		m_mesh->m_materials8.pushBack( (hkUint8)material );
	}
	else if ( chunk.m_materialInfo == 0xffffffff && m_mesh->m_materialType == hkpCompressedMeshShape::MATERIAL_SINGLE_VALUE_PER_CHUNK )
	{
		chunk.m_materialInfo = material;
	}
}

void HK_CALL hkpCompressedMeshShapeBuilder::quantizeVertices( hkReal quantization, const hkVector4& offset, const hkArray<hkVector4>& verticesIn, hkArray<hkUint16>& verticesOut )
{
	hkReal invError = 1.0f / quantization;
	const int nv = verticesIn.getSize();
	verticesOut.setSize( nv * 3);
	for ( int i = 0; i < nv; ++i )
	{
		hkVector4 v = verticesIn[i];
		v.sub( offset );
		v.mul( hkSimdReal::fromFloat(invError) );
		for ( int j = 0; j < 3; ++j )
		{
			int quant = hkMath::hkFloatToInt( v(j) + 0.5f );
			HK_ASSERT2(0x1e4361b1, quant < MAX_UINT16 , "");
			verticesOut[i * 3 + j] = (hkUint16)quant;
		}
	}
}

void hkpCompressedMeshShapeBuilder::createChunk( const hkGeometry& geometry, const hkVector4& offset, hkpCompressedMeshShape::Chunk& chunk,
												MappingTree* mapping )
{
	// set the offset
	chunk.m_offset = offset;

	// set the material info field
	if ( m_mesh->m_materialType == hkpCompressedMeshShape::MATERIAL_ONE_BYTE_PER_TRIANGLE )
	{
		chunk.m_materialInfo = m_mesh->m_materials8.getSize();
	}
	else if ( m_mesh->m_materialType == hkpCompressedMeshShape::MATERIAL_TWO_BYTES_PER_TRIANGLE )
	{
		chunk.m_materialInfo = m_mesh->m_materials16.getSize();
	}
	else if ( m_mesh->m_materialType == hkpCompressedMeshShape::MATERIAL_FOUR_BYTES_PER_TRIANGLE )
	{
		chunk.m_materialInfo = m_mesh->m_materials.getSize();
	}
	else
	{
		chunk.m_materialInfo = 0xffffffff;
	}

	// add the vertices
	quantizeVertices( m_error, offset, geometry.m_vertices, chunk.m_vertices );

	// generate triangle strips
	hkgpIndexedMesh indexedMesh;
	indexedMesh.appendFromGeometry( geometry );
	hkArray< hkArray<int> > strips;
	hkArray<int> lists;
	hkArray<int> stripMap;
	hkgpIndexedMesh::StripConfig stripConfig;
	//stripConfig.m_searchPasses = m_stripperPasses;
	//stripConfig.m_maxLength = 6;
	indexedMesh.generateStrips( strips, lists, stripMap, stripConfig );

	// rebuild the mapping data
	if ( mapping != HK_NULL )
	{
		hkArray<TriangleMapping> newMapping;
		newMapping.setSize( stripMap.getSize() );
		HK_ASSERT2(0x735e3137, stripMap.getSize() == newMapping.getSize() ," strip mapping not consistent with non degenerate mapping - the length of stripMap should be equal to that of newMapping");
		for ( int i = 0; i < stripMap.getSize(); ++i )
		{
			newMapping[i] = mapping->m_triangles[stripMap[i]];
		}
		mapping->m_triangles = newMapping;
	}

	// copy the strips to the chunk
	chunk.m_stripLengths.setSize( strips.getSize() );
	int stripOffset = 0;
	int currTri = 0;
	int material;
	
	for (int k = 0; k < strips.getSize(); ++k )
	{
		const int stripLen = strips[k].getSize();
		chunk.m_stripLengths[k] = (hkUint16)stripLen;		
		for ( int i = 0; i < stripLen; ++i )
		{
			HK_ASSERT2( 0x735e3138, strips[k][i] <= m_mesh->m_indexMask, "You have too many triangles in a chunk. Try setting the bits per index in the shape key to a larger value."
				"See hkpCompressedMeshShape::setShapeKeyBitsPerIndex." );
			chunk.m_indices.pushBack( (hkUint16)strips[k][i] );			
			// set the index and winding for the triangle mapping
			if ( i < stripLen - 2 )
			{
				if ( mapping != HK_NULL )
				{
					mapping->m_triangles[currTri].m_triangleIndex = stripOffset + i;
					mapping->m_triangles[currTri].m_winding = i & 1;
				}
				material = geometry.m_triangles[stripMap[currTri]].m_material;
				++currTri;
			}
			else
			{
				material = -1;
			}
			addMaterial( chunk, material );
		}
		stripOffset += stripLen;
	}

	// copy the left over triangles too
	for ( int i = 0; i < lists.getSize(); ++i )
	{
		HK_ASSERT2( 0x735e3138, lists[i] <= m_mesh->m_indexMask, "You have too many triangles in a chunk. Try setting the bits per index in the shape key to a larger value."
			"See hkpCompressedMeshShape::setShapeKeyBitsPerIndex." );
		chunk.m_indices.pushBack( (hkUint16)lists[i] );		
		if ( i % 3 == 0 )
		{
			if ( mapping != HK_NULL )
			{
				mapping->m_triangles[currTri].m_triangleIndex = stripOffset + i;
			}
			material = geometry.m_triangles[stripMap[currTri]].m_material;
			++currTri;
		}
		else
		{
			material = -1;
		}
		addMaterial( chunk, material );
	}

	// optimize the indices array
	chunk.m_indices.optimizeCapacity( 0, true );
}

int hkpCompressedMeshShapeBuilder::splitCriterion( const hkVector4& v1, const hkVector4& v2, 
													const hkVector4& v3, hkReal half, int dir)
{
	hkReal min = v1(dir);
	if ( v2(dir) < min )
		min = v2(dir);
	if ( v3(dir) < min )
		min = v3(dir);

	hkReal max = v1(dir);
	if ( v2(dir) > max )
		max = v2(dir);
	if ( v3(dir) > max )
		max = v3(dir);

	if ( min < half && max > half)
	{
		hkReal overlap = max - min;
		if ( overlap > MAX_UINT16 * m_error * m_overlapRatio )
		{
			return -1;
		}
		if ( overlap > m_statistics.m_maxOverlap )
		{
			m_statistics.m_maxOverlap = overlap;
		}
	}

	hkReal mid = ( min + max ) * 0.5f;
	return ( mid < half ) ? 1 : 0; 
}

void addTriangle( int a, int b, int c, int material, 
						 const hkVector4& vertexA, const hkVector4& vertexB, const hkVector4& vertexC, 
						 hkGeometry& geom, hkArray<int>& vertexIndex )
{
	// add the corners to the vertices list
	if ( vertexIndex[a] == -1 )
	{
		vertexIndex[a] = geom.m_vertices.getSize();
		geom.m_vertices.pushBack( vertexA );
	}
	if ( vertexIndex[b] == -1 )
	{
		vertexIndex[b] = geom.m_vertices.getSize();
		geom.m_vertices.pushBack( vertexB );
	}
	if ( vertexIndex[c] == -1 )
	{
		vertexIndex[c] = geom.m_vertices.getSize();
		geom.m_vertices.pushBack( vertexC );
	}

	// add triangle struct
	hkGeometry::Triangle tri;
	tri.m_a = vertexIndex[a];
	tri.m_b = vertexIndex[b];
	tri.m_c = vertexIndex[c];
	tri.m_material = material;
	geom.m_triangles.pushBack( tri );
}

void hkpCompressedMeshShapeBuilder::splitGeometry( const hkGeometry& source, hkGeometry& out1, hkGeometry& out2, hkGeometry& leftOver )
{
	// calculate the AABB of the source geometry
	hkAabb bounds;
	hkAabbUtil::calcAabb( source.m_vertices.begin(), source.m_vertices.getSize(), bounds );

	// find the longest extent
	hkVector4 extents;
	extents.setSub( bounds.m_max, bounds.m_min );
	int dir = extents.getIndexOfMaxAbsComponent<3>();

	// find the half plane (passing through the center of the AABB)
	hkVector4 center;
	bounds.getCenter( center );
	hkReal half = center( dir );

	splitGeometry( source, out1, out2, dir, half, HK_NULL );
}

void hkpCompressedMeshShapeBuilder::splitGeometry( const hkGeometry& source, hkGeometry& out1, hkGeometry& out2, 
												  int dir, hkReal half, MappingTree* mapping )
{
	// create the lookup tables used to store the new vertex indices
	int nv = source.m_vertices.getSize();
	hkArray<int> vertexIndex1;
	hkArray<int> vertexIndex2;
	vertexIndex1.setSize( nv );
	vertexIndex2.setSize( nv );
	for ( int i = 0; i < nv; ++i )
	{
		vertexIndex1[i] = -1;
		vertexIndex2[i] = -1;
	}

	// if mapping is required, instantiate the two sub-trees
	MappingTree* mapLeft = HK_NULL;
	MappingTree* mapRight = HK_NULL;
	if ( mapping != HK_NULL )
	{
		mapLeft = new MappingTree;
		mapRight = new MappingTree;
		mapping->m_left = mapLeft;
		mapping->m_right = mapRight;
	}

	// the actual division
	for ( int i = 0; i < source.m_triangles.getSize(); ++i )
	{
		const int a = source.m_triangles[i].m_a;
		const int b = source.m_triangles[i].m_b;
		const int c = source.m_triangles[i].m_c;
		const int material = source.m_triangles[i].m_material;
		
		const hkVector4& vertexA = source.m_vertices[a];
		const hkVector4& vertexB = source.m_vertices[b];
		const hkVector4& vertexC = source.m_vertices[c];

		int sc = splitCriterion( vertexA, vertexB, vertexC, half, dir );

		if ( sc == -1 )
		{
			// add the triangle to the left over geometry
			TriangleMapping* map = HK_NULL;
			if ( mapping != HK_NULL )
			{
				map = &mapping->m_triangles[i];
			}
			addLeftOverTriangle( vertexA, vertexB, vertexC, material, map );
		}
		else if ( sc == 1 )
		{
			// add the triangle to the first output geometry
			addTriangle( a, b, c, material, vertexA, vertexB, vertexC, out1, vertexIndex1 );
			if ( mapLeft != HK_NULL )
			{
				mapLeft->m_triangles.pushBack( mapping->m_triangles[i] );
			}
		}
		else if ( sc == 0 )
		{
			// add the triangle to the second output geometry
			addTriangle( a, b, c, material, vertexA, vertexB, vertexC, out2, vertexIndex2 );
			if ( mapRight != HK_NULL )
			{
				mapRight->m_triangles.pushBack( mapping->m_triangles[i] );
			}
		}
	}

	if ( mapping != HK_NULL )
	{
		mapping->m_triangles.clear();
	}
}

int addVertex( hkGeometry& geometry, const hkVector4& vertex )
{
	// linear search
	for ( int i = 0; i < geometry.m_vertices.getSize(); ++i )
	{
		if ( vertex.equal( geometry.m_vertices[i] ).allAreSet() )
		{
			return i;
		}
	}
	
	geometry.m_vertices.pushBack( vertex );

	return geometry.m_vertices.getSize() - 1;
}

void hkpCompressedMeshShapeBuilder::addLeftOverTriangle( const hkVector4& v0, const hkVector4& v1, const hkVector4& v2, int material, 
														TriangleMapping* map )
{
	hkGeometry::Triangle tri;
	tri.m_a = addVertex( m_leftOver, v0 );
	tri.m_b = addVertex( m_leftOver, v1 );
	tri.m_c = addVertex( m_leftOver, v2 );
	tri.m_material = material;

	m_leftOver.m_triangles.pushBack( tri );

	if ( map != HK_NULL )
	{
		m_bigMapping.pushBack( *map );
		//HK_ON_DEBUG( testVertices( map->m_originalIndex, m_leftOver.m_vertices, tri.m_a, tri.m_b, tri.m_c ) );
	}
}

void HK_CALL hkpCompressedMeshShapeBuilder::chunkToGeometry( hkpCompressedMeshShape* compressedMesh, int id, hkGeometry& geometry )
{
	hkQsTransform transform; transform.setIdentity();
	if ( compressedMesh->m_chunks[id].m_transformIndex != 0xffff )
	{
		transform = compressedMesh->m_transforms[ compressedMesh->m_chunks[id].m_transformIndex ];
	}
	if ( compressedMesh->m_chunks[id].m_reference != 0xffff )
	{
		id = compressedMesh->m_chunks[id].m_reference;
	}
	const hkpCompressedMeshShape::Chunk& chunk = compressedMesh->m_chunks[id];
	const int numVerts = chunk.m_vertices.getSize() / 3;
	hkArray<int> indexMap( numVerts );
	for ( int i = 0; i < numVerts; ++i )
	{
		indexMap[i] = -1;
	}

	int winding = 0;
	
	for ( int index = 0 ; index != -1; index = chunk.getNextIndex( index, winding ) )
	{
		hkVector4 v[3];
		chunk.getTriangle( index, compressedMesh->m_error , v[0], v[1], v[2] );
		for ( int i = 0; i < 3; ++i )
		{
			const int k = chunk.m_indices[index + i];
			if ( indexMap[k] == -1 )
			{
				indexMap[k] = geometry.m_vertices.getSize();
				v[i].setTransformedPos( transform, v[i] );
				geometry.m_vertices.pushBack( v[i] );
			}
		}

		hkGeometry::Triangle tri;
		tri.m_a = indexMap[chunk.m_indices[index + ( winding << 1 )]];
		tri.m_b = indexMap[chunk.m_indices[index + 1]];
		tri.m_c = indexMap[chunk.m_indices[index + ( ( 1 - winding ) << 1 )]];
		tri.m_material = 0;
		geometry.m_triangles.pushBack( tri );
	}
}

void HK_CALL hkpCompressedMeshShapeBuilder::convexPieceToGeometry( const hkpCompressedMeshShape* compressedMesh, int index, hkGeometry& geometry )
{
	hkQsTransform transform; transform.setIdentity();
	if ( compressedMesh->m_convexPieces[index].m_transformIndex != 0xffff )
	{
		transform = compressedMesh->m_transforms[ compressedMesh->m_convexPieces[index].m_transformIndex  ];
	}

	if ( compressedMesh->m_convexPieces[index].m_reference != 0xffff )
	{
		index = compressedMesh->m_convexPieces[index].m_reference;
	}

	hkArray<hkVector4> vertices;
	compressedMesh->m_convexPieces[index].getVertices( compressedMesh->m_error, transform, vertices );

	
	hkGeometryUtility::createConvexGeometry( vertices, geometry );
}

void HK_CALL hkpCompressedMeshShapeBuilder::bigTrianglesToGeometry( const hkpCompressedMeshShape* compressedMesh, hkGeometry& geometryOut )
{
	
	for ( int i = 0; i < compressedMesh->m_bigTriangles.getSize(); ++i )
	{
		hkVector4 v0, v1, v2;
		compressedMesh->getBigTriangle( i, v0, v1, v2 );

		const int newIndex = geometryOut.m_vertices.getSize();
		geometryOut.m_vertices.pushBack( v0 );
		geometryOut.m_vertices.pushBack( v1 );
		geometryOut.m_vertices.pushBack( v2 );

		hkGeometry::Triangle newTriangle;
		newTriangle.m_a = newIndex;
		newTriangle.m_b = newIndex + 1;
		newTriangle.m_c = newIndex + 2;
		newTriangle.m_material = 0;

		geometryOut.m_triangles.pushBack( newTriangle );
	}
}

void hkpCompressedMeshShapeBuilder::MappingTree::getKeys( hkArray<hkpShapeKey>& keys, const int keysBase )
{
	for ( int i = 0; i < m_triangles.getSize(); ++i )
	{
		keys[m_triangles[i].m_originalIndex + keysBase] = m_triangles[i].m_key;
	}
	if ( m_left != HK_NULL && m_right != HK_NULL )
	{
		m_left->getKeys( keys, keysBase );
		m_right->getKeys( keys, keysBase );
	}
}

int hkpCompressedMeshShapeBuilder::addBigTriangles( hkpCompressedMeshShape* meshShape )
{
	hkGeometryUtils::weldVertices( m_leftOver );
	const int bigVerticesBase = meshShape->m_bigVertices.getSize();
	meshShape->m_bigVertices.append( m_leftOver.m_vertices.begin(), m_leftOver.m_vertices.getSize() );	

	const int bigTrianglesBase = meshShape->m_bigTriangles.getSize();
	hkpCompressedMeshShape::BigTriangle* list = meshShape->m_bigTriangles.expandBy( m_leftOver.m_triangles.getSize() );
	HK_ASSERT2( 0x735e3138, meshShape->m_bigTriangles.getSize() <= (1 << meshShape->m_bitsPerIndex), "You have too many triangles in a chunk."
		"Try setting the bits per index in the shape key to a larger value. See hkpCompressedMeshShape::setShapeKeyBitsPerIndex." );	
	for ( int i = 0; i < m_leftOver.m_triangles.getSize(); ++i )
	{
		hkpCompressedMeshShape::BigTriangle tri;
		tri.m_a = (hkUint16)( bigVerticesBase + m_leftOver.m_triangles[i].m_a );
		tri.m_b = (hkUint16)( bigVerticesBase + m_leftOver.m_triangles[i].m_b );
		tri.m_c = (hkUint16)( bigVerticesBase + m_leftOver.m_triangles[i].m_c );
		tri.m_material = m_leftOver.m_triangles[i].m_material;
		tri.m_weldingInfo = 0;
		
		HK_ASSERT2( 0x23674f5a, !hkpTriangleUtil::isDegenerate( m_leftOver.m_vertices[ m_leftOver.m_triangles[i].m_a ],
			m_leftOver.m_vertices[ m_leftOver.m_triangles[i].m_b ],
			m_leftOver.m_vertices[ m_leftOver.m_triangles[i].m_c ] ), "degenerate" );

		*list++ = tri;
	}

	// clear the left over geometry for the next input geometry
	m_statistics.m_numBigTriangles += m_leftOver.m_triangles.getSize();
	m_statistics.m_numBigVertices += m_leftOver.m_vertices.getSize();
	m_leftOver.m_vertices.clear();
	m_leftOver.m_triangles.clear();

	return bigTrianglesBase;
}

void HK_CALL hkpCompressedMeshShapeBuilder::snapGeometry( hkGeometry& geometry, hkReal quantizationError )
{
	for ( int i = 0; i < geometry.m_vertices.getSize(); ++i )
	{
		snapToGrid( geometry.m_vertices[i], quantizationError );
	}
}

void projectOnEdge( hkVector4& v, const hkVector4& a, const hkVector4& b )
{
	hkVector4 edge;
	edge.setSub( b, a );
	hkSimdReal len = edge.length<3>();

	hkVector4 delta1;
	delta1.setSub( v, a );
	//hkReal dot1 = delta1.dot3( edge );
	hkSimdReal len1 = delta1.length<3>();

	hkVector4 delta2;
	delta2.setSub( b, v );
	//hkReal dot2 = delta2.dot3( edge );
	hkSimdReal len2 = delta2.length<3>();

	hkSimdReal ratio = len1 / len;
	//hkReal ratio1 = ratio;
	hkSimdReal ratio2 = hkSimdReal_1 - ( len2 / len ) ;

	if ( ratio.isGreaterEqualZero() && ratio.isLessEqual(hkSimdReal_1) )
	{
		v.setAddMul( a, edge, ratio );
	}
	else
	{
		HK_REPORT( "ratio: " << ratio.getReal() << ", " << ratio2.getReal() );
	}
}

void hkpCompressedMeshShapeBuilder::snapGeometry( MappingTree* mapping )
{
	// optionally weld the vertices before detecting T-junctions and snapping the geometry 
	if ( m_weldVertices )
	{
		if ( mapping != HK_NULL )
		{
			hkArray<int> vertexRemap, triangleRemap;
			hkArray<TriangleMapping> newMapping;
			hkGeometryUtils::weldVertices( m_geometry, m_weldTolerance, false, vertexRemap, triangleRemap );
			
			for ( int i = 0; i < triangleRemap.getSize(); ++i )
			{
				const int remap = triangleRemap[i];
				if ( remap != -1 )
				{
					newMapping.pushBack( mapping->m_triangles[i] );
				}
				else
				{
					++m_statistics.m_numDegenerate;
				}
			}

			//HK_ON_DEBUG( testMapping( m_geometry, newMapping ) );

			mapping->m_triangles.swap( newMapping );

		}
		else
		{
			hkGeometryUtils::weldVertices( m_geometry, m_weldTolerance );
		}
	}

	// detect the T-junctions
	m_TjunctionsBase = m_Tjunctions.getSize();
	hkTjunctionDetector::detect( m_geometry, m_Tjunctions, m_weldedVertices, m_TjunctionTolerance, m_weldTolerance );

	// snap the geometry
	snapGeometry( m_geometry, m_error );

	// move the T-vertices
	for ( int i = m_TjunctionsBase; i < m_Tjunctions.getSize(); ++i )
	{
		hkTjunctionDetector::ProximityInfo& info = m_Tjunctions[i];
		hkVector4& Tvertex = m_geometry.m_vertices[info.m_index];

		// get triangle
		const hkGeometry::Triangle& triangle = m_geometry.m_triangles[info.m_key];

		const hkVector4& v0 = m_geometry.m_vertices[triangle.m_a];
		const hkVector4& v1 = m_geometry.m_vertices[triangle.m_b];
		const hkVector4& v2 = m_geometry.m_vertices[triangle.m_c];

		if ( info.m_type == hkTjunctionDetector::NEAR_FACE )
		{
			// compute normal
			hkVector4 normal;
			hkpTriangleUtil::calcNormal( normal, v0, v1, v2 );
			normal.normalize<3>();

			// compute distance to plane
			hkVector4 delta;
			delta.setSub( Tvertex, v0 );
			hkSimdReal dot = delta.dot<3>( normal );

			// compute projection
			Tvertex.subMul( dot, normal );
		}
		else if ( info.m_type == hkTjunctionDetector::NEAR_EDGE0 )
		{
			projectOnEdge( Tvertex, v0, v1 );
		}
		else if ( info.m_type == hkTjunctionDetector::NEAR_EDGE1 )
		{
			projectOnEdge( Tvertex, v1, v2 );
		}
		else if ( info.m_type == hkTjunctionDetector::NEAR_EDGE2 )
		{
			projectOnEdge( Tvertex, v2, v0 );
		}

		info.m_vertex = Tvertex;
		info.m_v0 = v0;
		info.m_v1 = v1;
		info.m_v2 = v2;

	}
}

void hkpCompressedMeshShapeBuilder::filterGeometry( MappingTree* mapping )
{
	hkArray<int> TvertexMap;
	if ( m_preserveTjunctions )
	{
		// mark the T-vertices
		TvertexMap.setSize( m_geometry.m_vertices.getSize() );
		for ( int i = 0; i < TvertexMap.getSize(); ++i )
		{
			TvertexMap[i] = 0;
		}
		for ( int i = m_TjunctionsBase; i < m_Tjunctions.getSize(); ++i )
		{
			TvertexMap[ m_Tjunctions[i].m_index ] = -1;
		}
	}

	// filter unwanted triangles
	hkArray<TriangleMapping> newMapping; 
	hkGeometry::Triangle* nonDegenerate = m_geometry.m_triangles.begin();
	for ( int i = 0; i < m_geometry.m_triangles.getSize(); ++i )
	{
		const int a = m_geometry.m_triangles[i].m_a;
		const int b = m_geometry.m_triangles[i].m_b;
		const int c = m_geometry.m_triangles[i].m_c;

		const hkVector4& v0 = m_geometry.m_vertices[a];
		const hkVector4& v1 = m_geometry.m_vertices[b];
		const hkVector4& v2 = m_geometry.m_vertices[c];

		// check for degenerate triangles
		if ( hkpTriangleUtil::isDegenerate( v0, v1, v2 ) )
		{
			++m_statistics.m_numDegenerate;
			continue;
		}

		TriangleMapping* map = ( mapping != HK_NULL ) ? &mapping->m_triangles[i] : HK_NULL;

		// check for triangles containing T-vertices
		if ( m_preserveTjunctions && ( ( TvertexMap[a] == -1 ) || ( TvertexMap[b] == -1 ) || ( TvertexMap[c] == -1 ) ) )
		{
			addLeftOverTriangle( v0, v1, v2, m_geometry.m_triangles[i].m_material, map );
			continue;
		}

		// check for big triangles
		hkAabb bounds;
		bounds.setEmpty();
		bounds.includePoint( v0 );
		bounds.includePoint( v1 );
		bounds.includePoint( v2 );

		hkVector4 extents;
		extents.setSub( bounds.m_max, bounds.m_min );
		hkReal maxExtent = extents( extents.getIndexOfMaxAbsComponent<3>() );

		if ( maxExtent > ( MAX_UINT16 - 1 ) * m_error )
		{
			addLeftOverTriangle( v0, v1, v2, m_geometry.m_triangles[i].m_material, map );
			continue;
		}

		*nonDegenerate++ = m_geometry.m_triangles[i];
		if ( mapping != HK_NULL )
		{
			newMapping.pushBack( mapping->m_triangles[i] );
		}
	}
	m_geometry.m_triangles.setSize( (int) ( nonDegenerate - m_geometry.m_triangles.begin() ) );
	
	// replace the mapping
	if ( mapping != HK_NULL )
	{
		mapping->m_triangles.swap( newMapping );
	}

	if ( m_preserveTjunctions )
	{
		// remove T-vertices from the vertex list
		hkVector4* nonTjunction = m_geometry.m_vertices.begin();
		int index = 0;
		for ( int i = 0; i < m_geometry.m_vertices.getSize(); ++i )
		{
			if ( TvertexMap[i] != -1 )
			{
				*nonTjunction++ = m_geometry.m_vertices[i];
				TvertexMap[i] = index++;
			}
		}
		m_geometry.m_vertices.setSize( (int) ( nonTjunction - m_geometry.m_vertices.begin() ) );

		// remap the vertex indices
		for ( int i = 0; i < m_geometry.m_triangles.getSize(); ++i )
		{
			hkGeometry::Triangle& triangle = m_geometry.m_triangles[i];
			triangle.m_a = TvertexMap[triangle.m_a];
			triangle.m_b = TvertexMap[triangle.m_b];
			triangle.m_c = TvertexMap[triangle.m_c];
		}
	}
}

bool hkpCompressedMeshShapeBuilder::addConvexPiece( const hkArray<hkVector4>& vertices, hkpCompressedMeshShape* compressedMesh )
{
	hkAabb aabb;
	hkAabbUtil::calcAabb( vertices.begin(), vertices.getSize(), aabb );

	hkVector4 extents; extents.setSub( aabb.m_max, aabb.m_min );
	int maxDir = extents.getIndexOfMaxAbsComponent<3>();
	if ( extents( maxDir ) >= ( MAX_UINT16 * m_error - 1 ) )
	{
		return false;
	}

	hkgpConvexHull::BuildConfig hullConfig;
	hullConfig.m_allowLowerDimensions = true;

	hkgpConvexHull hull;
	hull.build( vertices, hullConfig );

	hkArray<hkVector4> hullVertices;
	hull.fetchPositions( hkgpConvexHull::INTERNAL_VERTICES, hullVertices );
	if ( hullVertices.getSize() > hkpCompressedMeshShape::MAX_CONVEX_VERTICES )
	{
		HK_WARN_ALWAYS(0x1CC91291, "Number of convex hull vertices ("<<hullVertices.getSize()<<") too large, maximum allowed: "<<hkpCompressedMeshShape::MAX_CONVEX_VERTICES);
		return false;
	}
	m_statistics.m_numVertices += hullVertices.getSize();

	hkpCompressedMeshShape::ConvexPiece convexPiece;
	convexPiece.m_offset = aabb.m_min;
	snapToGrid( convexPiece.m_offset, m_error );
	quantizeVertices( m_error, convexPiece.m_offset, hullVertices, convexPiece.m_vertices );

	compressedMesh->m_convexPieces.pushBack( convexPiece );
	HK_ASSERT2( 0x735e3138, compressedMesh->m_convexPieces.getSize() <= (1 << compressedMesh->m_bitsPerIndex), "You have too many triangles in a chunk."
		"Try setting the bits per index in the shape key to a larger value. See hkpCompressedMeshShape::setShapeKeyBitsPerIndex." );	

	return true;
}

bool hkpCompressedMeshShapeBuilder::addConvexPiece( const hkGeometry& geometry, hkpCompressedMeshShape* compressedMesh )
{
	bool ret = addConvexPiece( geometry.m_vertices, compressedMesh );

	if ( ret && m_createMapping )
	{
		// initialize the keys to invalid
		const int keysBase = m_shapeKeys.getSize();
		m_shapeKeys.expandBy( geometry.m_triangles.getSize() );
		for ( int i = keysBase; i < m_shapeKeys.getSize(); ++i )
		{
			m_shapeKeys[i] = HK_INVALID_SHAPE_KEY;
		}
	}

	return ret;
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
