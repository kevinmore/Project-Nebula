/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>
#include <Physics2012/Collide/Shape/Deprecated/CompressedMesh/hkpCompressedMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/Mesh/hkpMeshMaterial.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>

#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Base/Thread/Job/ThreadPool/Cpu/hkCpuJobThreadPool.h>

#include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>

#ifdef HK_PLATFORM_SPU
#	include <Common/Base/Spu/Dma/Iterator/hkSpuReadOnlyIterator.h>
#endif

#if !defined(HK_PLATFORM_SPU)
hkpCompressedMeshShape::hkpCompressedMeshShape( int bitsPerIndex, hkReal radius )
:	hkpShapeCollection( HKCD_SHAPE_TYPE_FROM_CLASS(hkpCompressedMeshShape), COLLECTION_COMPRESSED_MESH ),
	m_bitsPerIndex ( bitsPerIndex ),
	m_bitsPerWIndex ( m_bitsPerIndex + 1 ),
	m_wIndexMask ( ( 1 << m_bitsPerWIndex ) - 1 ),
	m_indexMask( ( 1 << m_bitsPerIndex ) - 1 ),
	m_radius ( radius ),
	m_defaultCollisionFilterInfo( 0 ),
	m_meshMaterials( HK_NULL ),
	m_numMaterials( 0 )
{
	m_weldingType = hkpWeldingUtility::WELDING_TYPE_NONE;
	m_materialStriding = sizeof(hkpMeshMaterial);
}

hkpCompressedMeshShape::hkpCompressedMeshShape( hkFinishLoadedObjectFlag flag ) : 
	hkpShapeCollection(flag),
	m_materials( flag ),
	m_materials16( flag ),
	m_materials8( flag ),
	m_transforms( flag ),
	m_bigVertices( flag ),
	m_bigTriangles( flag ),
	m_chunks( flag ),
	m_convexPieces( flag ),
	m_namedMaterials( flag )
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpCompressedMeshShape));
		m_collectionType = COLLECTION_COMPRESSED_MESH;
		if ( m_namedMaterials.getSize() != 0 )
		{
			m_meshMaterials = m_namedMaterials.begin();
			m_materialStriding = sizeof(hkpNamedMeshMaterial);
		}
	}
}

int hkpCompressedMeshShape::Chunk::getNextIndex( int index, int& winding ) const
{
	winding = 0; // default winding value

	// if index is negative (-1) then return the first index (0)
	if ( index < 0 )
	{
		HK_ASSERT2( 0x4582d4f3, m_indices.getSize() != 0, "empty chunk found" );
		return 0;
	}
	
	++index;

	// find the strip
	int strip = 0;
	const int numStrips = m_stripLengths.getSize();
	int top = 0;
	int base = 0;
	if ( numStrips != 0 )
	{
		top = m_stripLengths[strip];
		while ( index >= top )
		{	
			++strip;
			if ( strip >= numStrips )
				break;
			base = top;
			top += m_stripLengths[strip];
		}
	}

	if ( strip >= numStrips )
	{
		// we are dealing with lists now
		while ( ( index - top ) % 3 != 0 )
		{
			++index;
		}
	}
	else if ( index >= top - 2 )
	{
		// invalid triangle - we are between strips
		index = top;
	}
	else
	{
		// we are dealing with strips
		winding = ( index - base ) & 1;
	}

	if ( index >= m_indices.getSize() )
	{
		return -1;
	}
	
	return index;
}

// the key structure is as following:
// the value stored in the first (bitsPerChunk = 32 - m_bitsPerWIndex) bits tells us the type of the child shape
// *	0 means the key describes a big triangle
// *	1 to ( 1 << bitsPerChunk ) - 2 is a chunk
// *	( 1 << bitsPerChunk ) - 1 is a convex piece
// the rest of the bits (m_bitsWPerIndex) store an index; in the case of chunks one bit is reserved for winding (1 = revert winding)
//
// big triangle: < 000.. | index >
// chunk: < id | w | index >, id = chunk index + 1, w = 0/1
// convex piece: < 111.. | index >

hkpShapeKey hkpCompressedMeshShape::getFirstKey() const
{
	return getNextKey( HK_INVALID_SHAPE_KEY );
}

hkpShapeKey hkpCompressedMeshShape::getNextKey( hkpShapeKey oldKey ) const
{
	int chunkId, start;
	if ( oldKey == HK_INVALID_SHAPE_KEY )
	{
		// this is for the first key
		chunkId = 0;
		start = -1;
	}
	else
	{
		chunkId = oldKey >> m_bitsPerWIndex;
		start = oldKey & m_indexMask;
	}
	
	// if true look for the key in the big triangle list
	if ( chunkId == 0 )
	{
		for ( int i = start + 1; i < m_bigTriangles.getSize(); ++i )
		{
			const hkVector4& v0 = m_bigVertices[m_bigTriangles[i].m_a];
			const hkVector4& v1 = m_bigVertices[m_bigTriangles[i].m_b];
			const hkVector4& v2 = m_bigVertices[m_bigTriangles[i].m_c];

			if ( !hkpTriangleUtil::isDegenerate( v0, v1, v2 ) )
			{
				return i; // < 0 | i >
			}
		}

		// if no big triangle found go to the chunks next
		++chunkId;
		start = -1;
	}

	const int bitsPerChunk = 32 - m_bitsPerWIndex;
	const int mask = ( 1 << bitsPerChunk ) - 1;

	// if the key does not point to a convex piece, look in the chunks
	if ( chunkId != mask )
	{
		for (int i = chunkId - 1; i < m_chunks.getSize(); ++i )
		{
			const Chunk& chunk = m_chunks[i];
			int winding = 0;
			int index = 0;

			// loop until we reach the end of the chunk
			while ( 1 )
			{
				if ( chunk.m_reference != HK_CMS_NULL_REF )
				{
					// use the referenced chunk to get the next index
					index = m_chunks[chunk.m_reference].getNextIndex( start, winding );
				}
				else
				{
					index = chunk.getNextIndex( start, winding );
				}

				HK_ASSERT2( 0x4582d4f4, index <= m_indexMask, "Not enough bits for storing chunk indices. You need to increase m_bitsPerIndex." );
				if ( index == -1 )
				{
					break;
				}

				// build the shape key and get the triangle shape to test it
				{
					hkpShapeKey key = ( ( i + 1 ) << m_bitsPerWIndex ) | ( ( winding & 1 ) << m_bitsPerIndex ) | ( index & m_wIndexMask ); // < i + 1 | winding | index >
					hkpShapeBuffer buffer;
					const hkpShape* shape = getChildShape( key, buffer );
					HK_ASSERT2( 0x4582d4f4, shape->getType() == hkcdShapeType::TRIANGLE, "The shape should be a triangle" );
					const hkpTriangleShape* triangle = static_cast<const hkpTriangleShape*>( shape );
					const hkVector4* v = triangle->getVertices();
					// if the triangle is not degenerate return the key
					if ( !hkpTriangleUtil::isDegenerate( v[0], v[1], v[2] ) )
					{
						return key;
					}
					// else try and get the next index
					start = index;
				}
			}

			// if no next index found move on to the next chunk or to the convex pieces
			start = -1;
		}
	}

	// last get the convex pieces
	start++;
	if ( start < m_convexPieces.getSize() )
	{
		return ( ( mask << m_bitsPerWIndex ) | start ); // < 111... | oldIndex + 1 >
	}

	return HK_INVALID_SHAPE_KEY;
}

void hkpCompressedMeshShape::getChunkAabb( const Chunk& chunk, hkAabb& bounds )
{
	
	hkQsTransform transform;
	transform.setIdentity();
	if ( chunk.m_transformIndex != HK_CMS_NULL_REF )
	{
		transform = m_transforms[chunk.m_transformIndex];
	}

	hkArray<hkVector4> vertices;
	if ( chunk.m_reference == HK_CMS_NULL_REF )
	{
		chunk.getVertices( m_error, transform, vertices );
	}
	else
	{
		m_chunks[chunk.m_reference].getVertices( m_error, transform, vertices );
	}

	hkAabbUtil::calcAabb( vertices.begin(), vertices.getSize(), bounds );
	bounds.expandBy( hkSimdReal::fromFloat(m_radius) );
}

void hkpCompressedMeshShape::getConvexPieceAabb( const ConvexPiece& piece, hkAabb& bounds )
{
	hkQsTransform transform;
	transform.setIdentity();
	if ( piece.m_transformIndex != HK_CMS_NULL_REF )
	{
		transform = m_transforms[piece.m_transformIndex];
	}

	hkArray<hkVector4> vertices;
	if ( piece.m_reference == HK_CMS_NULL_REF )
	{
		piece.getVertices( m_error, transform, vertices );
	}
	else
	{
		m_convexPieces[piece.m_reference].getVertices( m_error, transform, vertices );
	}

	hkAabbUtil::calcAabb( vertices.begin(), vertices.getSize(), bounds );
}

#if !defined(HK_PLATFORM_SPU)

int hkpCompressedMeshShape::calcSizeForSpu(const CalcSizeForSpuInput& input, int spuBufferSizeLeft) const
{
#if defined(HK_PLATFORM_HAS_SPU)
	// no dynamic compressed mesh shapes on the spu
	if ( !input.m_isFixedOrKeyframed && input.m_midphaseAgent3Registered )
	{
		HK_WARN(0xdbc1ffbc, "This hkpCompressedMeshShape cannot run on SPU - midphase agent is registered and rigid body is not fixed or keyframed");
		return -1;
	}
#endif
	return sizeof(*this);
}

#endif

void hkpCompressedMeshShape::Chunk::getTriangle( int a, int b, int c, hkReal error, hkVector4& v1, hkVector4& v2, hkVector4& v3 ) const
{
	hkSimdReal err = hkSimdReal::fromFloat( error );
	hkVector4 v1Ints; v1Ints.set( m_vertices[a], m_vertices[a + 1], m_vertices[a + 2] );
	v1.setAddMul( m_offset, v1Ints, err);

	hkVector4 v2Ints; v2Ints.set( m_vertices[b], m_vertices[b + 1], m_vertices[b + 2] );
	v2.setAddMul( m_offset, v2Ints, err );

	hkVector4 v3Ints; v3Ints.set( m_vertices[c], m_vertices[c + 1], m_vertices[c + 2] );
	v3.setAddMul( m_offset, v3Ints, err );
}

int hkpCompressedMeshShape::Chunk::getNumTriangles() const
{
	int numTriangles = 0;
	int top = 0;
	for ( int i = 0; i < m_stripLengths.getSize(); ++i )
	{
		top += m_stripLengths[i];
		numTriangles += m_stripLengths[i] - 2;
	}
	numTriangles += ( m_indices.getSize() - top ) / 3;
	return numTriangles;
}

int hkpCompressedMeshShape::isValidShapeKey( hkpShapeKey key )
{
	int chunkId = key >> m_bitsPerWIndex;
	int index = key & m_wIndexMask;
	int winding = index >> m_bitsPerIndex;
	index &= m_indexMask;

	if ( chunkId == 0 )
	{
		if ( winding != 0 )
		{
			// wrong winding for big triangle
			return 1;
		}
	}
	else
	{
		--chunkId;
		if ( chunkId >= m_chunks.getSize() )
		{
			// chunk index too large
			return 2;
		}
		Chunk* chunk = &m_chunks[chunkId];
		if ( chunk->m_reference != HK_CMS_NULL_REF )
		{
			chunkId = chunk->m_reference;
			chunk = &m_chunks[chunkId];
		}
		int w;
		int test = chunk->getNextIndex( index - 1, w );
		if ( test != index )
		{
			// wrong index
			return 3;
		}
		if ( w != winding )
		{
			// wrong winding
			return 4;
		}
	}

	return 0;
}

void hkpCompressedMeshShape::setWeldingInfo( hkpShapeKey key, hkInt16 weldingInfo)
{
	int chunkId = key >> m_bitsPerWIndex;
	int index = key & m_wIndexMask;

	if ( chunkId == 0 )
	{
		m_bigTriangles[index].m_weldingInfo = weldingInfo;
	}
	else
	{
		chunkId--;
		index = key & m_indexMask;
		if ( m_chunks[chunkId].m_reference != HK_CMS_NULL_REF )
		{
			chunkId = m_chunks[chunkId].m_reference;
		}		
		m_chunks[chunkId].m_weldingInfo[index] = weldingInfo;
	}
}

void hkpCompressedMeshShape::initWeldingInfo( hkpWeldingUtility::WeldingType weldingType )
{
	m_weldingType = weldingType;
	if ( weldingType == hkpWeldingUtility::WELDING_TYPE_NONE )
	{
		return;
	}

	for ( int i = 0; i < m_chunks.getSize(); ++i )
	{
		m_chunks[i].m_weldingInfo.setSize( m_chunks[i].m_indices.getSize(), 0 );
	}
}

void hkpCompressedMeshShape::setShapeKeyBitsPerIndex( int bitsPerIndex )
{
	HK_ASSERT2( 0x34f6a7c0, bitsPerIndex > 0 && bitsPerIndex < 30, "Invalid value for bits per index" );
	if ( bitsPerIndex == m_bitsPerWIndex )
	{
		return;
	}
	HK_ASSERT2( 0x34f6a7c1, m_chunks.getSize() == 0 && m_bigTriangles.getSize() == 0 && m_convexPieces.getSize() == 0,
		"The compressed mesh shape already contains data. You cannot change the shape key structure now." );
	m_bitsPerIndex = bitsPerIndex;
	m_bitsPerWIndex = m_bitsPerIndex + 1;
	m_wIndexMask = ( 1 << m_bitsPerWIndex ) - 1;
	m_indexMask = ( 1 << m_bitsPerIndex ) - 1;
}

void hkpCompressedMeshShape::Chunk::getTriangle( int index, hkReal error, hkVector4& v1, hkVector4& v2, hkVector4& v3 ) const
{
	int a = m_indices[index] * 3;
	int b = m_indices[index + 1] * 3;
	int c = m_indices[index + 2] * 3;

	getTriangle( a, b, c, error, v1, v2, v3 );
}

void hkpCompressedMeshShape::Chunk::getVertices( hkReal quantization, const hkQsTransform& transform, hkArray<hkVector4>& vertices ) const
{
	HK_ASSERT( 0x34f627c3, m_reference == HK_CMS_NULL_REF );

	vertices.setSize( m_vertices.getSize() / 3 );
	const hkSimdReal quant = hkSimdReal::fromFloat(quantization);
	for ( int i = 0; i < m_vertices.getSize(); i += 3 )
	{
		hkVector4 v;
		v.set( m_vertices[i], m_vertices[i + 1], m_vertices[i + 2] );
		v.setAddMul( m_offset, v, quant );
		v.setTransformedPos( transform, v );
		vertices[i / 3] = v;
	}
}

void hkpCompressedMeshShape::ConvexPiece::getVertices( hkReal quantization, const hkQsTransform& transform, hkArray<hkVector4>& vertices ) const
{
	HK_ASSERT( 0x34f627c3, m_reference == HK_CMS_NULL_REF );

	vertices.setSize( m_vertices.getSize() / 3 );
	const hkSimdReal quant = hkSimdReal::fromFloat(quantization);
	for ( int i = 0; i < m_vertices.getSize(); i += 3 )
	{
		hkVector4 v;
		v.set( m_vertices[i], m_vertices[i + 1], m_vertices[i + 2] );
		v.setAddMul( m_offset, v, quant );
		v.setTransformedPos( transform, v );
		vertices[i / 3] = v;
	}
}

#else

const void* HK_CALL GetArrayElemWithByteStridingHalfCacheSize( const void* base, int index, int elemsize, int dmaGroup, bool waitForCompletion )
{	
	hkUlong arrayAddrPpu = hkUlong(base) + ( index * elemsize );
	// We will virtually use only half the cache size while still bringing in the full size. That way we have a spill-over buffer of half the cache size that can be used for otherwise out-of-bounds accesses.
	const hkUlong mask  = ~static_cast<hkUlong>((HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE/2)-1);
	hkUlong arrayAddrAligned = arrayAddrPpu & mask;
	hkUlong alignedDataSpu = (hkUlong)g_SpuCollideUntypedCache->getFromMainMemoryInlined( (const void*)arrayAddrAligned , HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE, dmaGroup, waitForCompletion );
	return reinterpret_cast<const void*> ( alignedDataSpu + (arrayAddrPpu & ~mask) );
}

void hkpCompressedMeshShape::Chunk::decompressVertex( hkInt16 index, const hkSimdReal& error, hkVector4& v )
{
	const hkUint16* cache = fetchDma<hkUint16>( m_vertices.begin(), index * HK_HINT_SIZE16(3) );
	hkVector4 vint; vint.set( cache[0], cache[1], cache[2] );
	v.setAddMul( m_offset, vint, error );
}

HK_ALIGN16( char convexPiece[ sizeof( hkpCompressedMeshShape::ConvexPiece ) ] );

HK_ALIGN16( hkVector4 hkpCompressedMeshShape::m_spuPlanes[hkpCompressedMeshShape::MAX_CONVEX_FACES] );

#endif

void hkpCompressedMeshShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out ) const
{
	
	hkVector4 center; m_bounds.getCenter( center );
	hkVector4 halfExtents; m_bounds.getHalfExtents( halfExtents );
	hkAabbUtil::calcAabb( localToWorld, halfExtents, center, hkSimdReal::fromFloat(tolerance), out );
}

#define MAX_ROTATED_VERTICES HK_NEXT_MULTIPLE_OF( 4, hkpCompressedMeshShape::MAX_CONVEX_VERTICES / 4 )

struct ShortVector
{
	hkUint16 x, y, z;
};

#if defined(HK_PLATFORM_SPU)
	HK_ALIGN16( static hkFourTransposedPoints rotatedVertices[MAX_ROTATED_VERTICES] );
	// Want this for memClear16 below
	HK_COMPILE_TIME_ASSERT ( sizeof(hkpConvexVerticesShape) % 16 == 0);
#elif defined(HK_PLATFORM_PPU)
	HK_ALIGN_REAL( hkFourTransposedPoints rotatedVerticesPool[1][MAX_ROTATED_VERTICES] );
#else
	HK_ALIGN_REAL( hkFourTransposedPoints rotatedVerticesPool[HK_MAX_NUM_THREADS][MAX_ROTATED_VERTICES] );
#endif

const hkpShape* hkpCompressedMeshShape::getChildShape(hkpShapeKey key, hkpShapeBuffer& buffer) const
{
	int chunkId = key >> m_bitsPerWIndex;
	int index = key & m_wIndexMask;
	const int winding = index >> m_bitsPerIndex;

	//
	// convex pieces case
	//

	if ( chunkId == (1 << (32 - m_bitsPerWIndex)) - 1 )
	{
		hkQsTransform transform; transform = hkQsTransform::getIdentity();
		HK_COMPILE_TIME_ASSERT( sizeof(hkpConvexVerticesShape) <= HK_SHAPE_BUFFER_SIZE );
#ifdef HK_PLATFORM_SPU
		hkpConvexVerticesShape* polytope = (hkpConvexVerticesShape*)&buffer;
		hkString::memClear16( polytope, sizeof(hkpConvexVerticesShape) / 16 );

		polytope->setRadius( m_radius );
		polytope->setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexVerticesShape));

		const ConvexPiece* cachedPiece = FETCH_DMA( ConvexPiece, m_convexPieces.begin(), index );
		const ConvexPiece* piece = (const ConvexPiece*)&convexPiece;
		hkString::memCpy16< sizeof(ConvexPiece) >( (void*)piece, cachedPiece );

		if ( piece->m_transformIndex != HK_CMS_NULL_REF )
		{
			transform = *FETCH_DMA( hkQsTransform, m_transforms.begin(), piece->m_transformIndex );
		}

		if ( piece->m_reference != HK_CMS_NULL_REF )
		{
			cachedPiece = FETCH_DMA( ConvexPiece, m_convexPieces.begin(), piece->m_reference );
			hkString::memCpy16< sizeof(ConvexPiece) >( (void*)piece, cachedPiece );
		}

#else
		hkpConvexVerticesShape* polytope = new (&buffer) hkpConvexVerticesShape( m_radius );
		if ( m_convexPieces[index].m_transformIndex != HK_CMS_NULL_REF )
		{
			transform = m_transforms[ m_convexPieces[index].m_transformIndex ];
		}

		if ( m_convexPieces[index].m_reference != HK_CMS_NULL_REF )
		{
			index = m_convexPieces[index].m_reference;
		}
		const ConvexPiece* piece = &m_convexPieces[index];

		const int threadNo = HK_THREAD_LOCAL_GET( hkThreadNumber );
		hkFourTransposedPoints* rotatedVertices = rotatedVerticesPool[threadNo];
#endif

		HKCD_PATCH_SHAPE_VTABLE( polytope );

		const int numVertices = piece->m_vertices.getSize() / 3 ;
		HK_ASSERT2(0x593f7a2f, numVertices <= MAX_CONVEX_VERTICES, " too many vertices in the convex piece");
		const int paddedSize = HK_NEXT_MULTIPLE_OF(4, numVertices);
		const int numRotatedVertices = paddedSize >> 2;

		// write the rotated vertices buffer
		hkAabb aabb;
		aabb.setEmpty();
		{
#ifdef HK_PLATFORM_SPU
			hkSpuReadOnlyIterator<ShortVector, MAX_CONVEX_VERTICES * 3, 0> ints( (ShortVector*)piece->m_vertices.begin() );
#else
			ShortVector* ints = (ShortVector*)piece->m_vertices.begin();
#endif

			hkVector4 v[4];
			const hkVector4& offset = piece->m_offset;
			const hkSimdReal error = hkSimdReal::fromFloat( m_error );
			int k = 0;
			for ( int i = 0; i < numRotatedVertices; ++i )
			{
				// decompress 4 vertices
				for ( int j = 0; j < 4; ++j )
				{					
					hkVector4 vertex; vertex.set( (*ints).x, (*ints).y, (*ints).z );
					vertex.setAddMul( offset, vertex, error );
					v[j].setTransformedPos( transform, vertex );
					aabb.includePoint( v[j] );
					if ( k < numVertices - 1 )
					{
						++k;
						ints++;
					}
				}

				rotatedVertices[i].set(v[0], v[1], v[2], v[3]);
			}
		}
			
		polytope->m_rotatedVertices.setDataAutoFree( rotatedVertices, numRotatedVertices, MAX_ROTATED_VERTICES);
		polytope->m_numVertices = numVertices;		
		polytope->m_useSpuBuffer = true;
		aabb.getHalfExtents( polytope->m_aabbHalfExtents );
		aabb.getCenter( polytope->m_aabbCenter );
		return polytope;
	}

	//
	// triangles case
	//
	
	HK_ASSERT(0x73f97fa7, sizeof( hkpTriangleShape ) <= HK_SHAPE_BUFFER_SIZE );

#if !defined ( HK_PLATFORM_SPU )
	hkpTriangleShape* HK_RESTRICT triangleShape = new (&buffer) hkpTriangleShape();
#else
	hkpTriangleShape* HK_RESTRICT triangleShape = (hkpTriangleShape*)&buffer;
	triangleShape->setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpTriangleShape));
	triangleShape->setExtrusion( hkVector4::getZero() );
#endif
	HKCD_PATCH_SHAPE_VTABLE( triangleShape );

	triangleShape->setRadius( m_radius );
	triangleShape->setWeldingType( m_weldingType );

	// if the key corresponds to a big (unquantized) triangle
	if ( chunkId == 0 )
	{
		HK_ASSERT2(0x593f7a2f, index >= 0 && index < m_bigTriangles.getSize(), "hkpShapeKey invalid");

		hkVector4 v0, v1, v2;
		hkUint16 welding;
		getBigTriangle( index, v0, v1, v2, &welding );

		triangleShape->setWeldingInfo( welding );
		triangleShape->setVertex<0>( v0 );
		triangleShape->setVertex<1>( v1 );
		triangleShape->setVertex<2>( v2 );
	}
	else
	{
		index &= m_indexMask;
		--chunkId;
		hkQsTransform transform; transform.setIdentity();

#ifdef HK_PLATFORM_SPU
		// get the chunk from main memory
		HK_DECLARE_ALIGNED_LOCAL_PTR( Chunk, chunk, sizeof(Chunk) );

		const Chunk* cachedChunk = FETCH_DMA( Chunk, m_chunks.begin(), chunkId );

		HK_COMPILE_TIME_ASSERT( (sizeof(Chunk) & 0xf) == 0 );
		hkString::memCpy16< sizeof(Chunk) >( chunk, cachedChunk );

		// get transform before referencing another chunk
		if ( chunk->m_transformIndex != HK_CMS_NULL_REF )
		{
			transform = *FETCH_DMA( hkQsTransform, m_transforms.begin(), chunk->m_transformIndex );
		}
	
		
		// get the referenced chunk (if any)
		if ( chunk->m_reference != HK_CMS_NULL_REF )
		{
			chunkId = chunk->m_reference;

			cachedChunk = FETCH_DMA( Chunk, m_chunks.begin(), chunkId );
			hkString::memCpy16< sizeof(Chunk) >( chunk, cachedChunk );
		}

		if ( m_weldingType != hkpWeldingUtility::WELDING_TYPE_NONE )
		{
			const hkUint16* weldingInfo = FETCH_DMA( hkUint16, chunk->m_weldingInfo.begin(), index );
			triangleShape->setWeldingInfo( *weldingInfo );
		}

#else
		// get transform before referencing another chunk
		if ( m_chunks[chunkId].m_transformIndex != HK_CMS_NULL_REF )
		{
			transform = m_transforms[ m_chunks[chunkId].m_transformIndex ];
		}
		// get the referenced chunk (if any)
		if ( m_chunks[chunkId].m_reference != HK_CMS_NULL_REF )
		{
			chunkId = m_chunks[chunkId].m_reference;
		}
		const Chunk* chunk = &m_chunks[chunkId];
		if ( m_weldingType != hkpWeldingUtility::WELDING_TYPE_NONE )
		{
			triangleShape->setWeldingInfo( chunk->m_weldingInfo[index] );
		}
#endif

		HK_ASSERT2(0x593f7a2f, index >= 0 && index < chunk->m_indices.getSize() - 2, "hkpShapeKey invalid");

		hkVector4 v1, v2, v3;
#ifdef HK_PLATFORM_SPU
		const hkUint16* triangle = FETCH_DMA( hkUint16, chunk->m_indices.begin(), index );

		const hkSimdReal simdError = hkSimdReal::fromFloat(m_error);

		chunk->decompressVertex( triangle[0], simdError, v1 );
		chunk->decompressVertex( triangle[1], simdError, v2 );
		chunk->decompressVertex( triangle[2], simdError, v3 );
#else
		int a = chunk->m_indices[index] * 3;
		int b = chunk->m_indices[index + 1] * 3;
		int c = chunk->m_indices[index + 2] * 3;

		chunk->getTriangle( a, b, c, m_error, v1, v2, v3 );

#endif

		v1.setTransformedPos( transform, v1 );
		v2.setTransformedPos( transform, v2 );
		v3.setTransformedPos( transform, v3 );

		triangleShape->setVertex( winding << 1, v1 );
		triangleShape->setVertex<1>( v2 );
		triangleShape->setVertex( ( 1 ^ winding ) << 1, v3 );
	}

	return triangleShape;
}

const hkpMeshMaterial* hkpCompressedMeshShape::getMaterial( hkpShapeKey key ) const
{
	if ( m_meshMaterials == HK_NULL )
	{
		return HK_NULL;
	}

	int chunkId = key >> m_bitsPerWIndex;
	const int index = key & m_indexMask;
	int material = 0;

	const int bitsPerChunk = 32 - m_bitsPerWIndex;
	const int mask = ( 1 << bitsPerChunk ) - 1;

	if ( chunkId == 0 )
	{
		const BigTriangle* triangle = FETCH_DMA( BigTriangle, m_bigTriangles.begin(), index );
		material = triangle->m_material;
	}
	else if ( chunkId != mask )
	{
		const Chunk* chunk = FETCH_DMA( Chunk, m_chunks.begin(), chunkId - 1 );
		if ( chunk->m_reference != HK_CMS_NULL_REF )
		{
			chunkId = chunk->m_reference;
			chunk = FETCH_DMA( Chunk, m_chunks.begin(), chunkId );
		}
		hkUint32 materialInfo = chunk->m_materialInfo;
		switch( m_materialType )
		{
			case MATERIAL_SINGLE_VALUE_PER_CHUNK:
				material = materialInfo;
				break;
			case MATERIAL_FOUR_BYTES_PER_TRIANGLE:
				material = m_materials[materialInfo + index];
				break;
			case MATERIAL_TWO_BYTES_PER_TRIANGLE:
				material = m_materials16[materialInfo + index];
				break;
			case MATERIAL_ONE_BYTE_PER_TRIANGLE:
				material = m_materials8[materialInfo + index];
				break;
			default:
				return HK_NULL;
		}
	}
	else
	{
		
		return HK_NULL;
	}

#ifndef HK_PLATFORM_SPU
	return hkAddByteOffset( m_meshMaterials, material * m_materialStriding );
#else
	return (const hkpMeshMaterial*)GetArrayElemWithByteStridingHalfCacheSize( m_meshMaterials, material, m_materialStriding );
#endif
}

hkUint32 hkpCompressedMeshShape::getCollisionFilterInfo(hkpShapeKey key) const
{
	const hkpMeshMaterial* material = getMaterial( key );
	if ( material )
	{
		return material->m_filterInfo;
	}
	else
	{
		return m_defaultCollisionFilterInfo;
	}
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
