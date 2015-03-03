/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>

#include <Physics2012/Collide/Shape/Compound/Collection/Mesh/hkpMeshMaterial.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>

#include <Physics2012/Collide/Shape/Deprecated/Mesh/hkpMeshShape.h>

#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/ExtendedMeshShape/hkpExtendedMeshShape.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

// Ensure this shape and the MOPP both fit in a single cache line
#if defined(HK_PLATFORM_HAS_SPU)
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#endif

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>
#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Memory/PlatformUtils/Spu/SpuDmaCache/hkSpu4WayCache.h>
#endif

// Make sure that the memCpy below are valid
HK_COMPILE_TIME_ASSERT( (sizeof(hkpExtendedMeshShape::TrianglesSubpart)&0xF) == 0);
HK_COMPILE_TIME_ASSERT( (sizeof(hkpExtendedMeshShape::ShapesSubpart)&0xF) == 0);

#if !defined(HK_PLATFORM_SPU)

#include <Common/Base/Reflection/hkTypeInfo.h>

#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>

extern hkReal hkDefaultTriangleDegeneracyTolerance;

hkpExtendedMeshShape::hkpExtendedMeshShape( hkReal radius, int numBitsForSubpartIndex )
:	hkpShapeCollection(HKCD_SHAPE_TYPE_FROM_CLASS(hkpExtendedMeshShape), COLLECTION_EXTENDED_MESH )
{
	m_triangleRadius = radius;
	m_defaultCollisionFilterInfo = 0;

	HK_ASSERT2(0x16aa7e0a, numBitsForSubpartIndex > 0 && numBitsForSubpartIndex < 32,\
		"cinfo.m_numBitsForSubpartIndex must be greater than zero and less than 32."\
		"See comment in construction info for details on how this parameter is used.");

	m_numBitsForSubpartIndex = numBitsForSubpartIndex;

	m_weldingType = hkpWeldingUtility::WELDING_TYPE_NONE;

	// Initialize the cached AABB
	m_aabbHalfExtents = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();
	m_aabbCenter.setZero();

	m_cachedNumChildShapes = 0;
	m_materialClass = HK_NULL;
}

hkpExtendedMeshShape::hkpExtendedMeshShape( const class hkpMeshShape* meshShape )
:	hkpShapeCollection( HKCD_SHAPE_TYPE_FROM_CLASS(hkpExtendedMeshShape), COLLECTION_EXTENDED_MESH )
{
	m_triangleRadius	 	 = meshShape->m_radius;
	m_numBitsForSubpartIndex = meshShape->m_numBitsForSubpartIndex;
	m_weldingType			 = meshShape->m_weldingType;
	m_weldingInfo			 = meshShape->m_weldingInfo;

	if (meshShape->getNumSubparts() == 1)
	{
		m_trianglesSubparts.setDataUserFree( &m_embeddedTrianglesSubpart, 1, 1);
	}
	else
	{
		m_trianglesSubparts.setSize( meshShape->getNumSubparts() );
	}

	m_defaultCollisionFilterInfo = 0;

	for (int i = 0; i < meshShape->getNumSubparts(); ++i)
	{
		m_trianglesSubparts[i].m_numTriangleShapes = meshShape->getSubpartAt(i).m_numTriangles;
		m_trianglesSubparts[i].m_materialIndexBase = meshShape->getSubpartAt(i).m_materialIndexBase;
		m_trianglesSubparts[i].m_materialIndexStriding = (hkUint8)meshShape->getSubpartAt(i).m_materialIndexStriding;
		m_trianglesSubparts[i].setMaterialIndexStridingType((MaterialIndexStridingType)((hkInt8)meshShape->getSubpartAt(i).m_materialIndexStridingType));
		m_trianglesSubparts[i].m_materialBase = meshShape->getSubpartAt(i).m_materialBase;
		m_trianglesSubparts[i].m_materialStriding = (hkInt16)meshShape->getSubpartAt(i).m_materialStriding;
		m_trianglesSubparts[i].setNumMaterials((hkUint16)meshShape->getSubpartAt(i).m_numMaterials);
		m_trianglesSubparts[i].m_vertexBase = meshShape->getSubpartAt(i).m_vertexBase;
		m_trianglesSubparts[i].m_vertexStriding = (hkUint16)meshShape->getSubpartAt(i).m_vertexStriding;
		m_trianglesSubparts[i].m_numVertices = meshShape->getSubpartAt(i).m_numVertices;
		m_trianglesSubparts[i].m_indexBase = meshShape->getSubpartAt(i).m_indexBase;
		*(hkInt8*)&m_trianglesSubparts[i].m_stridingType = (hkInt8)meshShape->getSubpartAt(i).m_stridingType;
		m_trianglesSubparts[i].m_indexStriding = (hkUint16)meshShape->getSubpartAt(i).m_indexStriding;
		m_trianglesSubparts[i].m_triangleOffset = meshShape->getSubpartAt(i).m_triangleOffset;
		m_trianglesSubparts[i].m_extrusion.setZero();
		m_trianglesSubparts[i].m_flipAlternateTriangles = (hkInt8)meshShape->getSubpartAt(i).m_flipAlternateTriangles;
		m_trianglesSubparts[i].m_transform = hkQsTransform::getIdentity();

		HK_ON_DEBUG( assertTrianglesSubpartValidity(m_trianglesSubparts[i]); )
	}

	recalcAabbExtents();

	m_cachedNumChildShapes = hkpShapeCollection::getNumChildShapes();
}

hkpExtendedMeshShape::~hkpExtendedMeshShape()
{
}

hkpExtendedMeshShape::hkpExtendedMeshShape( hkFinishLoadedObjectFlag flag )
:	hkpShapeCollection(flag),
	m_embeddedTrianglesSubpart(flag),
	m_trianglesSubparts(flag),
	m_shapesSubparts(flag),
	m_weldingInfo(flag)
{
	if( flag.m_finishing )
	{
		m_collectionType = COLLECTION_EXTENDED_MESH;

		// 3.0 compatibility. m_materialIndexStridingType is loaded as binary zero
		// For 3.0 files material indices are always int8
		{
			for( int i = 0; i < m_trianglesSubparts.getSize(); ++i )
			{
				if( m_trianglesSubparts[i].getMaterialIndexStridingType() == MATERIAL_INDICES_INVALID )
				{
					m_trianglesSubparts[i].setMaterialIndexStridingType(MATERIAL_INDICES_INT8);
				}
			}
		}
		{
			for( int i = 0; i < m_shapesSubparts.getSize(); ++i )
			{
				// HVK-5435
				new( &m_shapesSubparts[i] ) ShapesSubpart(flag);

				if( m_shapesSubparts[i].getMaterialIndexStridingType() == MATERIAL_INDICES_INVALID )
				{
					m_shapesSubparts[i].setMaterialIndexStridingType(MATERIAL_INDICES_INT8);
				}
			}
		}

		// Embed triangle subparts if possible
		if( m_trianglesSubparts.getSize() == 1 )
		{
			//Assumes memCpy(dest,dest,X) is safe
			hkMemUtil::memCpyOneAligned<sizeof(hkpExtendedMeshShape::TrianglesSubpart), 16>( &m_embeddedTrianglesSubpart, m_trianglesSubparts.begin() );
			m_trianglesSubparts.clearAndDeallocate();
			m_trianglesSubparts.setDataUserFree( &m_embeddedTrianglesSubpart, 1, 1);
		}

		// Init cached num child shapes
		if (m_cachedNumChildShapes == -1)
		{
			// Set it to large negative to, so that it's still <0 even when we modify it and add new subparts before the num of current
			// child shapes is actually calculated.
			m_cachedNumChildShapes = HK_INT32_MIN;
		}
	}
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpExtendedMeshShape));
}

void hkpExtendedMeshShape::setWeldingInfo(hkpShapeKey key, hkInt16 weldingInfo)
{
	// If we do not have welding info yet then allocate adequate room for it
	if( m_weldingInfo.getSize() == 0 )
	{
		int totalSize = 0;
		for (int i = 0; i < m_trianglesSubparts.getSize(); ++i)
		{
			m_trianglesSubparts[i].m_triangleOffset = totalSize;
			totalSize += m_trianglesSubparts[i].m_numTriangleShapes;
		}
		m_weldingInfo.reserveExactly(totalSize);
		m_weldingInfo.setSize(totalSize, 0);
	}

	// set welding
	if ( getSubpartType(key) == SUBPART_TRIANGLES)
	{
		const hkUint32 terminalIndex = getTerminalIndexInSubPart(key);

		// Grab a handle to the sub-part
		const TrianglesSubpart& part = m_trianglesSubparts[getSubPartIndex(key)];

		const int index = part.m_triangleOffset + terminalIndex;
		HK_ASSERT2(0xad45bb32, part.m_indexBase, "Invalid mesh shape. Subpart has no elements/triangles.");
		HK_ASSERT3(0x3b082fa1, index >= 0 && index < m_weldingInfo.getSize(), "hkpExtendedMeshShape does not have a triangle at index" << index);
		m_weldingInfo[index] = weldingInfo;
	}
}

void hkpExtendedMeshShape::initWeldingInfo( hkpWeldingUtility::WeldingType weldingType )
{
	HK_ASSERT2(0x897654dd, m_trianglesSubparts.getSize() != 0, "You must add subparts before calling computeWeldingInfo.");

	m_weldingType = weldingType;

	if (weldingType != hkpWeldingUtility::WELDING_TYPE_NONE )
	{
		HK_ON_DEBUG( if ( m_weldingInfo.getSize() != 0) HK_WARN(0x798d7651, "You are calling computeWeldingInfo more than once on a mesh."); )

			int totalSize = 0;
		for (int i = 0; i < m_trianglesSubparts.getSize(); ++i)
		{
			m_trianglesSubparts[i].m_triangleOffset = totalSize;
			totalSize += m_trianglesSubparts[i].m_numTriangleShapes;
		}
		m_weldingInfo.reserveExactly(totalSize);
		m_weldingInfo.setSize(totalSize, 0);
	}
	else
	{
		m_weldingInfo.clearAndDeallocate();
	}
}

int hkpExtendedMeshShape::getNumChildShapes() const
{
	if (m_cachedNumChildShapes < 0)
	{
		m_cachedNumChildShapes = hkpShapeCollection::getNumChildShapes(); 
	}

	HK_ON_DEBUG( int realNumKeys = hkpShapeContainer::getNumChildShapes() );
	HK_ASSERT2(0xad903231, realNumKeys == m_cachedNumChildShapes, "Internal error: Miscalculating number of child shapes for hkpExtendedMeshShape.");

	return m_cachedNumChildShapes;
}


hkpShapeKey hkpExtendedMeshShape::getFirstKey() const
{
	if ( ( getNumTrianglesSubparts() + getNumShapesSubparts() ) == 0 )
	{
		return HK_INVALID_SHAPE_KEY;
	}

	hkpShapeBuffer buffer;
	hkpShapeKey firstKey = (m_trianglesSubparts.getSize())? 0 : hkpShapeKey(0x80000000);

	const hkpShape* shape = getChildShape( firstKey, buffer );

	if ( shape->getType() != hkcdShapeType::TRIANGLE )
	{
		return firstKey; // first shape key is 0, but flagged as SHAPE
	}

	const hkpTriangleShape* tri = static_cast<const hkpTriangleShape*>(shape);
	if ( !hkpTriangleUtil::isDegenerate( tri->getVertex<0>(), tri->getVertex<1>(), tri->getVertex<2>(), hkDefaultTriangleDegeneracyTolerance ) )
	{
		return firstKey;
	}

	return getNextKey( firstKey );
}

// Get the next child shape key.
hkpShapeKey hkpExtendedMeshShape::getNextKey( hkpShapeKey initialKey ) const
{
	hkpShapeBuffer buffer;

	unsigned int subpartIndex  = getSubPartIndex (initialKey);
	int          terminalIndex = getTerminalIndexInSubPart(initialKey);

	int subpartType = initialKey & 0x80000000;

	while (1)
	{
		if ( subpartType == 0  )
		{
			if ( ++terminalIndex >= m_trianglesSubparts[subpartIndex].m_numTriangleShapes )
			{
				terminalIndex = 0;
				if ( ++subpartIndex >= unsigned(m_trianglesSubparts.getSize()) )
				{
					if ( m_shapesSubparts.getSize() == 0 )
					{
						return HK_INVALID_SHAPE_KEY;
					}

					// continue with shape list
					subpartType = 0x80000000;
					subpartIndex = 0;
					terminalIndex = -1;
					continue;
				}
			}
		}
		else // subpartType == 1<<31
		{
			if ( ++terminalIndex >= m_shapesSubparts[subpartIndex].m_childShapes.getSize() )
			{
				if ( ++subpartIndex >= unsigned(m_shapesSubparts.getSize()) )
				{
					return HK_INVALID_SHAPE_KEY;
				}
				terminalIndex = 0;
			}
		}

		// calculate shape key for subpart
		hkpShapeKey key = subpartType | ( subpartIndex << ( 32 - m_numBitsForSubpartIndex )) | terminalIndex;

		//
		//	check for valid triangle
		//

		const hkpShape* shape = getChildShape(key, buffer );

		if ( shape->getType() != hkcdShapeType::TRIANGLE )
		{
			return key;
		}

		const hkpTriangleShape* tri = static_cast<const hkpTriangleShape*>(shape);
		if ( !hkpTriangleUtil::isDegenerate( tri->getVertex<0>(), tri->getVertex<1>(), tri->getVertex<2>(), hkDefaultTriangleDegeneracyTolerance ) )
		{
			return key;
		}
	}
}

int hkpExtendedMeshShape::_getNumChildShapesInTrianglesSubpart(const hkpExtendedMeshShape::TrianglesSubpart& subpart, int subpartIndex) const
{
	hkpShapeBuffer buffer;
	int subpartType = 0;
	int childShapeCount = 0;

	for (int terminalIndex = 0; terminalIndex < subpart.m_numTriangleShapes; terminalIndex++)
	{
		// calculate shape key
		hkpShapeKey key = subpartType | ( subpartIndex << ( 32 - m_numBitsForSubpartIndex )) | terminalIndex;
		const hkpTriangleShape* tri = static_cast<const hkpTriangleShape*>( hkpExtendedMeshShape::getChildShape(key, buffer ) );
		HK_ASSERT2(0xad678293, tri->getType() == hkcdShapeType::TRIANGLE, "Child shape expected to be a triangle.");
		if ( !hkpTriangleUtil::isDegenerate( tri->getVertex<0>(), tri->getVertex<1>(), tri->getVertex<2>(), hkDefaultTriangleDegeneracyTolerance ) )
		{
			childShapeCount++;
		}
	}
	return childShapeCount;
}

int hkpExtendedMeshShape::_getNumChildShapesInShapesSubpart(const hkpExtendedMeshShape::ShapesSubpart& subpart) const
{
	return subpart.m_childShapes.getSize();
}

HK_COMPILE_TIME_ASSERT( sizeof(hkpConvexTransformShape ) < HK_SHAPE_BUFFER_SIZE);
HK_COMPILE_TIME_ASSERT( sizeof(hkpConvexTranslateShape ) < HK_SHAPE_BUFFER_SIZE);


#endif

#if defined HK_PLATFORM_SPU

HK_ALWAYS_INLINE const void* HK_CALL hkGetArrayElemWithByteStriding( const void* base, int index, int elemsize, int dmaGroup = HK_SPU_DMA_GROUP_STALL, bool waitForCompletion = true  )
{
	HK_ASSERT2(0xdbcf8890, base != HK_NULL, "null array base pointer passed to hkGetArrayElem");
	HK_ASSERT2(0xdbcf8891, index >= 0,		"Negative array index passed to hkGetArrayElem");

	hkUlong arrayAddrPpu = hkUlong(base) + ( index * elemsize );
	const hkUlong mask  = ~static_cast<hkUlong>(HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE-1);
	hkUlong arrayAddrAligned = arrayAddrPpu & mask;
	hkUlong alignedDataSpu = (hkUlong)g_SpuCollideUntypedCache->getFromMainMemoryInlined( (const void*)arrayAddrAligned , HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE, dmaGroup, waitForCompletion );
	return reinterpret_cast<const void*> ( alignedDataSpu + (arrayAddrPpu & ~mask) );
}

HK_ALWAYS_INLINE const void* HK_CALL hkGetArrayElemWithByteStridingHalfCacheSize( const void* base, int index, int elemsize, int dmaGroup = HK_SPU_DMA_GROUP_STALL, bool waitForCompletion = true  )
{
	hkUlong arrayAddrPpu = hkUlong(base) + ( index * elemsize );
	// We will virtually use only half the cache size while still bringing in the full size. That way we have a spill-over buffer of half the cache size that can be used for otherwise out-of-bounds accesses.
	const hkUlong mask  = ~static_cast<hkUlong>((HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE/2)-1);
	hkUlong arrayAddrAligned = arrayAddrPpu & mask;
	hkUlong alignedDataSpu = (hkUlong)g_SpuCollideUntypedCache->getFromMainMemoryInlined( (const void*)arrayAddrAligned , HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE, dmaGroup, waitForCompletion );
	return reinterpret_cast<const void*> ( alignedDataSpu + (arrayAddrPpu & ~mask) );
}

#else

template <typename TYPE>
HK_ALWAYS_INLINE const TYPE* HK_CALL hkGetArrayElem( const TYPE* base, int index, int dmaGroup = 0, bool waitForCompletion = true )
{
	HK_ASSERT2(0xdbcf8890, base != HK_NULL, "null array base pointer passed to hkGetArrayElem");
	HK_ASSERT2(0xdbcf8891, index >= 0,		"Negative array index passed to hkGetArrayElem");

	return reinterpret_cast<const TYPE*>( reinterpret_cast<const char*>(base) + ( index * sizeof(TYPE) ) );
}

template <typename TYPE>
HK_ALWAYS_INLINE const TYPE* HK_CALL hkGetArrayElemWithByteStriding( const TYPE* base, int index, int striding, int dmaGroup = 0, bool waitForCompletion = true )
{
	HK_ASSERT2(0xdbcf8890, base != HK_NULL, "null array base pointer passed to hkGetArrayElem");
	HK_ASSERT2(0xdbcf8891, index >= 0,		"Negative array index passed to hkGetArrayElem");

	return reinterpret_cast<const TYPE*>( reinterpret_cast<const char*>(base) + ( index * striding ) );
}

template <typename TYPE>
HK_ALWAYS_INLINE const TYPE* HK_CALL hkGetArrayElemWithByteStridingHalfCacheSize( const TYPE* base, int index, int striding, int dmaGroup = 0, bool waitForCompletion = true )
{
	return reinterpret_cast<const TYPE*>( reinterpret_cast<const char*>(base) + ( index * striding ) );
}
#endif

enum {
	HK_SPU_DMA_GROUP_PREFETCH_VERTICES = 16
};

#ifdef HK_PLATFORM_PS3_SPU
HK_ALWAYS_INLINE void unalignedLoad(hkVector4& v, const hkReal* ptr)
{
	vector float qw0, qw1;
	int shift;

	qw0 = *(const vector float*)ptr;
	qw1 = * hkAddByteOffsetConst((const vector float*)ptr,15) ;
	shift = (unsigned) ptr & 15;

	v.m_quad =  spu_or(	spu_slqwbyte(qw0, shift),	spu_rlmaskqwbyte(qw1, shift-16) );
}
#endif

const hkpShape* hkpExtendedMeshShape::getChildShape(hkpShapeKey key, hkpShapeBuffer& buffer) const
{
	// Extract triangle/child shape index and sub-part index
	const hkUint32 subpartIndex  = getSubPartIndex (key);
	const hkUint32 terminalIndex = getTerminalIndexInSubPart(key);

	// Grab a handle to the sub-part
	// We need the 'm_trianglesSubparts.getSize() > 0' check for the case when key == 0 (e.g. called from getFirstKey() ).
	if ( getSubpartType(key) == SUBPART_TRIANGLES)
	{
		HK_DECLARE_ALIGNED_LOCAL_PTR(TrianglesSubpart, localPart, sizeof(TrianglesSubpart));
		const TrianglesSubpart* part;

		if (m_trianglesSubparts.getSize() == 1)
		{
			part = &m_embeddedTrianglesSubpart;
		}
		else
		{
			const TrianglesSubpart* cachedPart = (const TrianglesSubpart*)hkGetArrayElemWithByteStridingHalfCacheSize( m_trianglesSubparts.begin(), subpartIndex, sizeof(TrianglesSubpart) );

			hkMemUtil::memCpyOneAligned<sizeof(hkpExtendedMeshShape::TrianglesSubpart), 16>( localPart, cachedPart );
			part = localPart;
		}

		HK_ASSERT2(0xad45bb32, part->m_indexBase, "Invalid mesh shape. First subpart has no elements/triangles.");


		HK_ASSERT2(0x978f756, part->m_flipAlternateTriangles == 0 || part->m_flipAlternateTriangles == 1, "m_flipAlternateTriangles must equal 0 or 1");

		// m_flipAlternateTriangles is 1 if flip is enabled, 0 is disabled
		int triangleWindingFlip = terminalIndex & part->m_flipAlternateTriangles;

		// Extract the triangle indices and vertices
		hkPadSpu<int> index[3];

		const void* data = hkGetArrayElemWithByteStridingHalfCacheSize( part->m_indexBase, terminalIndex, part->m_indexStriding );
		switch( part->m_stridingType )
		{
			case INDICES_INT8:
			{
				const hkUint8* triangle = (const hkUint8*)data;

				index[0] = triangle[ HK_HINT_SIZE16( 0 )];
				index[1] = triangle[ HK_HINT_SIZE16( 1 + triangleWindingFlip ) ];
				index[2] = triangle[ HK_HINT_SIZE16( 1 + (1 ^ triangleWindingFlip)) ];
			}
			break;

			case INDICES_INT16:
			{
				const hkUint16* triangle = (const hkUint16*)data;

				index[0] = triangle[ HK_HINT_SIZE16( 0 )];
				index[1] = triangle[ HK_HINT_SIZE16( 1 + triangleWindingFlip ) ];
				index[2] = triangle[ HK_HINT_SIZE16( 1 + (1 ^ triangleWindingFlip)) ];
			}
			break;

			case INDICES_INT32:
			{
				const hkUint32* triangle = (const hkUint32*)data;

				index[0] = triangle[ HK_HINT_SIZE16( 0 ) ];
				index[1] = triangle[ HK_HINT_SIZE16( 1 + triangleWindingFlip ) ];
				index[2] = triangle[ HK_HINT_SIZE16( 1 + (1 ^ triangleWindingFlip) ) ];
			}
			break;

			default:
				// Initialize index vals to prevent 'uninitialized data' compiler warning.
				index[0] = 0;
				index[1] = 0;
				index[2] = 0;
				HK_ASSERT2( 0x12131a31,  part->m_stridingType == INDICES_INT32, "Subpart index type is not set or out of range (8, 16, or 32 bit only)." );

		}

		// Grab the vertices
		// The three triangle vertices to be filled in
		HK_PAD_ON_SPU(const hkReal*) vertex[3];	// pointer into the cache
		const hkReal* base = part->m_vertexBase;
		const int striding = part->m_vertexStriding;
#if defined(HK_PLATFORM_SPU)
		// Fetch verts
		{
			for (int i = 0; i < 3; i++)
			{
				vertex[i] = (const hkReal*)hkGetArrayElemWithByteStridingHalfCacheSize( base, index[i], striding, HK_SPU_DMA_GROUP_PREFETCH_VERTICES, false );
			}
		}
#endif

		HK_ASSERT(0x73f97fa7,  sizeof( hkpTriangleShape ) <= HK_SHAPE_BUFFER_SIZE );
#if !defined ( HK_PLATFORM_SPU )
		hkpTriangleShape* HK_RESTRICT triangleShape = new (&buffer) hkpTriangleShape();
#else
		hkpTriangleShape* HK_RESTRICT triangleShape = (hkpTriangleShape*)&buffer;
#endif

		triangleShape->setType( HKCD_SHAPE_TYPE_FROM_CLASS(hkpTriangleShape) );
		triangleShape->setUserData( part->m_userData );
		triangleShape->setRadius( m_triangleRadius );
		triangleShape->setWeldingType( m_weldingType );
		HKCD_PATCH_SHAPE_VTABLE( triangleShape );

		// Set the extrusion
		triangleShape->setExtrusion( part->m_extrusion );

		// get welding
		HK_ASSERT(0x54654323, (m_weldingInfo.getSize() == 0) || (m_weldingInfo.getSize() > ( part->m_triangleOffset + (int)terminalIndex ) ) );
		const hkUint16 weldingInfo = (m_weldingInfo.getSize() == 0) ? 0 : *(const hkUint16*)hkGetArrayElemWithByteStriding( m_weldingInfo.begin(), part->m_triangleOffset + terminalIndex, sizeof(hkUint16) );
 		triangleShape->setWeldingInfo( weldingInfo );

#if defined ( HK_PLATFORM_SPU )
		// Wait for verts
		hkSpuDmaManager::waitForDmaCompletion( HK_SPU_DMA_GROUP_PREFETCH_VERTICES );
#else
		// Keep this unrolled for other platforms
		vertex[0] = hkAddByteOffsetConst( base, index[0] * striding );
		vertex[1] = hkAddByteOffsetConst( base, index[1] * striding );
		vertex[2] = hkAddByteOffsetConst( base, index[2] * striding );
#endif

		// The vertex pointers might not be 16-byte aligned (e.g. if the striding is a multiple of 3), so we can't just cast to an hkVector4 
		// hkVector4::load4 won't work either, because the 4th component might be out of bounds. So we have to use load3.
		hkVector4 p0; p0.load<3,HK_IO_NATIVE_ALIGNED>(vertex[0]);
		hkVector4 p1; p1.load<3,HK_IO_NATIVE_ALIGNED>(vertex[1]);
		hkVector4 p2; p2.load<3,HK_IO_NATIVE_ALIGNED>(vertex[2]);

		hkVector4 t0; t0._setTransformedPos( part->m_transform, p0 );
		hkVector4 t1; t1._setTransformedPos( part->m_transform, p1 );
		hkVector4 t2; t2._setTransformedPos( part->m_transform, p2 );

		triangleShape->setVertex<0>( t0 );
		triangleShape->setVertex<1>( t1 );
		triangleShape->setVertex<2>( t2 );

		return triangleShape;
	}
	else // SUBPART_SHAPE
	{
		HK_DECLARE_ALIGNED_LOCAL_PTR(ShapesSubpart, part, sizeof(hkpExtendedMeshShape::ShapesSubpart));
		{
#if (HK_POINTER_SIZE == 8)
			const ShapesSubpart* cachedSubpart = (const ShapesSubpart*)hkGetArrayElem( m_shapesSubparts.begin(), subpartIndex);
#else
			const ShapesSubpart* cachedSubpart = (const ShapesSubpart*)hkGetArrayElemWithByteStridingHalfCacheSize( m_shapesSubparts.begin(), subpartIndex, sizeof(ShapesSubpart) );
#endif
			hkMemUtil::memCpyOneAligned<sizeof(hkpExtendedMeshShape::ShapesSubpart), 16>( part, cachedSubpart );
		}

		HK_ASSERT2( 0xf0323445, terminalIndex < (hkUint32)(part->m_childShapes.getSize()), "Invalid shape key");

		const hkpConvexShape* const childShape = *(const hkpConvexShape**)hkGetArrayElemWithByteStriding( part->m_childShapes.begin(), terminalIndex, sizeof( hkpConvexShape* ) );

#if !defined( HK_PLATFORM_SPU )
		if( part->getFlags() == ShapesSubpart::FLAG_NONE )
		{
			return childShape;
		}
		else
		{
			const hkpConvexShape* shape;

#ifdef HK_DEBUG
			if(childShape->getType() == hkcdShapeType::CONVEX_TRANSLATE ||
				childShape->getType() == hkcdShapeType::CONVEX_TRANSFORM )
			{
				HK_WARN(0x4de62416, "Returned Shape will have a nested convex transform or translate shape from a hkpExtendedMeshShape::ShapesSubpart.  Bake the tranform into the child shape for best performance." );
			}
#endif

			if ( part->getFlags() == ShapesSubpart::FLAG_TRANSLATE_SET )
			{
				shape = new (buffer) hkpConvexTranslateShape( childShape, part->getTranslation(), hkpShapeContainer::REFERENCE_POLICY_IGNORE );
			}
			else //Any rotation requires a hkpConvexTranformShape
			{
				hkTransform transform; transform.set(part->getRotation(), part->getTranslation());
				shape = new (buffer) hkpConvexTransformShape( childShape, transform, hkpShapeContainer::REFERENCE_POLICY_IGNORE );
			}

			return shape;
		}
#else

		// Retrieve shape
		HK_COMPILE_TIME_ASSERT( HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE == 256 );
		const hkpConvexShape* childShapeSpu = (const hkpConvexShape*)g_SpuCollideUntypedCache->getFromMainMemory( (const void*)childShape , HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE);
		HKCD_PATCH_SHAPE_VTABLE( const_cast<hkpConvexShape*>(childShapeSpu) );

		if ( part->getFlags() == ShapesSubpart::FLAG_NONE )
		{
			// copy shape to buffer
			hkString::memCpy256( &buffer, childShapeSpu);			
			return reinterpret_cast<hkpShape*>(&buffer);
		}
		else
		{
			if ( part->getFlags() == ShapesSubpart::FLAG_TRANSLATE_SET )
			{
				hkpConvexTranslateShape* translateShape = (hkpConvexTranslateShape*)&buffer;

				// Put child shape behind the transform shape in LS
				hkpConvexShape* childShapeBehind = (hkpConvexShape*)( translateShape + 1 );
				int shapeSize = ( HK_SHAPE_BUFFER_SIZE - sizeof( hkpConvexTranslateShape ) ) >> 4;
				hkString::memCpy16( childShapeBehind, childShapeSpu, shapeSize);

				translateShape->initializeSpu( childShapeBehind, part->getTranslation(), childShapeSpu->getRadius() );
			}
			else //Any rotation requires a hkpConvexTransformShape
			{
				hkpConvexTransformShape* transformShape = (hkpConvexTransformShape*)&buffer;

				// Put child shape behind the transform shape in LS
				hkpConvexShape* childShapeBehind = (hkpConvexShape*)( transformShape + 1 );
				int shapeSize = (HK_SHAPE_BUFFER_SIZE - sizeof( hkpConvexTransformShape ) ) >> 4;
				hkString::memCpy16( childShapeBehind, childShapeSpu, shapeSize);

				hkQsTransform transform;
				transform.set( part->getTranslation(), part->getRotation() );
				transformShape->initializeSpu( childShapeBehind, transform, childShapeSpu->getRadius() );
			}
			
			return reinterpret_cast<hkpShape*>(&buffer);
		}

#endif
	}
}

const hkpMeshMaterial* hkpExtendedMeshShape::getMeshMaterial( hkpShapeKey key ) const
{
	const hkUint32 subpartIndex  = getSubPartIndex (key);
	const hkUint32 terminalIndex = getTerminalIndexInSubPart(key);

	MaterialIndexStridingType	materialIndexStridingType;
	int							materialIndexStriding;
	const void*					materialIndexBase;
	const hkpMeshMaterial*		materialBase;
	int							materialStriding;
	HK_ON_DEBUG( int			numMaterials );

	// Grab a handle to the sub-part
	{
		const Subpart* part;
		const Subpart* partOnPPu;
		if ( getSubpartType(key) == SUBPART_TRIANGLES  )
		{
			if (m_trianglesSubparts.getSize()==1)
			{
				part = &m_embeddedTrianglesSubpart;
				goto GOT_PART;
			}
			else
			{
				partOnPPu = &m_trianglesSubparts[subpartIndex];
			}
		}
		else
		{
			partOnPPu = &m_shapesSubparts[subpartIndex];
		}
		part = (const Subpart*)hkGetArrayElemWithByteStridingHalfCacheSize( partOnPPu, 0, sizeof(Subpart));

GOT_PART:
		materialIndexStridingType		= part->getMaterialIndexStridingType();
		materialIndexStriding			= part->m_materialIndexStriding;
		materialIndexBase				= part->m_materialIndexBase;
		materialBase					= part->m_materialBase;
		materialStriding				= part->m_materialStriding;
		HK_ON_DEBUG( numMaterials	= part->getNumMaterials() );
	}

	if ( materialIndexBase && materialStriding > 0 )
	{
		int materialIdx;
		HK_ASSERT2(0xad453fa2, materialIndexStridingType == MATERIAL_INDICES_INT8 || materialIndexStridingType == MATERIAL_INDICES_INT16, "Invalid hkpExtendedMeshShape::SubPart::m_materialIndexStridingType.");

		const void* data = hkGetArrayElemWithByteStriding( materialIndexBase, terminalIndex, materialIndexStriding );
		if (materialIndexStridingType == MATERIAL_INDICES_INT8)
		{
			materialIdx = *static_cast<const hkUint8*>( data );
		}
		else
		{
			materialIdx = *static_cast<const hkUint16*>( data );
		}

		HK_ASSERT2(0x26d359f1, materialIdx < numMaterials, "Your mesh references a material which does not exist" );
		return (const hkpMeshMaterial*)hkGetArrayElemWithByteStriding( materialBase, materialIdx, materialStriding );
	}
	else
	{
		return HK_NULL;
	}
}

hkUint32 hkpExtendedMeshShape::getCollisionFilterInfo(hkpShapeKey key) const
{
	const hkpMeshMaterial* material = getMeshMaterial(key);
	if (material)
	{
		return material->m_filterInfo;
	}
	else
	{
		return m_defaultCollisionFilterInfo;
	}
}



static inline void HK_CALL addToAabb(hkAabb& aabb, const hkQsTransform& localToWorld, const hkReal* v )
{
	hkVector4 vLocal;
	vLocal.load<3,HK_IO_NATIVE_ALIGNED>(v);
	vLocal.zeroComponent<3>();
	hkVector4 vWorld; vWorld._setTransformedPos( localToWorld, vLocal );

	aabb.m_min.setMin( aabb.m_min, vWorld );
	aabb.m_max.setMax( aabb.m_max, vWorld );
}



void hkpExtendedMeshShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out ) const
{
#if defined(HK_DEBUG) && !defined(HK_PLATFORM_SPU)
	for ( int i = 0; i < m_trianglesSubparts.getSize(); ++i )
	{
		HK_ASSERT2(0x6541f916, m_trianglesSubparts[i].getTranslation()(3) == 0, "The triangles subpart transform has been modified since last time the AABB was computed" );
	}
#endif
	
	hkAabbUtil::calcAabb( localToWorld, m_aabbHalfExtents, m_aabbCenter, hkSimdReal::fromFloat(tolerance), out );
}

void hkpExtendedMeshShape::calcAabbExtents( TrianglesSubpart& part, hkAabb& out )
{
	out.m_min = hkVector4::getConstant<HK_QUADREAL_MAX>();
	out.m_max = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();

	HK_ASSERT2(0x6541f816, part.m_indexBase, "No indices provided in a subpart of a hkpExtendedMeshShape." );
	HK_ASSERT2(0x6541f817, part.m_vertexBase, "No vertices provided in a subpart of a hkpExtendedMeshShape." );
	for (int v = 0; v < part.m_numTriangleShapes; v++ )
	{
		const hkReal* vf0;
		const hkReal* vf1;
		const hkReal* vf2;

		switch ( part.m_stridingType )
		{
			case INDICES_INT8:
			{
				const hkUint8* tri = hkAddByteOffsetConst<hkUint8>( (const hkUint8*)part.m_indexBase, part.m_indexStriding * v);
				vf0 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[0] );
				vf1 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[1] );
				vf2 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[2] );
			}
			break;

			case INDICES_INT16:
			{
				const hkUint16* tri = hkAddByteOffsetConst<hkUint16>( (const hkUint16*)part.m_indexBase, part.m_indexStriding * v);
				vf0 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[0] );
				vf1 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[1] );
				vf2 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[2] );
			}
			break;

			default:
			{
				HK_ASSERT2( 0x12131a31, part.m_stridingType == INDICES_INT32, "Subpart index type is not set or out of range (8, 16, or 32 bit only)." );

				const hkUint32* tri = hkAddByteOffsetConst<hkUint32>( (const hkUint32*)part.m_indexBase, part.m_indexStriding * v);
				vf0 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[0] );
				vf1 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[1] );
				vf2 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[2] );
			}
				
		}
		
		addToAabb( out, part.m_transform, vf0 );
		addToAabb( out, part.m_transform, vf1 );
		addToAabb( out, part.m_transform, vf2 );
	}

	HK_ON_DEBUG( hkVector4 t = part.m_transform.getTranslation(); t.zeroComponent(3); part.m_transform.setTranslation(t) );
}

void hkpExtendedMeshShape::calcAabbExtents( const ShapesSubpart& part, hkAabb& out )
{
	out.m_min = hkVector4::getConstant<HK_QUADREAL_MAX>();
	out.m_max = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();

	hkTransform transform; transform.set(part.getRotation(), part.getTranslation());

	for (int i =0 ; i < part.m_childShapes.getSize(); i++)
	{
		hkAabb childAabb;
		part.m_childShapes[i]->getAabb( transform, 0.0f, childAabb );
		out.m_min.setMin( out.m_min, childAabb.m_min );
		out.m_max.setMax( out.m_max, childAabb.m_max );
	}
}

void hkpExtendedMeshShape::recalcAabbExtents()
{
	hkAabb out;

	out.m_min = hkVector4::getConstant<HK_QUADREAL_MAX>();
	out.m_max = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();

	hkSimdReal tol4; tol4.load<1>( &m_triangleRadius );

	for (int s = 0; s < m_trianglesSubparts.getSize(); s++)
	{
		hkAabb childAabb;
		calcAabbExtents( m_trianglesSubparts[s], childAabb );

		// Increment by triangle radius
		childAabb.m_min.setSub( childAabb.m_min,tol4 );
		childAabb.m_max.setAdd( childAabb.m_max,tol4 );
		out.m_min.setMin( out.m_min, childAabb.m_min );
		out.m_max.setMax( out.m_max, childAabb.m_max );
	}

	for (int s = 0; s < m_shapesSubparts.getSize(); s++)
	{
		hkAabb childAabb;
		calcAabbExtents( m_shapesSubparts[s], childAabb);
		out.m_min.setMin( out.m_min, childAabb.m_min );
		out.m_max.setMax( out.m_max, childAabb.m_max );
	}

	out.getCenter( m_aabbCenter );
	out.getHalfExtents( m_aabbHalfExtents );
}

#ifndef HK_PLATFORM_SPU

void hkpExtendedMeshShape::assertTrianglesSubpartValidity( const TrianglesSubpart& part )
{
	HK_ASSERT2(0x68fb31d4, m_trianglesSubparts.getSize() < ((1 << (m_numBitsForSubpartIndex-1)) ), "You are adding too many triangle subparts for the mesh shape. "\
		"You can change the number of bits usable for the subpart index by changing the m_numBitsForSubpartIndex in the mesh construction info.");

	HK_ASSERT2(0x6541f716,  part.m_vertexBase, "Subpart vertex base pointer is not set or null.");
	HK_ASSERT2(0x426c5d43,  part.m_vertexStriding >= 4, "Subpart vertex striding is not set or invalid (less than 4 bytes stride).");
	HK_ASSERT2(0x2223ecab,  part.m_numVertices > 0, "Subpart num vertices is not set or negative.");
	HK_ASSERT2(0x5a93ebb6,  part.m_indexBase, "Subpart index base pointer is not set or null.");
	HK_ASSERT2(0x12131a31,  ((part.m_stridingType == INDICES_INT8) || (part.m_stridingType == INDICES_INT16) || (part.m_stridingType == INDICES_INT32)),
		"Subpart index type is not set or out of range (8, 16, or 32 bit only).");
	HK_ASSERT2(0x492cb07c,  part.m_indexStriding >= 2,
		"Subpart index striding pointer is not set or invalid (less than 2 bytes stride).");
	HK_ASSERT2(0x53c3cd4f,  part.m_numTriangleShapes > 0, "Subpart num shapes is not set or negative.");
	HK_ASSERT2(0xad5aae43,  part.m_materialIndexBase == HK_NULL || part.getMaterialIndexStridingType() == MATERIAL_INDICES_INT8 || part.getMaterialIndexStridingType() == MATERIAL_INDICES_INT16, "Subpart materialIndexStridingType is not set or out of range (8 or 16 bit only).");

	HK_ASSERT2(0x7b8c4c78,	part.m_numTriangleShapes-1 < (1<<(32-m_numBitsForSubpartIndex)),
		"There are only 32 bits available to index the subpart and triangle in a "
		"hkpExtendedMeshShape. This subpart has too many triangles. Attempts to index a "
		"triangle could overflow the available bits. Try decreasing the number of "
		"bits reserved for the subpart index.");

}

void hkpExtendedMeshShape::assertShapesSubpartValidity( const ShapesSubpart& part )
{
	HK_ASSERT2(0x68fb32d4, m_shapesSubparts.getSize() < ((1 << (m_numBitsForSubpartIndex-1)) ), "You are adding too many shape subparts for the mesh shape. "\
		"You can change the number of bits usable for the subpart index by changing the m_numBitsForSubpartIndex in the mesh construction info.");

	HK_ASSERT2(0x51c3cd4f,  part.m_childShapes.getSize() > 0, "Subpart num shapes is not set or negative.");

	HK_ASSERT2(0x7b834c78,	part.m_childShapes.getSize()-1 < (1<<(32-m_numBitsForSubpartIndex)),
		"There are only 32 bits available to index the subpart and terminal shape in a "
		"hkpExtendedMeshShape. This subpart has too many terminal shapes. Attempts to index a "
		"terminal shape could overflow the available bits. Try decreasing the number of "
		"bits reserved for the subpart index.");
}

void hkpExtendedMeshShape::addTrianglesSubpart( const TrianglesSubpart& part )
{
	TrianglesSubpart& tsp = *expandOneTriangleSubparts();

	tsp = static_cast<const TrianglesSubpart&>(part);
	HK_ON_DEBUG( assertTrianglesSubpartValidity(tsp); )

	HK_ASSERT2(0x09fe8645, m_weldingInfo.getSize() == 0, "You must add all subparts prior to building welding information" );

	//Incrementally update AABB
	{
		hkAabb current;
		current.m_min.setSub( m_aabbCenter, m_aabbHalfExtents );
		current.m_max.setAdd( m_aabbCenter, m_aabbHalfExtents );

		hkAabb aabbPart;
		{
			calcAabbExtents( tsp, aabbPart );

			// Increment by triangle radius
			hkSimdReal tol4; tol4.load<1>( &m_triangleRadius );
			aabbPart.m_min.setSub( aabbPart.m_min,tol4 );
			aabbPart.m_max.setAdd( aabbPart.m_max,tol4 );
			current.m_min.setMin( current.m_min, aabbPart.m_min );
			current.m_max.setMax( current.m_max, aabbPart.m_max );
		}

		current.getCenter( m_aabbCenter );
		current.getHalfExtents( m_aabbHalfExtents );
	}

	// Update cached num of child shapes
	m_cachedNumChildShapes += _getNumChildShapesInTrianglesSubpart(part, m_trianglesSubparts.getSize()-1);
}

int hkpExtendedMeshShape::addShapesSubpart( const ShapesSubpart& part )
{
	ShapesSubpart& ssp = *expandOneShapesSubparts();

	const ShapesSubpart& in = static_cast<const ShapesSubpart&>(part);
	ssp = in;
	HK_ON_DEBUG( assertShapesSubpartValidity(ssp); )

	// Incrementally update the AABB
	{
		hkAabb current;
		current.m_min.setSub( m_aabbCenter, m_aabbHalfExtents );
		current.m_max.setAdd( m_aabbCenter, m_aabbHalfExtents );

		hkAabb aabbPart;
		{
			calcAabbExtents( ssp, aabbPart );
			current.m_min.setMin( current.m_min, aabbPart.m_min );
			current.m_max.setMax( current.m_max, aabbPart.m_max );
		}

		current.getCenter( m_aabbCenter );
		current.getHalfExtents( m_aabbHalfExtents );
	}

	// Update cached num of child shapes
	m_cachedNumChildShapes += _getNumChildShapesInShapesSubpart(part);

	return m_shapesSubparts.getSize()-1;
}

hkpExtendedMeshShape::ShapesSubpart::ShapesSubpart(hkFinishLoadedObjectFlag flag)
: Subpart(flag)
, m_childShapes(flag)
{
	if( flag.m_finishing )
	{
		int flags = m_translation.allEqualZero<3>(hkSimdReal::fromFloat(1e-3f)) ? FLAG_NONE : FLAG_TRANSLATE_SET;
		flags |= m_rotation.getAngle() < 1e-3f ? FLAG_NONE : FLAG_ROTATE_SET;
		setFlags(flags);
	}
}


hkpExtendedMeshShape::ShapesSubpart::ShapesSubpart( const hkpConvexShape*const* childShapes, int numChildShapes, const hkVector4& offset )
: Subpart( SUBPART_SHAPE )
, m_translation(offset)
{
	hkRefPtr<hkpConvexShape>* c = m_childShapes.expandBy(numChildShapes);
	for( int i = 0; i < numChildShapes; ++i )
	{
		c[i] = const_cast<hkpConvexShape*>( childShapes[i] );
	}
	// init offset
	{
		m_rotation.setIdentity();
		setFlags( offset.allEqualZero<3>(hkSimdReal::fromFloat(1e-3f)) ? FLAG_NONE : FLAG_TRANSLATE_SET );
	}
}


hkpExtendedMeshShape::ShapesSubpart::ShapesSubpart( const hkpConvexShape*const* childShapes, int numChildShapes, const hkTransform& transform ): Subpart( SUBPART_SHAPE )
, m_translation(transform.getTranslation())
{
	m_rotation.set(transform.getRotation());
	hkRefPtr<hkpConvexShape>* c = m_childShapes.expandBy(numChildShapes);
	for( int i = 0; i < numChildShapes; ++i )
	{
		c[i] = const_cast<hkpConvexShape*>( childShapes[i] );
	}
	int flags = m_translation.allEqualZero<3>(hkSimdReal::fromFloat(1e-3f)) ? FLAG_NONE : FLAG_TRANSLATE_SET;
	flags |= transform.getRotation().isApproximatelyEqual( hkTransform::getIdentity().getRotation()) ? FLAG_NONE : FLAG_ROTATE_SET;
	setFlags(flags);
}

hkpExtendedMeshShape::ShapesSubpart::~ShapesSubpart()
{
}

#if !defined(HK_PLATFORM_SPU)

int hkpExtendedMeshShape::calcSizeForSpu(const CalcSizeForSpuInput& input, int spuBufferSizeLeft) const
{
#if defined(HK_PLATFORM_HAS_SPU)
	// no dynamic extended mesh shapes on the spu
	if ( (!input.m_isFixedOrKeyframed || input.m_hasDynamicMotionSaved) && input.m_midphaseAgent3Registered )
	{
		HK_WARN(0xdbc1ffbc, "This hkpExtendedMeshShape cannot run on SPU - midphase agent is registered and rigid body is not fixed or keyframed or has a dynamic motion saved.");
		if (input.m_isFixedOrKeyframed)
		{
			HK_WARN(0xad906291, "This shape can be run on SPU only for fixed and keyframed bodies. Now it defaults to be run on PPU. You can make it run on SPU by removing the saved dynamic motion.");
		}
		return -1;
	}

#endif

	for (int s=0 ; s < m_shapesSubparts.getSize(); s++)
	{
		const ShapesSubpart& part = m_shapesSubparts[s];

		for (int sIdx=0; sIdx < part.m_childShapes.getSize(); sIdx++)
		{
			int childSize = part.m_childShapes[sIdx]->calcSizeForSpu( input, HK_SPU_AGENT_SECTOR_JOB_MAX_SHAPE_SIZE );

			// The buffer size depends on the shapes returned in hkpExtendedMeshShape::getChildShape
			// If rotation or offsets are set we need to check the shape will fit in the shape buffer with the transform or translate shape
			int maxAvailableBufferSizeForChild;
			if( part.getFlags() == ShapesSubpart::FLAG_NONE )
			{
				maxAvailableBufferSizeForChild = HK_SHAPE_BUFFER_SIZE;
			}
			else
			{
				if( part.getFlags() == ShapesSubpart::FLAG_TRANSLATE_SET )
				{
					maxAvailableBufferSizeForChild = HK_SHAPE_BUFFER_SIZE - sizeof(hkpConvexTranslateShape);
				}
				else
				{
					maxAvailableBufferSizeForChild = HK_SHAPE_BUFFER_SIZE - sizeof(hkpConvexTransformShape);
				}
			}

			if ( childSize < 0 )
			{
				HK_WARN(0x54342345, "Child " << sIdx << " is not supported on SPU");
				return -1;
			}
			if ( childSize > maxAvailableBufferSizeForChild )
			{
				HK_WARN(0x54342346, "Child " << sIdx << " overflows available space, so shape does not fit on SPU");
				return -1;
			}

		}
	}

	return sizeof(*this);
}

#endif

hkpExtendedMeshShape::TrianglesSubpart* hkpExtendedMeshShape::expandOneTriangleSubparts()
{
	// Expand the triangle subpart array
	if (m_trianglesSubparts.getSize() == 0)
	{
		// Embed the first subpart directly with the shape
		m_trianglesSubparts.setDataUserFree( &m_embeddedTrianglesSubpart, 1, 1);
		return m_trianglesSubparts.begin();
	}
	else
	{
		return m_trianglesSubparts.expandBy(1);
	}
}

hkpExtendedMeshShape::ShapesSubpart*	 hkpExtendedMeshShape::expandOneShapesSubparts()
{
	// Expand the shapes subpart array
	return m_shapesSubparts.expandBy(1);
}

void hkpExtendedMeshShape::freeTriangleSubparts()
{
}

void hkpExtendedMeshShape::freeShapesSubparts()
{
}

#endif

// Ensure this shape and the MOPP both fit in a single cache line
#if defined(HK_PLATFORM_HAS_SPU)
HK_COMPILE_TIME_ASSERT( ( sizeof(hkpExtendedMeshShape::TrianglesSubpart) & 0xf ) == 0 );
HK_COMPILE_TIME_ASSERT( ( sizeof(hkpExtendedMeshShape::ShapesSubpart) & 0xf ) == 0 );
HK_COMPILE_TIME_ASSERT( sizeof(hkpExtendedMeshShape::ShapesSubpart) == 64 ); // HVK-4093
#endif

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
