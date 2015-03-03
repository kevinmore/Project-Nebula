/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Compound/Collection/StorageExtendedMesh/hkpStorageExtendedMeshShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>

extern const hkClass hkpStorageExtendedMeshShapeMaterialClass;


hkpStorageExtendedMeshShape::MeshSubpartStorage::MeshSubpartStorage( hkFinishLoadedObjectFlag flag )
	: hkReferencedObject(flag), m_vertices(flag), m_indices8(flag), m_indices16(flag), m_indices32(flag),
	m_materialIndices(flag), m_materials(flag), m_namedMaterials(flag), m_materialIndices16(flag)
{
}

hkpStorageExtendedMeshShape::ShapeSubpartStorage::ShapeSubpartStorage( hkFinishLoadedObjectFlag flag )
	: hkReferencedObject(flag),
	m_materialIndices(flag), m_materials(flag), m_materialIndices16(flag)
{
}

hkpStorageExtendedMeshShape::hkpStorageExtendedMeshShape( hkFinishLoadedObjectFlag flag )
	: hkpExtendedMeshShape(flag), m_meshstorage(flag), m_shapestorage(flag)
{
	if( flag.m_finishing )
	{
		int i;
		for( i = 0; i < m_trianglesSubparts.getSize(); ++i )
		{
			TrianglesSubpart& part = m_trianglesSubparts[i];
			
			// Fix v-table to MeshSubpartStorage in case it needs to get serialized again
			new (m_meshstorage[i]) MeshSubpartStorage(flag);

			MeshSubpartStorage& store = *m_meshstorage[i];

			part.m_vertexBase = &store.m_vertices.begin()[0](0);
			if (part.m_stridingType == INDICES_INT8)
			{
				part.m_indexBase = store.m_indices8.begin();
			}
			else if (part.m_stridingType == INDICES_INT16)
			{
				part.m_indexBase = store.m_indices16.begin();
			}
			else // INDICES_INT32
			{
				HK_ASSERT2( 0x12131a31, part.m_stridingType == INDICES_INT32, "Subpart index type is not set or out of range (8, 16, or 32 bit only)." );
				part.m_indexBase = store.m_indices32.begin();
			}

			if (part.getMaterialIndexStridingType() == MATERIAL_INDICES_INT8)
			{
				part.m_materialIndexBase = store.m_materialIndices.begin();
			}
			else
			{
				part.m_materialIndexBase = store.m_materialIndices16.begin();
			}

			if( store.m_namedMaterials.getSize() )
			{
				part.m_materialBase = reinterpret_cast<hkpNamedMeshMaterial*>(store.m_namedMaterials.begin());
				part.m_materialStriding = sizeof(hkpNamedMeshMaterial);
			}
			else
			{
				part.m_materialBase = reinterpret_cast<hkpMeshMaterial*>(store.m_materials.begin());
				part.m_materialStriding = sizeof(Material);
			}
		}

		for( i = 0; i < m_shapesSubparts.getSize(); ++i )
		{
			ShapesSubpart& part = m_shapesSubparts[i];
			
			// Fix v-table to ShapeSubpartStorage in case it needs to get serialized again
			new (m_shapestorage[i]) ShapeSubpartStorage(flag);

			ShapeSubpartStorage& store = *m_shapestorage[i];
			
			if (part.getMaterialIndexStridingType() == MATERIAL_INDICES_INT8)
			{
				part.m_materialIndexBase = store.m_materialIndices.begin();
			}
			else
			{
				part.m_materialIndexBase = store.m_materialIndices16.begin();
			}

			part.m_materialBase = reinterpret_cast<hkpMeshMaterial*>(store.m_materials.begin());
		}
	}	
}

hkpStorageExtendedMeshShape::hkpStorageExtendedMeshShape( hkReal radius, int numbits )
: hkpExtendedMeshShape(radius, numbits)
{
}

hkpStorageExtendedMeshShape::~hkpStorageExtendedMeshShape()
{
	for( int i = 0; i < m_meshstorage.getSize(); ++i )
	{
		m_meshstorage[i]->removeReference();
	}

	// Take over what the hkpExtendedMeshShape destructor would do since our ShapesSubparts don't have an allocated array
	// of hkpConvexShape*, they actually point directly to the array.	
	// Ensure that hkpExtendedMeshShape destructor doesn't try to do the deallocation of the m_childShapes array
	m_shapesSubparts.clear();

	for( int i = 0; i < m_shapestorage.getSize(); ++i )
	{
		m_shapestorage[i]->removeReference();
	}		

	m_materialClass = &hkpStorageExtendedMeshShapeMaterialClass;
}

static int numIndices( int stride, int numTri )
{
	switch( stride )
	{
		case 1: // tri strip
			return 2 + numTri;
		case 2: // strange!?
			return 1 + 2*numTri;
		default: // independent
			return 4*numTri;
	}
}

hkpStorageExtendedMeshShape::hkpStorageExtendedMeshShape( const hkpExtendedMeshShape* mesh )
: hkpExtendedMeshShape( mesh->getRadius(), mesh->getNumBitsForSubpartIndex() )
{
	m_userData = mesh->m_userData; // hkpShape
	m_disableWelding = mesh->m_disableWelding; // hkpShapeCollection
	
	// now add the triangle parts
	int i;
	for( i = 0; i < mesh->getNumTrianglesSubparts(); ++i )
	{
		addTrianglesSubpart(mesh->getTrianglesSubpartAt(i));
	}

	// now add the convex parts
	for( i = 0; i < mesh->getNumShapesSubparts(); ++i )
	{
		addShapesSubpart(mesh->getShapesSubpartAt(i));
	}

	m_weldingInfo = mesh->m_weldingInfo;
	m_weldingType = mesh->m_weldingType;

		// Confirm all data got copied over correctly
		// Not easy to do this for convex vertices shapes, so for the moment just do it for triangles.
#ifdef HK_DEBUG
	hkpShapeKey korig = mesh->getFirstKey();
	hkpShapeKey kthis = this->getFirstKey();
	while(1)
	{
		HK_ASSERT(0x2f720403, korig == kthis);
		if( korig == HK_INVALID_SHAPE_KEY ) break;

		if ( getSubpartType(korig) == SUBPART_TRIANGLES )
		{
			hkpShapeBuffer borig;
			hkpShapeBuffer bthis;

			const hkpTriangleShape* torig = static_cast<const hkpTriangleShape*>(mesh->getChildShape(korig, borig));
			const hkpTriangleShape* tthis = static_cast<const hkpTriangleShape*>(this->getChildShape(kthis, bthis));

			for( int j = 0; j < 3; ++j )
			{
				HK_ASSERT(0x7ffdd20a, torig->getVertex(j).allExactlyEqual<3>( tthis->getVertex(j) ) );
			}
		}

		korig = mesh->getNextKey(korig);
		kthis = mesh->getNextKey(kthis);
	}
#endif
	
	recalcAabbExtents();
}

void hkpStorageExtendedMeshShape::addTrianglesSubpart( const TrianglesSubpart& partIn )
{
	HK_ON_DEBUG( assertTrianglesSubpartValidity(partIn); )

		// Copy all arguments first ( consistent with addShapesSubpart() )
		TrianglesSubpart& part = *expandOneTriangleSubparts();
	part = partIn; 
	/*
	{
	// expandOne doesn't call constructor, so force it
	part = TrianglesSubpart();

	// copy all fields over otherwise we might get crashes in the welding
	part.m_extrusion = partIn.m_extrusion;
	part.m_triangleOffset = partIn.m_triangleOffset;
	}
	*/
	m_meshstorage.pushBack( new MeshSubpartStorage );
	MeshSubpartStorage& store = *m_meshstorage.back();

	// vertices

	{
		hkVector4* dst = store.m_vertices.expandBy(partIn.m_numVertices);
		const hkReal* src = partIn.m_vertexBase;
		for( int j = 0; j < partIn.m_numVertices; ++j )
		{
			dst->set(src[0], src[1], src[2], 0.0f);
			dst ++;
			src = hkAddByteOffsetConst( src, partIn.m_vertexStriding );
		}
		part.m_vertexBase = &store.m_vertices.begin()[0](0);
		part.m_vertexStriding = sizeof(hkVector4);
		part.m_numVertices = partIn.m_numVertices;
	}

	// indices

	{
		void* newIndicesBegin;
		if( partIn.m_stridingType == INDICES_INT8 )
		{
			hkInt16 strideIn = partIn.m_indexStriding;
			part.m_indexStriding = (hkUint16)( (strideIn <= 2 ? strideIn : 4) * sizeof(hkUint8) );

			int nindices = numIndices(strideIn, partIn.m_numTriangleShapes);
			hkUint8* dst = store.m_indices8.expandBy( nindices );
			hkString::memSet( dst, 0, nindices*sizeof(hkUint8));
			newIndicesBegin = dst;
			const hkUint8* src = static_cast<const hkUint8*>(partIn.m_indexBase);
			for( int j = 0; j < partIn.m_numTriangleShapes; ++j )
			{
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
				dst = hkAddByteOffset( dst, part.m_indexStriding );
				src = hkAddByteOffsetConst( src, partIn.m_indexStriding );
			}
		}
		else if( partIn.m_stridingType == INDICES_INT16 )
		{
			hkInt16 strideIn = partIn.m_indexStriding/sizeof(hkUint16);
			part.m_indexStriding = (hkUint16)( (strideIn <= 2 ? strideIn : 4) * sizeof(hkUint16) );

			int nindices = numIndices(strideIn, partIn.m_numTriangleShapes);
			hkUint16* dst = store.m_indices16.expandBy( nindices );
			hkString::memSet( dst, 0, nindices*sizeof(hkUint16));
			newIndicesBegin = dst;
			const hkUint16* src = static_cast<const hkUint16*>(partIn.m_indexBase);
			for( int j = 0; j < partIn.m_numTriangleShapes; ++j )
			{
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
				dst = hkAddByteOffset( dst, part.m_indexStriding );
				src = hkAddByteOffsetConst( src, partIn.m_indexStriding );
			}
		}
		else //if( partIn.m_stridingType == INDICES_INT32 )
		{
			HK_ASSERT2( 0x12131a31, part.m_stridingType == INDICES_INT32, "Subpart index type is not set or out of range (8, 16, or 32 bit only)." );

			hkInt16 strideIn = partIn.m_indexStriding/sizeof(hkUint32);
			part.m_indexStriding = (hkUint16)( (strideIn <= 2 ? strideIn : 4) * sizeof(hkUint32) );

			int nindices = numIndices(strideIn, partIn.m_numTriangleShapes);
			hkUint32* dst = store.m_indices32.expandBy( nindices );
			hkString::memSet( dst, 0, nindices*sizeof(hkUint32));
			newIndicesBegin = dst;
			const hkUint32* src = static_cast<const hkUint32*>(partIn.m_indexBase);
			for( int j = 0; j < partIn.m_numTriangleShapes; ++j )
			{
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
				dst = hkAddByteOffset( dst, part.m_indexStriding );
				src = hkAddByteOffsetConst( src, partIn.m_indexStriding );
			}
		}

		part.m_stridingType = partIn.m_stridingType;
		part.m_numTriangleShapes = partIn.m_numTriangleShapes;
		part.m_flipAlternateTriangles = partIn.m_flipAlternateTriangles;
		part.m_indexBase = newIndicesBegin;
	}

	// material indices

	part.setMaterialIndexStridingType(partIn.getMaterialIndexStridingType());

	if( partIn.m_materialIndexBase )
	{
		if( partIn.getMaterialIndexStridingType() == MATERIAL_INDICES_INT8 )
		{
			if( partIn.m_materialIndexStriding == 0 )
			{
				store.m_materialIndices.pushBack( static_cast<const hkUint8*>(partIn.m_materialIndexBase)[0] );
				part.m_materialIndexBase = &store.m_materialIndices.back();
			}
			else
			{
				hkUint8* dst = store.m_materialIndices.expandBy(partIn.m_numTriangleShapes);
				part.m_materialIndexBase = dst;
				const hkUint8* src = static_cast<const hkUint8*>(partIn.m_materialIndexBase);
				for( int j = 0; j < partIn.m_numTriangleShapes; ++j )
				{
					dst[j] = *src;
					src = hkAddByteOffsetConst( src, partIn.m_materialIndexStriding );
				}
			}
		}
		else // if( partIn.m_materialIndexType == MATERIAL_INDICES_INT16 )
		{
			if( partIn.m_materialIndexStriding == 0 )
			{
				store.m_materialIndices16.pushBack( static_cast<const hkUint16*>(partIn.m_materialIndexBase)[0] );
				part.m_materialIndexBase = &store.m_materialIndices16.back();
			}
			else 
			{
				hkUint16* dst = store.m_materialIndices16.expandBy(partIn.m_numTriangleShapes);
				part.m_materialIndexBase = dst;
				const hkUint16* src = static_cast<const hkUint16*>(partIn.m_materialIndexBase);
				for( int j = 0; j < partIn.m_numTriangleShapes; ++j )
				{
					dst[j] = *src;
					src = hkAddByteOffsetConst( src, partIn.m_materialIndexStriding );
				}
			}
		}

		part.m_materialIndexStriding = partIn.m_materialIndexStriding;

		if (part.m_materialIndexStriding != 0)
		{
			switch( part.getMaterialIndexStridingType() )
			{
			case MATERIAL_INDICES_INT8 : part.m_materialIndexStriding = sizeof(hkUint8); break;
			case MATERIAL_INDICES_INT16: part.m_materialIndexStriding = sizeof(hkUint16); break;
			default                    : HK_ASSERT2(0XAD45F3F3,0, "hkpMeshShape::m_materialIndexType not properly specified."); break;
			}
		}

	}
	else
	{
		part.m_materialIndexBase = HK_NULL;
		part.m_materialIndexStriding = 0;
	}

	// materials
	if( part.m_materialIndexBase )
	{
		HK_ASSERT2(0xad87fb7b, partIn.m_materialBase, "Materials not defined for a hkpMeshShape::SubPart.");

		if( m_materialClass == &hkpNamedMeshMaterialClass )
		{
			hkpNamedMeshMaterial* dst = store.m_namedMaterials.expandBy(partIn.getNumMaterials());
			const hkpMeshMaterial* src = partIn.m_materialBase;

			for( int j = 0; j < partIn.getNumMaterials(); ++j )
			{
				hkpNamedMeshMaterial& namedMaterial = * new (dst+j) hkpNamedMeshMaterial();
				const hkpNamedMeshMaterial* m = static_cast< const hkpNamedMeshMaterial* >( src );
				namedMaterial = *m;			

				src = hkAddByteOffsetConst( src, partIn.m_materialStriding );
			}

			part.m_materialBase = reinterpret_cast<const hkpMeshMaterial*>( store.m_namedMaterials.begin() );

			if ( partIn.m_materialStriding )
			{
				part.m_materialStriding = sizeof( hkpNamedMeshMaterial );
				part.setNumMaterials(partIn.getNumMaterials());
			}
			else
			{
				part.m_materialStriding = 0;
				part.setNumMaterials(1);
			}

		}
		else
		{
			hkpStorageExtendedMeshShape::Material* dst = store.m_materials.expandBy(partIn.getNumMaterials());
			const hkpMeshMaterial* src = partIn.m_materialBase;

			for( int j = 0; j < partIn.getNumMaterials(); ++j )
			{
				hkpStorageExtendedMeshShape::Material& material = * new (dst+j) hkpStorageExtendedMeshShape::Material();
				if ( m_materialClass == &hkpStorageExtendedMeshShapeMaterialClass )
				{
					const hkpStorageExtendedMeshShape::Material* m = static_cast< const hkpStorageExtendedMeshShape::Material* >( src );
					material = *m;
				}
				else
				{
					material.m_filterInfo = src->m_filterInfo;
					material.m_friction.setOne();
					material.m_restitution.setZero();
					material.m_userData = 0;

					HK_WARN_ONCE( 0x75971a9c, "Only the m_filterInfo member is copied from the mesh material." );
				}

				src = hkAddByteOffsetConst( src, partIn.m_materialStriding );
			}
			part.m_materialBase = reinterpret_cast<const hkpMeshMaterial*>( store.m_materials.begin() );

			if ( partIn.m_materialStriding )
			{
				part.m_materialStriding = sizeof( hkpStorageExtendedMeshShape::Material );
				part.setNumMaterials(partIn.getNumMaterials());
			}
			else
			{
				part.m_materialStriding = 0;
				part.setNumMaterials(1);
			}

		}
		
	}


	// For internal debugging
	HK_ON_DEBUG( assertTrianglesSubpartValidity(part); )

	{
		hkAabb current;
		current.m_min.setSub( m_aabbCenter, m_aabbHalfExtents );
		current.m_max.setAdd( m_aabbCenter, m_aabbHalfExtents );

		hkAabb aabbPart;
		{
			calcAabbExtents( part, aabbPart );

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

	m_cachedNumChildShapes += _getNumChildShapesInTrianglesSubpart(part, m_trianglesSubparts.getSize()-1);
}

int hkpStorageExtendedMeshShape::addShapesSubpart( const ShapesSubpart& partIn )
{
	HK_ON_DEBUG( assertShapesSubpartValidity(partIn); )

	ShapesSubpart& part = *expandOneShapesSubparts();

	m_shapestorage.pushBack( new ShapeSubpartStorage );
	ShapeSubpartStorage& store = *m_shapestorage.back();

	// shapes
	part = partIn;	// Copy most of data

	// material indices

	part.setMaterialIndexStridingType(partIn.getMaterialIndexStridingType());

	if( partIn.m_materialIndexBase )
	{
		if( partIn.getMaterialIndexStridingType() == MATERIAL_INDICES_INT8 )
		{
			if( partIn.m_materialIndexStriding == 0 )
			{
				store.m_materialIndices.pushBack( static_cast<const hkUint8*>(partIn.m_materialIndexBase)[0] );
				part.m_materialIndexBase = &store.m_materialIndices.back();
			}
			else
			{
				int numChildShapes = partIn.m_childShapes.getSize();
				hkUint8* dst = store.m_materialIndices.expandBy(numChildShapes);
				part.m_materialIndexBase = dst;
				const hkUint8* src = static_cast<const hkUint8*>(partIn.m_materialIndexBase);
				for( int j = 0; j < numChildShapes; ++j )
				{
					dst[j] = *src;
					src = hkAddByteOffsetConst( src, partIn.m_materialIndexStriding );
				}
			}
		}
		else // if( partIn.m_materialIndexType == MATERIAL_INDICES_INT16 )
		{
			if( partIn.m_materialIndexStriding == 0 )
			{
				store.m_materialIndices16.pushBack( static_cast<const hkUint16*>(partIn.m_materialIndexBase)[0] );
				part.m_materialIndexBase = &store.m_materialIndices16.back();
			}
			else 
			{
				int numChildShapes = partIn.m_childShapes.getSize();
				hkUint16* dst = store.m_materialIndices16.expandBy(numChildShapes);
				part.m_materialIndexBase = dst;
				const hkUint16* src = static_cast<const hkUint16*>(partIn.m_materialIndexBase);
				for( int j = 0; j < numChildShapes; ++j )
				{
					dst[j] = *src;
					src = hkAddByteOffsetConst( src, partIn.m_materialIndexStriding );
				}
			}
		}

		part.m_materialIndexStriding = partIn.m_materialIndexStriding;

		if (part.m_materialIndexStriding != 0)
		{
			switch(part.getMaterialIndexStridingType())
			{
			case MATERIAL_INDICES_INT8 : part.m_materialIndexStriding = sizeof(hkUint8); break;
			case MATERIAL_INDICES_INT16: part.m_materialIndexStriding = sizeof(hkUint16); break;
			default                    : HK_ASSERT2(0XAD45F3F3,0, "hkpMeshShape::m_materialIndexType not properly specified."); break;
			}
		}

	}
	else
	{
		part.m_materialIndexBase = HK_NULL;
		part.m_materialIndexStriding = 0;
	}

	
	// materials
	if( part.m_materialIndexBase )
	{
		HK_ASSERT2(0xad87fb7b, partIn.m_materialBase, "Materials not defined for a hkpMeshShape::SubPart.");

		hkpStorageExtendedMeshShape::Material* dst = store.m_materials.expandBy(partIn.getNumMaterials());
		const hkpMeshMaterial* src = partIn.m_materialBase;

		for( int j = 0; j < partIn.getNumMaterials(); ++j )
		{
			hkpStorageExtendedMeshShape::Material& material = *new ( dst + j ) hkpStorageExtendedMeshShape::Material();
			if ( m_materialClass == &hkpStorageExtendedMeshShapeMaterialClass )
			{
				const hkpStorageExtendedMeshShape::Material* m = static_cast< const hkpStorageExtendedMeshShape::Material* >( src );
				material = *m;
			}
			else
			{
				material.m_filterInfo = src->m_filterInfo;
				material.m_friction.setOne();
				material.m_restitution.setZero();
				material.m_userData = 0;

				HK_WARN_ONCE( 0x75971a9b, "Only the m_filterInfo member is copied from the mesh material." );
			}

			src = hkAddByteOffsetConst( src, partIn.m_materialStriding );
		}

		if ( partIn.m_materialStriding )
		{
			part.m_materialStriding = sizeof( hkpStorageExtendedMeshShape::Material );
			part.setNumMaterials(partIn.getNumMaterials());
		}
		else
		{
			part.m_materialStriding = 0;
			part.setNumMaterials(1);
		}

		part.m_materialBase = reinterpret_cast<const hkpMeshMaterial*>( store.m_materials.begin() );
	}

	{
		hkAabb current;
		current.m_min.setSub( m_aabbCenter, m_aabbHalfExtents );
		current.m_max.setAdd( m_aabbCenter, m_aabbHalfExtents );

		hkAabb aabbPart;
		{
			calcAabbExtents( part, aabbPart );
			current.m_min.setMin( current.m_min, aabbPart.m_min );
			current.m_max.setMax( current.m_max, aabbPart.m_max );
		}

		current.getCenter( m_aabbCenter );
		current.getHalfExtents( m_aabbHalfExtents );
	}

	m_cachedNumChildShapes += _getNumChildShapesInShapesSubpart(part);

	return m_shapesSubparts.getSize()-1;
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
