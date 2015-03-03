/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Deprecated/StorageMesh/hkpStorageMeshShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>

hkpStorageMeshShape::SubpartStorage::SubpartStorage( hkFinishLoadedObjectFlag flag )
	: hkReferencedObject(flag), m_vertices(flag), m_indices16(flag), m_indices32(flag),
	m_materialIndices(flag), m_materials(flag), m_materialIndices16(flag)
{
}

hkpStorageMeshShape::hkpStorageMeshShape( hkFinishLoadedObjectFlag flag )
	: hkpMeshShape(flag), m_storage(flag)
{
	if( flag.m_finishing )
	{
		for( int i = 0; i < m_subparts.getSize(); ++i )
		{
			Subpart& part = m_subparts[i];
			SubpartStorage& store = *m_storage[i];

			part.m_vertexBase = store.m_vertices.begin();
			if (part.m_stridingType == INDICES_INT16)
				part.m_indexBase = store.m_indices16.begin();
			else
				part.m_indexBase = store.m_indices32.begin();
			if (part.m_materialIndexStridingType == MATERIAL_INDICES_INT8)
				part.m_materialIndexBase = store.m_materialIndices.begin();
			else
				part.m_materialIndexBase = store.m_materialIndices16.begin();
			part.m_materialBase = reinterpret_cast<hkpMeshMaterial*>(store.m_materials.begin());
		}
	}	
}

hkpStorageMeshShape::hkpStorageMeshShape( hkReal radius, int numbits )
: hkpMeshShape(radius, numbits)
{
}

hkpStorageMeshShape::~hkpStorageMeshShape()
{
	for( int i = 0; i < m_storage.getSize(); ++i )
	{
		m_storage[i]->removeReference();
	}
}

static int storageMeshShape_numIndices( int stride, int numTri )
{
	switch( stride )
	{
		case 1: // tri strip
			return 2 + numTri;
		case 2: // strange!?
			return 1 + 2*numTri;
		default: // independent
			return 3*numTri;
	}
}

hkpStorageMeshShape::hkpStorageMeshShape( const hkpMeshShape* mesh )
: hkpMeshShape( mesh->getRadius(), mesh->getNumBitsForSubpartIndex() )
{
	m_scaling = mesh->m_scaling;
	
	// [HVK-2295] missing mesh atributes
	m_userData = mesh->m_userData; // hkpShape
	m_disableWelding = mesh->m_disableWelding; // hkpShapeCollection
	
	// now add the parts
	for( int i = 0; i < mesh->getNumSubparts(); ++i )
	{
		addSubpart(mesh->getSubpartAt(i));
	}


	m_weldingInfo = mesh->m_weldingInfo;
	m_weldingType = mesh->m_weldingType;

#ifdef HK_DEBUG
	hkpShapeKey korig = mesh->getFirstKey();
	hkpShapeKey kthis = this->getFirstKey();
	while(1)
	{
		HK_ASSERT(0x30fa7b46, korig == kthis);
		if( korig == HK_INVALID_SHAPE_KEY ) break;

		hkpShapeBuffer borig;
		hkpShapeBuffer bthis;

		const hkpTriangleShape* torig = static_cast<const hkpTriangleShape*>(mesh->getChildShape(korig, borig));
		const hkpTriangleShape* tthis = static_cast<const hkpTriangleShape*>(this->getChildShape(kthis, bthis));

		for( int j = 0; j < 3; ++j )
		{
			HK_ASSERT(0x55c9ebfc, torig->getVertex(j).allExactlyEqual<3>( tthis->getVertex(j) ) );
		}

		korig = mesh->getNextKey(korig);
		kthis = mesh->getNextKey(kthis);
	}
#endif
}

void hkpStorageMeshShape::addSubpart( const Subpart& partIn )
{
	HK_ON_DEBUG( assertSubpartValidity(partIn); )

	Subpart& part = m_subparts.expandOne();
	m_storage.pushBack( new SubpartStorage );
	SubpartStorage& store = *m_storage.back();

	// vertices

	{
		hkReal* dst = store.m_vertices.expandBy(3*partIn.m_numVertices);
		const hkReal* src = partIn.m_vertexBase;
		for( int j = 0; j < partIn.m_numVertices; ++j )
		{
			dst[0] = src[0];
			dst[1] = src[1];
			dst[2] = src[2];
			dst += 3;
			src = hkAddByteOffsetConst( src, partIn.m_vertexStriding );
		}
		part.m_vertexBase = store.m_vertices.begin();
		part.m_vertexStriding = 12;
		part.m_numVertices = partIn.m_numVertices;
	}

	// indices

	{
		void* newIndicesBegin;
		if( partIn.m_stridingType == INDICES_INT16 )
		{
			int strideIn = partIn.m_indexStriding/sizeof(hkUint16);
			part.m_indexStriding = (strideIn <= 2 ? strideIn : 3) * sizeof(hkUint16);
			
			int nindices = storageMeshShape_numIndices(strideIn, partIn.m_numTriangles);
			hkUint16* dst = store.m_indices16.expandBy( nindices );
			newIndicesBegin = dst;
			const hkUint16* src = static_cast<const hkUint16*>(partIn.m_indexBase);
			for( int j = 0; j < partIn.m_numTriangles; ++j )
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
			int strideIn = partIn.m_indexStriding/sizeof(hkUint32);
			part.m_indexStriding = (strideIn <= 2 ? strideIn : 3) * sizeof(hkUint32);

			int nindices = storageMeshShape_numIndices(strideIn, partIn.m_numTriangles);
			hkUint32* dst = store.m_indices32.expandBy( nindices );
			newIndicesBegin = dst;
			const hkUint32* src = static_cast<const hkUint32*>(partIn.m_indexBase);
			for( int j = 0; j < partIn.m_numTriangles; ++j )
			{
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
				dst = hkAddByteOffset( dst, part.m_indexStriding );
				src = hkAddByteOffsetConst( src, partIn.m_indexStriding );
			}
		}

		part.m_stridingType = partIn.m_stridingType;
		part.m_numTriangles = partIn.m_numTriangles;
		part.m_flipAlternateTriangles = partIn.m_flipAlternateTriangles;
		part.m_indexBase = newIndicesBegin;
	}

	// material indices

	part.m_materialIndexStridingType = partIn.m_materialIndexStridingType;

	if( partIn.m_materialIndexBase )
	{
		if( partIn.m_materialIndexStridingType == MATERIAL_INDICES_INT8 )
		{
			if( partIn.m_materialIndexStriding == 0 )
			{
				store.m_materialIndices.pushBack( static_cast<const hkUint8*>(partIn.m_materialIndexBase)[0] );
				part.m_materialIndexBase = &store.m_materialIndices.back();
			}
			else
			{
				hkUint8* dst = store.m_materialIndices.expandBy(partIn.m_numTriangles);
				part.m_materialIndexBase = dst;
				const hkUint8* src = static_cast<const hkUint8*>(partIn.m_materialIndexBase);
				for( int j = 0; j < partIn.m_numTriangles; ++j )
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
				hkUint16* dst = store.m_materialIndices16.expandBy(partIn.m_numTriangles);
				part.m_materialIndexBase = dst;
				const hkUint16* src = static_cast<const hkUint16*>(partIn.m_materialIndexBase);
				for( int j = 0; j < partIn.m_numTriangles; ++j )
				{
					dst[j] = *src;
					src = hkAddByteOffsetConst( src, partIn.m_materialIndexStriding );
				}
			}
		}

		part.m_materialIndexStriding = partIn.m_materialIndexStriding;

		if (part.m_materialIndexStriding != 0)
		{
			switch(part.m_materialIndexStridingType)
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
		if( partIn.m_materialStriding == 0 )
		{
			store.m_materials.pushBack( *reinterpret_cast<const hkUint32*>(partIn.m_materialBase) );
			part.m_materialStriding = 0;
			part.m_numMaterials = 1;
		}
		else
		{
			hkUint32* dst = store.m_materials.expandBy(partIn.m_numMaterials);
			const hkUint32* src = reinterpret_cast<const hkUint32*>(partIn.m_materialBase);
			for( int j = 0; j < partIn.m_numMaterials; ++j )
			{
				dst[j] = *src;
				src = hkAddByteOffsetConst( src, partIn.m_materialStriding );
			}
			part.m_materialStriding = sizeof(hkUint32*);
			part.m_numMaterials = partIn.m_numMaterials;
		}
		part.m_materialBase = reinterpret_cast<const hkpMeshMaterial*>(store.m_materials.begin());
	}
	else
	{
		part.m_numMaterials = 1;
		part.m_materialBase = reinterpret_cast<const hkpMeshMaterial*>( &hkVector4::getZero() );
		part.m_materialStriding = 0;
		part.m_materialIndexBase = reinterpret_cast<const hkUint8*>( &hkVector4::getZero() );
	}

	part.m_triangleOffset = partIn.m_triangleOffset;

	HK_ASSERT2(0x75971a9b, partIn.m_materialStriding==0 || partIn.m_materialStriding==sizeof(hkUint32),
		"Only the m_filterInfo member is copied from the mesh material.");
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
