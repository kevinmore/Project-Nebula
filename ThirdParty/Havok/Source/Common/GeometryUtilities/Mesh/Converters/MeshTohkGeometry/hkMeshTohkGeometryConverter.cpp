/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Converters/MeshTohkGeometry/hkMeshTohkGeometryConverter.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindUniquePositionsUtil/hkFindUniquePositionsUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/PrimitiveUtil/hkMeshPrimitiveUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionBuilder/hkMeshSectionBuilder.h>

/*static*/ void HK_CALL hkMeshTohkGeometryConverter::getVerticesIndicesFromMeshSection(const hkMeshShape* shape, int sectionIndex, hkArray<hkVector4>& vertices, hkArray<int>& indices)
{
	HK_ASSERT(0x82358, sectionIndex < shape->getNumSections());

	hkMeshSection meshSection;
	shape->lockSection(sectionIndex, hkMeshShape::ACCESS_VERTEX_BUFFER | hkMeshShape::ACCESS_INDICES, meshSection);
	hkRefPtr<hkMeshVertexBuffer> vertexBuffer = meshSection.m_vertexBuffer;
	shape->unlockSection(meshSection);

    // Get the positions
    hkVertexFormat vertexFormat;
    vertexBuffer->getVertexFormat(vertexFormat);

    int elementIndex = vertexFormat.findElementIndex(hkVertexFormat::USAGE_POSITION, 0);
    if (elementIndex < 0)
    {
        HK_ASSERT(0x8d7292c4, !"Vertex buffer does not contain position");
		return;
    }

    //
    hkMeshVertexBuffer::LockInput lockInput;
    hkMeshVertexBuffer::PartialLockInput partialLockInput;

    partialLockInput.m_numLockFlags = 1;
    partialLockInput.m_elementIndices[0] = elementIndex;
    partialLockInput.m_lockFlags[0] = hkMeshVertexBuffer::ACCESS_READ | hkMeshVertexBuffer::ACCESS_ELEMENT_ARRAY;

    hkMeshVertexBuffer::LockedVertices lockedVertices;
    hkMeshVertexBuffer::LockResult lockRes = vertexBuffer->partialLock( lockInput, partialLockInput, lockedVertices);
    if (lockRes != hkMeshVertexBuffer::RESULT_SUCCESS)
    {
		HK_ASSERT(0xd8279a38, !"Unable to lock the vertex buffer");
		return;
    }

	hkArray<hkUint16> shortIndices;
	hkMeshPrimitiveUtil::appendTriangleIndices(meshSection, shortIndices);
	int vertexOffset = vertices.getSize();

	// copy-convert the indices as mesh system uses 16bit while the rest of the geometry util uses 32bit
	int* indicesAppendStart = indices.expandBy(shortIndices.getSize());
	for (int i = 0; i < shortIndices.getSize(); ++i)
	{
		indicesAppendStart[i] = int(shortIndices[i]) + vertexOffset;
	}

    // Get the positions
    hkVector4* verticesAppendStart = vertices.expandBy(lockedVertices.m_numVertices);
	{
		hkArray<hkFloat32>::Temp va; va.setSize(4*lockedVertices.m_numVertices);
		vertexBuffer->getElementVectorArray(lockedVertices, 0, va.begin());
		for (int i=0; i<lockedVertices.m_numVertices; ++i)
		{
			verticesAppendStart[i].load<4,HK_IO_NATIVE_ALIGNED>(&va[4*i]);
		}
	}
    vertexBuffer->unlock(lockedVertices);
}

/* static */hkGeometry* HK_CALL hkMeshTohkGeometryConverter::convert( const hkMeshShape* shape )
{
	hkGeometry* geom = new hkGeometry;

    hkFindUniquePositionsUtil posUtil;

	hkArray<int> indices;
    hkArray<hkVector4> vertices;

    const int numSections = shape->getNumSections();
    for (int i = 0; i < numSections; i++)
    {
		getVerticesIndicesFromMeshSection(shape, i, vertices, indices);
	}

	int numTriangles = indices.getSize()/3;
	geom->m_triangles.reserve(numTriangles);
	posUtil.m_positions.reserve(vertices.getSize());
	for (int j = 0; j < numTriangles; j ++)
	{
		hkGeometry::Triangle& triangle = geom->m_triangles.expandOne();
		for (int k=0;k<3;k++)
		{
			(&triangle.m_a)[k] = posUtil.addPosition( vertices[ indices[j*3 + k ]]);
        }
    }

	geom->m_vertices.swap(posUtil.m_positions);

	return geom;
}

hkMeshShape* HK_CALL hkMeshTohkGeometryConverter::convert(const hkGeometry* geom, hkMeshSystem* meshSystem, hkMeshMaterial* overrideMaterial )
{

	if (!geom || !meshSystem || (geom->m_triangles.getSize() < 1))
	{
		return HK_NULL;
	}

	hkMeshSectionBuilder builder;
	
	// Vertices
	hkVertexFormat dstVertexFormat;
	dstVertexFormat.addElement(hkVertexFormat::USAGE_POSITION, hkVertexFormat::TYPE_FLOAT32, 3);
	hkMeshVertexBuffer* vertexBuffer = meshSystem->createVertexBuffer(dstVertexFormat, geom->m_vertices.getSize());
	{
		hkMeshVertexBuffer::LockInput lockInput;
		hkMeshVertexBuffer::LockedVertices lockedVertices;
		lockInput.m_lockFlags = hkMeshVertexBuffer::ACCESS_WRITE | hkMeshVertexBuffer::ACCESS_WRITE_DISCARD;

		hkMeshVertexBuffer::LockResult lockRes = vertexBuffer->lock(lockInput, lockedVertices);
		if (lockRes != hkMeshVertexBuffer::RESULT_SUCCESS)
		{
			return HK_NULL;
		}

		const int numBuffers = lockedVertices.m_numBuffers;
		for (int i = 0; i < numBuffers; i++)
		{
			hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer = lockedVertices.m_buffers[i];
			if (dstBuffer.m_element.m_usage == hkVertexFormat::USAGE_POSITION)
			{
				hkFloat32* v = (hkFloat32*) dstBuffer.m_start;
				int floatStride = dstBuffer.m_stride / sizeof(hkFloat32);
				for (int vi=0; vi < geom->m_vertices.getSize(); ++vi)
				{
					const hkVector4& vertFrom = geom->m_vertices[vi]; 
					vertFrom.store<3,HK_IO_NATIVE_ALIGNED>( v );
					v += floatStride;
				}

				break;
			}
		}

		vertexBuffer->unlock(lockedVertices);
	}


	// builder
	builder.startMeshSection(vertexBuffer, overrideMaterial);

	hkUint32 indexBase = 0xffffffff;
	{
		for (int triangleIndex = 0; triangleIndex < geom->m_triangles.getSize(); triangleIndex++)
		{
			const hkGeometry::Triangle& triangle = geom->m_triangles[triangleIndex];
			if ( (hkUint32)triangle.m_a < indexBase) { indexBase = (hkUint32)triangle.m_a; }
			if ( (hkUint32)triangle.m_b < indexBase) { indexBase = (hkUint32)triangle.m_b; }
			if ( (hkUint32)triangle.m_c < indexBase) { indexBase = (hkUint32)triangle.m_c; }
		}
	}

	int numIndices = (geom->m_triangles.getSize()) * 3;
	hkArray<hkUint16> reducedIndices;
	reducedIndices.setSize(numIndices);
	int reducedIndicesCounter = 0;

	{
		for (int triangleIndex = 0; triangleIndex < geom->m_triangles.getSize(); triangleIndex++)
		{
			const hkGeometry::Triangle& triangle = geom->m_triangles[triangleIndex];
			hkUint32* triangleIndices = (hkUint32*)&triangle.m_a;
			{
				for (int i = 0; i < 3; i++)
				{
					hkUint32 ri = triangleIndices[i] - indexBase;
					if (ri > 0x0ffff)
					{
						HK_WARN(0x432645de, "Indices exceed the hkUint16 hkMesh section builder limit" );
					}
					reducedIndices[reducedIndicesCounter++] = (hkUint16)ri;
				}
			}
		}
	}

	builder.concatPrimitives(hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST, reducedIndices.begin(), reducedIndices.getSize(), indexBase);

	builder.endMeshSection();
	
	hkMeshShape* meshShape = meshSystem->createShape(builder.getSections(), builder.getNumSections());
	return meshShape;
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
