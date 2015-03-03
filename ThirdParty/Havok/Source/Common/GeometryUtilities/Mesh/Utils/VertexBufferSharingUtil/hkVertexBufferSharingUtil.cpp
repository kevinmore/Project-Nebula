/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferSharingUtil/hkVertexBufferSharingUtil.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshVertexBuffer.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindUniquePositionsUtil/hkFindUniquePositionsUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/IndexSet/hkIndexSet.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionBuilder/hkMeshSectionBuilder.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>
#include <Common/GeometryUtilities/Mesh/Utils/PrimitiveUtil/hkMeshPrimitiveUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexSharingUtil/hkVertexSharingUtil.h>

class hkMeshSectionVertexRemap: public hkReferencedObject
{
    public:
        HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SCENE_DATA);

        int m_sectionSetIndex;                          ///< The section index (in the hkMeshSectionLockSet)
        int m_vertexFormatIndex;                        ///< The vertex format index

		hkArray<hkUint16> m_triangleIndices;			///< The indices into the target vertex buffer
		hkRefPtr<hkMeshVertexBuffer> m_vertexBuffer;	///< The target mesh vertex buffer

        int m_sectionIndex;                             ///< The section index
        const hkMeshShape* m_meshShape;                 ///< The mesh shape this belongs to

        hkIndexSet m_indexSet;                          ///< The unique indices
        hkArray<int> m_globalTriangleIndices;           ///< The triangle indices into the global vertex buffer (the buffer shared between all of the same format)
};

HK_FORCE_INLINE static bool hkVertexBufferSharingUtil_orderNumIndices(const hkMeshSectionVertexRemap* a, const hkMeshSectionVertexRemap* b)
{
	return a->m_indexSet.getSize() < b->m_indexSet.getSize();
}

static void hkVertexBufferSharingUtil_copyVertices(const hkIndexSet& indexSet, const hkVertexSharingUtil& globalVertices, hkMemoryMeshVertexBuffer& vertexBuffer)
{
    const hkArray<int>& indices = indexSet.getIndices();
    const int numIndices = indices.getSize();
    vertexBuffer.setNumVerticesAndZero(numIndices);

    hkUint8* dstVertices = vertexBuffer.getVertexData();
    const hkUint8* srcVertices = globalVertices.getVertexData();

    const int vertexStride = vertexBuffer.getVertexStride();
	HK_ASSERT(0xd8279a2e, vertexStride == globalVertices.getVertexStride());

	HK_ON_DEBUG(const int numVertices =) globalVertices.getNumVertices();

    for (int i = 0; i < numIndices; i++)
    {
		int index = indices[i];
		HK_ASSERT(0xbd838dda, index >= 0 && index < numVertices);

        hkString::memCpy(dstVertices, srcVertices + (index * vertexStride), vertexStride);
        dstVertices += vertexStride;
    }
}

static void hkVertexBufferSharingUtil_flushVertexBuffer(hkMeshSystem* meshSystem, const hkIndexSet& indexSet, const hkVertexFormat& vertexFormat, const hkArray<hkMeshSectionVertexRemap*>& remapSet, const hkVertexSharingUtil& globalVertices)
{
	hkMemoryMeshVertexBuffer workVertexBuffer(vertexFormat, 0);

	// Flush what is in the work vertex buffer
	if (indexSet.getSize() <= 0)
	{
		return;
	}

	// Pull all from the index set, and place contiguously in the work vertex buffer
	hkVertexBufferSharingUtil_copyVertices(indexSet, globalVertices, workVertexBuffer);

	// Okay we have a vertex buffer, we need to construct and output
	hkMeshVertexBuffer* vertexBuffer = meshSystem->createVertexBuffer(vertexFormat, workVertexBuffer.getNumVertices());
	HK_ASSERT(0x3243243, vertexBuffer != HK_NULL);

	// I could lookup all of the vertices overt
	hkMeshVertexBufferUtil::convert(&workVertexBuffer, vertexBuffer);

	// Now I need to remap all of the indices
	for (int i = 0; i < remapSet.getSize(); i++)
	{
		hkMeshSectionVertexRemap* curRemap = remapSet[i];

		// We're using this vertex buffer
		curRemap->m_vertexBuffer = vertexBuffer;

		// set up for remap
		const hkArray<int>& srcIndices = curRemap->m_globalTriangleIndices;
		const int numIndices = srcIndices.getSize();

		hkArray<hkUint16>& dstIndices = curRemap->m_triangleIndices;
		dstIndices.setSize(numIndices);

		// Remap the indices
		for (int j = 0; j < numIndices; j++)
		{
			int dstIndex = indexSet.findIndex(srcIndices[j]);
			HK_ASSERT(0xd8279a2d, dstIndex >= 0);
			dstIndices[j] = hkUint16(dstIndex);
		}

		// We don't need these anymore, so we can deallocate
		curRemap->m_indexSet.clearAndDeallocate();
		curRemap->m_globalTriangleIndices.clearAndDeallocate();
	}

	// The vertex buffer is ref'd in all of the remaps that hold it
	vertexBuffer->removeReference();

}

static void hkVertexBufferSharingUtil_shareVertexBuffers(hkMeshSystem* meshSystem, const hkVertexFormat& vertexFormat, const hkArray<hkMeshSectionVertexRemap*>& remapsIn, hkVertexSharingUtil& globalVertices, int maxVertices)
{
	// Okay this is tricky.
	// We want to merge smallest first.
	// The more vertices that are shared then the better. Any amount shared is better than none shared.

	hkArray<hkMeshSectionVertexRemap*> remaps;
	remaps = remapsIn;

    // Sort by increasing number of indices
	hkSort(remaps.begin(), remaps.getSize(), hkVertexBufferSharingUtil_orderNumIndices);

    // The set of all the remaps that are being merged
    hkArray<hkMeshSectionVertexRemap*> remapSet;

    // The (global) indices we currently have
    hkIndexSet indexSet;

	while (remaps.getSize() > 0)
	{
		hkMeshSectionVertexRemap* remap = remaps[0];
		remaps.removeAtAndCopy(0);

        if (indexSet.getSize() + remap->m_indexSet.getSize() >= maxVertices)
        {
            // Flush what is in the work vertex buffer
			hkVertexBufferSharingUtil_flushVertexBuffer(meshSystem, indexSet, vertexFormat, remapSet, globalVertices);

            // Empty
            indexSet.clear();
            remapSet.clear();
        }

        // We can add this the remap set
        remapSet.pushBack(remap);

        // I need to add these to the index set
        hkIndexSet unionSet;
        unionSet.setUnion(remap->m_indexSet, indexSet);
        indexSet.swap(unionSet);
	}

	// Flush anything remaining
	hkVertexBufferSharingUtil_flushVertexBuffer(meshSystem, indexSet, vertexFormat, remapSet, globalVertices);
}

void hkVertexBufferSharingUtil::shareVertexBuffers(hkMeshSystem* meshSystem, const hkArray<const hkMeshShape*>& shapes, int maxVertices, hkArray<hkMeshShape*>& shapesOut)
{
    shapesOut.clear();

    // Okay lets try locking all of the input meshes
    hkMeshSectionLockSet sectionSet;

    {
        const int numShapes = shapes.getSize();
        for (int i = 0; i < numShapes; i++)
        {
            const hkMeshShape* meshShape = shapes[i];
            sectionSet.addMeshSections(meshShape, hkMeshShape::ACCESS_VERTEX_BUFFER | hkMeshShape::ACCESS_INDICES);
        }
    }
    const int numSections = sectionSet.getNumSections();

    hkArray<hkVertexFormat> uniqueVertexFormats;
    {

        for (int i = 0; i < numSections; i++)
        {
            const hkMeshSection& section = sectionSet.getSection(i);

            hkVertexFormat vertexFormat;
            section.m_vertexBuffer->getVertexFormat(vertexFormat);

            if (uniqueVertexFormats.indexOf(vertexFormat) < 0)
            {
                uniqueVertexFormats.pushBack(vertexFormat);
            }
        }
    }

	hkArray<hkMeshSectionVertexRemap*>::Temp sectionRemaps;
    sectionRemaps.setSize(numSections, HK_NULL);

    hkArray<hkMeshSectionVertexRemap*> vertexFormatRemaps;

    {
        hkVertexFormat vf;
        hkMemoryMeshVertexBuffer workVertexBuffer(vf, 0);
        hkArray<hkUint16> indices;

        const int numVertexFormats = uniqueVertexFormats.getSize();

        hkVertexSharingUtil sharingUtil;
		
        sharingUtil.setAllThresholds(hkReal(1.0e-3f));
        hkFindUniquePositionsUtil uniquePositionsUtil;

        for (int i = 0; i < numVertexFormats; i++)
        {
            const hkVertexFormat& vertexFormat = uniqueVertexFormats[i];
            workVertexBuffer.setVertexFormat(vertexFormat);

			int numVerticesInSections = 0;
			for (int j = 0; j < numSections; j++)
			{
				numVerticesInSections += sectionSet.getSection(j).m_vertexBuffer->getNumVertices();
			}
			uniquePositionsUtil.reset(numVerticesInSections);
            sharingUtil.begin(vertexFormat, numVerticesInSections);

			vertexFormatRemaps.clear();

            int positionIndex = vertexFormat.findElementIndex(hkVertexFormat::USAGE_POSITION, 0);
            HK_ASSERT(0x827faabb, positionIndex >= 0);

            for (int j = 0; j < numSections; j++)
            {
                const hkMeshSection& meshSection = sectionSet.getSection(j);
                hkMeshVertexBuffer* vertexBuffer = meshSection.m_vertexBuffer;

                // Only process sections with the same format
                hkVertexFormat sectionVertexFormat;
                vertexBuffer->getVertexFormat(sectionVertexFormat);
                if (sectionVertexFormat != vertexFormat)
                {
                    continue;
                }

                // Need to store this index somewhere
                hkMeshSectionVertexRemap* remap = new hkMeshSectionVertexRemap;

                remap->m_vertexFormatIndex = i;
                remap->m_sectionSetIndex = j;

                remap->m_meshShape = sectionSet.getShape(j);
                remap->m_sectionIndex = meshSection.m_sectionIndex;

                // These are in the same order as in the hkMeshSectionLockSet
                HK_ASSERT(0x34234, sectionRemaps[j] == HK_NULL);
                sectionRemaps[j] = remap;

                // Add to the remaps associated with this vertex format
                vertexFormatRemaps.pushBack(remap);

                //
                const int numVertices = vertexBuffer->getNumVertices();


                // Extract all of the vertices
                workVertexBuffer.setNumVerticesAndZero(numVertices);
                hkMeshVertexBufferUtil::convert(vertexBuffer, &workVertexBuffer);

				hkMeshVertexBuffer::LockedVertices::Buffer buffer;
				workVertexBuffer.getLockedVerticesBuffer(positionIndex, buffer);

                // We need the triangle indices
                indices.clear(); indices.reserve(meshSection.m_numPrimitives * 3);
                hkMeshPrimitiveUtil::appendTriangleIndices(meshSection, indices);

                const hkUint8* vertexData = workVertexBuffer.getVertexData();

                hkArray<int>& indexSet = remap->m_indexSet.startUpdate();

                const int numIndices = indices.getSize();
                for (int k = 0; k < numIndices; k++)
                {
                    const int index = indices[k];
					HK_ASSERT(0x995827, buffer.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32);
                    hkFloat32* position = (hkFloat32*)( ((hkUint8*)buffer.m_start) + (buffer.m_stride * index) );

                    hkVector4 pos; pos.load<3,HK_IO_NATIVE_ALIGNED>(position);
                    hkUint32 vertexHash = uniquePositionsUtil.addPosition(pos);

                    // Get the vertex
                    const hkUint8* vertex = vertexData + (index * buffer.m_stride);

                    // position index
                    const int vertexIndex = sharingUtil.addVertex((vertexHash << 4) + 1, vertex);

                    indexSet.pushBack(vertexIndex);
                    remap->m_globalTriangleIndices.pushBack(vertexIndex);
                }
                remap->m_indexSet.endUpdate();
                remap->m_indexSet.optimizeAllocation();
            }

#if 0
			// sharing consistency check: adding a shared vertex must return its own index
			int ok = 0;
			for (int j = 0; j < vertexFormatRemaps.getSize(); ++j)
			{
				const hkArray<int>& indexSet = vertexFormatRemaps[j]->m_indexSet.getIndices();
				hkMeshVertexBuffer* vtxBuf = sectionSet.getSection(vertexFormatRemaps[j]->m_sectionSetIndex).m_vertexBuffer;
				hkMeshVertexBuffer::LockedVertices lockedVertices;
				hkMeshVertexBuffer::LockInput lockInput;
				lockInput.m_lockFlags = hkMeshVertexBuffer::ACCESS_READ;
				hkMeshVertexBuffer::LockResult res = vtxBuf->lock(lockInput, lockedVertices);
				HK_ASSERT(0x323b6e86, res == hkMeshVertexBuffer::RESULT_SUCCESS);

				int bufIdx = lockedVertices.findBufferIndex(hkVertexFormat::USAGE_POSITION, 0);
				HK_ASSERT(0x2447e62a, bufIdx >= 0);
				hkMeshVertexBuffer::LockedVertices::Buffer buffer = lockedVertices.m_buffers[bufIdx];

				for (int k = 0; k < indexSet.getSize(); ++k)
				{
					const int index = indexSet[k];
					HK_ASSERT(0x995827, buffer.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32);
					hkFloat32* position = (hkFloat32*)(((hkUint8*)buffer.m_start) + buffer.m_stride * index);

					hkVector4 pos; pos.load<3>(position);
					int vertexHash = uniquePositionsUtil.findPosition(pos);
					if (vertexHash < 0)
					{
						HK_WARN(0x232004, "inconsistent uniquePositionUtil");
						ok++;
						continue;
					}

					// this only works for memory vtx buffers
					const hkUint8* vertex = (const hkUint8*)position;
					const int vertexIndex = sharingUtil.findVertexIndex(((hkUint32)vertexHash << 4) + 1, vertex);

					if (vertexIndex < 0)
					{
						HK_WARN(0x232003, "inconsistent sharingUtil");
						ok++;
						continue;
					}

					if (vertexIndex != index)
					{
						HK_WARN(0x232002, "inconsistent indexSet");
						ok++;
					}
				}

				vtxBuf->unlock(lockedVertices);
			}

			HK_ASSERT2(0x23957202,ok == 0, "vertex sharing produced inconsistent vertex buffer");
#endif

            hkMeshVertexBuffer::LockedVertices lockedVertices;
            sharingUtil.end(lockedVertices);

            // I can now optimize for the vertex format I have extracted
            hkVertexBufferSharingUtil_shareVertexBuffers(meshSystem, vertexFormat, vertexFormatRemaps, sharingUtil, maxVertices);
        }
    }

    // The section remaps are in the right order
    {
        hkMeshSectionBuilder builder;

        int start = 0;
        const int numShapes = shapes.getSize();
        for (int i = 0; i < numShapes; i ++)
        {
            builder.clear();

            const hkMeshShape* shape = shapes[i];
            const int numShapeSections = shape->getNumSections();

            for (int j = 0; j < numShapeSections; j++)
            {
                hkMeshSectionVertexRemap* remap = sectionRemaps[start + j];
                HK_ASSERT(0x2342354, remap->m_meshShape == shape);

                const hkMeshSection& meshSection = sectionSet.getSection(start + j);

                // Set up the section
                builder.startMeshSection(remap->m_vertexBuffer, meshSection.m_material);
				builder.concatPrimitives(hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST, remap->m_triangleIndices.begin(), remap->m_triangleIndices.getSize());
                builder.endMeshSection();

                // Destroy the remap
                remap->removeReference();
            }

            // Next ones
            start += numShapeSections;

            // Okay try creating it
            hkMeshShape* dstMeshShape = meshSystem->createShape(builder.getSections(), builder.getNumSections());
            shapesOut.pushBack(dstMeshShape);
        }
    }
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
