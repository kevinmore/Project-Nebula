/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionMergeUtil/hkMeshSectionMergeUtil.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionBuilder/hkMeshSectionBuilder.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>

static bool hkMeshSectionMergeUtil_hasVertexBufferInSet(const hkMeshSection* sections, int numSections, const hkArray<hkMeshVertexBuffer*>& buffers)
{
    for (int i = 0; i < numSections; i++)
	{
        if (buffers.indexOf(sections[i].m_vertexBuffer) >= 0)
		{
            return true;
		}
	}

    return false;
}


static hkMeshShape* hkMeshSectionMergeUtil_merge(hkMeshSystem* system, const hkMeshSection* sections, const int* numMergeSections, int numSections)
{
    // Find the total number of sections
    int totalNumSections = 0;
    for (int i = 0; i < numSections; i++)
    {
        totalNumSections += numMergeSections[i];
    }

    hkInplaceArray<hkMeshVertexBuffer*,16> mergedBuffers;
    hkInplaceArray<int,16> mergedStartIndex;

    // Maps an original vertex buffer to an index in the buffers array
    hkPointerMap<hkMeshVertexBuffer*, int> map;

    {
        hkInplaceArray<hkMeshVertexBuffer*,16> uniqueVertexBuffers;
        for (int i = 0; i < totalNumSections; i++)
        {
            if (uniqueVertexBuffers.indexOf(sections[i].m_vertexBuffer) < 0)
            {
                uniqueVertexBuffers.pushBack(sections[i].m_vertexBuffer);
            }
        }

        hkInplaceArray<hkMeshVertexBuffer*,16> srcVertexBuffers;

        while (! uniqueVertexBuffers.isEmpty())
        {
			// Get the starting vertex buffer, and work out the ones it must be combined with
			hkMeshVertexBuffer* startVertexBuffer = uniqueVertexBuffers.back();

			hkVertexFormat vertexFormat;
			startVertexBuffer->getVertexFormat(vertexFormat);

			// Clear the merge set
			srcVertexBuffers.clear();
			srcVertexBuffers.pushBack(startVertexBuffer);

            const hkMeshSection* curSections = sections;
            for (int i = 0; i < numSections; i++)
            {
                const int numMerge = numMergeSections[i];

                // If this section contains the vertex buffer, we add it to the merge set
                if (hkMeshSectionMergeUtil_hasVertexBufferInSet(curSections, numMerge, srcVertexBuffers))
                {
                    for (int j = 0; j < numMerge; j++)
                    {
                        hkMeshVertexBuffer* vertexBuffer = curSections[j].m_vertexBuffer;
                        // Add to the merge set
                        if (srcVertexBuffers.indexOf(vertexBuffer) < 0)
                        {
                            hkVertexFormat srcVertexFormat;
                            vertexBuffer->getVertexFormat(srcVertexFormat);

                            // We can only merge if they are the same format
                            if (vertexFormat == srcVertexFormat)
                            {
                                srcVertexBuffers.pushBack(vertexBuffer);
                            }
                        }
                    }
                }

                // Next merge section
                curSections += numMerge;
            }

			// Remove the buffers that will be merged from the original set
            for (int i = 0; i < srcVertexBuffers.getSize(); i++)
            {
                hkMeshVertexBuffer* vertexBuffer = srcVertexBuffers[i];

                int index = uniqueVertexBuffers.indexOf(vertexBuffer);
                HK_ASSERT2(0x234f9324, index >= 0, "Buffer not found in the original set?\n");

                // Remove it
                uniqueVertexBuffers.removeAt(index);
            }

            // Merge all of the vertex buffers (just ref counts if just a single one)
            hkMeshVertexBuffer* mergedVertexBuffer = hkMeshVertexBufferUtil::concatVertexBuffers(system, srcVertexBuffers.begin(), srcVertexBuffers.getSize());

            int base = 0;
            for (int i = 0; i < srcVertexBuffers.getSize(); i++)
            {
                hkMeshVertexBuffer* vertexBuffer = srcVertexBuffers[i];

                const int index = mergedBuffers.getSize();
                map.insert(vertexBuffer, index);

				mergedVertexBuffer->addReference();
                mergedBuffers.pushBack(mergedVertexBuffer);
                mergedStartIndex.pushBack(base);

                // Next set of indices
                base += vertexBuffer->getNumVertices();
            }

			mergedVertexBuffer->removeReference();
        }
    }

    hkMeshSectionBuilder builder;
    for (int i = 0; i < numSections; i++)
    {
        const int numMerge = numMergeSections[i];
        // Merge these to create a new section

        HK_ASSERT2(0x234f9325, numMerge > 0, "Nothing to merge\n");

        const hkMeshSection& startSection = sections[0];
        hkMeshVertexBuffer* mergedBuffer;
        {
            const int bufferIndex = map.getWithDefault(startSection.m_vertexBuffer, -1);
            HK_ASSERT2(0x234f9326, bufferIndex >= 0, "Buffer not found?\n");
            mergedBuffer = mergedBuffers[bufferIndex];
        }

        builder.startMeshSection(mergedBuffer, startSection.m_material);

        for (int j = 0; j < numMerge; j++)
        {
            const hkMeshSection& section = sections[j];

            const int bufferIndex = map.getWithDefault(section.m_vertexBuffer, -1);

            HK_ASSERT2(0x234f9326, bufferIndex >= 0, "Buffer not found?\n");
            HK_ASSERT2(0x234f9327, mergedBuffers[bufferIndex] == mergedBuffer, "Shared buffer not the same?");

            const int startIndex = mergedStartIndex[bufferIndex];

            // Add it
            builder.concatPrimitives(section.m_primitiveType, (const hkUint16*)section.m_indices, section.m_numIndices, startIndex);
        }

        builder.endMeshSection();

        // Next section
        sections += numMerge;
    }

    hkMeshShape* meshShape = system->createShape(builder.getSections(), builder.getNumSections());

    // Remove references
    hkReferencedObject::removeReferences(mergedBuffers.begin(), mergedBuffers.getSize());

    // Done
	return meshShape;
}


HK_FORCE_INLINE static hkBool hkMeshSectionMergeUtil_orderSections(const hkMeshSection& a, const hkMeshSection& b)
{
    return ( a.m_material < b.m_material );
}

static void hkMeshSectionMergeUtil_calcMergeSectionsByMaterial(const hkMeshSection* sections, int numSections, hkArray<hkMeshSection>& sectionsOut, hkArray<int>& numMergeSectionsOut)
{
    sectionsOut.setSize(numSections);
    hkString::memCpy(sectionsOut.begin(), sections, numSections * sizeof(hkMeshSection));

    // Sort them by material
    hkSort(sectionsOut.begin(), sectionsOut.getSize(), hkMeshSectionMergeUtil_orderSections);

    // Count the amount
    {
        numMergeSectionsOut.clear();

        const hkMeshSection* cur = sectionsOut.begin();
        const hkMeshSection* end = sectionsOut.end();

        while (cur < end)
        {
            const hkMeshSection* start = cur;
            hkMeshMaterial* material = start->m_material;
            cur++;

            // Look for the transition
            for (; cur < end && cur->m_material == material; cur++) ;

            // Save off the amount found
			const int numSectionsWithMaterial = int(cur - start);

            numMergeSectionsOut.pushBack(numSectionsWithMaterial);
        }
    }
}

/* static */hkMeshShape* HK_CALL hkMeshSectionMergeUtil::mergeShapeSectionsByMaterial(hkMeshSystem* system, hkMeshShape* meshShape)
{
	// Try doing merge on it
	if (meshShape->getNumSections() <= 1)
	{
		meshShape->addReference();
		return meshShape;
	}

	hkMeshSectionLockSet sectionSet;
	sectionSet.addMeshSections(meshShape, hkMeshShape::ACCESS_INDICES | hkMeshShape::ACCESS_VERTEX_BUFFER);

	hkArray<int> numMerge;
	hkArray<hkMeshSection> sections;

	hkMeshSectionMergeUtil_calcMergeSectionsByMaterial(sectionSet.getSections(), sectionSet.getNumSections(), sections, numMerge);

	return hkMeshSectionMergeUtil_merge(system, sections.begin(), numMerge.begin(), numMerge.getSize());
}

/* static */bool HK_CALL hkMeshSectionMergeUtil::hasUniqueMaterials( hkMeshShape* meshShape )
{
	hkMeshSectionLockSet sectionSet;
	sectionSet.addMeshSections(meshShape, 0);

	hkPointerMap<hkMeshMaterial*, int> foundMaterials;

	const int numSections = sectionSet.getNumSections();
	for (int i = 0; i < numSections; i++)
	{
		const hkMeshSection& section = sectionSet.getSection(i);
		if ( section.m_material == HK_NULL )
		{
			continue;
		}

		if (foundMaterials.hasKey(section.m_material))
		{
			return false;
		}

		foundMaterials.insert(section.m_material, 1);
	}

	// The materials must all be unique
	return true;
}

/*static*/ hkMeshShape* HK_CALL hkMeshSectionMergeUtil::mergeShapes(hkMeshSystem* system, const hkMeshShape* one, const hkMeshShape* two)
{
	hkMeshSectionLockSet sectionSet;
	sectionSet.addMeshSections(one, hkMeshShape::ACCESS_INDICES | hkMeshShape::ACCESS_VERTEX_BUFFER);
	sectionSet.addMeshSections(two, hkMeshShape::ACCESS_INDICES | hkMeshShape::ACCESS_VERTEX_BUFFER);

	hkArray<int> numMerge;
	hkArray<hkMeshSection> sections;

	hkMeshSectionMergeUtil_calcMergeSectionsByMaterial(sectionSet.getSections(), sectionSet.getNumSections(), sections, numMerge);

	return hkMeshSectionMergeUtil_merge(system, sections.begin(), numMerge.begin(), numMerge.getSize());
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
