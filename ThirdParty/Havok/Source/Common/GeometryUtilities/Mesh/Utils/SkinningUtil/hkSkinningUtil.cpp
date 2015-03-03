/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>

// Needed for the class reflection
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindVertexWeightsUtil/hkFindVertexWeightsUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshShapeUtil/hkMeshShapeUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>

void hkSkinningUtil_findMatrixIndexRange(const hkMeshVertexBuffer::LockedVertices::Buffer& bufferIn, int numVertices, int& minIndexOut, int& maxIndexOut)
{
	hkMeshVertexBuffer::LockedVertices::Buffer buffer = bufferIn;
	const int numValues = buffer.m_element.m_numValues;

	int maxIndex = *(const hkUint8*)buffer.m_start;
	int minIndex = maxIndex;

	for (int i = 0; i < numVertices; i++)
	{
		const hkUint8* values = (const hkUint8*)buffer.m_start;

		for (int j = 0; j < numValues; j++)
		{
			int index = int(values[j]);

			if (index > maxIndex)
			{
				maxIndex = index;
			}
			if (index < minIndex)
			{
				minIndex = index;
			}
		}

		buffer.next();
	}
	maxIndexOut = maxIndex;
	minIndexOut = minIndex;
}

/* static */hkResult hkSkinningUtil::findMatrixIndexRange(hkMeshVertexBuffer* vertexBuffer, int& minIndexOut, int& maxIndexOut)
{
    hkVertexFormat vertexFormat;
    vertexBuffer->getVertexFormat(vertexFormat);

    int elementIndex = vertexFormat.findElementIndex(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0);
    if (elementIndex < 0)
    {
        return HK_FAILURE;
    }

    // Lets do a lock and maybe a conversion
    hkVertexFormat::Element& srcElement = vertexFormat.m_elements[elementIndex];

    // I need to lock this
    hkMeshVertexBuffer::LockInput lockInput;
    hkMeshVertexBuffer::PartialLockInput partialLockInput;
    partialLockInput.m_elementIndices[0] = elementIndex;
    partialLockInput.m_lockFlags[0] = hkMeshVertexBuffer::ACCESS_READ;
    partialLockInput.m_numLockFlags = 1;

    hkMeshVertexBuffer::LockedVertices lockedVertices;

    hkMeshVertexBuffer::LockResult result = vertexBuffer->partialLock(lockInput, partialLockInput, lockedVertices);
    if (result != hkMeshVertexBuffer::RESULT_SUCCESS || lockedVertices.m_numVertices <= 0)
    {
        return HK_FAILURE;
    }

	hkResult res = HK_SUCCESS;
    // Okay if the type is UINT8 we can just run tru them
    if (srcElement.m_dataType == hkVertexFormat::TYPE_UINT8)
    {
		hkSkinningUtil_findMatrixIndexRange(lockedVertices.m_buffers[0], lockedVertices.m_numVertices, minIndexOut, maxIndexOut);
    }
	else
	{
		hkLocalBuffer<hkUint8> indices(lockedVertices.m_numVertices * srcElement.m_numValues);

		hkMeshVertexBuffer::LockedVertices::Buffer dstBuffer;
		dstBuffer.m_start = indices.begin();
		dstBuffer.m_stride = srcElement.m_numValues;
		dstBuffer.m_element.set(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, hkVertexFormat::TYPE_UINT8, srcElement.m_numValues, 0, 0);

		// Do the conversion
		hkMeshVertexBufferUtil::convert(lockedVertices.m_buffers[0], dstBuffer, lockedVertices.m_numVertices);

		// Work out the range
		hkSkinningUtil_findMatrixIndexRange(dstBuffer, lockedVertices.m_numVertices, minIndexOut, maxIndexOut);
	}

	vertexBuffer->unlock(lockedVertices);

	return res;
}

/* static */ hkResult hkSkinningUtil::findMatrixIndexRange(hkMeshShape* meshShape, int& minIndexOut, int& maxIndexOut)
{
	hkMeshSectionLockSet sectionSet;
	sectionSet.addMeshSections(meshShape, hkMeshShape::ACCESS_VERTEX_BUFFER);

	hkArray<hkMeshVertexBuffer*> uniqueBuffers;
	sectionSet.findUniqueVertexBuffers(uniqueBuffers);

    int shapeMinIndex = 0x7fffffff;
    int shapeMaxIndex = -1;

    const int numBuffers = uniqueBuffers.getSize();
    for (int i = 0; i < numBuffers; i++)
    {
        hkMeshVertexBuffer* buffer = uniqueBuffers[i];

        int minIndex,maxIndex;
        hkResult res = findMatrixIndexRange(buffer, minIndex, maxIndex);
        if (res != HK_SUCCESS)
        {
            continue;
        }

        if (minIndex < shapeMinIndex)
        {
            shapeMinIndex = minIndex;
        }
        if (maxIndex > shapeMaxIndex)
        {
            shapeMaxIndex = maxIndex;
        }
    }

    minIndexOut = shapeMinIndex;
    maxIndexOut = shapeMaxIndex;

    return (shapeMinIndex <= shapeMaxIndex) ? HK_SUCCESS : HK_FAILURE;
}

/* static */void hkSkinningUtil::findDistances(const hkArray<hkVector4>& vertices, const hkArray<hkVector4>& boneCenters, int maxDistances, hkReal maxDistance, hkArray<Entry>& entriesOut)
{
    const int numVertices = vertices.getSize();
    const int numEntries = maxDistances * numVertices;
    entriesOut.setSize(numEntries);

    const int numBones = boneCenters.getSize();

    hkReal maxDistanceSq = maxDistance * maxDistance;

    Entry* entries = entriesOut.begin();
    for (int i = 0; i < numVertices; i++, entries += maxDistances)
    {
        const hkVector4& pos = vertices[i];

        for (int j = 0; j < maxDistances; j++)
        {
            entries[j].m_index = -1;
            entries[j].m_distanceSquared = maxDistanceSq;
        }

        hkReal maxDistSq = maxDistanceSq;
		int numValues = 0;

        for (int j = 0; j < numBones; j++)
        {
            // Work out the distance
            hkVector4 diff; diff.setSub(boneCenters[j], pos);

            hkReal distSq = diff.lengthSquared<3>().getReal();

            if (distSq < maxDistSq)
            {
				numValues = hkFindVertexWeightsUtil::insertEntry(entries, numValues, maxDistances, distSq, j);
				maxDistSq = hkFindVertexWeightsUtil::getMaxValue(entries, maxDistances);
			}
        }
    }
}

void hkSkinningUtil::extractBoneCenters(const hkArray<hkMatrix4>& bones, hkArray<hkVector4>& boneCenters)
{
	const int numBones = bones.getSize();
	boneCenters.setSize(numBones);

	for (int i = 0; i < numBones; i++)
	{
		boneCenters[i] = bones[i].getColumn<3>();
	}
}

int hkSkinningUtil::findNumBoneIndices(const hkVertexFormat& vertexFormat)
{
	int matrixIndexElementIndex = vertexFormat.findElementIndex( hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0);

	if (matrixIndexElementIndex < 0 )
	{
		return 0;
	}

	const hkVertexFormat::Element& element = vertexFormat.m_elements[matrixIndexElementIndex];
	return element.m_numValues;
}

hkResult hkSkinningUtil::calculateBoneIndicesAndWeights(const hkArray<hkVector4>& boneCenters, hkMeshVertexBuffer* vertexBuffer, hkReal maxDistance)
{
	const int numVertices = vertexBuffer->getNumVertices();

    // We need the vertex positions
	hkLocalArray<hkVector4> positions(numVertices);
    hkMeshVertexBufferUtil::getElementVectorArray(vertexBuffer, hkVertexFormat::USAGE_POSITION, 0, positions);

	hkVertexFormat vertexFormat;
	vertexBuffer->getVertexFormat(vertexFormat);

	const int numValues = findNumBoneIndices(vertexFormat);

	if (numVertices <= 0)
	{
		HK_ASSERT(0x32432, !"Bone weights/transform indices not present");
		return HK_FAILURE;
	}

    hkArray<Entry> entries;
	findDistances(positions, boneCenters, numValues, maxDistance, entries);

	return setSkinningValues(entries, vertexBuffer, maxDistance);
}

hkResult hkSkinningUtil::setSkinningValues(const hkArray<Entry>& entries, hkMeshVertexBuffer* vertexBuffer, hkReal maxDistance)
{
	const int numVertices = vertexBuffer->getNumVertices();

	hkVertexFormat vertexFormat;
	vertexBuffer->getVertexFormat(vertexFormat);

    int matrixIndexElementIndex = vertexFormat.findElementIndex( hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0);
    int blendWeightsIndex = vertexFormat.findElementIndex( hkVertexFormat::USAGE_BLEND_WEIGHTS, 0);
    if ( blendWeightsIndex < 0)
    {
        blendWeightsIndex = vertexFormat.findElementIndex( hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED, 0);
    }

    if (matrixIndexElementIndex < 0 || blendWeightsIndex < 0)
    {
        HK_ASSERT(0x32432, !"Bone weights/transform indices not present");
        return HK_FAILURE;
    }

    const hkVertexFormat::Element& element = vertexFormat.m_elements[matrixIndexElementIndex];
    int numValues = element.m_numValues;


	hkLocalBuffer<hkFloat32> weights(numVertices * numValues);
	hkLocalBuffer<hkUint8> boneIndices(numVertices * numValues);

	HK_ASSERT(0x32432, numVertices * numValues == entries.getSize());
	computeBoneIndicesAndWeights(entries, maxDistance, numValues, weights.begin(), boneIndices.begin());

	{
		hkMeshVertexBuffer::LockInput lockInput;
		hkMeshVertexBuffer::PartialLockInput partialLock;

		partialLock.m_numLockFlags = 2;
		partialLock.m_elementIndices[0] = matrixIndexElementIndex;
		partialLock.m_lockFlags[0] = hkMeshVertexBuffer::ACCESS_WRITE | hkMeshVertexBuffer::ACCESS_WRITE_DISCARD;
		partialLock.m_elementIndices[1] = blendWeightsIndex;
		partialLock.m_lockFlags[1] = hkMeshVertexBuffer::ACCESS_WRITE | hkMeshVertexBuffer::ACCESS_WRITE_DISCARD;

		hkMeshVertexBuffer::LockedVertices::Buffer srcWeights;
		srcWeights.m_start = weights.begin();
		srcWeights.m_stride = sizeof(hkFloat32) * numValues;
		srcWeights.m_element.set(hkVertexFormat::USAGE_BLEND_WEIGHTS, hkVertexFormat::TYPE_FLOAT32, numValues);

		hkMeshVertexBuffer::LockedVertices::Buffer srcBoneIndices;
		srcBoneIndices.m_start = boneIndices.begin();
		srcBoneIndices.m_stride = sizeof(hkUint8) * numValues;
		srcBoneIndices.m_element.set(hkVertexFormat::USAGE_BLEND_WEIGHTS, hkVertexFormat::TYPE_UINT8, numValues);

		hkMeshVertexBuffer::LockedVertices lockedVertices;
		hkMeshVertexBuffer::LockResult lockRes = vertexBuffer->partialLock(lockInput, partialLock, lockedVertices);
		if (lockRes != hkMeshVertexBuffer::RESULT_SUCCESS)
		{
			return HK_FAILURE;
		}

		hkMeshVertexBufferUtil::convert(srcBoneIndices, lockedVertices.m_buffers[0], numVertices);
		hkMeshVertexBufferUtil::convert(srcWeights, lockedVertices.m_buffers[1], numVertices);

		vertexBuffer->unlock(lockedVertices);
	}

	return HK_SUCCESS;
}


/* static */hkBool hkSkinningUtil::isSkinnedVertexFormat(const hkVertexFormat& vertexFormat)
{
	hkBool hasBoneIndices = false;
	hkBool hasWeights = false;
	for (int i = 0; i < vertexFormat.m_numElements; i++)
	{
		const hkVertexFormat::Element& element = vertexFormat.m_elements[i];
		// We are only interested in sub usage 0
		if (element.m_subUsage != 0)
		{
			continue;
		}
		if (element.m_usage == hkVertexFormat::USAGE_BLEND_MATRIX_INDEX)
		{
			hasBoneIndices = true;
		}
		else if (element.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS ||
			element.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED)
		{
			hasWeights = true;
		}
	}

	return hasBoneIndices && hasWeights;
}

/* static */hkMeshVertexBuffer* hkSkinningUtil::createSkinnedVertexBuffer(hkMeshSystem* meshSystem, hkMeshVertexBuffer* srcVertexBuffer, int numWeights)
{
    hkVertexFormat srcFormat;
    srcVertexBuffer->getVertexFormat(srcFormat);

    hkVertexFormat dstFormat;

    for (int i = 0; i < srcFormat.m_numElements; i++)
    {
        const hkVertexFormat::Element& srcElement = srcFormat.m_elements[i];
        if (srcElement.m_usage == hkVertexFormat::USAGE_BLEND_MATRIX_INDEX ||
            srcElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS ||
            srcElement.m_usage == hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED)
        {
            // Ignore
        }
        else
        {
            dstFormat.addElement(srcElement);
        }
    }

    dstFormat.addElement(hkVertexFormat::USAGE_BLEND_WEIGHTS, hkVertexFormat::TYPE_FLOAT32, numWeights);
    dstFormat.addElement(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, hkVertexFormat::TYPE_UINT8, numWeights);

    dstFormat.makeCanonicalOrder();

    hkVertexFormat finalFormat;
    meshSystem->findSuitableVertexFormat(dstFormat, finalFormat);

    hkMeshVertexBuffer* dstVertexBuffer = meshSystem->createVertexBuffer(finalFormat, srcVertexBuffer->getNumVertices());

    // Convert all the src stuff over
    hkMeshVertexBufferUtil::convert(srcVertexBuffer, dstVertexBuffer);

    return dstVertexBuffer;
}

/* static */hkMeshVertexBuffer* hkSkinningUtil::ensureSkinnedVertexBuffer(hkMeshSystem* meshSystem, hkMeshVertexBuffer* srcVertexBuffer, int numWeights)
                {
	hkVertexFormat vertexFormat;
	srcVertexBuffer->getVertexFormat(vertexFormat);
	if (isSkinnedVertexFormat(vertexFormat))
	{
		srcVertexBuffer->addReference();
		return srcVertexBuffer;
	}
	else
	{
		return createSkinnedVertexBuffer(meshSystem, srcVertexBuffer, numWeights);
	}
}

/* static */hkBool hkSkinningUtil::isSkinnedShape(hkMeshShape* meshShape)
{
	hkMeshSectionLockSet sectionSet;
	sectionSet.addMeshSections(meshShape, hkMeshShape::ACCESS_VERTEX_BUFFER);

	hkArray<hkMeshVertexBuffer*> uniqueBuffers;
	sectionSet.findUniqueVertexBuffers(uniqueBuffers);

	const int numVertexBuffers = uniqueBuffers.getSize();
	for (int i = 0; i < numVertexBuffers; i++)
	{

		hkMeshVertexBuffer* buffer = uniqueBuffers[i];

		hkVertexFormat vertexFormat;
		buffer->getVertexFormat(vertexFormat);

		if (!isSkinnedVertexFormat(vertexFormat))
		{
			return false;
		}
	}
	return true;
}

/* static */hkResult hkSkinningUtil::setSkinningValues(hkMeshShape* meshShape, const hkArray<hkVector4>& boneCenters, hkReal maxDistance)
{
    hkMeshSectionLockSet sectionSet;
    sectionSet.addMeshSections(meshShape, hkMeshShape::ACCESS_VERTEX_BUFFER | hkMeshShape::ACCESS_INDICES);

    hkArray<hkMeshVertexBuffer*> uniqueBuffers;
    sectionSet.findUniqueVertexBuffers(uniqueBuffers);

	const int numUniqueBuffers = uniqueBuffers.getSize();
    for (int i = 0; i< numUniqueBuffers; i++)
    {
        hkMeshVertexBuffer* buffer = uniqueBuffers[i];
        hkVertexFormat vertexFormat;
        buffer->getVertexFormat(vertexFormat);

        int numBones = findNumBoneIndices(vertexFormat);

        if (numBones > 0)
        {
            hkResult res = calculateBoneIndicesAndWeights(boneCenters, buffer, maxDistance);
            if (res != HK_SUCCESS)
            {
                return res;
            }
        }
    }

    return HK_SUCCESS;
}

/* static */hkMeshShape* hkSkinningUtil::createEmptySkinnedShape(hkMeshSystem* meshSystem, const hkMeshShape* meshShape, int numWeights)
                {
    hkMeshSectionLockSet sectionSet;
    sectionSet.addMeshSections(meshShape, hkMeshShape::ACCESS_VERTEX_BUFFER);

    hkArray<hkMeshVertexBuffer*> uniqueVertexBuffers;
    sectionSet.findUniqueVertexBuffers(uniqueVertexBuffers);
    hkArray<hkMeshVertexBuffer*> uniqueSkinnedVertexBuffers;
    hkPointerMap<hkMeshVertexBuffer*, hkMeshVertexBuffer*> bufferMap;

    hkBool bufferReplaced = false;
    const int numBuffers = uniqueVertexBuffers.getSize();
    for (int i = 0; i < numBuffers; i++)
    {
        hkMeshVertexBuffer* buffer = uniqueVertexBuffers[i];

        hkVertexFormat vertexFormat;
        buffer->getVertexFormat(vertexFormat);

        hkMeshVertexBuffer* skinnedBuffer;
        if (hkSkinningUtil::isSkinnedVertexFormat(vertexFormat))
        {
            buffer->addReference();
            skinnedBuffer = buffer;
        }
        else
        {
            skinnedBuffer = hkSkinningUtil::createSkinnedVertexBuffer(meshSystem, buffer, numWeights);
            bufferReplaced = true;
        }

        uniqueSkinnedVertexBuffers.pushBack(skinnedBuffer);
        bufferMap.insert(buffer, skinnedBuffer);
    }

    hkMeshShape* skinnedMeshShape;
    if (bufferReplaced)
    {
		skinnedMeshShape = hkMeshShapeUtil::replaceShapeVertexBuffers(meshSystem, meshShape, bufferMap);
    }
    else
    {
        meshShape->addReference();
        skinnedMeshShape = const_cast<hkMeshShape*>(meshShape);
    }

    hkReferencedObject::removeReferences(uniqueSkinnedVertexBuffers.begin(), uniqueSkinnedVertexBuffers.getSize());
    return skinnedMeshShape;
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
