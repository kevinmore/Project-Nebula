/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkMeshSimplifierConverter.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindUniquePositionsUtil/hkFindUniquePositionsUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionBuilder/hkMeshSectionBuilder.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>
#include <Common/GeometryUtilities/Mesh/Utils/PrimitiveUtil/hkMeshPrimitiveUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexFloat32Converter/hkVertexFloat32Converter.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexSharingUtil/hkVertexSharingUtil.h>

#include <Common/Base/Container/String/hkStringBuf.h>

hkMeshSimplifierConverter::hkMeshSimplifierConverter()
{
	clearWeights();
	m_positionOnly = false;
	m_defaultMaterialWeight = hkSimdReal_1;
}

int hkMeshSimplifierConverter::_findGroup(int modelIndex, hkMeshMaterial* material, const hkVertexFormat& vertexFormat) const
{
	for (int i = 0; i < m_groups.getSize(); i++)
	{
		const Group& group = m_groups[i];

		if (group.m_material == material &&
			group.m_vertexFormat == vertexFormat &&
			group.m_modelIndex == modelIndex)
		{
			return i;
		}
	}
	return -1;
}

int hkMeshSimplifierConverter::_addGroup(int modelIndex, hkMeshMaterial* material, const hkVertexFormat& vertexFormat)
{
	Group& group = m_groups.expandOne();

	// Store the format
	group.m_vertexFormat = vertexFormat;
	group.m_material = material;
	group.m_modelIndex = modelIndex;
	group.m_qemGroupIndex = -1;

	// Return the index
	return m_groups.getSize() - 1;
}

void hkMeshSimplifierConverter::_getVertexFormat(hkMeshVertexBuffer* buffer, hkVertexFormat& vertexFormat)
{
	if (m_positionOnly)
	{
		vertexFormat.clear();
		vertexFormat.addElement(hkVertexFormat::USAGE_POSITION, hkVertexFormat::TYPE_FLOAT32, 3);
	}
	else
	{
		buffer->getVertexFormat(vertexFormat);
	}
}


void hkMeshSimplifierConverter::_addMeshSection(hkFindUniquePositionsUtil& positionUtil, hkVertexSharingUtil& sharingUtil, const hkMeshSection& section, int groupIndex, hkQemSimplifier& simplifier)
{
	//
	hkVertexFormat vertexFormat;
	_getVertexFormat(section.m_vertexBuffer, vertexFormat);
	
	// See if we already have this vertex format
	hkRefPtr<hkMemoryMeshVertexBuffer> vertexBuffer = new hkMemoryMeshVertexBuffer(vertexFormat, section.m_vertexBuffer->getNumVertices());
	vertexBuffer->removeReference();

	// We now have vertex buffer in memory format
	hkMeshVertexBufferUtil::convert(section.m_vertexBuffer, vertexBuffer);

	// Work out the position elements offset in the vertex
	const int positionOffset = hkMemoryMeshVertexBuffer::calculateElementOffset(vertexFormat, hkVertexFormat::USAGE_POSITION, 0);

	const int numSrcVertices = vertexBuffer->getNumVertices();

	// Used to mark if its already been added
	hkArray<int> attributeMap;
	attributeMap.setSize(numSrcVertices, -1);
	hkArray<int> vertexPositionMap;
	vertexPositionMap.setSize(numSrcVertices, -1);

	// We need the triangle indices
	hkArray<hkUint32> srcTriIndices;
	hkMeshPrimitiveUtil::appendTriangleIndices(section, srcTriIndices);

	const int numTris = srcTriIndices.getSize() / 3;

	{
		const hkUint32* srcIndices = srcTriIndices.begin();

		for (int i = 0; i < numTris; i++, srcIndices += 3)
		{
			int positionIndices[3];
			int attributeIndices[3];

			for (int j = 0; j < 3; j++)
			{
				const int srcVertexIndex = srcIndices[j];
				int dstAttributeIndex = attributeMap[srcVertexIndex];
				int dstPositionIndex = vertexPositionMap[srcVertexIndex];

				if (dstAttributeIndex < 0)
				{
					hkVector4 pos;

					const hkUint8* vertex = vertexBuffer->getVertexData() + (vertexBuffer->getVertexStride() * srcVertexIndex);

					const hkFloat32* vv = (const hkFloat32*)(vertex + positionOffset);
					pos.load<3,HK_IO_NATIVE_ALIGNED>(vv);
					dstPositionIndex = positionUtil.addPosition(pos);

					// Add the vertex
					dstAttributeIndex = sharingUtil.addVertex(hkUint32(dstPositionIndex), vertex);

					attributeMap[srcVertexIndex] = dstAttributeIndex;
					vertexPositionMap[srcVertexIndex] = dstPositionIndex;
				}

				HK_ASSERT(0x4534543, dstPositionIndex >= 0 && dstAttributeIndex >= 0);

				positionIndices[j] = dstPositionIndex;
				attributeIndices[j] = dstAttributeIndex;
			}

			if (positionIndices[0] == positionIndices[1] || positionIndices[1] == positionIndices[2] || positionIndices[0] == positionIndices[2])
			{
				// It has a null edge (same start and end vertex index - ignore it)
			}
			else
			{
				// Add the triangle
				simplifier.addTriangle(positionIndices, attributeIndices);
			}
		}
	}
}

/* static */void hkMeshSimplifierConverter::_calcAabb(const hkVector4* pos, int numPos, hkAabb& aabb)
{
	aabb.setEmpty();

	const hkVector4* cur = pos; 
	const hkVector4* end = pos + numPos;
	for (; cur < end; cur++)
	{
		aabb.includePoint(*cur);
	}

	// In the space I want areas, and lengths to keep the same proportions to each other.
	// So make the AABB a cube, with the longest extent.
	hkVector4 size; size.setSub(aabb.m_max, aabb.m_min);

	const hkSimdReal maxSize = size.horizontalMax<3>();

	aabb.m_max.setAdd(aabb.m_min, maxSize);
}

/* static */hkResult hkMeshSimplifierConverter::getPositions(const hkMeshSection& section, hkArray<hkVector4>& positions)
{
	hkMeshVertexBuffer* vertexBuffer = section.m_vertexBuffer;

	hkResult res = hkMeshVertexBufferUtil::getElementVectorArray(vertexBuffer, hkVertexFormat::USAGE_POSITION, 0, positions);

	if (res != HK_SUCCESS)
	{
		HK_WARN(0x32432432, "Unable to access position info. Cannot continue.");
	}

	return res;
}

/* static */hkResult hkMeshSimplifierConverter::_addPositions(const hkMeshSectionLockSet& sectionLockSet, hkFindUniquePositionsUtil& positionUtil )
{
	hkArray<hkVector4> positions;
	// Go through all, adding vertices
	for (int i = 0; i < sectionLockSet.getNumSections(); i++)
	{
		const hkMeshSection& section = sectionLockSet.getSection(i);

		hkResult res = getPositions(section, positions);
		if (res != HK_SUCCESS)
		{
			return res;
		}

		// Add to the position information
		positionUtil.addPositions(positions.begin(), positions.getSize());
	}

	return HK_SUCCESS;
}

void hkMeshSimplifierConverter::_addModelGroups(hkMeshSectionLockSet& sectionLockSet, int modelIndex, hkArray<int>& sectionMapOut)
{
	int* sectionMap = sectionMapOut.expandBy(sectionLockSet.getNumSections());

	for (int i = 0; i < sectionLockSet.getNumSections(); i++)
	{
		const hkMeshSection& section = sectionLockSet.getSection(i);

		hkMeshVertexBuffer* vertexBuffer = section.m_vertexBuffer;

		hkVertexFormat vertexFormat;
		vertexBuffer->getVertexFormat(vertexFormat);

		int groupIndex = _findGroup(modelIndex, section.m_material, vertexFormat);
		if (groupIndex < 0)
		{
			groupIndex = _addGroup(modelIndex, section.m_material, vertexFormat);
		}

		// Save off what it maps to
		sectionMap[i] = groupIndex;
	}
}

int hkMeshSimplifierConverter::calcNumModelTriangles(int modelIndex, const hkQemSimplifier& simplifier) const
{
	hkInplaceArray<int, 16> trisPerGroup;
	simplifier.calcNumTrianglesPerGroup(trisPerGroup);

	int total = 0;
	for (int i = 0; i < m_groups.getSize(); i++)
	{
		const Group& group = m_groups[i];

		if (group.m_modelIndex == modelIndex)
		{
			total += trisPerGroup[group.m_qemGroupIndex];
		}
	}

	return total;
}

hkResult hkMeshSimplifierConverter::initMesh(	const hkMeshShape* srcMeshShape,
												hkQemSimplifier& simplifier,
												hkFindUniquePositionsUtil& positionUtil,
												const Threshold* thresholds,
												bool unitScalePosition)
{
	return initMeshes(&srcMeshShape, 1, simplifier, positionUtil, thresholds, unitScalePosition);
}

hkResult hkMeshSimplifierConverter::initMeshes(	const hkMeshShape*const* srcMeshes, int numMeshes,
												hkQemSimplifier& simplifier,
												hkFindUniquePositionsUtil& positionUtil,
												const Threshold* thresholds,
												bool unitScalePosition)
{
	// Map goes in the order of the shapes
	
	for (int i = 0; i < numMeshes; i++)
	{
		hkMeshSectionLockSet sectionLockSet;

		const hkMeshShape* srcMeshShape = srcMeshes[i];

		sectionLockSet.addMeshSections(srcMeshShape, hkMeshShape::ACCESS_INDICES | hkMeshShape::ACCESS_VERTEX_BUFFER);

		hkArray<int> sectionMap;
		_addModelGroups(sectionLockSet, i, sectionMap);
		hkResult res = _addPositions(sectionLockSet, positionUtil);
		if (res != HK_SUCCESS) return res;
	}

	// Work out the AABB
	_calcAabb(positionUtil.m_positions.begin(), positionUtil.m_positions.getSize(), m_srcAabb);

	// Perhaps I can start now
	simplifier.startMesh(m_srcAabb, positionUtil.m_positions.begin(), positionUtil.m_positions.getSize(), unitScalePosition);

	{
		// Configure the vertex sharing utility
		hkVertexSharingUtil sharingUtil;
		if ( thresholds )	{		sharingUtil.setThresholds(*thresholds);		}
		else				{		sharingUtil.setAllThresholds(hkReal(1.0e-6f));		}

		const int numGroups = m_groups.getSize();
		for (int i = 0; i < numGroups; i++)
		{
			Group& group = m_groups[i];

			hkMeshSectionLockSet sectionLockSet;

			// Get the model
			const hkMeshShape* srcMeshShape = srcMeshes[group.m_modelIndex];
			sectionLockSet.addMeshSections(srcMeshShape, hkMeshShape::ACCESS_INDICES | hkMeshShape::ACCESS_VERTEX_BUFFER);

			// Get the map from sections to the group its in
			hkArray<int> sectionMap;
			_addModelGroups(sectionLockSet, group.m_modelIndex, sectionMap);

			// No new groups should have been added
			HK_ASSERT(0x4323432, m_groups.getSize() == numGroups);

			// Go through the groups...
			hkVertexFloat32Converter converter;
			converter.init(group.m_vertexFormat, m_srcAabb, unitScalePosition);

			_setWeights(group.m_material, group.m_vertexFormat, converter);

			hkSimdReal positionWeight; converter.getWeight(hkVertexFormat::USAGE_POSITION, 0, positionWeight);

			// Set the qem group index
			group.m_qemGroupIndex = simplifier.getGroups().getSize();

			// Set up the group
			hkQemSimplifier::AttributeFormat attribFmt;
			attribFmt.set(group.m_vertexFormat);
			simplifier.startGroup(converter.getNumReals(), attribFmt, positionWeight);

			// 
			sharingUtil.begin(group.m_vertexFormat);
			for (int j = 0; j < sectionMap.getSize(); j++)
			{
				if (sectionMap[j] == i)
				{
					const hkMeshSection& section = sectionLockSet.getSection(j);

					// Okay we can try and sort this vertex format out
					_addMeshSection(positionUtil, sharingUtil, section, i, simplifier);
				}
			}

			// Set up the attributes
			{
				hkMeshVertexBuffer::LockedVertices srcLockedVertices;
				sharingUtil.end(srcLockedVertices);	

				const hkUint8* srcVertex = sharingUtil.getVertexData();
				const int numVertices = sharingUtil.getNumVertices();
				const int vertexStride = sharingUtil.getVertexStride();

				// Okay - I could now convert this into attributes
				hkArray<hkFloat32> fattrib;
				fattrib.setSize(converter.countVertexToFloat32());
				for (int j = 0; j < numVertices; j++)
				{
					int attribIndex;
					hkReal* attrib = (hkReal*)simplifier.addAttribute(attribIndex); // force unaligned

					HK_ASSERT(0x42343242, attribIndex == j);

					// Convert over
					converter.convertVertexToFloat32(srcVertex, fattrib.begin());
					for (int a=0; a<fattrib.getSize(); ++a)
					{
						attrib[a] = hkReal(fattrib[a]);
					}
					// Next
					srcVertex += vertexStride;
				}
			}

			// End the group
			simplifier.endGroup();

			// Verify the groups content
			_verifyLastGroupAttributes(simplifier, converter, group.m_material);
		}
	}

	// End the mesh on the simplifier
	return simplifier.endMesh();
}

void hkMeshSimplifierConverter::_verifyLastGroupAttributes(hkQemSimplifier& simplifier, const hkVertexFloat32Converter& converter, hkMeshMaterial* material)
{
	// Check to see if the values are valid
	const hkQemSimplifier::Group& group = simplifier.getGroups().back();
	const int numAttribs = group.getNumAttributes();

	hkInplaceArray<char, 32> hasProblem;

	const int numVals = group.m_attributeSize;
	hasProblem.setSize(numVals, 0);

	for (int i = 0; i < numAttribs; i++)
	{
		hkReal* v = (hkReal*)group.getAttribute(i); // force unaligned

		for (int j = 0; j < numVals; j++)
		{
			if ( !hkMath::isFinite(v[j]) )
			{
				v[j] = 0;
				hasProblem[j] = 1;
			}
		}
	}

	// Need to work out what components that have a problem
	const hkVertexFormat& format = converter.getVertexFormat();
	const int numEle = format.m_numElements;

	hkInplaceArray<int, 16> problemElements;
	for (int i = 0; i < hasProblem.getSize(); i++)
	{
		if (!hasProblem[i])
		{
			continue;
		}

		// Find what it belongs to
		for (int j = 0; j < numEle; j++)
		{
			int start = converter.getOffsetByIndex(j);
			int size = converter.getSizeByIndex(j);

			if ( (i >= start) && (i < start + size) )
			{
				// Found it, add it if its not added
				if (problemElements.indexOf(j) < 0)
				{
					problemElements.pushBack(j);
				}
			}
		}
	}

	if ( problemElements.getSize() <= 0 )
	{
		return;
	}

	hkStringBuf buf;
	for (int i = 0; i < problemElements.getSize(); i++)
	{
		const hkVertexFormat::Element& ele = format.m_elements[problemElements[i]];
		ele.getText(buf);

		const char* materialName = material->getName();
		if (materialName == HK_NULL)
		{
			materialName = "(null)";
		}
			
		HK_WARN(0x243423aa, "Element " << buf.cString() << " on material '" << materialName << "' has non finite values. Non finite values have been set to zero.");
	}
}

void hkMeshSimplifierConverter::_setWeights(hkMeshMaterial* material, const hkVertexFormat& format, hkVertexFloat32Converter& converter)
{
	for (int i = 0; i < format.m_numElements; i++)
	{
		const hkVertexFormat::Element& ele = format.m_elements[i];

		hkSimdReal weight; getWeight(ele.m_usage, ele.m_subUsage, weight);
		if (ele.m_usage == hkVertexFormat::USAGE_POSITION && ele.m_subUsage == 0)
		{
			hkSimdReal matWeight; getMaterialWeight(material, matWeight);
			weight.mul(matWeight);
		}
		converter.setWeight(ele.m_usage, ele.m_subUsage, weight);
	}
}

hkMeshShape* hkMeshSimplifierConverter::createMesh(hkMeshSystem* system, int modelIndex, hkQemSimplifier& simplifier, bool unitScalePosition)
{
	const int numTris = simplifier.getNumTriangles();

	hkMeshSectionBuilder builder;

	for (int i = 0; i < m_groups.getSize(); i++)
	{
		Group& group = m_groups[i];
		if (group.m_modelIndex != modelIndex)
		{
			continue;
		}

		const hkQemSimplifier::Group& qemGroup = simplifier.getGroups()[i];
		const int numVertices = qemGroup.m_attributes.getSize();

		if (numVertices < 3)
		{
			// If its not a triangle, don't bother adding it
			continue;
		}

		// Create the vertex buffer
		hkVertexFloat32Converter converter;
		converter.init(group.m_vertexFormat, m_srcAabb, unitScalePosition);
		_setWeights(group.m_material, group.m_vertexFormat, converter);

		// Convert back
		hkMemoryMeshVertexBuffer buffer(group.m_vertexFormat, numVertices);

		{
			hkUint8* vertex = buffer.getVertexData();
			const int stride = buffer.getVertexStride();
			hkArray<hkFloat32> fattrib;
			fattrib.setSize(converter.countVertexToFloat32());
			for (int j = 0; j < numVertices; j++, vertex += stride)
			{
				const hkReal* attrib = (const hkReal*)qemGroup.getAttribute(j);
				for (int a=0; a<fattrib.getSize(); ++a)
				{
					fattrib[a] = hkFloat32(attrib[a]);
				}
				converter.convertFloat32ToVertex(fattrib.begin(), vertex);
			}
		}
	
		// I could copy over the attributes unique positions...
		{
			const int numAttribs = qemGroup.m_attributes.getSize();
			hkArray<int> attribToPositionMap(numAttribs, -1);

			for (int j = 0; j < numTris; j++)
			{
				const hkQemSimplifier::Triangle* tri = simplifier.getTriangle(j);
				if (tri->m_groupIndex != i)
				{
					continue;
				}

				for (int k = 0; k < 3; k++)
				{
					const int positionIndex = tri->m_vertexIndices[k];
					const int attribIndex = tri->m_attributeIndices[k];

					int dstPositionIndex = attribToPositionMap[attribIndex];
					if (dstPositionIndex >= 0)
					{
						// They should all map to the same position index...
						HK_ASSERT(0x4224a342, dstPositionIndex == positionIndex);
					}
					else
					{
						attribToPositionMap[attribIndex] = positionIndex;
					}
				}
			}

			// Okay I can now run through all of the attributes.. and set their positions...
			hkUint8* vertex = buffer.getVertexData();
			const int stride = buffer.getVertexStride();

			const int positionOffset = hkMemoryMeshVertexBuffer::calculateElementOffset(group.m_vertexFormat, hkVertexFormat::USAGE_POSITION, 0);
			HK_ASSERT(0x2424324, positionOffset >= 0 );

			vertex += positionOffset;

			const hkArray<hkVector4>& positions = simplifier.getPositions();

			for (int j = 0; j < numVertices; j++, vertex += stride)
			{
				const int positionIndex = attribToPositionMap[j];
				HK_ASSERT(0x424324a2, positionIndex >= 0);

				// Set the position
				const hkVector4& position = positions[positionIndex];

				// Save the position
				position.store<3,HK_IO_NATIVE_ALIGNED>((hkFloat32*)vertex);
			}
		}
	
		// Create the buffer on the device
		hkMeshVertexBuffer* vertexBuffer = system->createVertexBuffer(group.m_vertexFormat, numVertices);

		// Convert
		hkMeshVertexBufferUtil::convert(&buffer, vertexBuffer);

		// Start the section
		builder.startMeshSection(vertexBuffer, group.m_material);
		vertexBuffer->removeReference();
		
		for (int j = 0; j < numTris; j++)
		{
			const Triangle* tri = simplifier.getTriangle(j);
			if (tri->m_groupIndex != i)
			{
				continue;
			}
			if (qemGroup.getNumAttributes() < 0x10000)
			{
				// 16 bit will do
				hkUint16 indices[3] = { hkUint16(tri->m_attributeIndices[0]), hkUint16(tri->m_attributeIndices[1]), hkUint16(tri->m_attributeIndices[2])}; 
				/// Add indices to the current primitive
				builder.concatPrimitives(hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST, indices, 3);
			}
			else
			{
				// Have to do 32 bit
				hkUint32 indices[3] = { hkUint32(tri->m_attributeIndices[0]), hkUint32(tri->m_attributeIndices[1]), hkUint32(tri->m_attributeIndices[2])}; 
				/// Add indices to the current primitive
				builder.concatPrimitives(hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST, indices, 3);
			}
		}

		builder.endMeshSection();		
	}

	if (builder.getNumSections() == 0)
	{
		// There isn't anything left...
		return HK_NULL;
	}

	return system->createShape(builder.getSections(), builder.getNumSections());
}


void hkMeshSimplifierConverter::getWeight(hkVertexFormat::ComponentUsage usage, int subUsage, hkSimdReal& weight) const
{
	hkUint32 key = (hkUint32(usage) << 8) | subUsage;
	const int index = m_weightElements.indexOf(key);
	if (index >= 0)
	{
		weight = m_weights[index];
	}

	weight = m_defaultWeights[usage];
}

void hkMeshSimplifierConverter::setMaterialWeight(hkMeshMaterial* material, hkSimdRealParameter weight)
{
	int index = m_materials.indexOf(material);
	if (index >= 0)
	{
		m_materialWeight[index] = weight;
		return;
	}
	m_materials.pushBack(material);
	m_materialWeight.pushBack(weight);
}

void hkMeshSimplifierConverter::getMaterialWeight(hkMeshMaterial* material, hkSimdReal& weight) const
{
	int index = m_materials.indexOf(material);
	weight = (index >= 0) ? m_materialWeight[index] : m_defaultMaterialWeight;
}

void hkMeshSimplifierConverter::setWeight(hkVertexFormat::ComponentUsage usage, int subUsage, hkSimdRealParameter weight)
{
	hkUint32 key = (hkUint32(usage) << 8) | subUsage;
	const int index = m_weightElements.indexOf(key);
	if (index >= 0)
	{
		m_weights[index] = weight;
		
	}
	else
	{
		m_weightElements.pushBack(key);
		m_weights.pushBack(weight);
	}
}

void hkMeshSimplifierConverter::clearWeights()
{
	m_materials.clear();
	m_materialWeight.clear();

	m_weightElements.clear();
	m_weights.clear();

	const hkSimdReal defaultWeight = hkSimdReal::fromFloat(hkReal(0.02f));
	for (unsigned int i = 0; i < HK_COUNT_OF(m_defaultWeights); i++)
	{
		m_defaultWeights[i] = defaultWeight;
	}
	m_defaultWeights[hkVertexFormat::USAGE_POSITION] = hkSimdReal_1;
	m_defaultWeights[hkVertexFormat::USAGE_TEX_COORD] = hkSimdReal_Half;
}

void hkMeshSimplifierConverter::scaleWeights(hkSimdRealParameter scale)
{
	for (int i = 0; i < m_materialWeight.getSize(); i++)
	{
		m_materialWeight[i].mul(scale);
	}

	for (int i = 0; i < m_weights.getSize(); i++)
	{
		m_weights[i].mul(scale);
	}
	for (unsigned int i = 0; i < HK_COUNT_OF(m_defaultWeights); i++)
	{
		m_defaultWeights[i].mul(scale);
	}
}

hkMeshShape* hkMeshSimplifierConverter::simplifyCoplanar(	hkMeshSystem* meshSystem,
															const hkMeshShape* meshShape,
															const Threshold* thresholds,
															hkReal maxCoplanarError, bool allowOpen, bool unitScalePosition)
{
	hkQemSimplifier simplifier;
	simplifier.setEnableOpenGeometry(allowOpen);

	// Don't have any scaling calculator
	simplifier.setScaleCalculator(HK_NULL);
	simplifier.setDiscardInvalidAttributes(true);
		
	hkFindUniquePositionsUtil positionUtil;
	hkResult res = initMesh(meshShape, simplifier, positionUtil, thresholds, unitScalePosition);
	if (res != HK_SUCCESS)
	{
		return HK_NULL;
	}

	bool hasSimplified = false;
	const hkSimdReal maxCoplanar = hkSimdReal::fromFloat(maxCoplanarError);
	while (true)
	{
		hkQemSimplifier::EdgeContraction* contraction = simplifier.getTopContraction();
		if (contraction == HK_NULL)
		{
			break;
		}

		hkSimdReal absError; absError.setAbs(contraction->m_error);
		if (absError > maxCoplanar)
		{
			break;
		}

		// If it flips something, we'll still ignore it
		if (simplifier.doesTopContractionFlipTriangle())
		{
			simplifier.discardTopContraction();
		}
		else
		{
			// Apply the contraction
			hasSimplified = true;
			simplifier.applyTopContraction();
		}
	}

	if (!hasSimplified)
	{
		// If it hasn't simplified, the original is good enough
		meshShape->addReference();
		return const_cast<hkMeshShape*>(meshShape);
	}

	// 
	simplifier.finalize();

	return createMesh(meshSystem, 0, simplifier, unitScalePosition);
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
