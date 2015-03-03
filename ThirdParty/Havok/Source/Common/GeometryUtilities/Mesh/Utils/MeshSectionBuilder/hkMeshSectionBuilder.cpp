/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>

#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>
#include <Common/GeometryUtilities/Mesh/Utils/PrimitiveUtil/hkMeshPrimitiveUtil.h>

// this
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionBuilder/hkMeshSectionBuilder.h>

hkMeshSectionBuilder::~hkMeshSectionBuilder()
{
	clear();
}

void hkMeshSectionBuilder::clear()
{
    const int numSections = m_sections.getSize();
    for (int i = 0; i < numSections; i++)
    {
        hkMeshSectionCinfo& section = m_sections[i];

        if (section.m_vertexBuffer)
        {
            section.m_vertexBuffer->removeReference();
        }
        if (section.m_material)
        {
            section.m_material->removeReference();
        }
    }

    m_sections.clear();
    m_indices16.clear();
	m_indices32.clear();
}

void hkMeshSectionBuilder::startMeshSection(hkMeshVertexBuffer* vertexBuffer, hkMeshMaterial* material)
{
    hkMeshSectionCinfo& section = m_sections.expandOne();

	section.m_material = material;
	if (material)
	{
		material->addReference();
	}

	section.m_vertexBuffer = vertexBuffer;
	if (vertexBuffer)
	{
		vertexBuffer->addReference();
	}

	section.m_primitiveType = hkMeshSection::PRIMITIVE_TYPE_UNKNOWN;
	section.m_indexType = hkMeshSection::INDEX_TYPE_UINT16;
    section.m_indices = HK_NULL;

    section.m_numPrimitives = 0;
    section.m_vertexStartIndex = 0;
    section.m_transformIndex = -1;

	m_indexBase16 = m_indices16.getSize();
	m_indexBase32 = m_indices32.getSize();
}

void hkMeshSectionBuilder::setMaterial(hkMeshMaterial* material)
{
	hkMeshSectionCinfo& section = m_sections.back();
	if (section.m_material != HK_NULL)
	{
		HK_ASSERT(0x324323, !"The material can only be set once for a section");
		return;
	}
	material->addReference();
	section.m_material = material;
}

void hkMeshSectionBuilder::setVertexBuffer(hkMeshVertexBuffer* vertexBuffer)
{
	hkMeshSectionCinfo& section = m_sections.back();
	if (section.m_vertexBuffer != HK_NULL)
	{
		HK_ASSERT(0x324323, !"The vertexBuffer can only be set once for a section");
		return;
	}
	vertexBuffer->addReference();
	section.m_vertexBuffer = vertexBuffer;
}

hkBool hkMeshSectionBuilder::_isIndexed() const 
{
	return m_indices16.getSize() != m_indexBase16 || m_indices32.getSize() != m_indexBase32;
}

void hkMeshSectionBuilder::endMeshSection()
{
    hkMeshSectionCinfo& section = m_sections.back();

	if (section.m_vertexBuffer == HK_NULL )
	{
		HK_ASSERT(0xe4234234, !"The vertex buffer must be set before end of mesh section");
		m_sections.popBack();
		return;
	}

	if (_isIndexed())
	{
		// If indexed we need to set the amount of indices
		switch (section.m_indexType )
		{
			case hkMeshSection::INDEX_TYPE_UINT16:
			{
				const int numIndices = m_indices16.getSize() - m_indexBase16;
				section.m_indices = m_indices16.begin() + m_indexBase16;
				section.m_numPrimitives = hkMeshPrimitiveUtil::calculateNumPrimitives(section.m_primitiveType, numIndices);
				break;
			}
			case hkMeshSection::INDEX_TYPE_UINT32:
			{
				const int numIndices = m_indices32.getSize() - m_indexBase32;
				section.m_indices = m_indices32.begin() + m_indexBase32;
				section.m_numPrimitives = hkMeshPrimitiveUtil::calculateNumPrimitives(section.m_primitiveType, numIndices);
				break;
			}
			default:
				HK_ASSERT(0x324323, !"Unknown index type");
		}
	}

	if (section.m_numPrimitives == 0)
	{
		// Nothing was added, so remove it
		m_sections.popBack();
	}
}

void hkMeshSectionBuilder::_makeIndices32()
{
	hkMeshSectionCinfo& section = m_sections.back();
	if (section.m_indexType == hkMeshSection::INDEX_TYPE_UINT32)
	{
		return;
	}

	HK_ASSERT(0xabb34234, section.m_indexType == hkMeshSection::INDEX_TYPE_UINT16);

	// Work out how many indices
	const int numIndices = m_indices16.getSize() - m_indexBase16;
	if (numIndices > 0)
	{
		// Expand them up
		const hkUint16* srcIndices = m_indices16.begin() + m_indexBase16;
		// We need to make space 
		hkUint32* dstIndices = _addIndices32(numIndices);

		for (int i = 0; i < numIndices; i++)
		{
			dstIndices[i] = srcIndices[i];
		}

		// Remove the 16 bit indices
		m_indices16.setSize(m_indexBase16);
	}

	// Indices are now 32 bit
	section.m_indexType = hkMeshSection::INDEX_TYPE_UINT32;
}

void hkMeshSectionBuilder::_concatIndices(int vertexStartIndex, int numIndices)
{
	if (vertexStartIndex + numIndices > 0xffff)
	{
		// Make sure we have the range
		_makeIndices32();
	}

	hkMeshSectionCinfo& section = m_sections.back();
	switch (section.m_indexType)
	{
		case hkMeshSection::INDEX_TYPE_UINT16:
		{
			hkUint16* dstIndices = _addIndices16(numIndices);
			for (int i = 0; i < numIndices; i++)
			{
				dstIndices[i] = hkUint16(vertexStartIndex + i);
			}
			break;
		}
		case hkMeshSection::INDEX_TYPE_UINT32:
		{
			hkUint32* dstIndices = _addIndices32(numIndices);
			for (int i = 0; i < numIndices; i++)
			{
				dstIndices[i] = hkUint32(vertexStartIndex + i);
			}
			break;
		}
		default:
		{
			HK_ASSERT(0x3422342, !"Unknown index type");
		}
	}
}

/* static */bool hkMeshSectionBuilder::_needsIndices32(const hkUint16* srcIndices, int numIndices, int indexBase)
{
	if (indexBase == 0)
	{
		return false;
	}

	if (indexBase > 0xffff)
	{
		return true;
	}

	int max = srcIndices[0];
	for (int i = 1; i < numIndices; i++)
	{
		if (srcIndices[i] > max)
		{
			max = srcIndices[i];
		}
	}

	return (indexBase + max) > 0xffff;
}

void hkMeshSectionBuilder::_concatIndices(const hkUint16* srcIndices, int numIndices, int indexBase)
{
	if (numIndices == 0)
	{
		return;
	}
	if (_needsIndices32(srcIndices, numIndices, indexBase))
	{
		_makeIndices32();
	}

	hkMeshSectionCinfo& section = m_sections.back();

	switch (section.m_indexType)
	{
		case hkMeshSection::INDEX_TYPE_UINT16:
		{
			hkUint16* dstIndices = _addIndices16(numIndices);
			for (int i = 0; i < numIndices; i++)
			{
				dstIndices[i] = hkUint16(srcIndices[i] + indexBase);
			}
			break;
		}
		case hkMeshSection::INDEX_TYPE_UINT32:
		{
			hkUint32* dstIndices = _addIndices32(numIndices);
			for (int i = 0; i < numIndices; i++)
			{
				dstIndices[i] = hkUint32(srcIndices[i] + indexBase);
			}
			break;
		}
		default: HK_ASSERT(0x234234, !"Unknown index type");
	}
}

void hkMeshSectionBuilder::_concatIndices(const hkUint32* srcIndices, int numIndices, int indexBase)
{
	_makeIndices32();
	hkUint32* dstIndices = _addIndices32(numIndices);
	for (int i = 0; i < numIndices; i++)
	{
		dstIndices[i] = hkUint32(srcIndices[i] + indexBase);
	}
}

hkResult hkMeshSectionBuilder::concatUnindexed(hkMeshSection::PrimitiveType primType, int vertexStartIndex, int numIndices)
{
    hkMeshSectionCinfo& section = m_sections.back();
	if (section.m_primitiveType == hkMeshSection::PRIMITIVE_TYPE_UNKNOWN)
	{
		// If the section is empty, just set up as specified
		section.m_primitiveType = primType;
		section.m_indices = HK_NULL;
		section.m_indexType = hkMeshSection::INDEX_TYPE_NONE;
		section.m_numPrimitives = hkMeshPrimitiveUtil::calculateNumPrimitives(primType, numIndices);
		section.m_vertexStartIndex = vertexStartIndex;
		section.m_transformIndex = -1;
		return HK_SUCCESS;
	}

	// Make concatable
	hkResult res = _makeConcatable(primType);
	if (res != HK_SUCCESS)
	{
		return res;
	}

	if (primType == section.m_primitiveType)
	{
		_concatIndices(vertexStartIndex, numIndices);
		return HK_SUCCESS;
	}

	hkMeshPrimitiveUtil::PrimitiveStyle style = hkMeshPrimitiveUtil::getPrimitiveStyle(primType);
	if (style == hkMeshPrimitiveUtil::PRIMITIVE_STYLE_TRIANGLE && section.m_primitiveType == hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST)
	{
		if (vertexStartIndex + numIndices > 0xffff)
		{
			hkArray<hkUint32> triangleIndices;
			hkMeshPrimitiveUtil::appendTriangleIndices(primType, numIndices, vertexStartIndex, triangleIndices);
			_concatIndices(triangleIndices.begin(), triangleIndices.getSize());
		}
		else
		{
			hkArray<hkUint16> triangleIndices;
			hkMeshPrimitiveUtil::appendTriangleIndices(primType, numIndices, vertexStartIndex, triangleIndices);
			_concatIndices(triangleIndices.begin(), triangleIndices.getSize());
		}
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}


hkUint16* hkMeshSectionBuilder::_addIndices16(int numIndices)
{
	hkUint16* indexBase = m_indices16.begin();
	hkUint16* dstIndices = m_indices16.expandBy(numIndices);

	if (indexBase != m_indices16.begin())
	{
		// Fix up all the pointers if the index set has moved
		const int numSections = m_sections.getSize();
		for (int i = 0; i < numSections; i++)
		{
			hkMeshSectionCinfo& section = m_sections[i];

			if (section.m_indices && section.m_indexType == hkMeshSection::INDEX_TYPE_UINT16)
			{
				// Fix the pointers as the index sets have moved
				const hkUint16* ind = (const hkUint16*)section.m_indices;
				ind = ind - indexBase + m_indices16.begin();
				section.m_indices = ind;
			}
		}
	}

	return dstIndices;
}

hkUint32* hkMeshSectionBuilder::_addIndices32(int numIndices)
{
	hkUint32* indexBase = m_indices32.begin();
	hkUint32* dstIndices = m_indices32.expandBy(numIndices);

	if (indexBase != m_indices32.begin())
	{
		// Fix up all the pointers if the index set has moved
		const int numSections = m_sections.getSize();
		for (int i = 0; i < numSections; i++)
		{
			hkMeshSectionCinfo& section = m_sections[i];
			if (section.m_indices && section.m_indexType == hkMeshSection::INDEX_TYPE_UINT32)
			{
				// Fix the pointers as the index sets have moved
				const hkUint32* ind = (const hkUint32*)section.m_indices;
				ind = ind - indexBase + m_indices32.begin();
				section.m_indices = ind;
			}
		}
	}
	return dstIndices;
}

/* static */bool hkMeshSectionBuilder::canConcatPrimitives(hkMeshSection::PrimitiveType b, hkMeshSection::PrimitiveType a)
{
	if (a == b)
	{
		return true;
	}

	hkMeshPrimitiveUtil::PrimitiveStyle styleA  = hkMeshPrimitiveUtil::getPrimitiveStyle(a);
	if (styleA == hkMeshPrimitiveUtil::PRIMITIVE_STYLE_UNKNOWN)
	{
		return false;
	}

	hkMeshPrimitiveUtil::PrimitiveStyle styleB  = hkMeshPrimitiveUtil::getPrimitiveStyle(b);
	return styleA == styleB;
}


bool hkMeshSectionBuilder::_canConcatPrimitive(hkMeshSection::PrimitiveType primType)
{
	// We are safe if there are no sections
	if (m_sections.getSize() <= 0)
	{
		return true;
	}

	hkMeshSectionCinfo& section = m_sections.back();

	// If its not set, then try setting it
	if (section.m_primitiveType == hkMeshSection::PRIMITIVE_TYPE_UNKNOWN)
	{
		return true;
	}
	return canConcatPrimitives(section.m_primitiveType, primType);
}


hkResult hkMeshSectionBuilder::_makeConcatable(hkMeshSection::PrimitiveType primType)
{
	HK_ASSERT(0x43243232, _canConcatPrimitive(primType));

	hkMeshSectionCinfo& section = m_sections.back();

	// If the current section was started with unindexed - then make indexed
	if (!_isIndexed() && section.m_numPrimitives > 0)
	{
		// We should expand it out 
		HK_ASSERT(0x42423, section.m_vertexStartIndex >= 0);
		const int numSectionIndices = hkMeshPrimitiveUtil::calculateNumIndices(section.m_primitiveType, section.m_numPrimitives);
		// Convert it into indices
		section.m_indexType = hkMeshSection::INDEX_TYPE_UINT16;
		_concatIndices(section.m_vertexStartIndex, numSectionIndices);
	}

	// If its not set, then try setting it
	if (section.m_primitiveType == hkMeshSection::PRIMITIVE_TYPE_UNKNOWN)
	{
		section.m_primitiveType = primType;
	}

	if (section.m_primitiveType == primType)
	{
		// If same type - can concat easily
		return HK_SUCCESS;
	}

	// Okay must be triangle type - or the _canConcatPrimitive call would have failed

	if (section.m_primitiveType == hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST)
	{
		return HK_SUCCESS;
	}
	
	// Make it a triangle list them
	if (section.m_primitiveType == hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP)
	{
		const int numSectionIndices = hkMeshPrimitiveUtil::calculateNumIndices(section.m_primitiveType, section.m_numPrimitives);

		switch (section.m_indexType)
		{
			case hkMeshSection::INDEX_TYPE_UINT16:
			{
				hkArray<hkUint16> triangleIndices;
				const hkUint16* srcIndices = m_indices16.begin() + m_indexBase16;
				hkMeshPrimitiveUtil::appendTriangleIndices16(section.m_primitiveType, srcIndices, numSectionIndices, 0, triangleIndices);
				m_indices16.setSize(m_indices16.getSize() - numSectionIndices);

				hkString::memCpy(_addIndices16(triangleIndices.getSize()), triangleIndices.begin(), triangleIndices.getSize() * sizeof(hkUint16));
				break;
			}
			case hkMeshSection::INDEX_TYPE_UINT32:
			{
				hkArray<hkUint32> triangleIndices;
				const hkUint32* srcIndices = m_indices32.begin() + m_indexBase32;
				hkMeshPrimitiveUtil::appendTriangleIndices32(section.m_primitiveType, srcIndices, numSectionIndices, 0, triangleIndices);
				m_indices32.setSize(m_indices32.getSize() - numSectionIndices);

				hkString::memCpy(_addIndices32(triangleIndices.getSize()), triangleIndices.begin(), triangleIndices.getSize() * sizeof(hkUint32));
				break;
			}
			default: 
			{
				HK_ASSERT(0x353453, !"Unhandled indexing");
				return HK_FAILURE;
			}
		}

		// Its a triangle list now
		section.m_primitiveType = hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST;
		return HK_SUCCESS;
	}

	return HK_FAILURE;
}

hkResult hkMeshSectionBuilder::_concatPrimitives(hkMeshSection::PrimitiveType primType, const hkUint16* indices, int numIndices, int indexBase)
{
	hkMeshSectionCinfo& section = m_sections.back();
	if (section.m_primitiveType == primType)
	{
		_concatIndices(indices, numIndices, indexBase);
		return HK_SUCCESS;
	}

	if (section.m_primitiveType == hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST && primType == hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP)
	{
		hkArray<hkUint16> triIndices;
		hkMeshPrimitiveUtil::appendTriangleIndices16(primType, indices, numIndices, 0, triIndices);
		_concatIndices(triIndices.begin(), triIndices.getSize(), indexBase);
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

hkResult hkMeshSectionBuilder::_concatPrimitives(hkMeshSection::PrimitiveType primType, const hkUint32* indices, int numIndices, int indexBase)
{
	hkMeshSectionCinfo& section = m_sections.back();
	if (section.m_primitiveType == primType)
	{
		_concatIndices(indices, numIndices, indexBase);
		return HK_SUCCESS;
	}

	if (section.m_primitiveType == hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST && primType == hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP)
	{
		hkArray<hkUint32> triIndices;
		hkMeshPrimitiveUtil::appendTriangleIndices32(primType, indices, numIndices, 0, triIndices);
		_concatIndices(triIndices.begin(), triIndices.getSize(), indexBase);
		return HK_SUCCESS;
	}

	return HK_FAILURE;
}

hkResult hkMeshSectionBuilder::concatPrimitives(hkMeshSection::PrimitiveType primType, const hkUint16* indices, int numIndices, int indexBase)
{
	hkResult res = _makeConcatable(primType);
	if (res != HK_SUCCESS)
	{
		return res;
	}

	return _concatPrimitives(primType, indices, numIndices, indexBase);
}

hkResult hkMeshSectionBuilder::concatPrimitives(hkMeshSection::PrimitiveType primType, const hkUint32* indices, int numIndices, int indexBase)
{
	hkResult res = _makeConcatable(primType);
	if (res != HK_SUCCESS)
	{
		return res;
	}
	return _concatPrimitives(primType, indices, numIndices, indexBase);
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
