/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshShape.h>

#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>
#include <Common/GeometryUtilities/Mesh/Utils/PrimitiveUtil/hkMeshPrimitiveUtil.h>

hkMemoryMeshShape::hkMemoryMeshShape(const hkMeshSectionCinfo* sections, int numSections)
{
	m_name = HK_NULL;

    int numIndices16 = 0;
    int numIndices32 = 0;

    for (int i = 0; i < numSections; i++)
    {
        const hkMeshSectionCinfo& section = sections[i];
        int numIndices = hkMeshPrimitiveUtil::calculateNumIndices(section.m_primitiveType, section.m_numPrimitives);
        switch (section.m_indexType)
        {
            case hkMeshSection::INDEX_TYPE_NONE: break;
            case hkMeshSection::INDEX_TYPE_UINT16: numIndices16 += numIndices; break;
            case hkMeshSection::INDEX_TYPE_UINT32: numIndices32 += numIndices; break;
            default: HK_ASSERT(0x1312312, !"Unknown type");
        }
    }

    m_indices16.reserve(numIndices16);
    m_indices32.reserve(numIndices32);

    m_sections.setSize(numSections);

    for (int i = 0; i < numSections; i++)
    {
		const hkMeshSectionCinfo& srcSection = sections[i];
        hkMeshSectionCinfo& dstSection = m_sections[i];
        
        dstSection = srcSection;
        dstSection.m_vertexBuffer->addReference();
		if ( dstSection.m_material )
		{
			dstSection.m_material->addReference();
		}

        switch (srcSection.m_indexType)
        {
			case hkMeshSection::INDEX_TYPE_NONE: 
				break;
            case hkMeshSection::INDEX_TYPE_UINT16:
            {
                int numIndices = hkMeshPrimitiveUtil::calculateNumIndices(srcSection.m_primitiveType, srcSection.m_numPrimitives);
                hkUint16* dstIndices = m_indices16.expandBy(numIndices);
                hkString::memCpy(dstIndices, srcSection.m_indices, numIndices * sizeof(hkUint16));
                dstSection.m_indices = dstIndices;
                break;
            }
            case hkMeshSection::INDEX_TYPE_UINT32:
            {
                int numIndices = hkMeshPrimitiveUtil::calculateNumIndices(srcSection.m_primitiveType, srcSection.m_numPrimitives);
                hkUint32* dstIndices = m_indices32.expandBy(numIndices);
                hkString::memCpy(dstIndices, srcSection.m_indices, numIndices * sizeof(hkUint32));
                dstSection.m_indices = dstIndices;
                break;
            }
        }
    }
}


hkMemoryMeshShape::hkMemoryMeshShape( hkFinishLoadedObjectFlag flag )
: hkMeshShape(flag)
, m_sections(flag)
, m_indices16(flag)
, m_indices32(flag)
, m_name(flag)
{
	if( flag.m_finishing )
	{
		const hkUint16* offsetIntoIndices16 = m_indices16.begin();
		const hkUint32* offsetIntoIndices32 = m_indices32.begin();

		for (int i = 0; i < m_sections.getSize(); ++i)
		{
			switch(m_sections[i].m_indexType)
			{
				case hkMeshSection::INDEX_TYPE_NONE: 
					break;
				case hkMeshSection::INDEX_TYPE_UINT16:
					{
						int numIndices = hkMeshPrimitiveUtil::calculateNumIndices(m_sections[i].m_primitiveType, m_sections[i].m_numPrimitives);						
						m_sections[i].m_indices = offsetIntoIndices16;
						offsetIntoIndices16 += numIndices;
					}
					break;
				case hkMeshSection::INDEX_TYPE_UINT32:
					{
						int numIndices = hkMeshPrimitiveUtil::calculateNumIndices(m_sections[i].m_primitiveType, m_sections[i].m_numPrimitives);						
						m_sections[i].m_indices = offsetIntoIndices32;
						offsetIntoIndices32 += numIndices;
					}
					break;
			}
		}
	}
}

hkMemoryMeshShape::~hkMemoryMeshShape()
{
    const int numSections = m_sections.getSize();
    for (int i = 0; i < numSections; i++)
    {
        const hkMeshSectionCinfo& section = m_sections[i];

        section.m_vertexBuffer->removeReference();
		if ( section.m_material )
		{
			section.m_material->removeReference();
		}
    }
}

int hkMemoryMeshShape::getNumSections() const
{
    return m_sections.getSize();
}

void hkMemoryMeshShape::lockSection(int sectionIndex, hkUint8 accessFlags, hkMeshSection& sectionOut) const
{
    const hkMeshSectionCinfo& section = m_sections[sectionIndex];

    sectionOut.m_primitiveType = section.m_primitiveType;
    sectionOut.m_numPrimitives = section.m_numPrimitives;
    sectionOut.m_numIndices = hkMeshPrimitiveUtil::calculateNumIndices(section.m_primitiveType, section.m_numPrimitives);
    sectionOut.m_vertexStartIndex = section.m_vertexStartIndex;
    sectionOut.m_indexType = section.m_indexType;
    sectionOut.m_transformIndex = section.m_transformIndex;

    // Vertex buffer
    sectionOut.m_vertexBuffer = (accessFlags & ACCESS_VERTEX_BUFFER) ? section.m_vertexBuffer : HK_NULL;
    // Indices
    sectionOut.m_indices = HK_NULL;
    if (accessFlags & ACCESS_INDICES)
	{
		if (section.m_indexType != hkMeshSection::INDEX_TYPE_NONE)
		{
			sectionOut.m_indices = section.m_indices;
		}
    }

    // Material
	hkMeshMaterial* mtl = section.m_material;
    sectionOut.m_material = mtl;
	if ( mtl )
	{
		mtl->addReference();
	}

    // Save the section
    sectionOut.m_sectionIndex = sectionIndex;
}

void hkMemoryMeshShape::unlockSection(const hkMeshSection& section) const
{
	hkMeshSectionCinfo& srcSection = const_cast<hkMeshSectionCinfo&>(m_sections[section.m_sectionIndex]);

	// Add reference on new material
	hkMeshMaterial* newMtl = section.m_material;
	if ( newMtl != srcSection.m_material )
	{
		newMtl->addReference();
	}

	// Release reference on previous material
	if ( srcSection.m_material )
	{
		srcSection.m_material->removeReference();
		srcSection.m_material = HK_NULL;
	}

	srcSection.m_material = newMtl;
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
