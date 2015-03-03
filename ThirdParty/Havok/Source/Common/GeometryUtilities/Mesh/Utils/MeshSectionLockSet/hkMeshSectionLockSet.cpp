/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>
#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>

hkMeshSectionLockSet::~hkMeshSectionLockSet()
{
    clear();
}

void hkMeshSectionLockSet::clear()
{
    const int numSections = m_sections.getSize();
    for (int i = 0; i < numSections; i++)
    {
        const hkMeshShape* shape = m_shapes[i];
        hkMeshSection& section = m_sections[i];
        shape->unlockSection(section);
		shape->removeReference();
    }

    m_sections.clear();
    m_shapes.clear();
}

void hkMeshSectionLockSet::removeSectionAtIndex(int index)
{
	const hkMeshShape* shape = m_shapes[index];
    hkMeshSection& section = m_sections[index];
    shape->unlockSection(section);
	shape->removeReference();	

	m_sections.removeAt(index);
	m_shapes.removeAt(index);
}

void hkMeshSectionLockSet::findUniqueVertexBuffers(hkArray<hkMeshVertexBuffer*>& buffersOut)
{
    buffersOut.clear();

    const int numSections = m_sections.getSize();
    for (int i = 0; i < numSections; i++)
    {
        hkMeshSection& section = m_sections[i];

        hkMeshVertexBuffer* buffer = section.m_vertexBuffer;
        if (!buffer)
        {
            continue;
        }

        if (buffersOut.indexOf(buffer) < 0)
        {
            buffersOut.pushBack(buffer);
        }
    }
}

const hkMeshSection& hkMeshSectionLockSet::addMeshSection(const hkMeshShape* shape, int index, hkUint8 accessFlags)
{
    hkMeshSection& section = m_sections.expandOne();
    m_shapes.pushBack(shape);
	shape->addReference();

    shape->lockSection(index, accessFlags, section);

    return section;
}

void hkMeshSectionLockSet::addMeshSections(const hkMeshShape* shape, hkUint8 accessFlags)
{
    const int numSections = shape->getNumSections();
    hkMeshSection* sections = m_sections.expandBy(numSections);
    const hkMeshShape** shapes = m_shapes.expandBy(numSections);
    for (int i = 0; i < numSections; i++)
    {
        shape->lockSection(i, accessFlags, sections[i]);
		shape->addReference();
        shapes[i] = shape;
    }
}

const hkMeshSection* hkMeshSectionLockSet::findSection(const hkMeshShape* shape, int sectionIndex) const
{
    const int num = m_sections.getSize();
    for (int i = 0; i < num; i++)
    {
        if (m_shapes[i] == shape && m_sections[i].m_sectionIndex == sectionIndex)
        {
            return &m_sections[i];
        }
    }
    return HK_NULL;
}

//
//	Finds the index of a section - returns -1 if not found

int hkMeshSectionLockSet::findSectionIndex(const hkMeshShape* shape, int meshSectionIndex) const
{
	const int num = m_sections.getSize();
	for (int i = 0; i < num; i++)
	{
		if ( (m_shapes[i] == shape) && (m_sections[i].m_sectionIndex == meshSectionIndex) )
		{
			return i;
		}
	}
	return -1;
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
