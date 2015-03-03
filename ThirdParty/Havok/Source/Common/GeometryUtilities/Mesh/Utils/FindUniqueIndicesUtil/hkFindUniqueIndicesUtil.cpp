/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>

// this
#include <Common/GeometryUtilities/Mesh/Utils/FindUniqueIndicesUtil/hkFindUniqueIndicesUtil.h>

void hkFindUniqueIndicesUtil::initialize(int maxIndex)
{
    m_indicesMap.setSize(maxIndex);
    hkString::memSet4(m_indicesMap.begin(), -1, maxIndex);
    m_uniqueIndices.clear();
}

int hkFindUniqueIndicesUtil::addIndex(int index)
{
    int newIndex = m_indicesMap[index];
    if (newIndex >= 0)
    {
        return newIndex;
    }

    newIndex = m_uniqueIndices.getSize();
    m_indicesMap[index] = newIndex;
    m_uniqueIndices.pushBack(index);

    return newIndex;
}

void hkFindUniqueIndicesUtil::addIndices(const hkUint16* indices, int numIndices)
{
    for (int i = 0; i < numIndices; i++)
    {
        addIndex(indices[i]);
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
