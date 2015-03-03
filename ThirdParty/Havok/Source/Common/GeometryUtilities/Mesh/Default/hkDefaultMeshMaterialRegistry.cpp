/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Default/hkDefaultMeshMaterialRegistry.h>
#include <Common/Base/Reflection/hkClass.h>

#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>

HK_REFLECTION_DEFINE_STUB_VIRTUAL(hkDefaultMeshMaterialRegistry, hkMeshMaterialRegistry);

hkMeshMaterial* hkDefaultMeshMaterialRegistry::loadMaterial(const char* name)
{
    return findMaterial(name);
}

int hkDefaultMeshMaterialRegistry::_findEntryIndex(const char* name) const
{
    const int numEntries = m_entries.getSize();
    for (int i = 0; i < numEntries; i++)
    {
        const Entry& entry = m_entries[i];
		if (hkString::strCmp(entry.m_name.cString(), name) == 0)
        {
            return i;
        }
    }

    return -1;
}


hkMeshMaterial* hkDefaultMeshMaterialRegistry::findMaterial(const char* name)
{
    int index = _findEntryIndex(name);
    return (index >= 0) ? (hkMeshMaterial*)m_entries[index].m_material : HK_NULL;
}

void hkDefaultMeshMaterialRegistry::registerMaterial(const char* name, hkMeshMaterial* material)
{
    int index = _findEntryIndex(name);
    if (index >= 0)
    {
        m_entries[index].m_material = material;
        return;
    }

    Entry& entry = m_entries.expandOne();
    entry.m_name = name;
    entry.m_material = material;
}

void hkDefaultMeshMaterialRegistry::unregisterMaterial(const char* name)
{
    int index = _findEntryIndex(name);
    if (index >= 0)
    {
        m_entries.removeAt(index);
    }
}

void hkDefaultMeshMaterialRegistry::freeMaterials()
{
	m_entries.clearAndDeallocate();
}

void hkDefaultMeshMaterialRegistry::getMaterials(hkArrayBase<hkMeshMaterial*>& materials, hkMemoryAllocator& a)
{
    materials._setSize(a, m_entries.getSize());
    const int numEntries = m_entries.getSize();
    for (int i = 0; i < numEntries; i++)
    {
        const Entry& entry = m_entries[i];
        materials[i] = entry.m_material;
    }
}

const char* hkDefaultMeshMaterialRegistry::getMaterialName(hkMeshMaterial* material)
{
    const int numEntries = m_entries.getSize();
    for (int i = 0; i < numEntries; i++)
    {
        const Entry& entry = m_entries[i];

        if (entry.m_material == material)
        {
            return entry.m_name.cString();
        }
    }
    return HK_NULL;
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
