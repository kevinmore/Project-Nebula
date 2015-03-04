/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Material/hkxMaterial.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

hkxMaterial::~hkxMaterial()
{
	
}

/// function object that routes calls to operator<
class _texStageLessFn
{
public:

	HK_FORCE_INLINE hkBool operator() ( const hkxMaterial::TextureStage& a, const hkxMaterial::TextureStage& b )
	{
		return ((int)a.m_usageHint) < ((int)b.m_usageHint) || ( (a.m_usageHint == b.m_usageHint) && (a.m_tcoordChannel < b.m_tcoordChannel) );
	}
};


void hkxMaterial::sortTextureStageOrder()
{
	_texStageLessFn funktor;
	hkAlgorithm::insertionSort( m_stages.begin(), m_stages.getSize(), funktor); // Has to be a stable sort, so that can predict from modeller, based on depth first material traversal, the order in final material of shared hints and uv coords
}

// Adds a property to the material

void hkxMaterial::addProperty(int key, int value)
{
	if(!hasProperty(key))
	{
		Property * p = m_properties.expandBy(1);
		p->m_key = key;
		p->m_value = value;
	}
}

// Returns a property value

hkUint32 hkxMaterial::getProperty(int key)
{
	const hkUint32 theKey = (hkUint32)key;
	for(int i = m_properties.getSize() - 1; i >= 0; i--)
	{
		const Property & p = m_properties[i];
		if(p.m_key == theKey)
			return p.m_value;
	}

	// Not found!
	HK_ASSERT(0x6e6765d5, false);
	return 0xFFFFFFFF;
}

// Returns true if the material has the given property

hkBool hkxMaterial::hasProperty(int key)
{
	const hkUint32 theKey = (hkUint32)key;
	for(int i = m_properties.getSize() - 1; i >= 0; i--)
	{
		const Property & p = m_properties[i];
		if(p.m_key == theKey)
			return true;
	}

	// Not found!
	return false;
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
