/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Destruction/BreakableMaterial/hkpBreakableMaterial.h>

//
//	Called to add multiple shape keys

void hkpBreakableMaterial::ShapeKeyCollector::addShapeKeyBatch(const hkpShapeKey* shapeKeys, int numShapeKeys)
{
	for (int i = 0; i < numShapeKeys; i++)
	{
		addShapeKey(shapeKeys[i]);
	}
}

//
//	Called to add a contiguous range of shape keys, i.e. from shapeKeyBase to shapeKeyBase + n - 1

void hkpBreakableMaterial::ShapeKeyCollector::addContiguousShapeKeyRange(hkpShapeKey baseShapeKey, int numShapeKeys)
{
	for (int i = 0; i < numShapeKeys; i++)
	{
		addShapeKey(baseShapeKey | i);
	}
}

//
//	Sets the default mapping

void hkpBreakableMaterial::setDefaultMapping()
{
	setMapping(MATERIAL_MAPPING_UNKNOWN);
}

//
//	Destructor

hkpBreakableMaterial::~hkpBreakableMaterial()
{
	if ( m_properties && m_memSizeAndFlags )
	{
		delete m_properties;
	}

	m_properties = HK_NULL;
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
