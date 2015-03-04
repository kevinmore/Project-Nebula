/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Attributes/hkxAttributeHolder.h>
#include <Common/Base/Reflection/hkClass.h>

hkxAttributeHolder::hkxAttributeHolder(const hkxAttributeHolder& other)
: hkReferencedObject(other)
{
	m_attributeGroups = other.m_attributeGroups;
}

hkxAttributeHolder::~hkxAttributeHolder()
{
	// Make sure we release all mem in our owned groups
	
}

const hkxAttributeGroup* hkxAttributeHolder::findAttributeGroupByName (const char* name) const
{
	if( !name )
	{
		return HK_NULL;
	}

	for (int j=0; j < m_attributeGroups.getSize(); ++j)
	{
		const hkxAttributeGroup& group = m_attributeGroups[j];
		if ( group.m_name && hkString::strCasecmp(group.m_name, name)==0 )
		{
			return &group;
		}
	}
	return HK_NULL;
}


hkRefVariant hkxAttributeHolder::findAttributeVariantByName( const char* name ) const
{
	if( name )
	{
		for (int j=0; j < m_attributeGroups.getSize(); ++j)
		{
			const hkxAttributeGroup& g = m_attributeGroups[j];
			hkRefVariant var = g.findAttributeVariantByName( name );
			if (var) 
				return var;
		}
	}
	hkVariant v = {HK_NULL,HK_NULL};
	return v;
}

hkReferencedObject* hkxAttributeHolder::findAttributeObjectByName( const char* name, const hkClass* type ) const
{
	if( !name )
	{
		return HK_NULL;
	}

	hkRefVariant var = findAttributeVariantByName( name );

	// compare class by name so that it deals with serialized classes etc better (for instance in the filters)
	if (var && (!type || (hkString::strCasecmp(type->getName(), var.getClass()->getName()) == 0)) )
	{
		return var.val();
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
