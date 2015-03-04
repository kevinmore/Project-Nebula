/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkRootLevelContainer.h>

hkRootLevelContainer::NamedVariant::NamedVariant(const char* name, void* object, const hkClass* klass)
{
	set(name, object, klass);
}

hkRootLevelContainer::NamedVariant::NamedVariant(const char* name, const hkVariant& v)
{
	set(name, v);
}

hkRootLevelContainer::NamedVariant::NamedVariant(hkFinishLoadedObjectFlag f) :
	m_name(f), m_className(f), m_variant(f)
{
}

hkRootLevelContainer::hkRootLevelContainer(hkFinishLoadedObjectFlag f) : m_namedVariants(f)
{
}

void* hkRootLevelContainer::findObjectByType( const char* typeName, const void* prevObject ) const
{
	HK_ASSERT(0x07649ef4, typeName);
	int index = 0;
	while((prevObject) && (index < m_namedVariants.getSize()) && (m_namedVariants[index++].getObject() != prevObject) )  { }

	for( int i = index; i < m_namedVariants.getSize(); ++i )
	{
		if( m_namedVariants[i].getTypeName() && hkString::strCmp( typeName, m_namedVariants[i].getTypeName() ) == 0 )
		{
			return m_namedVariants[i].getObject();
		}
	}
	return HK_NULL;
}

void* hkRootLevelContainer::findObjectByName( const char* objectName, const void* prevObject ) const
{
	HK_ASSERT(0x07649ef5, objectName);
	int index = 0;

	while( (prevObject) && (index < m_namedVariants.getSize()) && (m_namedVariants[index++].getObject() != prevObject) )  { }

	for( int i = index; i < m_namedVariants.getSize(); ++i )
	{
		if( m_namedVariants[i].getName() && hkString::strCmp( objectName, m_namedVariants[i].getName() ) == 0 )
		{
			return m_namedVariants[i].getObject();
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
