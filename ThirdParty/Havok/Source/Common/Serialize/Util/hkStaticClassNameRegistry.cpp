/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkStaticClassNameRegistry.h>
#include <Common/Serialize/Version/hkVersionUtil.h>

hkStaticClassNameRegistry::hkStaticClassNameRegistry(hkClass*const* classes, int classVersion, const char* name)
		: hkClassNameRegistry(), m_classes(classes), m_classVersion(classVersion), m_ready(false)
{
	m_name = name;
}

hkStaticClassNameRegistry::hkStaticClassNameRegistry(const hkClass*const* classes, int classVersion, const char* name)
		: hkClassNameRegistry(), m_classes(classes), m_classVersion(classVersion), m_ready(true)
{
	m_name = name;
}

const char* hkStaticClassNameRegistry::getName() const
{
	return m_name;
}

void hkStaticClassNameRegistry::getClasses(hkArray<const hkClass*>& classes) const
{
	checkIfReady();
	int count = 0;
	for( int i = 0; m_classes[i] != HK_NULL; ++i )
	{
		count += 1;
	}
	const hkClass** c = classes.expandBy(count);
	for( int i = 0; m_classes[i] != HK_NULL; ++i )
	{
		c[i] = m_classes[i];
	}
}

const hkClass* hkStaticClassNameRegistry::getClassByName(const char* className) const
{
	checkIfReady();
	for( int i = 0; m_classes[i] != HK_NULL; ++i )
	{
		if( hkString::strCmp(className, m_classes[i]->getName()) == 0 )
		{
			return m_classes[i];
		}
	}
	return HK_NULL;
}

void hkStaticClassNameRegistry::checkIfReady() const
{
	if( !m_ready )
	{
		hkVersionUtil::recomputeClassMemberOffsets( const_cast<hkClass*const*>(m_classes), m_classVersion );
		m_ready = true;
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
