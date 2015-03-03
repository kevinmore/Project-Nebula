/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Resource/hkObjectResource.h>

static inline bool baseOrSameClass(const char* typeName, const char* topLevelClassName, const hkClassNameRegistry* classReg)
{
	HK_ASSERT(0x53ad1bc5, typeName && topLevelClassName);
	HK_ASSERT(0x58b56f4e, classReg);
	const hkClass* topLevelClass = classReg->getClassByName(topLevelClassName);
	HK_ASSERT3(0x720938bc, topLevelClass, "The class " << topLevelClassName << " is not registered.");
	const hkClass* baseClass = classReg->getClassByName(typeName);
	HK_ASSERT(0x11efc014, baseClass);
	return baseClass->isSuperClass(*topLevelClass);
}

hkObjectResource::hkObjectResource(const hkVariant& v)
: m_topLevelObject(v)
, m_classRegistry(hkBuiltinTypeRegistry::getInstance().getClassNameRegistry())
, m_typeRegistry(hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry())
{
	HK_ASSERT(0x36e7deb0, v.m_class && v.m_object);
}

hkObjectResource::~hkObjectResource()
{
	if( m_topLevelObject.m_class )
	{
		if( hkReferencedObjectClass.isSuperClass(*m_topLevelObject.m_class) )
		{
			static_cast<hkReferencedObject*>(m_topLevelObject.m_object)->removeReference();
		}
		else
		{
			if( m_typeRegistry )
			{
				m_typeRegistry->cleanupLoadedObject(m_topLevelObject.m_object, m_topLevelObject.m_class->getName());
			}
			hkMemHeapBlockFree(m_topLevelObject.m_object, m_topLevelObject.m_class->getObjectSize());
		}
	}	
}

void hkObjectResource::setClassNameRegistry(const hkClassNameRegistry* classReg)
{
	m_classRegistry = classReg;
}

void hkObjectResource::setTypeInfoRegistry(const hkTypeInfoRegistry* typeReg)
{
	m_typeRegistry = typeReg;
}

const char* hkObjectResource::getName() const
{
	return HK_NULL;
}

void hkObjectResource::getImportsExports( hkArray<Import>& impOut, hkArray<Export>& expOut ) const
{
}

void* hkObjectResource::getContentsPointer(const char* typeName, const hkTypeInfoRegistry* typeRegistry) const
{
	if( !typeName
		|| baseOrSameClass(typeName, m_topLevelObject.m_class->getName(), m_classRegistry) )
	{
		// the objects are already finished
		return m_topLevelObject.m_object;
	}
	else
	{
		HK_WARN(0x599a0b9c, "Requested '" << typeName << "', but packfile data contains '" << m_topLevelObject.m_class->getName() << "'.");
	}
	return HK_NULL;
}

void* hkObjectResource::stealContentsPointer(const char* typeName, const hkTypeInfoRegistry* typeRegistry)
{
	void* ret = getContentsPointer(typeName, typeRegistry);
	if( ret ) // successful steal,
	{
		m_topLevelObject.m_object = HK_NULL; // resource no longer owns memory
		m_topLevelObject.m_class = HK_NULL;
	}
	return ret;
}

const char* hkObjectResource::getContentsTypeName() const
{
	return m_topLevelObject.m_class->getName();
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
