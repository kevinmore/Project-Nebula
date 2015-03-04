/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkClassPointerVtable.h>
#include <Common/Base/Reflection/Registry/hkDynamicClassNameRegistry.h>

hkClassPointerVtable::TypeInfoRegistry::TypeInfoRegistry(const hkClassNameRegistry* classes)
{
	m_classes = classes;
}

hkClassPointerVtable::TypeInfoRegistry::TypeInfoRegistry(const hkStringMap<const hkClass*>& classes)
{
	hkDynamicClassNameRegistry* d = new hkDynamicClassNameRegistry();
	d->merge(classes);
	m_classes.setAndDontIncrementRefCount(d);
}

hkClassPointerVtable::TypeInfoRegistry::~TypeInfoRegistry()
{
	for( hkStringMap<hkTypeInfo*>::Iterator it = m_typeInfos.getIterator(); 
		m_typeInfos.isValid(it); it = m_typeInfos.getNext(it) )
	{
		delete m_typeInfos.getValue(it);
	}
}


const hkTypeInfo* hkClassPointerVtable::TypeInfoRegistry::finishLoadedObject( void* obj, const char* className ) const
{
	if( hkString::strCmp(className, "hkClass") == 0
		|| hkString::strCmp(className, "hkClassMember") == 0
		|| hkString::strCmp(className, "hkClassEnum") == 0
		|| hkString::strCmp(className, "hkClassEnumItem") == 0 )
	{
		static const hkTypeInfo s_dummyTypeInfo("dummyTypeInfo", HK_NULL, HK_NULL, HK_NULL, HK_NULL, HK_NULL);
		return &s_dummyTypeInfo;
	}
	const hkClass* k = m_classes->getClassByName(className);
	HK_ASSERT(0x22e51cb7, k);
	if( k->hasVtable() )
	{
		*(const void**)obj = k;
	}
	if( hkTypeInfo* t = m_typeInfos.getWithDefault(className, HK_NULL) )
	{
		return t;
	}
	hkTypeInfo* t = new hkTypeInfo(className, HK_NULL, HK_NULL, HK_NULL, HK_NULL, HK_NULL);
	m_typeInfos.insert(className, t );
	return t;
}

void hkClassPointerVtable::VtableRegistry::registerVtable( const void* vtable, const hkClass* klass )
{
	HK_ASSERT(0x30153e6c, 0);
}

const hkClass* hkClassPointerVtable::VtableRegistry::getClassFromVirtualInstance( const void* obj ) const
{
	return *(hkClass**)obj;
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
