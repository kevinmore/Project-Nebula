/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#if defined(HK_DEBUG)
#	include <Common/Base/Reflection/hkClass.h>
#	include <Common/Base/Container/String/hkStringBuf.h>
#endif

#if (HK_LINKONCE_VTABLES==1) || (HK_HASHCODE_VTABLE_REGISTRY==1)
#define HK_CAST_VTABLE(x) (x)
#else
#define HK_CAST_VTABLE(x) ((const char*)(x))
#endif

void hkVtableClassRegistry::registerVtable( const void* vtable, const hkClass* klass )
{
	HK_ASSERT2(0x2e231c83, vtable!=HK_NULL, "Nonvirtual classes should not be registered");
#if defined(HK_DEBUG)
	const hkClass* existingClass = HK_NULL;
	if (m_map.get(HK_CAST_VTABLE(vtable), &existingClass) == HK_SUCCESS)
	{
		if(klass != existingClass)
		{
			hkStringBuf errorMsg; errorMsg.printf("Vtable already registered for %s", existingClass->getName());
			HK_ASSERT2(0xabcdabcd, false, errorMsg.cString());
		}
	}
#endif

	m_map.insert(HK_CAST_VTABLE(vtable), klass);
}

void hkVtableClassRegistry::registerList( const hkTypeInfo* const * infos, const hkClass* const * classes)
{
	const hkTypeInfo* const * ti = infos;
	const hkClass* const * ci = classes;
	while(*ti != HK_NULL && *ci != HK_NULL)
	{
		if( const void* vtable = (*ti)->getVtable() )
		{
			registerVtable( vtable, *ci );
		}
		++ti;
		++ci;
	}
}

void hkVtableClassRegistry::merge(const hkVtableClassRegistry& mergeFrom)
{
	#if (HK_LINKONCE_VTABLES==1) || (HK_HASHCODE_VTABLE_REGISTRY==1)
		hkPointerMap<const void*, const hkClass*>::Iterator iter = mergeFrom.m_map.getIterator();
	#else
		hkStringMap<const hkClass*>::Iterator iter = mergeFrom.m_map.getIterator();
	#endif
	while (mergeFrom.m_map.isValid(iter))
	{
		m_map.insert( mergeFrom.m_map.getKey(iter), mergeFrom.m_map.getValue(iter) );
		iter = mergeFrom.m_map.getNext(iter);
	}
}

HK_SINGLETON_IMPLEMENTATION(hkVtableClassRegistry);

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
