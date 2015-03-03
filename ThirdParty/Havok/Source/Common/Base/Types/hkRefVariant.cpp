/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkRefVariant.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>

#if defined(HK_DEBUG)
static void checkObject(const void* o, const hkClass* k)
{
	if( !o )
	{
		return;
	}
	HK_ON_DEBUG(const hkVtableClassRegistry& vtable = hkVtableClassRegistry::getInstance());
	HK_ON_DEBUG(const hkClass* realClass = vtable.getClassFromVirtualInstance(o));
	HK_ASSERT3(0x204578f6, realClass,
		"Object at 0x" << o << " is not reference counted or '" << k->getName() << "' is not registered.\n"
		"hkRefVariant can be used with reference counted objects only.");
	HK_ASSERT2(0x4183e922, k, "Cannot set variant with hkClass pointer HK_NULL. Object must be HK_NULL too.");
	HK_ASSERT3(0x28cc2069, k->isSuperClass(*realClass),
		"The given class '" << k->getName() << "' is different from object's class '" << realClass->getName() << "'.");
}
#endif //defined(HK_DEBUG)

hkRefVariant::hkRefVariant(void* o, const hkClass* k)
: hkRefPtr<hkReferencedObject>(static_cast<hkReferencedObject*>(o))
{
	HK_ON_DEBUG(checkObject(o, k));
}

hkRefVariant::hkRefVariant(const hkVariant& v)
: hkRefPtr<hkReferencedObject>(static_cast<hkReferencedObject*>(v.m_object))
{
	HK_ON_DEBUG(checkObject(v.m_object, v.m_class));
}

void hkRefVariant::set(void* o, const hkClass* k)
{
	HK_ON_DEBUG(checkObject(o, k));
	hkRefPtr<hkReferencedObject>::operator=(static_cast<hkReferencedObject*>(o));
}

const hkClass* hkRefVariant::getClass() const
{
	if( HK_NULL != val() )
	{
		const hkVtableClassRegistry& vtable = hkVtableClassRegistry::getInstance();
		const hkClass* realClass = vtable.getClassFromVirtualInstance(val());
		HK_ON_DEBUG(checkObject(val(), realClass));
		return realClass;
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
