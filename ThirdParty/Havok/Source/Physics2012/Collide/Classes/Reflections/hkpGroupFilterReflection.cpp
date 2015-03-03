/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//HK_REFLECTION_PARSER_EXCLUDE_FILE

// Autogenerated by generateReflections.py (reflectedClasses.py)
// Changes will not be lost unless:
// - The workspace is re-generated using build.py
// - The corresponding reflection database (reflection.db) is deleted
// - The --force-output or --force-rebuild option is added to the pre-build generateReflection.py execution

// Generated from 'Physics2012/Collide/Filter/Group/hkpGroupFilter.h'
#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics2012/Collide/Filter/Group/hkpGroupFilter.h>
#define True true
#define False false


//
// Class hkpGroupFilter
//
extern const hkClass hkpCollisionFilterClass;

static const hkInternalClassMember hkpGroupFilterClass_Members[] =
{
	{ "nextFreeSystemGroup", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpGroupFilter,m_nextFreeSystemGroup), HK_NULL },
	{ "collisionLookupTable", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 32, 0, HK_OFFSET_OF(hkpGroupFilter,m_collisionLookupTable), HK_NULL },
	{ "pad256", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 4, 0, HK_OFFSET_OF(hkpGroupFilter,m_pad256), HK_NULL }
};
extern const hkClass hkpGroupFilterClass;
const hkClass hkpGroupFilterClass(
	"hkpGroupFilter",
	&hkpCollisionFilterClass, // parent
	sizeof(::hkpGroupFilter),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkpGroupFilterClass_Members),
	HK_COUNT_OF(hkpGroupFilterClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkpGroupFilter::staticClass()
{
	return hkpGroupFilterClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkpGroupFilter*>(0))) == sizeof(hkBool::CompileTimeTrueType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthkpGroupFilter(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hkpGroupFilter(f);
}
static void HK_CALL cleanupLoadedObjecthkpGroupFilter(void* p)
{
	static_cast<hkpGroupFilter*>(p)->~hkpGroupFilter();
}
static const void* HK_CALL getVtablehkpGroupFilter()
{
	#if HK_LINKONCE_VTABLES==0
	#if HK_HASHCODE_VTABLE_REGISTRY==1
	return ((const void*)(typeid(hkpGroupFilter).hash_code()));
	#else
	return ((const void*)(typeid(hkpGroupFilter).name()));
	#endif
	#else
	union { HK_ALIGN16(void* ptr); char buf[sizeof(hkpGroupFilter)]; } u;
	hkFinishLoadedObjectFlag f;
	new (u.buf) hkpGroupFilter(f);
	return u.ptr;
	#endif
}
extern const hkTypeInfo hkpGroupFilterTypeInfo;
const hkTypeInfo hkpGroupFilterTypeInfo(
	"hkpGroupFilter",
	"!hkpGroupFilter",
	finishLoadedObjecthkpGroupFilter,
	cleanupLoadedObjecthkpGroupFilter,
	getVtablehkpGroupFilter(),
	sizeof(hkpGroupFilter)
	);
#endif

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
