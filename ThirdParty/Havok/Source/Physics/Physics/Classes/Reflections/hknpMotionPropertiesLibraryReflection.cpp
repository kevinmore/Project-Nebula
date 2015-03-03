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

// Generated from 'Physics/Physics/Dynamics/Motion/hknpMotionPropertiesLibrary.h'
#include <Physics/Physics/hknpPhysics.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionPropertiesLibrary.h>
#define True true
#define False false


// External pointer and enum types
extern const hkClass hknpMotionPropertiesLibraryMotionPropertiesAddedSignalClass;
extern const hkClass hknpMotionPropertiesLibraryMotionPropertiesModifiedSignalClass;
extern const hkClass hknpMotionPropertiesLibraryMotionPropertiesRemovedSignalClass;

//
// Class hknpMotionPropertiesLibrary
//
extern const hkClass hkReferencedObjectClass;

extern const hkClass hkFreeListArrayhknpMotionPropertieshknpMotionPropertiesId8hknpMotionPropertiesFreeListArrayOperationsClass;

const hkInternalClassMember hknpMotionPropertiesLibrary::Members[] =
{
	{ "entryAddedSignal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hknpMotionPropertiesLibrary,m_entryAddedSignal), HK_NULL },
	{ "entryModifiedSignal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hknpMotionPropertiesLibrary,m_entryModifiedSignal), HK_NULL },
	{ "entryRemovedSignal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hknpMotionPropertiesLibrary,m_entryRemovedSignal), HK_NULL },
	{ "entries", &hkFreeListArrayhknpMotionPropertieshknpMotionPropertiesId8hknpMotionPropertiesFreeListArrayOperationsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hknpMotionPropertiesLibrary,m_entries), HK_NULL }
};
extern const hkClass hknpMotionPropertiesLibraryClass;
const hkClass hknpMotionPropertiesLibraryClass(
	"hknpMotionPropertiesLibrary",
	&hkReferencedObjectClass, // parent
	sizeof(::hknpMotionPropertiesLibrary),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hknpMotionPropertiesLibrary::Members),
	HK_COUNT_OF(hknpMotionPropertiesLibrary::Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hknpMotionPropertiesLibrary::staticClass()
{
	return hknpMotionPropertiesLibraryClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hknpMotionPropertiesLibrary*>(0))) == sizeof(hkBool::CompileTimeTrueType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthknpMotionPropertiesLibrary(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hknpMotionPropertiesLibrary(f);
}
static void HK_CALL cleanupLoadedObjecthknpMotionPropertiesLibrary(void* p)
{
	static_cast<hknpMotionPropertiesLibrary*>(p)->~hknpMotionPropertiesLibrary();
}
static const void* HK_CALL getVtablehknpMotionPropertiesLibrary()
{
	#if HK_LINKONCE_VTABLES==0
	#if HK_HASHCODE_VTABLE_REGISTRY==1
	return ((const void*)(typeid(hknpMotionPropertiesLibrary).hash_code()));
	#else
	return ((const void*)(typeid(hknpMotionPropertiesLibrary).name()));
	#endif
	#else
	union { HK_ALIGN16(void* ptr); char buf[sizeof(hknpMotionPropertiesLibrary)]; } u;
	hkFinishLoadedObjectFlag f;
	new (u.buf) hknpMotionPropertiesLibrary(f);
	return u.ptr;
	#endif
}
extern const hkTypeInfo hknpMotionPropertiesLibraryTypeInfo;
const hkTypeInfo hknpMotionPropertiesLibraryTypeInfo(
	"hknpMotionPropertiesLibrary",
	"!hknpMotionPropertiesLibrary",
	finishLoadedObjecthknpMotionPropertiesLibrary,
	cleanupLoadedObjecthknpMotionPropertiesLibrary,
	getVtablehknpMotionPropertiesLibrary(),
	sizeof(hknpMotionPropertiesLibrary)
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
