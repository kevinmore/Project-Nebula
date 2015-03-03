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

// Generated from 'Physics/Physics/Dynamics/Material/hknpMaterialLibrary.h'
#include <Physics/Physics/hknpPhysics.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterialLibrary.h>
#define True true
#define False false


// External pointer and enum types
extern const hkClass hknpMaterialLibraryMaterialAddedSignalClass;
extern const hkClass hknpMaterialLibraryMaterialModifiedSignalClass;
extern const hkClass hknpMaterialLibraryMaterialRemovedSignalClass;

//
// Class hknpMaterialLibrary
//
extern const hkClass hkReferencedObjectClass;

extern const hkClass hkFreeListArrayhknpMaterialhknpMaterialId8hknpMaterialFreeListArrayOperationsClass;

const hkInternalClassMember hknpMaterialLibrary::Members[] =
{
	{ "materialAddedSignal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hknpMaterialLibrary,m_materialAddedSignal), HK_NULL },
	{ "materialModifiedSignal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hknpMaterialLibrary,m_materialModifiedSignal), HK_NULL },
	{ "materialRemovedSignal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hknpMaterialLibrary,m_materialRemovedSignal), HK_NULL },
	{ "entries", &hkFreeListArrayhknpMaterialhknpMaterialId8hknpMaterialFreeListArrayOperationsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hknpMaterialLibrary,m_entries), HK_NULL }
};
extern const hkClass hknpMaterialLibraryClass;
const hkClass hknpMaterialLibraryClass(
	"hknpMaterialLibrary",
	&hkReferencedObjectClass, // parent
	sizeof(::hknpMaterialLibrary),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hknpMaterialLibrary::Members),
	HK_COUNT_OF(hknpMaterialLibrary::Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hknpMaterialLibrary::staticClass()
{
	return hknpMaterialLibraryClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hknpMaterialLibrary*>(0))) == sizeof(hkBool::CompileTimeTrueType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthknpMaterialLibrary(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hknpMaterialLibrary(f);
}
static void HK_CALL cleanupLoadedObjecthknpMaterialLibrary(void* p)
{
	static_cast<hknpMaterialLibrary*>(p)->~hknpMaterialLibrary();
}
static const void* HK_CALL getVtablehknpMaterialLibrary()
{
	#if HK_LINKONCE_VTABLES==0
	#if HK_HASHCODE_VTABLE_REGISTRY==1
	return ((const void*)(typeid(hknpMaterialLibrary).hash_code()));
	#else
	return ((const void*)(typeid(hknpMaterialLibrary).name()));
	#endif
	#else
	union { HK_ALIGN16(void* ptr); char buf[sizeof(hknpMaterialLibrary)]; } u;
	hkFinishLoadedObjectFlag f;
	new (u.buf) hknpMaterialLibrary(f);
	return u.ptr;
	#endif
}
extern const hkTypeInfo hknpMaterialLibraryTypeInfo;
const hkTypeInfo hknpMaterialLibraryTypeInfo(
	"hknpMaterialLibrary",
	"!hknpMaterialLibrary",
	finishLoadedObjecthknpMaterialLibrary,
	cleanupLoadedObjecthknpMaterialLibrary,
	getVtablehknpMaterialLibrary(),
	sizeof(hknpMaterialLibrary)
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
