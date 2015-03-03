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

// Generated from 'Physics2012/Collide/Shape/Compound/Collection/Mesh/hkpNamedMeshMaterial.h'
#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics2012/Collide/Shape/Compound/Collection/Mesh/hkpNamedMeshMaterial.h>
#define True true
#define False false


//
// Class hkpNamedMeshMaterial
//
extern const hkClass hkpMeshMaterialClass;

static const hkInternalClassMember hkpNamedMeshMaterialClass_Members[] =
{
	{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_STRINGPTR, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpNamedMeshMaterial,m_name), HK_NULL }
};
extern const hkClass hkpNamedMeshMaterialClass;
const hkClass hkpNamedMeshMaterialClass(
	"hkpNamedMeshMaterial",
	&hkpMeshMaterialClass, // parent
	sizeof(::hkpNamedMeshMaterial),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkpNamedMeshMaterialClass_Members),
	HK_COUNT_OF(hkpNamedMeshMaterialClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkpNamedMeshMaterial::staticClass()
{
	return hkpNamedMeshMaterialClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkpNamedMeshMaterial*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthkpNamedMeshMaterial(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hkpNamedMeshMaterial(f);
}
static void HK_CALL cleanupLoadedObjecthkpNamedMeshMaterial(void* p)
{
	static_cast<hkpNamedMeshMaterial*>(p)->~hkpNamedMeshMaterial();
}
extern const hkTypeInfo hkpNamedMeshMaterialTypeInfo;
const hkTypeInfo hkpNamedMeshMaterialTypeInfo(
	"hkpNamedMeshMaterial",
	"!hkpNamedMeshMaterial",
	finishLoadedObjecthkpNamedMeshMaterial,
	cleanupLoadedObjecthkpNamedMeshMaterial,
	HK_NULL,
	sizeof(hkpNamedMeshMaterial)
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
