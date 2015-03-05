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

// Generated from 'Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h'
#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>
#define True true
#define False false


// External pointer and enum types
extern const hkClass hkpSimpleContactConstraintDataInfoClass;

//
// Class hkpSimpleContactConstraintAtom
//
extern const hkClass hkpConstraintAtomClass;

static const hkInternalClassMember hkpSimpleContactConstraintAtomClass_Members[] =
{
	{ "sizeOfAllAtoms", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpSimpleContactConstraintAtom,m_sizeOfAllAtoms), HK_NULL },
	{ "numContactPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpSimpleContactConstraintAtom,m_numContactPoints), HK_NULL },
	{ "numReservedContactPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpSimpleContactConstraintAtom,m_numReservedContactPoints), HK_NULL },
	{ "numUserDatasForBodyA", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpSimpleContactConstraintAtom,m_numUserDatasForBodyA), HK_NULL },
	{ "numUserDatasForBodyB", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpSimpleContactConstraintAtom,m_numUserDatasForBodyB), HK_NULL },
	{ "contactPointPropertiesStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpSimpleContactConstraintAtom,m_contactPointPropertiesStriding), HK_NULL },
	{ "maxNumContactPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpSimpleContactConstraintAtom,m_maxNumContactPoints), HK_NULL },
	{ "info", &hkpSimpleContactConstraintDataInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, HK_OFFSET_OF(hkpSimpleContactConstraintAtom,m_info), HK_NULL }
};
extern const hkClass hkpSimpleContactConstraintAtomClass;
const hkClass hkpSimpleContactConstraintAtomClass(
	"hkpSimpleContactConstraintAtom",
	&hkpConstraintAtomClass, // parent
	sizeof(::hkpSimpleContactConstraintAtom),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkpSimpleContactConstraintAtomClass_Members),
	HK_COUNT_OF(hkpSimpleContactConstraintAtomClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkpSimpleContactConstraintAtom::staticClass()
{
	return hkpSimpleContactConstraintAtomClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkpSimpleContactConstraintAtom*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthkpSimpleContactConstraintAtom(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hkpSimpleContactConstraintAtom(f);
}
static void HK_CALL cleanupLoadedObjecthkpSimpleContactConstraintAtom(void* p)
{
	static_cast<hkpSimpleContactConstraintAtom*>(p)->~hkpSimpleContactConstraintAtom();
}
extern const hkTypeInfo hkpSimpleContactConstraintAtomTypeInfo;
const hkTypeInfo hkpSimpleContactConstraintAtomTypeInfo(
	"hkpSimpleContactConstraintAtom",
	"!hkpSimpleContactConstraintAtom",
	finishLoadedObjecthkpSimpleContactConstraintAtom,
	cleanupLoadedObjecthkpSimpleContactConstraintAtom,
	HK_NULL,
	sizeof(hkpSimpleContactConstraintAtom)
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