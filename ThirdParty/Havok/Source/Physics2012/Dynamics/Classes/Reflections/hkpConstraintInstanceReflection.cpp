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

// Generated from 'Physics2012/Dynamics/Constraint/hkpConstraintInstance.h'
#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>
#define True true
#define False false


// External pointer and enum types
extern const hkClass hkConstraintInternalClass;
extern const hkClass hkpConstraintAtomClass;
extern const hkClass hkpConstraintDataClass;
extern const hkClass hkpConstraintInstanceClass;
extern const hkClass hkpConstraintOwnerClass;
extern const hkClass hkpConstraintRuntimeClass;
extern const hkClass hkpEntityClass;
extern const hkClass hkpModifierConstraintAtomClass;
extern const hkClassEnum* hkpConstraintInstanceConstraintPriorityEnum;
extern const hkClassEnum* hkpConstraintInstanceInstanceTypeEnum;
extern const hkClassEnum* hkpConstraintInstanceOnDestructionRemapInfoEnum;

//
// Enum hkpConstraintInstance::ConstraintPriority
//
static const hkInternalClassEnumItem hkpConstraintInstanceConstraintPriorityEnumItems[] =
{
	{0, "PRIORITY_INVALID"},
	{1, "PRIORITY_PSI"},
	{2, "PRIORITY_SIMPLIFIED_TOI_UNUSED"},
	{3, "PRIORITY_TOI"},
	{4, "PRIORITY_TOI_HIGHER"},
	{5, "PRIORITY_TOI_FORCED"},
	{6, "NUM_PRIORITIES"},
};

//
// Enum hkpConstraintInstance::InstanceType
//
static const hkInternalClassEnumItem hkpConstraintInstanceInstanceTypeEnumItems[] =
{
	{0, "TYPE_NORMAL"},
	{1, "TYPE_CHAIN"},
	{2, "TYPE_DISABLE_SPU"},
};

//
// Enum hkpConstraintInstance::AddReferences
//
static const hkInternalClassEnumItem hkpConstraintInstanceAddReferencesEnumItems[] =
{
	{0, "DO_NOT_ADD_REFERENCES"},
	{1, "DO_ADD_REFERENCES"},
};

//
// Enum hkpConstraintInstance::CloningMode
//
static const hkInternalClassEnumItem hkpConstraintInstanceCloningModeEnumItems[] =
{
	{0, "CLONE_SHALLOW_IF_NOT_CONSTRAINED_TO_WORLD"},
	{1, "CLONE_DATAS_WITH_MOTORS"},
	{2, "CLONE_FORCE_SHALLOW"},
};

//
// Enum hkpConstraintInstance::OnDestructionRemapInfo
//
static const hkInternalClassEnumItem hkpConstraintInstanceOnDestructionRemapInfoEnumItems[] =
{
	{0, "ON_DESTRUCTION_REMAP"},
	{1, "ON_DESTRUCTION_REMOVE"},
	{2, "ON_DESTRUCTION_RESET_REMOVE"},
};
static const hkInternalClassEnum hkpConstraintInstanceEnums[] = {
	{"ConstraintPriority", hkpConstraintInstanceConstraintPriorityEnumItems, 7, HK_NULL, 0 },
	{"InstanceType", hkpConstraintInstanceInstanceTypeEnumItems, 3, HK_NULL, 0 },
	{"AddReferences", hkpConstraintInstanceAddReferencesEnumItems, 2, HK_NULL, 0 },
	{"CloningMode", hkpConstraintInstanceCloningModeEnumItems, 3, HK_NULL, 0 },
	{"OnDestructionRemapInfo", hkpConstraintInstanceOnDestructionRemapInfoEnumItems, 3, HK_NULL, 0 }
};
const hkClassEnum* hkpConstraintInstanceConstraintPriorityEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintInstanceEnums[0]);
const hkClassEnum* hkpConstraintInstanceInstanceTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintInstanceEnums[1]);
const hkClassEnum* hkpConstraintInstanceAddReferencesEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintInstanceEnums[2]);
const hkClassEnum* hkpConstraintInstanceCloningModeEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintInstanceEnums[3]);
const hkClassEnum* hkpConstraintInstanceOnDestructionRemapInfoEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintInstanceEnums[4]);

//
// Class hkpConstraintInstance::SmallArraySerializeOverrideType
//
static const hkInternalClassMember hkpConstraintInstance_SmallArraySerializeOverrideTypeClass_Members[] =
{
	{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkpConstraintInstance::SmallArraySerializeOverrideType,m_data), HK_NULL },
	{ "size", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpConstraintInstance::SmallArraySerializeOverrideType,m_size), HK_NULL },
	{ "capacityAndFlags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpConstraintInstance::SmallArraySerializeOverrideType,m_capacityAndFlags), HK_NULL }
};
extern const hkClass hkpConstraintInstanceSmallArraySerializeOverrideTypeClass;
const hkClass hkpConstraintInstanceSmallArraySerializeOverrideTypeClass(
	"hkpConstraintInstanceSmallArraySerializeOverrideType",
	HK_NULL, // parent
	sizeof(hkpConstraintInstance::SmallArraySerializeOverrideType),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkpConstraintInstance_SmallArraySerializeOverrideTypeClass_Members),
	HK_COUNT_OF(hkpConstraintInstance_SmallArraySerializeOverrideTypeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(1) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkpConstraintInstance::SmallArraySerializeOverrideType::staticClass()
{
	return hkpConstraintInstanceSmallArraySerializeOverrideTypeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkpConstraintInstance::SmallArraySerializeOverrideType*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkpConstraintInstanceSmallArraySerializeOverrideType(void* p)
{
	static_cast<hkpConstraintInstance::SmallArraySerializeOverrideType*>(p)->~SmallArraySerializeOverrideType();
}
extern const hkTypeInfo hkpConstraintInstanceSmallArraySerializeOverrideTypeTypeInfo;
const hkTypeInfo hkpConstraintInstanceSmallArraySerializeOverrideTypeTypeInfo(
	"hkpConstraintInstanceSmallArraySerializeOverrideType",
	"!hkpConstraintInstance::SmallArraySerializeOverrideType",
	HK_NULL,
	cleanupLoadedObjecthkpConstraintInstanceSmallArraySerializeOverrideType,
	HK_NULL,
	sizeof(hkpConstraintInstance::SmallArraySerializeOverrideType)
	);
#endif

//
// Class hkpConstraintInstance
//
extern const hkClass hkReferencedObjectClass;

const hkInternalClassMember hkpConstraintInstance::Members[] =
{
	{ "owner", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkpConstraintInstance,m_owner), HK_NULL },
	{ "data", &hkpConstraintDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkpConstraintInstance,m_data), HK_NULL },
	{ "constraintModifiers", &hkpModifierConstraintAtomClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkpConstraintInstance,m_constraintModifiers), HK_NULL },
	{ "entities", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 2, 0, HK_OFFSET_OF(hkpConstraintInstance,m_entities), HK_NULL },
	{ "priority", HK_NULL, hkpConstraintInstanceConstraintPriorityEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, HK_OFFSET_OF(hkpConstraintInstance,m_priority), HK_NULL },
	{ "wantRuntime", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpConstraintInstance,m_wantRuntime), HK_NULL },
	{ "destructionRemapInfo", HK_NULL, hkpConstraintInstanceOnDestructionRemapInfoEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, HK_OFFSET_OF(hkpConstraintInstance,m_destructionRemapInfo), HK_NULL },
	{ "listeners", &hkpConstraintInstanceSmallArraySerializeOverrideTypeClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkpConstraintInstance,m_listeners), HK_NULL },
	{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_STRINGPTR, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpConstraintInstance,m_name), HK_NULL },
	{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpConstraintInstance,m_userData), HK_NULL },
	{ "internal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkpConstraintInstance,m_internal), HK_NULL },
	{ "uid", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkpConstraintInstance,m_uid), HK_NULL }
};
namespace
{
	struct hkpConstraintInstance_DefaultStruct
	{
		int s_defaultOffsets[12];
		typedef hkInt8 _hkBool;
		typedef hkFloat32 _hkVector4[4];
		typedef hkReal _hkQuaternion[4];
		typedef hkReal _hkMatrix3[12];
		typedef hkReal _hkRotation[12];
		typedef hkReal _hkQsTransform[12];
		typedef hkReal _hkMatrix4[16];
		typedef hkReal _hkTransform[16];
	};
	const hkpConstraintInstance_DefaultStruct hkpConstraintInstance_Default =
	{
		{-1,-1,-1,-1,-1,-1,-1,-1,-1,hkClassMember::HK_CLASS_ZERO_DEFAULT,-1,-1},
		
	};
}
const hkClass hkpConstraintInstanceClass(
	"hkpConstraintInstance",
	&hkReferencedObjectClass, // parent
	sizeof(::hkpConstraintInstance),
	HK_NULL,
	0, // interfaces
	reinterpret_cast<const hkClassEnum*>(hkpConstraintInstanceEnums),
	5, // enums
	reinterpret_cast<const hkClassMember*>(hkpConstraintInstance::Members),
	HK_COUNT_OF(hkpConstraintInstance::Members),
	&hkpConstraintInstance_Default,
	HK_NULL, // attributes
	0, // flags
	hkUint32(1) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkpConstraintInstance::staticClass()
{
	return hkpConstraintInstanceClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkpConstraintInstance*>(0))) == sizeof(hkBool::CompileTimeTrueType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthkpConstraintInstance(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hkpConstraintInstance(f);
}
static void HK_CALL cleanupLoadedObjecthkpConstraintInstance(void* p)
{
	static_cast<hkpConstraintInstance*>(p)->~hkpConstraintInstance();
}
static const void* HK_CALL getVtablehkpConstraintInstance()
{
	#if HK_LINKONCE_VTABLES==0
	#if HK_HASHCODE_VTABLE_REGISTRY==1
	return ((const void*)(typeid(hkpConstraintInstance).hash_code()));
	#else
	return ((const void*)(typeid(hkpConstraintInstance).name()));
	#endif
	#else
	union { HK_ALIGN16(void* ptr); char buf[sizeof(hkpConstraintInstance)]; } u;
	hkFinishLoadedObjectFlag f;
	new (u.buf) hkpConstraintInstance(f);
	return u.ptr;
	#endif
}
extern const hkTypeInfo hkpConstraintInstanceTypeInfo;
const hkTypeInfo hkpConstraintInstanceTypeInfo(
	"hkpConstraintInstance",
	"!hkpConstraintInstance",
	finishLoadedObjecthkpConstraintInstance,
	cleanupLoadedObjecthkpConstraintInstance,
	getVtablehkpConstraintInstance(),
	sizeof(hkpConstraintInstance)
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
