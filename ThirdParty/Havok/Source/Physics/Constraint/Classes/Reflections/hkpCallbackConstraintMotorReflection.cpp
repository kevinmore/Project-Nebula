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

// Generated from 'Physics/Constraint/Motor/Callback/hkpCallbackConstraintMotor.h'
#include <Common/Base/hkBase.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics/Constraint/Motor/Callback/hkpCallbackConstraintMotor.h>
#define True true
#define False false


// External pointer and enum types
extern const hkClassEnum* hkpCallbackConstraintMotorCallbackTypeEnum;

//
// Enum hkpCallbackConstraintMotor::CallbackType
//
static const hkInternalClassEnumItem hkpCallbackConstraintMotorCallbackTypeEnumItems[] =
{
	{0, "CALLBACK_MOTOR_TYPE_HAVOK_DEMO_SPRING_DAMPER"},
	{1, "CALLBACK_MOTOR_TYPE_USER_0"},
	{2, "CALLBACK_MOTOR_TYPE_USER_1"},
	{3, "CALLBACK_MOTOR_TYPE_USER_2"},
	{4, "CALLBACK_MOTOR_TYPE_USER_3"},
};
static const hkInternalClassEnum hkpCallbackConstraintMotorEnums[] = {
	{"CallbackType", hkpCallbackConstraintMotorCallbackTypeEnumItems, 5, HK_NULL, 0 }
};
const hkClassEnum* hkpCallbackConstraintMotorCallbackTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpCallbackConstraintMotorEnums[0]);

//
// Class hkpCallbackConstraintMotor
//
extern const hkClass hkpLimitedForceConstraintMotorClass;

static const hkInternalClassMember hkpCallbackConstraintMotorClass_Members[] =
{
	{ "callbackFunc", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkpCallbackConstraintMotor,m_callbackFunc), HK_NULL },
	{ "callbackType", HK_NULL, hkpCallbackConstraintMotorCallbackTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, HK_OFFSET_OF(hkpCallbackConstraintMotor,m_callbackType), HK_NULL },
	{ "userData0", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpCallbackConstraintMotor,m_userData0), HK_NULL },
	{ "userData1", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpCallbackConstraintMotor,m_userData1), HK_NULL },
	{ "userData2", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpCallbackConstraintMotor,m_userData2), HK_NULL }
};
namespace
{
	struct hkpCallbackConstraintMotor_DefaultStruct
	{
		int s_defaultOffsets[5];
		typedef hkInt8 _hkBool;
		typedef hkFloat32 _hkVector4[4];
		typedef hkReal _hkQuaternion[4];
		typedef hkReal _hkMatrix3[12];
		typedef hkReal _hkRotation[12];
		typedef hkReal _hkQsTransform[12];
		typedef hkReal _hkMatrix4[16];
		typedef hkReal _hkTransform[16];
	};
	const hkpCallbackConstraintMotor_DefaultStruct hkpCallbackConstraintMotor_Default =
	{
		{-1,-1,hkClassMember::HK_CLASS_ZERO_DEFAULT,hkClassMember::HK_CLASS_ZERO_DEFAULT,hkClassMember::HK_CLASS_ZERO_DEFAULT},

	};
}
extern const hkClass hkpCallbackConstraintMotorClass;
const hkClass hkpCallbackConstraintMotorClass(
	"hkpCallbackConstraintMotor",
	&hkpLimitedForceConstraintMotorClass, // parent
	sizeof(::hkpCallbackConstraintMotor),
	HK_NULL,
	0, // interfaces
	reinterpret_cast<const hkClassEnum*>(hkpCallbackConstraintMotorEnums),
	1, // enums
	reinterpret_cast<const hkClassMember*>(hkpCallbackConstraintMotorClass_Members),
	HK_COUNT_OF(hkpCallbackConstraintMotorClass_Members),
	&hkpCallbackConstraintMotor_Default,
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkpCallbackConstraintMotor::staticClass()
{
	return hkpCallbackConstraintMotorClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkpCallbackConstraintMotor*>(0))) == sizeof(hkBool::CompileTimeTrueType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthkpCallbackConstraintMotor(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hkpCallbackConstraintMotor(f);
}
static void HK_CALL cleanupLoadedObjecthkpCallbackConstraintMotor(void* p)
{
	static_cast<hkpCallbackConstraintMotor*>(p)->~hkpCallbackConstraintMotor();
}
static const void* HK_CALL getVtablehkpCallbackConstraintMotor()
{
	#if HK_LINKONCE_VTABLES==0
	#if HK_HASHCODE_VTABLE_REGISTRY==1
	return ((const void*)(typeid(hkpCallbackConstraintMotor).hash_code()));
	#else
	return ((const void*)(typeid(hkpCallbackConstraintMotor).name()));
	#endif
	#else
	union { HK_ALIGN16(void* ptr); char buf[sizeof(hkpCallbackConstraintMotor)]; } u;
	hkFinishLoadedObjectFlag f;
	new (u.buf) hkpCallbackConstraintMotor(f);
	return u.ptr;
	#endif
}
extern const hkTypeInfo hkpCallbackConstraintMotorTypeInfo;
const hkTypeInfo hkpCallbackConstraintMotorTypeInfo(
	"hkpCallbackConstraintMotor",
	"!hkpCallbackConstraintMotor",
	finishLoadedObjecthkpCallbackConstraintMotor,
	cleanupLoadedObjecthkpCallbackConstraintMotor,
	getVtablehkpCallbackConstraintMotor(),
	sizeof(hkpCallbackConstraintMotor)
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
