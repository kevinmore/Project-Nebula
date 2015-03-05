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

// Generated from 'Common/Base/Reflection/Attributes/hkAttributes.h'
#include <Common/Base/hkBase.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#define True true
#define False false


// External pointer and enum types
extern const hkClassEnum* hkArrayTypeAttributeArrayTypeEnum;
extern const hkClassEnum* hkGizmoAttributeGizmoTypeEnum;
extern const hkClassEnum* hkLinkAttributeLinkEnum;
extern const hkClassEnum* hkModelerNodeTypeAttributeModelerTypeEnum;
extern const hkClassEnum* hkSemanticsAttributeSemanticsEnum;
extern const hkClassEnum* hkUiAttributeHideInModelerEnum;

//
// Class hkRangeRealAttribute
//
static const hkInternalClassMember hkRangeRealAttributeClass_Members[] =
{
	{ "absmin", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkRangeRealAttribute,m_absmin), HK_NULL },
	{ "absmax", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkRangeRealAttribute,m_absmax), HK_NULL },
	{ "softmin", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkRangeRealAttribute,m_softmin), HK_NULL },
	{ "softmax", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkRangeRealAttribute,m_softmax), HK_NULL }
};
extern const hkClass hkRangeRealAttributeClass;
const hkClass hkRangeRealAttributeClass(
	"hkRangeRealAttribute",
	HK_NULL, // parent
	sizeof(::hkRangeRealAttribute),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkRangeRealAttributeClass_Members),
	HK_COUNT_OF(hkRangeRealAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkRangeRealAttribute::staticClass()
{
	return hkRangeRealAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkRangeRealAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkRangeRealAttribute(void* p)
{
	static_cast<hkRangeRealAttribute*>(p)->~hkRangeRealAttribute();
}
extern const hkTypeInfo hkRangeRealAttributeTypeInfo;
const hkTypeInfo hkRangeRealAttributeTypeInfo(
	"hkRangeRealAttribute",
	"!hkRangeRealAttribute",
	HK_NULL,
	cleanupLoadedObjecthkRangeRealAttribute,
	HK_NULL,
	sizeof(hkRangeRealAttribute)
	);
#endif

//
// Class hkRangeInt32Attribute
//
static const hkInternalClassMember hkRangeInt32AttributeClass_Members[] =
{
	{ "absmin", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkRangeInt32Attribute,m_absmin), HK_NULL },
	{ "absmax", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkRangeInt32Attribute,m_absmax), HK_NULL },
	{ "softmin", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkRangeInt32Attribute,m_softmin), HK_NULL },
	{ "softmax", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkRangeInt32Attribute,m_softmax), HK_NULL }
};
extern const hkClass hkRangeInt32AttributeClass;
const hkClass hkRangeInt32AttributeClass(
	"hkRangeInt32Attribute",
	HK_NULL, // parent
	sizeof(::hkRangeInt32Attribute),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkRangeInt32AttributeClass_Members),
	HK_COUNT_OF(hkRangeInt32AttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkRangeInt32Attribute::staticClass()
{
	return hkRangeInt32AttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkRangeInt32Attribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkRangeInt32Attribute(void* p)
{
	static_cast<hkRangeInt32Attribute*>(p)->~hkRangeInt32Attribute();
}
extern const hkTypeInfo hkRangeInt32AttributeTypeInfo;
const hkTypeInfo hkRangeInt32AttributeTypeInfo(
	"hkRangeInt32Attribute",
	"!hkRangeInt32Attribute",
	HK_NULL,
	cleanupLoadedObjecthkRangeInt32Attribute,
	HK_NULL,
	sizeof(hkRangeInt32Attribute)
	);
#endif

//
// Enum hkUiAttribute::HideInModeler
//
static const hkInternalClassEnumItem hkUiAttributeHideInModelerEnumItems[] =
{
	{0, "NONE"},
	{1, "MAX"},
	{2, "MAYA"},
};
static const hkInternalClassEnum hkUiAttributeEnums[] = {
	{"HideInModeler", hkUiAttributeHideInModelerEnumItems, 3, HK_NULL, 0 }
};
const hkClassEnum* hkUiAttributeHideInModelerEnum = reinterpret_cast<const hkClassEnum*>(&hkUiAttributeEnums[0]);

//
// Class hkUiAttribute
//
static const hkInternalClassMember hkUiAttributeClass_Members[] =
{
	{ "visible", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkUiAttribute,m_visible), HK_NULL },
	{ "hideInModeler", HK_NULL, hkUiAttributeHideInModelerEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, HK_OFFSET_OF(hkUiAttribute,m_hideInModeler), HK_NULL },
	{ "label", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkUiAttribute,m_label), HK_NULL },
	{ "group", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkUiAttribute,m_group), HK_NULL },
	{ "hideBaseClassMembers", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkUiAttribute,m_hideBaseClassMembers), HK_NULL },
	{ "endGroup", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkUiAttribute,m_endGroup), HK_NULL },
	{ "endGroup2", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkUiAttribute,m_endGroup2), HK_NULL },
	{ "advanced", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkUiAttribute,m_advanced), HK_NULL }
};
extern const hkClass hkUiAttributeClass;
const hkClass hkUiAttributeClass(
	"hkUiAttribute",
	HK_NULL, // parent
	sizeof(::hkUiAttribute),
	HK_NULL,
	0, // interfaces
	reinterpret_cast<const hkClassEnum*>(hkUiAttributeEnums),
	1, // enums
	reinterpret_cast<const hkClassMember*>(hkUiAttributeClass_Members),
	HK_COUNT_OF(hkUiAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(2) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkUiAttribute::staticClass()
{
	return hkUiAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkUiAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkUiAttribute(void* p)
{
	static_cast<hkUiAttribute*>(p)->~hkUiAttribute();
}
extern const hkTypeInfo hkUiAttributeTypeInfo;
const hkTypeInfo hkUiAttributeTypeInfo(
	"hkUiAttribute",
	"!hkUiAttribute",
	HK_NULL,
	cleanupLoadedObjecthkUiAttribute,
	HK_NULL,
	sizeof(hkUiAttribute)
	);
#endif

//
// Enum hkGizmoAttribute::GizmoType
//
static const hkInternalClassEnumItem hkGizmoAttributeGizmoTypeEnumItems[] =
{
	{0, "POINT"},
	{1, "SPHERE"},
	{2, "PLANE"},
	{3, "ARROW"},
};
static const hkInternalClassEnum hkGizmoAttributeEnums[] = {
	{"GizmoType", hkGizmoAttributeGizmoTypeEnumItems, 4, HK_NULL, 0 }
};
const hkClassEnum* hkGizmoAttributeGizmoTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkGizmoAttributeEnums[0]);

//
// Class hkGizmoAttribute
//
static const hkInternalClassMember hkGizmoAttributeClass_Members[] =
{
	{ "visible", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkGizmoAttribute,m_visible), HK_NULL },
	{ "label", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkGizmoAttribute,m_label), HK_NULL },
	{ "type", HK_NULL, hkGizmoAttributeGizmoTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, HK_OFFSET_OF(hkGizmoAttribute,m_type), HK_NULL }
};
extern const hkClass hkGizmoAttributeClass;
const hkClass hkGizmoAttributeClass(
	"hkGizmoAttribute",
	HK_NULL, // parent
	sizeof(::hkGizmoAttribute),
	HK_NULL,
	0, // interfaces
	reinterpret_cast<const hkClassEnum*>(hkGizmoAttributeEnums),
	1, // enums
	reinterpret_cast<const hkClassMember*>(hkGizmoAttributeClass_Members),
	HK_COUNT_OF(hkGizmoAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkGizmoAttribute::staticClass()
{
	return hkGizmoAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkGizmoAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkGizmoAttribute(void* p)
{
	static_cast<hkGizmoAttribute*>(p)->~hkGizmoAttribute();
}
extern const hkTypeInfo hkGizmoAttributeTypeInfo;
const hkTypeInfo hkGizmoAttributeTypeInfo(
	"hkGizmoAttribute",
	"!hkGizmoAttribute",
	HK_NULL,
	cleanupLoadedObjecthkGizmoAttribute,
	HK_NULL,
	sizeof(hkGizmoAttribute)
	);
#endif

//
// Enum hkModelerNodeTypeAttribute::ModelerType
//
static const hkInternalClassEnumItem hkModelerNodeTypeAttributeModelerTypeEnumItems[] =
{
	{0, "DEFAULT"},
	{1, "LOCATOR"},
};
static const hkInternalClassEnum hkModelerNodeTypeAttributeEnums[] = {
	{"ModelerType", hkModelerNodeTypeAttributeModelerTypeEnumItems, 2, HK_NULL, 0 }
};
const hkClassEnum* hkModelerNodeTypeAttributeModelerTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkModelerNodeTypeAttributeEnums[0]);

//
// Class hkModelerNodeTypeAttribute
//
static const hkInternalClassMember hkModelerNodeTypeAttributeClass_Members[] =
{
	{ "type", HK_NULL, hkModelerNodeTypeAttributeModelerTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, HK_OFFSET_OF(hkModelerNodeTypeAttribute,m_type), HK_NULL }
};
extern const hkClass hkModelerNodeTypeAttributeClass;
const hkClass hkModelerNodeTypeAttributeClass(
	"hkModelerNodeTypeAttribute",
	HK_NULL, // parent
	sizeof(::hkModelerNodeTypeAttribute),
	HK_NULL,
	0, // interfaces
	reinterpret_cast<const hkClassEnum*>(hkModelerNodeTypeAttributeEnums),
	1, // enums
	reinterpret_cast<const hkClassMember*>(hkModelerNodeTypeAttributeClass_Members),
	HK_COUNT_OF(hkModelerNodeTypeAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkModelerNodeTypeAttribute::staticClass()
{
	return hkModelerNodeTypeAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkModelerNodeTypeAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkModelerNodeTypeAttribute(void* p)
{
	static_cast<hkModelerNodeTypeAttribute*>(p)->~hkModelerNodeTypeAttribute();
}
extern const hkTypeInfo hkModelerNodeTypeAttributeTypeInfo;
const hkTypeInfo hkModelerNodeTypeAttributeTypeInfo(
	"hkModelerNodeTypeAttribute",
	"!hkModelerNodeTypeAttribute",
	HK_NULL,
	cleanupLoadedObjecthkModelerNodeTypeAttribute,
	HK_NULL,
	sizeof(hkModelerNodeTypeAttribute)
	);
#endif

//
// Enum hkLinkAttribute::Link
//
static const hkInternalClassEnumItem hkLinkAttributeLinkEnumItems[] =
{
	{0, "NONE"},
	{1, "DIRECT_LINK"},
	{2, "CHILD"},
	{3, "MESH"},
	{4, "PARENT_NAME"},
};
static const hkInternalClassEnum hkLinkAttributeEnums[] = {
	{"Link", hkLinkAttributeLinkEnumItems, 5, HK_NULL, 0 }
};
const hkClassEnum* hkLinkAttributeLinkEnum = reinterpret_cast<const hkClassEnum*>(&hkLinkAttributeEnums[0]);

//
// Class hkLinkAttribute
//
static const hkInternalClassMember hkLinkAttributeClass_Members[] =
{
	{ "type", HK_NULL, hkLinkAttributeLinkEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, HK_OFFSET_OF(hkLinkAttribute,m_type), HK_NULL }
};
extern const hkClass hkLinkAttributeClass;
const hkClass hkLinkAttributeClass(
	"hkLinkAttribute",
	HK_NULL, // parent
	sizeof(::hkLinkAttribute),
	HK_NULL,
	0, // interfaces
	reinterpret_cast<const hkClassEnum*>(hkLinkAttributeEnums),
	1, // enums
	reinterpret_cast<const hkClassMember*>(hkLinkAttributeClass_Members),
	HK_COUNT_OF(hkLinkAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkLinkAttribute::staticClass()
{
	return hkLinkAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkLinkAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkLinkAttribute(void* p)
{
	static_cast<hkLinkAttribute*>(p)->~hkLinkAttribute();
}
extern const hkTypeInfo hkLinkAttributeTypeInfo;
const hkTypeInfo hkLinkAttributeTypeInfo(
	"hkLinkAttribute",
	"!hkLinkAttribute",
	HK_NULL,
	cleanupLoadedObjecthkLinkAttribute,
	HK_NULL,
	sizeof(hkLinkAttribute)
	);
#endif

//
// Enum hkSemanticsAttribute::Semantics
//
static const hkInternalClassEnumItem hkSemanticsAttributeSemanticsEnumItems[] =
{
	{0, "UNKNOWN"},
	{1, "DISTANCE"},
	{2, "ANGLE"},
	{3, "NORMAL"},
	{4, "POSITION"},
	{5, "COSINE_ANGLE"},
};
static const hkInternalClassEnum hkSemanticsAttributeEnums[] = {
	{"Semantics", hkSemanticsAttributeSemanticsEnumItems, 6, HK_NULL, 0 }
};
const hkClassEnum* hkSemanticsAttributeSemanticsEnum = reinterpret_cast<const hkClassEnum*>(&hkSemanticsAttributeEnums[0]);

//
// Class hkSemanticsAttribute
//
static const hkInternalClassMember hkSemanticsAttributeClass_Members[] =
{
	{ "type", HK_NULL, hkSemanticsAttributeSemanticsEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, HK_OFFSET_OF(hkSemanticsAttribute,m_type), HK_NULL }
};
extern const hkClass hkSemanticsAttributeClass;
const hkClass hkSemanticsAttributeClass(
	"hkSemanticsAttribute",
	HK_NULL, // parent
	sizeof(::hkSemanticsAttribute),
	HK_NULL,
	0, // interfaces
	reinterpret_cast<const hkClassEnum*>(hkSemanticsAttributeEnums),
	1, // enums
	reinterpret_cast<const hkClassMember*>(hkSemanticsAttributeClass_Members),
	HK_COUNT_OF(hkSemanticsAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkSemanticsAttribute::staticClass()
{
	return hkSemanticsAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkSemanticsAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkSemanticsAttribute(void* p)
{
	static_cast<hkSemanticsAttribute*>(p)->~hkSemanticsAttribute();
}
extern const hkTypeInfo hkSemanticsAttributeTypeInfo;
const hkTypeInfo hkSemanticsAttributeTypeInfo(
	"hkSemanticsAttribute",
	"!hkSemanticsAttribute",
	HK_NULL,
	cleanupLoadedObjecthkSemanticsAttribute,
	HK_NULL,
	sizeof(hkSemanticsAttribute)
	);
#endif

//
// Class hkDescriptionAttribute
//
static const hkInternalClassMember hkDescriptionAttributeClass_Members[] =
{
	{ "string", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkDescriptionAttribute,m_string), HK_NULL }
};
extern const hkClass hkDescriptionAttributeClass;
const hkClass hkDescriptionAttributeClass(
	"hkDescriptionAttribute",
	HK_NULL, // parent
	sizeof(::hkDescriptionAttribute),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkDescriptionAttributeClass_Members),
	HK_COUNT_OF(hkDescriptionAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkDescriptionAttribute::staticClass()
{
	return hkDescriptionAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkDescriptionAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkDescriptionAttribute(void* p)
{
	static_cast<hkDescriptionAttribute*>(p)->~hkDescriptionAttribute();
}
extern const hkTypeInfo hkDescriptionAttributeTypeInfo;
const hkTypeInfo hkDescriptionAttributeTypeInfo(
	"hkDescriptionAttribute",
	"!hkDescriptionAttribute",
	HK_NULL,
	cleanupLoadedObjecthkDescriptionAttribute,
	HK_NULL,
	sizeof(hkDescriptionAttribute)
	);
#endif

//
// Enum hkArrayTypeAttribute::ArrayType
//
static const hkInternalClassEnumItem hkArrayTypeAttributeArrayTypeEnumItems[] =
{
	{0, "NONE"},
	{1, "POINTSOUP"},
	{2, "ENTITIES"},
};
static const hkInternalClassEnum hkArrayTypeAttributeEnums[] = {
	{"ArrayType", hkArrayTypeAttributeArrayTypeEnumItems, 3, HK_NULL, 0 }
};
const hkClassEnum* hkArrayTypeAttributeArrayTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkArrayTypeAttributeEnums[0]);

//
// Class hkArrayTypeAttribute
//
static const hkInternalClassMember hkArrayTypeAttributeClass_Members[] =
{
	{ "type", HK_NULL, hkArrayTypeAttributeArrayTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, HK_OFFSET_OF(hkArrayTypeAttribute,m_type), HK_NULL }
};
extern const hkClass hkArrayTypeAttributeClass;
const hkClass hkArrayTypeAttributeClass(
	"hkArrayTypeAttribute",
	HK_NULL, // parent
	sizeof(::hkArrayTypeAttribute),
	HK_NULL,
	0, // interfaces
	reinterpret_cast<const hkClassEnum*>(hkArrayTypeAttributeEnums),
	1, // enums
	reinterpret_cast<const hkClassMember*>(hkArrayTypeAttributeClass_Members),
	HK_COUNT_OF(hkArrayTypeAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkArrayTypeAttribute::staticClass()
{
	return hkArrayTypeAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkArrayTypeAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkArrayTypeAttribute(void* p)
{
	static_cast<hkArrayTypeAttribute*>(p)->~hkArrayTypeAttribute();
}
extern const hkTypeInfo hkArrayTypeAttributeTypeInfo;
const hkTypeInfo hkArrayTypeAttributeTypeInfo(
	"hkArrayTypeAttribute",
	"!hkArrayTypeAttribute",
	HK_NULL,
	cleanupLoadedObjecthkArrayTypeAttribute,
	HK_NULL,
	sizeof(hkArrayTypeAttribute)
	);
#endif

//
// Class hkDataObjectTypeAttribute
//
static const hkInternalClassMember hkDataObjectTypeAttributeClass_Members[] =
{
	{ "typeName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkDataObjectTypeAttribute,m_typeName), HK_NULL }
};
extern const hkClass hkDataObjectTypeAttributeClass;
const hkClass hkDataObjectTypeAttributeClass(
	"hkDataObjectTypeAttribute",
	HK_NULL, // parent
	sizeof(::hkDataObjectTypeAttribute),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkDataObjectTypeAttributeClass_Members),
	HK_COUNT_OF(hkDataObjectTypeAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkDataObjectTypeAttribute::staticClass()
{
	return hkDataObjectTypeAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkDataObjectTypeAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkDataObjectTypeAttribute(void* p)
{
	static_cast<hkDataObjectTypeAttribute*>(p)->~hkDataObjectTypeAttribute();
}
extern const hkTypeInfo hkDataObjectTypeAttributeTypeInfo;
const hkTypeInfo hkDataObjectTypeAttributeTypeInfo(
	"hkDataObjectTypeAttribute",
	"!hkDataObjectTypeAttribute",
	HK_NULL,
	cleanupLoadedObjecthkDataObjectTypeAttribute,
	HK_NULL,
	sizeof(hkDataObjectTypeAttribute)
	);
#endif

//
// Class hkDocumentationAttribute
//
static const hkInternalClassMember hkDocumentationAttributeClass_Members[] =
{
	{ "docsSectionTag", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkDocumentationAttribute,m_docsSectionTag), HK_NULL }
};
extern const hkClass hkDocumentationAttributeClass;
const hkClass hkDocumentationAttributeClass(
	"hkDocumentationAttribute",
	HK_NULL, // parent
	sizeof(::hkDocumentationAttribute),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkDocumentationAttributeClass_Members),
	HK_COUNT_OF(hkDocumentationAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkDocumentationAttribute::staticClass()
{
	return hkDocumentationAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkDocumentationAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkDocumentationAttribute(void* p)
{
	static_cast<hkDocumentationAttribute*>(p)->~hkDocumentationAttribute();
}
extern const hkTypeInfo hkDocumentationAttributeTypeInfo;
const hkTypeInfo hkDocumentationAttributeTypeInfo(
	"hkDocumentationAttribute",
	"!hkDocumentationAttribute",
	HK_NULL,
	cleanupLoadedObjecthkDocumentationAttribute,
	HK_NULL,
	sizeof(hkDocumentationAttribute)
	);
#endif

//
// Class hkPostFinishAttribute
//
static const hkInternalClassMember hkPostFinishAttributeClass_Members[] =
{
	{ "postFinishFunction", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkPostFinishAttribute,m_postFinishFunction), HK_NULL }
};
extern const hkClass hkPostFinishAttributeClass;
const hkClass hkPostFinishAttributeClass(
	"hkPostFinishAttribute",
	HK_NULL, // parent
	sizeof(::hkPostFinishAttribute),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkPostFinishAttributeClass_Members),
	HK_COUNT_OF(hkPostFinishAttributeClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkPostFinishAttribute::staticClass()
{
	return hkPostFinishAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkPostFinishAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkPostFinishAttribute(void* p)
{
	static_cast<hkPostFinishAttribute*>(p)->~hkPostFinishAttribute();
}
extern const hkTypeInfo hkPostFinishAttributeTypeInfo;
const hkTypeInfo hkPostFinishAttributeTypeInfo(
	"hkPostFinishAttribute",
	"!hkPostFinishAttribute",
	HK_NULL,
	cleanupLoadedObjecthkPostFinishAttribute,
	HK_NULL,
	sizeof(hkPostFinishAttribute)
	);
#endif

//
// Class hkScriptableAttribute
//
extern const hkClass hkScriptableAttributeClass;
const hkClass hkScriptableAttributeClass(
	"hkScriptableAttribute",
	HK_NULL, // parent
	sizeof(::hkScriptableAttribute),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	HK_NULL,
	0,
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkScriptableAttribute::staticClass()
{
	return hkScriptableAttributeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkScriptableAttribute*>(0))) == sizeof(hkBool::CompileTimeFalseType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL cleanupLoadedObjecthkScriptableAttribute(void* p)
{
	static_cast<hkScriptableAttribute*>(p)->~hkScriptableAttribute();
}
extern const hkTypeInfo hkScriptableAttributeTypeInfo;
const hkTypeInfo hkScriptableAttributeTypeInfo(
	"hkScriptableAttribute",
	"!hkScriptableAttribute",
	HK_NULL,
	cleanupLoadedObjecthkScriptableAttribute,
	HK_NULL,
	sizeof(hkScriptableAttribute)
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