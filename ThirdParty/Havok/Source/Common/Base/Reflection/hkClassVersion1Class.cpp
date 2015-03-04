/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>

HK_REFLECTION_CLASSFILE_HEADER("../hkBase.h")

extern const hkClass hkClassVersion1Class;
extern const hkClass hkClassVersion2Class;
extern const hkClass hkClassMemberVersion1Class;
extern const hkClass hkClassEnumVersion1Class;

class hkClassVersion1
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_HKCLASS, hkClassVersion1 );
		HK_DECLARE_REFLECTION();

	public:

		hkClassVersion1() { }

	protected:

		const char* m_name;
		const hkClass* m_parent;
		int m_objectSize;
		int m_numImplementedInterfaces;
		const class hkClassEnumVersion1* m_declaredEnums;
		int m_numDeclaredEnums;
		const class hkClassMember* m_declaredMembers;
		int m_numDeclaredMembers;
		hkBool m_hasVtable;
		char m_padToSizeOfClass[sizeof(void*) - sizeof(hkBool)];
};

class hkClassVersion2
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_HKCLASS, hkClassVersion2 );
		HK_DECLARE_REFLECTION();

	public:

		hkClassVersion2() { }

	protected:

		const char* m_name;
		const hkClass* m_parent;
		int m_objectSize;
		int m_numImplementedInterfaces;
		const class hkClassEnumVersion1* m_declaredEnums;
		int m_numDeclaredEnums;
		const class hkClassMember* m_declaredMembers;
		int m_numDeclaredMembers;
		void* m_defaults;
};

class hkClassMemberVersion1
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_HKCLASS, hkClassMemberVersion1);
		HK_DECLARE_REFLECTION();

	public:

		hkClassMemberVersion1() { }

		enum Type
		{
			TYPE_VOID = 0,
			TYPE_BOOL,
			TYPE_CHAR,
			TYPE_INT8,
			TYPE_UINT8,
			TYPE_INT16,
			TYPE_UINT16,
			TYPE_INT32,
			TYPE_UINT32,
			TYPE_INT64,
			TYPE_UINT64,
			TYPE_REAL,
			TYPE_VECTOR4,
			TYPE_QUATERNION,
			TYPE_MATRIX3,
			TYPE_ROTATION,
			TYPE_QSTRANSFORM,
			TYPE_MATRIX4,
			TYPE_TRANSFORM,
			TYPE_ZERO,
			TYPE_POINTER,
			TYPE_FUNCTIONPOINTER,
			TYPE_ARRAY,
			TYPE_INPLACEARRAY,
			TYPE_ENUM,
			TYPE_STRUCT,
			TYPE_SIMPLEARRAY,
			TYPE_HOMOGENEOUSARRAY,
			TYPE_VARIANT,
			TYPE_CSTRING,
			TYPE_ULONG,
			TYPE_FLAGS,
			TYPE_MAX
		};

		enum Flags
		{
			POINTER_OPTIONAL = 1,
			POINTER_VOIDSTAR = 2,
			ENUM_8 = 8,
			ENUM_16 = 16,
			ENUM_32 = 32,
			ARRAY_RAWDATA = 64
		};

		enum Range
		{
			INVALID = 0,
			DEFAULT = 1,
			ABS_MIN = 2,
			ABS_MAX = 4,
			SOFT_MIN = 8,
			SOFT_MAX = 16,
			RANGE_MAX = 32
		};

		const char* m_name;
		const hkClass* m_class;
		const hkClassEnumVersion1* m_enum;
		hkEnum<Type,hkUint8> m_type;
		hkEnum<Type,hkUint8> m_subtype;
		hkInt16 m_cArraySize;
		hkUint16 m_flags;
		hkUint16 m_offset;
};

class hkClassEnumVersion1
{
	public:

        HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_HKCLASS, hkClassEnumVersion1);
		HK_DECLARE_REFLECTION();

		class Item
        {
            public:

                HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE, Item);
				HK_DECLARE_REFLECTION();
                int m_value;
                const char* m_name;
        };

        const char* m_name;
        const class Item* m_items;
        int m_numItems;
};

//
// Class hkClass
//
const hkInternalClassMember hkClassVersion1::Members[] =
{
	{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion1,m_name), HK_NULL },
	{ "parent", &hkClassVersion1Class, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassVersion1,m_parent), HK_NULL },
	{ "objectSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion1,m_objectSize), HK_NULL },
	{ "numImplementedInterfaces", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion1,m_numImplementedInterfaces), HK_NULL },
	{ "declaredEnums", &hkClassEnumVersion1Class, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassVersion1,m_declaredEnums), HK_NULL },
	{ "declaredMembers", &hkClassMemberVersion1Class, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassVersion1,m_declaredMembers), HK_NULL },
	{ "hasVtable", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion1,m_hasVtable), HK_NULL }
};

const hkClass hkClassVersion1Class(
	"hkClass",
	HK_NULL, // parent
	sizeof(hkClassVersion1),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkClassVersion1::Members),
	HK_COUNT_OF(hkClassVersion1::Members), // members
	HK_NULL // defaults
	);
const hkInternalClassMember hkClassVersion2::Members[] =
{
	{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion2,m_name), HK_NULL },
	{ "parent", &hkClassVersion2Class, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassVersion2,m_parent), HK_NULL },
	{ "objectSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion2,m_objectSize), HK_NULL },
	{ "numImplementedInterfaces", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion2,m_numImplementedInterfaces), HK_NULL },
	{ "declaredEnums", &hkClassEnumVersion1Class, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassVersion2,m_declaredEnums), HK_NULL },
	{ "declaredMembers", &hkClassMemberVersion1Class, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassVersion2,m_declaredMembers), HK_NULL },
	{ "defaults", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion2,m_defaults), HK_NULL }
};

const hkClass hkClassVersion2Class(
	"hkClass",
	HK_NULL, // parent
	sizeof(hkClassVersion2),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkClassVersion2::Members),
	HK_COUNT_OF(hkClassVersion2::Members), // members
	HK_NULL // defaults
	);


static const hkInternalClassEnumItem hkClassMemberVersion1TypeEnumItems[] =
{
	{0, "TYPE_VOID"},
	{1, "TYPE_BOOL"},
	{2, "TYPE_CHAR"},
	{3, "TYPE_INT8"},
	{4, "TYPE_UINT8"},
	{5, "TYPE_INT16"},
	{6, "TYPE_UINT16"},
	{7, "TYPE_INT32"},
	{8, "TYPE_UINT32"},
	{9, "TYPE_INT64"},
	{10, "TYPE_UINT64"},
	{11, "TYPE_REAL"},
	{12, "TYPE_VECTOR4"},
	{13, "TYPE_QUATERNION"},
	{14, "TYPE_MATRIX3"},
	{15, "TYPE_ROTATION"},
	{16, "TYPE_QSTRANSFORM"},
	{17, "TYPE_MATRIX4"},
	{18, "TYPE_TRANSFORM"},
	{19, "TYPE_ZERO"},
	{20, "TYPE_POINTER"},
	{21, "TYPE_FUNCTIONPOINTER"},
	{22, "TYPE_ARRAY"},
	{23, "TYPE_INPLACEARRAY"},
	{24, "TYPE_ENUM"},
	{25, "TYPE_STRUCT"},
	{26, "TYPE_SIMPLEARRAY"},
	{27, "TYPE_HOMOGENEOUSARRAY"},
	{28, "TYPE_VARIANT"},
	{29, "TYPE_CSTRING"},
	{30, "TYPE_ULONG"},
	{31, "TYPE_FLAGS"},
	{32, "TYPE_MAX"}
};
static const hkInternalClassEnumItem hkClassMemberVersion1FlagsEnumItems[] =
{
	{1, "POINTER_OPTIONAL"},
	{2, "POINTER_VOIDSTAR"},
	{8, "ENUM_8"},
	{16, "ENUM_16"},
	{32, "ENUM_32"},
	{64, "ARRAY_RAWDATA"},
};
static const hkInternalClassEnumItem hkClassMemberVersion1RangeEnumItems[] =
{
	{0, "INVALID"},
	{1, "DEFAULT"},
	{2, "ABS_MIN"},
	{4, "ABS_MAX"},
	{8, "SOFT_MIN"},
	{16, "SOFT_MAX"},
	{32, "RANGE_MAX"},
};
static const hkInternalClassEnum hkClassMemberVersion1Enums[] = {
	{"Type", hkClassMemberVersion1TypeEnumItems, HK_COUNT_OF(hkClassMemberVersion1TypeEnumItems), HK_NULL, 0 },
	{"Flags", hkClassMemberVersion1FlagsEnumItems, HK_COUNT_OF(hkClassMemberVersion1FlagsEnumItems), HK_NULL, 0 },
	{"Range", hkClassMemberVersion1RangeEnumItems, HK_COUNT_OF(hkClassMemberVersion1RangeEnumItems), HK_NULL, 0 }
};
static const hkClassEnum* hkClassMemberVersion1TypeEnum = reinterpret_cast<const hkClassEnum*>(&hkClassMemberVersion1Enums[0]);
static const hkClassEnum* hkClassMemberVersion1FlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkClassMemberVersion1Enums[1]);
static const hkClassEnum* hkClassMemberVersion1RangeEnum = reinterpret_cast<const hkClassEnum*>(&hkClassMemberVersion1Enums[2]);

static hkInternalClassMember hkClassMemberVersion1Class_Members[] =
{
	{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassMemberVersion1,m_name), HK_NULL },
	{ "class", &hkClassVersion1Class, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassMemberVersion1,m_class), HK_NULL },
	{ "enum", &hkClassEnumVersion1Class, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassMemberVersion1,m_enum), HK_NULL },
	{ "type", HK_NULL, hkClassMemberVersion1TypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, HK_OFFSET_OF(hkClassMemberVersion1,m_type), HK_NULL },
	{ "subtype", HK_NULL, hkClassMemberVersion1TypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, HK_OFFSET_OF(hkClassMemberVersion1,m_subtype), HK_NULL },
	{ "cArraySize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassMemberVersion1,m_cArraySize), HK_NULL },
	{ "flags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassMemberVersion1,m_flags), HK_NULL },
	{ "offset", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassMemberVersion1,m_offset), HK_NULL }
};

const hkClass hkClassMemberVersion1Class(
	"hkClassMember",
	HK_NULL, // parent
	sizeof(hkClassMemberVersion1),
	HK_NULL,
	0, // interfaces
	reinterpret_cast<const hkClassEnum*>(hkClassMemberVersion1Enums),
	HK_COUNT_OF(hkClassMemberVersion1Enums), // enums
	reinterpret_cast<const hkClassMember*>(hkClassMemberVersion1Class_Members),
	HK_COUNT_OF(hkClassMemberVersion1Class_Members), // members
	HK_NULL // defaults
	);

const hkInternalClassMember hkClassEnumVersion1::Item::Members[] =
{
    { "value", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassEnumVersion1::Item,m_value), HK_NULL },
    { "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassEnumVersion1::Item,m_name), HK_NULL }
};
const hkClass hkClassEnumItemVersion1Class(
    "hkClassEnumItem",
    HK_NULL, // parent
    sizeof(hkClassEnumVersion1::Item),
    HK_NULL,
    0, // interfaces
    HK_NULL,
    0, // enums
    reinterpret_cast<const hkClassMember*>(hkClassEnumVersion1::Item::Members),
    HK_COUNT_OF(hkClassEnumVersion1::Item::Members),
    HK_NULL // defaults
    );      
const hkInternalClassMember hkClassEnumVersion1::Members[] =
{           
    { "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassEnumVersion1,m_name), HK_NULL },
    { "items", &hkClassEnumItemVersion1Class, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassEnumVersion1,m_items), HK_NULL }
};
const hkClass hkClassEnumVersion1Class(
    "hkClassEnum",
    HK_NULL, // parent
    sizeof(hkClassEnumVersion1),
    HK_NULL,
    0, // interfaces
    HK_NULL,
    0, // enums
    reinterpret_cast<const hkClassMember*>(hkClassEnumVersion1::Members),
    HK_COUNT_OF(hkClassEnumVersion1::Members),
    HK_NULL // defaults
    );

class hkClassMemberVersion3
{
public:

	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_HKCLASS, hkClassMemberVersion3);
	HK_DECLARE_REFLECTION();

	enum Type
	{
		/// No type
		TYPE_VOID = 0,
		/// hkBool,  boolean type
		TYPE_BOOL,
		/// hkChar, signed char type
		TYPE_CHAR,
		/// hkInt8, 8 bit signed integer type
		TYPE_INT8,
		/// hkUint8, 8 bit unsigned integer type
		TYPE_UINT8,
		/// hkInt16, 16 bit signed integer type
		TYPE_INT16,
		/// hkUint16, 16 bit unsigned integer type
		TYPE_UINT16,
		/// hkInt32, 32 bit signed integer type
		TYPE_INT32,
		/// hkUint32, 32 bit unsigned integer type
		TYPE_UINT32,
		/// hkInt64, 64 bit signed integer type
		TYPE_INT64,
		/// hkUint64, 64 bit unsigned integer type
		TYPE_UINT64,
		/// hkReal, float type
		TYPE_REAL,
		/// hkVector4 type
		TYPE_VECTOR4,
		/// hkQuaternion type
		TYPE_QUATERNION,
		/// hkMatrix3 type
		TYPE_MATRIX3,
		/// hkRotation type
		TYPE_ROTATION,
		/// hkQsTransform type
		TYPE_QSTRANSFORM,
		/// hkMatrix4 type
		TYPE_MATRIX4,
		/// hkTransform type
		TYPE_TRANSFORM,
		/// Serialize as zero - deprecated.
		TYPE_ZERO,
		/// Generic pointer, see member flags for more info
		TYPE_POINTER,
		/// Function pointer
		TYPE_FUNCTIONPOINTER,
		/// hkArray<T>, array of items of type T
		TYPE_ARRAY,
		/// hkInplaceArray<T,N> or hkInplaceArrayAligned16<T,N>, array of N items of type T
		TYPE_INPLACEARRAY,
		/// hkEnum<ENUM,STORAGE> - enumerated values
		TYPE_ENUM,
		/// Object
		TYPE_STRUCT,
		/// Simple array (ptr(typed) and size only)
		TYPE_SIMPLEARRAY,
		/// Simple array of homogeneous types, so is a class id followed by a void* ptr and size
		TYPE_HOMOGENEOUSARRAY,
		/// hkVariant (void* and hkClass*) type
		TYPE_VARIANT,
		/// char*, null terminated string
		TYPE_CSTRING,
		/// hkUlong, unsigned long, defined to always be the same size as a pointer
		TYPE_ULONG,
		/// hkFlags<ENUM,STORAGE> - 8,16,32 bits of named values.
		TYPE_FLAGS,
		/// hkHalf, 16-bit float value
		TYPE_HALF,
		/// hkStringPtr, c-string
		TYPE_STRINGPTR,
		TYPE_MAX
	};

	/// Special member properties.
	enum FlagValues
	{
		FLAGS_NONE = 0,
		/// Member has forced 8 byte alignment.
		ALIGN_8 = 128,
		/// Member has forced 16 byte alignment.
		ALIGN_16 = 256,
		/// The members memory contents is not owned by this object
		NOT_OWNED = 512,
		/// This member should not be written when serializing
		SERIALIZE_IGNORED = 1024
	};
	typedef hkFlags<FlagValues, hkUint16> Flags;

	enum DeprecatedFlagValues
	{
		DEPRECATED_SIZE_8 = 8,
		DEPRECATED_ENUM_8 = 8,
		DEPRECATED_SIZE_16 = 16,
		DEPRECATED_ENUM_16 = 16,
		DEPRECATED_SIZE_32 = 32,
		DEPRECATED_ENUM_32 = 32
	};

	enum
	{
		HK_CLASS_ZERO_DEFAULT = -2,
	};

	/// Properties of the builtin types.
	struct TypeProperties
	{
		/// The type associated with this
		hkEnum<hkClassMember::Type,hkUint8> m_type;
		/// Zero terminated name
		const char* m_name;
		/// Size of the type in bytes <=0 it is not defined
		short m_size;
		/// Alignment in bytes, if <=0 it is not defined
		short m_align;

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE_CLASS, hkClassMemberVersion3::TypeProperties );
	};

private:

	/// The name of this member.
	const char* m_name;

	///
	const hkClass* m_class; //
	const hkClassEnum* m_enum; // Usually null except for enums
	hkEnum<Type,hkUint8> m_type; // An hkMemberType.
	hkEnum<Type,hkUint8> m_subtype; // An hkMemberType.
	hkInt16 m_cArraySize; // Usually zero, nonzero for cstyle array..
	Flags m_flags; // Pointers:optional, voidstar, rawdata. Enums:sizeinbytes.
	hkUint16 m_offset; // Address offset from start of struct.
	const hkCustomAttributes* m_attributes; //+serialized(false)
};

//
// Enum hkClassMember::Type
//
static const hkInternalClassEnumItem hkClassMemberVersion3TypeEnumItems[] =
{
	{0, "TYPE_VOID"},
	{1, "TYPE_BOOL"},
	{2, "TYPE_CHAR"},
	{3, "TYPE_INT8"},
	{4, "TYPE_UINT8"},
	{5, "TYPE_INT16"},
	{6, "TYPE_UINT16"},
	{7, "TYPE_INT32"},
	{8, "TYPE_UINT32"},
	{9, "TYPE_INT64"},
	{10, "TYPE_UINT64"},
	{11, "TYPE_REAL"},
	{12, "TYPE_VECTOR4"},
	{13, "TYPE_QUATERNION"},
	{14, "TYPE_MATRIX3"},
	{15, "TYPE_ROTATION"},
	{16, "TYPE_QSTRANSFORM"},
	{17, "TYPE_MATRIX4"},
	{18, "TYPE_TRANSFORM"},
	{19, "TYPE_ZERO"},
	{20, "TYPE_POINTER"},
	{21, "TYPE_FUNCTIONPOINTER"},
	{22, "TYPE_ARRAY"},
	{23, "TYPE_INPLACEARRAY"},
	{24, "TYPE_ENUM"},
	{25, "TYPE_STRUCT"},
	{26, "TYPE_SIMPLEARRAY"},
	{27, "TYPE_HOMOGENEOUSARRAY"},
	{28, "TYPE_VARIANT"},
	{29, "TYPE_CSTRING"},
	{30, "TYPE_ULONG"},
	{31, "TYPE_FLAGS"},
	{32, "TYPE_HALF"},
	{33, "TYPE_STRINGPTR"},
	{34, "TYPE_MAX"},
};

//
// Enum hkClassMember::FlagValues
//
static const hkInternalClassEnumItem hkClassMemberVersion3FlagValuesEnumItems[] =
{
	{0, "FLAGS_NONE"},
	{128, "ALIGN_8"},
	{256, "ALIGN_16"},
	{512, "NOT_OWNED"},
	{1024, "SERIALIZE_IGNORED"},
};

//
// Enum hkClassMember::DeprecatedFlagValues
//
static const hkInternalClassEnumItem hkClassMemberVersion3DeprecatedFlagValuesEnumItems[] =
{
	{8, "DEPRECATED_SIZE_8"},
	{8, "DEPRECATED_ENUM_8"},
	{16, "DEPRECATED_SIZE_16"},
	{16, "DEPRECATED_ENUM_16"},
	{32, "DEPRECATED_SIZE_32"},
	{32, "DEPRECATED_ENUM_32"},
};
static const hkInternalClassEnum hkClassMemberVersion3Enums[] = {
	{"Type", hkClassMemberVersion3TypeEnumItems, 35, HK_NULL, 0 },
	{"FlagValues", hkClassMemberVersion3FlagValuesEnumItems, 5, HK_NULL, 0 },
	{"DeprecatedFlagValues", hkClassMemberVersion3DeprecatedFlagValuesEnumItems, 6, HK_NULL, 0 }
};
const hkClassEnum* hkClassMemberVersion3TypeEnum = reinterpret_cast<const hkClassEnum*>(&hkClassMemberVersion3Enums[0]);
const hkClassEnum* hkClassMemberVersion3FlagValuesEnum = reinterpret_cast<const hkClassEnum*>(&hkClassMemberVersion3Enums[1]);
const hkClassEnum* hkClassMemberVersion3DeprecatedFlagValuesEnum = reinterpret_cast<const hkClassEnum*>(&hkClassMemberVersion3Enums[2]);

extern const class hkClass hkCustomAttributesClass;
extern const class hkClass hkClassVersion3Class;
//
// Class hkClassMember
//
const hkInternalClassMember hkClassMemberVersion3::Members[] =
{
	{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassMemberVersion3,m_name), HK_NULL },
	{ "class", &hkClassVersion3Class, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassMemberVersion3,m_class), HK_NULL },
	{ "enum", &hkClassEnumClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassMemberVersion3,m_enum), HK_NULL },
	{ "type", HK_NULL, hkClassMemberVersion3TypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, HK_OFFSET_OF(hkClassMemberVersion3,m_type), HK_NULL },
	{ "subtype", HK_NULL, hkClassMemberVersion3TypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, HK_OFFSET_OF(hkClassMemberVersion3,m_subtype), HK_NULL },
	{ "cArraySize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassMemberVersion3,m_cArraySize), HK_NULL },
	{ "flags", HK_NULL, hkClassMemberVersion3FlagValuesEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT16, 0, 0, HK_OFFSET_OF(hkClassMemberVersion3,m_flags), HK_NULL },
	{ "offset", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassMemberVersion3,m_offset), HK_NULL },
	{ "attributes", &hkCustomAttributesClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkClassMemberVersion3,m_attributes), HK_NULL }
};
extern const hkClass hkClassMemberVersion3Class;
const hkClass hkClassMemberVersion3Class(
								 "hkClassMember",
								 HK_NULL, // parent
								 sizeof(hkClassMemberVersion3),
								 HK_NULL,
								 0, // interfaces
								 reinterpret_cast<const hkClassEnum*>(hkClassMemberVersion3Enums),
								 3, // enums
								 reinterpret_cast<const hkClassMember*>(hkClassMemberVersion3::Members),
								 HK_COUNT_OF(hkClassMemberVersion3::Members),
								 HK_NULL, // defaults
								 HK_NULL, // attributes
								 0, // flags
								 0 // version
								 );

// External pointer and enum types
extern const hkClass hkClassClass;
extern const hkClass hkClassEnumClass;
extern const hkClass hkClassMemberClass;
extern const hkClass hkCustomAttributesClass;
extern const hkClassEnum* hkClassFlagValuesEnum;


class hkClassVersion3
{
public:

	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_HKCLASS, hkClass );
	HK_DECLARE_REFLECTION();

public:

	enum SignatureFlags
	{
		SIGNATURE_LOCAL = 1	// don't include signature of parents
	};

	enum FlagValues
	{
		FLAGS_NONE = 0,
		FLAGS_NOT_SERIALIZABLE = 1
	};
	typedef hkFlags<FlagValues, hkUint32> Flags;

protected:

	/// Name of this type.
	const char* m_name;

	/// Parent class.
	const hkClass* m_parent;

	/// Size of the live instance.
	int m_objectSize;

	/// Interfaces implemented by this class.
	//const hkClass** m_implementedInterfaces;

	/// Number of interfaces implemented by this class.
	int m_numImplementedInterfaces;

	/// Declared enum members.
	const class hkClassEnum* m_declaredEnums;

	/// Number of enums declared in this class.
	int m_numDeclaredEnums;

	/// Declared members.
	const class hkClassMember* m_declaredMembers;

	/// Number of members declared in this class.
	int m_numDeclaredMembers;

	/// Default values for this class.
	const void* m_defaults; //+nosave

	/// Default values for this class.
	const hkCustomAttributes* m_attributes; //+serialized(false)

	/// Flag values.
	Flags m_flags;

	/// Version of described object.
	int m_describedVersion;
};



//
// Enum hkClass::SignatureFlags
//
static const hkInternalClassEnumItem hkClassVersion3SignatureFlagsEnumItems[] =
{
	{1, "SIGNATURE_LOCAL"},
};

//
// Enum hkClass::FlagValues
//
static const hkInternalClassEnumItem hkClassVersion3FlagValuesEnumItems[] =
{
	{0, "FLAGS_NONE"},
	{1, "FLAGS_NOT_SERIALIZABLE"},
};
static const hkInternalClassEnum hkClassVersion3Enums[] = {
	{"SignatureFlags", hkClassVersion3SignatureFlagsEnumItems, 1, HK_NULL, 0 },
	{"FlagValues", hkClassVersion3FlagValuesEnumItems, 2, HK_NULL, 0 }
};
const hkClassEnum* hkClassVersion3SignatureFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkClassVersion3Enums[0]);
const hkClassEnum* hkClassVersion3FlagValuesEnum = reinterpret_cast<const hkClassEnum*>(&hkClassVersion3Enums[1]);

extern const hkClass hkClassVersion3Class;

//
// Class hkClass
//
const hkInternalClassMember hkClassVersion3::Members[] =
{
	{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion3,m_name), HK_NULL },
	{ "parent", &hkClassVersion3Class, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassVersion3,m_parent), HK_NULL },
	{ "objectSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion3,m_objectSize), HK_NULL },
	{ "numImplementedInterfaces", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion3,m_numImplementedInterfaces), HK_NULL },
	{ "declaredEnums", &hkClassEnumClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassVersion3,m_declaredEnums), HK_NULL },
	{ "declaredMembers", &hkClassMemberVersion3Class, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkClassVersion3,m_declaredMembers), HK_NULL },
	{ "defaults", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkClassVersion3,m_defaults), HK_NULL },
	{ "attributes", &hkCustomAttributesClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkClassVersion3,m_attributes), HK_NULL },
	{ "flags", HK_NULL, hkClassVersion3FlagValuesEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT32, 0, 0, HK_OFFSET_OF(hkClassVersion3,m_flags), HK_NULL },
	{ "describedVersion", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkClassVersion3,m_describedVersion), HK_NULL }
};
const hkClass hkClassVersion3Class(
						   "hkClass",
						   HK_NULL, // parent
						   sizeof(hkClass),
						   HK_NULL,
						   0, // interfaces
						   reinterpret_cast<const hkClassEnum*>(hkClassVersion3Enums),
						   2, // enums
						   reinterpret_cast<const hkClassMember*>(hkClassVersion3::Members),
						   HK_COUNT_OF(hkClassVersion3::Members),
						   HK_NULL, // defaults
						   HK_NULL, // attributes
						   0, // flags
						   0 // version
						   );

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
