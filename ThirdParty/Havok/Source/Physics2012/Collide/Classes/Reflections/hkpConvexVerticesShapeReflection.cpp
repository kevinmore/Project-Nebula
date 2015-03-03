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

// Generated from 'Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h'
#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#define True true
#define False false


// External pointer and enum types
extern const hkClass hkFourTransposedPointsfClass;
extern const hkClass hkVector4fClass;
extern const hkClass hkpConvexVerticesConnectivityClass;

//
// Class hkpConvexVerticesShape
//
extern const hkClass hkpConvexShapeClass;

static const hkInternalClassMember hkpConvexVerticesShapeClass_Members[] =
{
	{ "aabbHalfExtents", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpConvexVerticesShape,m_aabbHalfExtents), HK_NULL },
	{ "aabbCenter", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpConvexVerticesShape,m_aabbCenter), HK_NULL },
	{ "rotatedVertices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_MATRIX3, 0, 0, HK_OFFSET_OF(hkpConvexVerticesShape,m_rotatedVertices), HK_NULL },
	{ "numVertices", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hkpConvexVerticesShape,m_numVertices), HK_NULL },
	{ "useSpuBuffer", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, HK_OFFSET_OF(hkpConvexVerticesShape,m_useSpuBuffer), HK_NULL },
	{ "planeEquations", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, HK_OFFSET_OF(hkpConvexVerticesShape,m_planeEquations), HK_NULL },
	{ "connectivity", &hkpConvexVerticesConnectivityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hkpConvexVerticesShape,m_connectivity), HK_NULL }
};
namespace
{
	struct hkpConvexVerticesShape_DefaultStruct
	{
		int s_defaultOffsets[7];
		typedef hkInt8 _hkBool;
		typedef hkFloat32 _hkVector4[4];
		typedef hkReal _hkQuaternion[4];
		typedef hkReal _hkMatrix3[12];
		typedef hkReal _hkRotation[12];
		typedef hkReal _hkQsTransform[12];
		typedef hkReal _hkMatrix4[16];
		typedef hkReal _hkTransform[16];
	};
	const hkpConvexVerticesShape_DefaultStruct hkpConvexVerticesShape_Default =
	{
		{-1,-1,-1,-1,hkClassMember::HK_CLASS_ZERO_DEFAULT,-1,-1},
		
	};
}
extern const hkClass hkpConvexVerticesShapeClass;
const hkClass hkpConvexVerticesShapeClass(
	"hkpConvexVerticesShape",
	&hkpConvexShapeClass, // parent
	sizeof(::hkpConvexVerticesShape),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkpConvexVerticesShapeClass_Members),
	HK_COUNT_OF(hkpConvexVerticesShapeClass_Members),
	&hkpConvexVerticesShape_Default,
	HK_NULL, // attributes
	0, // flags
	hkUint32(6) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkpConvexVerticesShape::staticClass()
{
	return hkpConvexVerticesShapeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkpConvexVerticesShape*>(0))) == sizeof(hkBool::CompileTimeTrueType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthkpConvexVerticesShape(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hkpConvexVerticesShape(f);
}
static void HK_CALL cleanupLoadedObjecthkpConvexVerticesShape(void* p)
{
	static_cast<hkpConvexVerticesShape*>(p)->~hkpConvexVerticesShape();
}
static const void* HK_CALL getVtablehkpConvexVerticesShape()
{
	#if HK_LINKONCE_VTABLES==0
	#if HK_HASHCODE_VTABLE_REGISTRY==1
	return ((const void*)(typeid(hkpConvexVerticesShape).hash_code()));
	#else
	return ((const void*)(typeid(hkpConvexVerticesShape).name()));
	#endif
	#else
	union { HK_ALIGN16(void* ptr); char buf[sizeof(hkpConvexVerticesShape)]; } u;
	hkFinishLoadedObjectFlag f;
	new (u.buf) hkpConvexVerticesShape(f);
	return u.ptr;
	#endif
}
extern const hkTypeInfo hkpConvexVerticesShapeTypeInfo;
const hkTypeInfo hkpConvexVerticesShapeTypeInfo(
	"hkpConvexVerticesShape",
	"!hkpConvexVerticesShape",
	finishLoadedObjecthkpConvexVerticesShape,
	cleanupLoadedObjecthkpConvexVerticesShape,
	getVtablehkpConvexVerticesShape(),
	sizeof(hkpConvexVerticesShape)
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
