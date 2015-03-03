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

// Generated from 'Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShape.h'
#include <Physics/Physics/hknpPhysics.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShape.h>
#define True true
#define False false


// External pointer and enum types
extern const hkClass hknpExternMeshShapeMeshClass;
extern const hkClass hknpExternMeshShapeTreeClass;

//
// Class hknpExternMeshShape::Mesh
//
extern const hkClass hkReferencedObjectClass;

const hkClass hknpExternMeshShapeMeshClass(
	"hknpExternMeshShapeMesh",
	&hkReferencedObjectClass, // parent
	sizeof(hknpExternMeshShape::Mesh),
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
const hkClass& HK_CALL hknpExternMeshShape::Mesh::staticClass()
{
	return hknpExternMeshShapeMeshClass;
}
#endif

//
// Class hknpExternMeshShape
//
extern const hkClass hknpCompositeShapeClass;

const hkInternalClassMember hknpExternMeshShape::Members[] =
{
	{ "mesh", &hknpExternMeshShapeMeshClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::ALIGN_16, HK_OFFSET_OF(hknpExternMeshShape,m_mesh), HK_NULL },
	{ "tree", &hknpExternMeshShapeTreeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, HK_OFFSET_OF(hknpExternMeshShape,m_tree), HK_NULL },
	{ "numIndexKeyBits", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hknpExternMeshShape,m_numIndexKeyBits), HK_NULL },
	{ "ownTree", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, HK_OFFSET_OF(hknpExternMeshShape,m_ownTree), HK_NULL }
};
extern const hkClass hknpExternMeshShapeClass;
const hkClass hknpExternMeshShapeClass(
	"hknpExternMeshShape",
	&hknpCompositeShapeClass, // parent
	sizeof(::hknpExternMeshShape),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hknpExternMeshShape::Members),
	HK_COUNT_OF(hknpExternMeshShape::Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hknpExternMeshShape::staticClass()
{
	return hknpExternMeshShapeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hknpExternMeshShape*>(0))) == sizeof(hkBool::CompileTimeTrueType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthknpExternMeshShape(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hknpExternMeshShape(f);
}
static void HK_CALL cleanupLoadedObjecthknpExternMeshShape(void* p)
{
	static_cast<hknpExternMeshShape*>(p)->~hknpExternMeshShape();
}
static const void* HK_CALL getVtablehknpExternMeshShape()
{
	#if HK_LINKONCE_VTABLES==0
	#if HK_HASHCODE_VTABLE_REGISTRY==1
	return ((const void*)(typeid(hknpExternMeshShape).hash_code()));
	#else
	return ((const void*)(typeid(hknpExternMeshShape).name()));
	#endif
	#else
	union { HK_ALIGN16(void* ptr); char buf[sizeof(hknpExternMeshShape)]; } u;
	hkFinishLoadedObjectFlag f;
	new (u.buf) hknpExternMeshShape(f);
	return u.ptr;
	#endif
}
extern const hkTypeInfo hknpExternMeshShapeTypeInfo;
const hkTypeInfo hknpExternMeshShapeTypeInfo(
	"hknpExternMeshShape",
	"!hknpExternMeshShape",
	finishLoadedObjecthknpExternMeshShape,
	cleanupLoadedObjecthknpExternMeshShape,
	getVtablehknpExternMeshShape(),
	sizeof(hknpExternMeshShape)
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
