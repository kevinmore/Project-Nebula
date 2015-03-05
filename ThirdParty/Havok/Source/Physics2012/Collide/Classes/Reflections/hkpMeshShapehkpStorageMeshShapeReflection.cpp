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

// Generated from 'Physics2012/Collide/Shape/Deprecated/StorageMesh/hkpStorageMeshShape.h'
#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Physics2012/Collide/Shape/Deprecated/FastMesh/hkpFastMeshShape.h>
#include <Physics2012/Collide/Shape/Deprecated/Mesh/hkpMeshShape.h>
#include <Physics2012/Collide/Shape/Deprecated/StorageMesh/hkpStorageMeshShape.h>
#define True true
#define False false


// External pointer and enum types
extern const hkClass hkpStorageMeshShapeSubpartStorageClass;

//
// Class hkpStorageMeshShape::SubpartStorage
//
extern const hkClass hkReferencedObjectClass;

static const hkInternalClassMember hkpStorageMeshShape_SubpartStorageClass_Members[] =
{
	{ "vertices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, HK_OFFSET_OF(hkpStorageMeshShape::SubpartStorage,m_vertices), HK_NULL },
	{ "indices16", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, HK_OFFSET_OF(hkpStorageMeshShape::SubpartStorage,m_indices16), HK_NULL },
	{ "indices32", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, HK_OFFSET_OF(hkpStorageMeshShape::SubpartStorage,m_indices32), HK_NULL },
	{ "materialIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, HK_OFFSET_OF(hkpStorageMeshShape::SubpartStorage,m_materialIndices), HK_NULL },
	{ "materials", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, HK_OFFSET_OF(hkpStorageMeshShape::SubpartStorage,m_materials), HK_NULL },
	{ "materialIndices16", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, HK_OFFSET_OF(hkpStorageMeshShape::SubpartStorage,m_materialIndices16), HK_NULL }
};
const hkClass hkpStorageMeshShapeSubpartStorageClass(
	"hkpStorageMeshShapeSubpartStorage",
	&hkReferencedObjectClass, // parent
	sizeof(hkpStorageMeshShape::SubpartStorage),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkpStorageMeshShape_SubpartStorageClass_Members),
	HK_COUNT_OF(hkpStorageMeshShape_SubpartStorageClass_Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkpStorageMeshShape::SubpartStorage::staticClass()
{
	return hkpStorageMeshShapeSubpartStorageClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkpStorageMeshShape::SubpartStorage*>(0))) == sizeof(hkBool::CompileTimeTrueType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthkpStorageMeshShapeSubpartStorage(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hkpStorageMeshShape::SubpartStorage(f);
}
static void HK_CALL cleanupLoadedObjecthkpStorageMeshShapeSubpartStorage(void* p)
{
	static_cast<hkpStorageMeshShape::SubpartStorage*>(p)->~SubpartStorage();
}
static const void* HK_CALL getVtablehkpStorageMeshShapeSubpartStorage()
{
	#if HK_LINKONCE_VTABLES==0
	#if HK_HASHCODE_VTABLE_REGISTRY==1
	return ((const void*)(typeid(hkpStorageMeshShape::SubpartStorage).hash_code()));
	#else
	return ((const void*)(typeid(hkpStorageMeshShape::SubpartStorage).name()));
	#endif
	#else
	union { HK_ALIGN16(void* ptr); char buf[sizeof(hkpStorageMeshShape::SubpartStorage)]; } u;
	hkFinishLoadedObjectFlag f;
	new (u.buf) hkpStorageMeshShape::SubpartStorage(f);
	return u.ptr;
	#endif
}
extern const hkTypeInfo hkpStorageMeshShapeSubpartStorageTypeInfo;
const hkTypeInfo hkpStorageMeshShapeSubpartStorageTypeInfo(
	"hkpStorageMeshShapeSubpartStorage",
	"!hkpStorageMeshShape::SubpartStorage",
	finishLoadedObjecthkpStorageMeshShapeSubpartStorage,
	cleanupLoadedObjecthkpStorageMeshShapeSubpartStorage,
	getVtablehkpStorageMeshShapeSubpartStorage(),
	sizeof(hkpStorageMeshShape::SubpartStorage)
	);
#endif

//
// Class hkpStorageMeshShape
//
extern const hkClass hkpMeshShapeClass;

// hkpStorageMeshShape attributes
const hkInternalClassMember hkpStorageMeshShape::Members[] =
{
	{ "storage", &hkpStorageMeshShapeSubpartStorageClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, HK_OFFSET_OF(hkpStorageMeshShape,m_storage), HK_NULL }
};
extern const hkClass hkpStorageMeshShapeClass;
const hkClass hkpStorageMeshShapeClass(
	"hkpStorageMeshShape",
	&hkpMeshShapeClass, // parent
	sizeof(::hkpStorageMeshShape),
	HK_NULL,
	0, // interfaces
	HK_NULL,
	0, // enums
	reinterpret_cast<const hkClassMember*>(hkpStorageMeshShape::Members),
	HK_COUNT_OF(hkpStorageMeshShape::Members),
	HK_NULL, // defaults
	HK_NULL, // attributes
	0, // flags
	hkUint32(0) // version
	);
#ifndef HK_HKCLASS_DEFINITION_ONLY
const hkClass& HK_CALL hkpStorageMeshShape::staticClass()
{
	return hkpStorageMeshShapeClass;
}
HK_COMPILE_TIME_ASSERT2( \
	sizeof(hkIsVirtual(static_cast<hkpStorageMeshShape*>(0))) == sizeof(hkBool::CompileTimeTrueType), \
	REFLECTION_PARSER_VTABLE_DETECTION_FAILED );
static void HK_CALL finishLoadedObjecthkpStorageMeshShape(void* p, int finishing = 1)
{
	hkFinishLoadedObjectFlag f;
	f.m_finishing = finishing;
	new (p) hkpStorageMeshShape(f);
}
static void HK_CALL cleanupLoadedObjecthkpStorageMeshShape(void* p)
{
	static_cast<hkpStorageMeshShape*>(p)->~hkpStorageMeshShape();
}
static const void* HK_CALL getVtablehkpStorageMeshShape()
{
	#if HK_LINKONCE_VTABLES==0
	#if HK_HASHCODE_VTABLE_REGISTRY==1
	return ((const void*)(typeid(hkpStorageMeshShape).hash_code()));
	#else
	return ((const void*)(typeid(hkpStorageMeshShape).name()));
	#endif
	#else
	union { HK_ALIGN16(void* ptr); char buf[sizeof(hkpStorageMeshShape)]; } u;
	hkFinishLoadedObjectFlag f;
	new (u.buf) hkpStorageMeshShape(f);
	return u.ptr;
	#endif
}
extern const hkTypeInfo hkpStorageMeshShapeTypeInfo;
const hkTypeInfo hkpStorageMeshShapeTypeInfo(
	"hkpStorageMeshShape",
	"!hkpStorageMeshShape",
	finishLoadedObjecthkpStorageMeshShape,
	cleanupLoadedObjecthkpStorageMeshShape,
	getVtablehkpStorageMeshShape(),
	sizeof(hkpStorageMeshShape)
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