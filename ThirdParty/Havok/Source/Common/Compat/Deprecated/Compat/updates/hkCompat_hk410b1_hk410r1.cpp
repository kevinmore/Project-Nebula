/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Compat/Deprecated/Compat/hkCompatUtil.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
	
#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

static void hkExtendedMeshShapeShapesSubpart_hk410b1_hk410r1(
										hkVariant& oldObj,
										hkVariant& newObj,
										hkObjectUpdateTracker& )
{
	hkClassMemberAccessor newNumChildShapes(newObj,		"numChildShapes");
	hkClassMemberAccessor oldNumShapes(oldObj,	"numShapes");

	if( newNumChildShapes.isOk() && oldNumShapes.isOk() )
	{
		newNumChildShapes.asInt32() = oldNumShapes.asInt32() = 0;
	}
}

static void hkExtendedMeshShapeTrianglesSubpart_hk410b1_hk410r1(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{
	hkClassMemberAccessor newNumTriangleShapes(newObj,		"numTriangleShapes");
	hkClassMemberAccessor oldNumShapes(oldObj,	"numShapes");

	if( newNumTriangleShapes.isOk() && oldNumShapes.isOk() )
	{
		newNumTriangleShapes.asInt32() = oldNumShapes.asInt32() = 0;
	}
}

static void hkExtendedMeshSubpartsArray_hk410b1_hk410r1(
	const hkClassMemberAccessor& oldObjMem,
	const hkClassMemberAccessor& newObjMem,
	hkObjectUpdateTracker& tracker, hkVersionRegistry::VersionFunc versionFuncPtr)
{
	hkClassMemberAccessor::SimpleArray& oldArray = oldObjMem.asSimpleArray();
	hkClassMemberAccessor::SimpleArray& newArray = newObjMem.asSimpleArray();
	HK_ASSERT( 0xad78555b, oldArray.size == newArray.size );
	hkVariant oldVariant = {HK_NULL, &oldObjMem.object().getClass()};
	hkVariant newVariant = {HK_NULL, &newObjMem.object().getClass()};
	hkInt32 oldSize = oldVariant.m_class->getObjectSize();
	hkInt32 newSize = newVariant.m_class->getObjectSize();
	for( int i = 0; i < oldArray.size; ++i )
	{
		oldVariant.m_object = static_cast<char*>(oldArray.data) + oldSize*i;
		newVariant.m_object = static_cast<char*>(newArray.data) + newSize*i;
		(versionFuncPtr)(oldVariant, newVariant, tracker);
	}
}

static void hkExtendedMeshShape_hk410b1_hk410r1(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& tracker)
{
	hkExtendedMeshSubpartsArray_hk410b1_hk410r1(
					hkClassMemberAccessor(oldObj, "trianglesSubparts"),
					hkClassMemberAccessor(newObj, "trianglesSubparts"),
					tracker,
					hkExtendedMeshShapeTrianglesSubpart_hk410b1_hk410r1);

	hkExtendedMeshSubpartsArray_hk410b1_hk410r1(
					hkClassMemberAccessor(oldObj, "shapesSubparts"),
					hkClassMemberAccessor(newObj, "shapesSubparts"),
					tracker,
					hkExtendedMeshShapeShapesSubpart_hk410b1_hk410r1);
}

namespace hkCompat_hk410b1_hk410r1
{

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	//hkExtended Mesh Shape
	{ 0x22cb60f2, 0x03b79c63, hkVersionRegistry::VERSION_COPY, "hkExtendedMeshShapeShapesSubpart", hkExtendedMeshShapeShapesSubpart_hk410b1_hk410r1 },
	{ 0x573ee2c4, 0xb782ddda, hkVersionRegistry::VERSION_COPY, "hkExtendedMeshShapeTrianglesSubpart", hkExtendedMeshShapeTrianglesSubpart_hk410b1_hk410r1 },
	{ 0xbfeecff5, 0x32dd318e, hkVersionRegistry::VERSION_COPY, "hkExtendedMeshShapeSubpart", HK_NULL },
	{ 0xa22a9842, 0x4c103864, hkVersionRegistry::VERSION_COPY, "hkExtendedMeshShape", hkExtendedMeshShape_hk410b1_hk410r1 },

	// common
	{ 0x8bdd3e9a, 0x8bdd3e9a, hkVersionRegistry::VERSION_VARIANT, "hkBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL }, 
	{ 0x3d43489c, 0x3d43489c, hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", HK_NULL },
	{ 0x914da6c1, 0x914da6c1, hkVersionRegistry::VERSION_VARIANT, "hkxAttribute", HK_NULL },
	{ 0x1667c01c, 0x1667c01c, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeGroup", HK_NULL }, 
	{ 0x0a62c79f, 0x0a62c79f, hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL }, 
	{ 0xe085ba9f, 0xe085ba9f, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL },
	{ 0x57061454, 0x57061454, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	{ 0, 0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok410b1Classes
#define HK_COMPAT_VERSION_TO hkHavok410r1Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO
} // namespace hkCompat_hk410b1_hk410r1

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
