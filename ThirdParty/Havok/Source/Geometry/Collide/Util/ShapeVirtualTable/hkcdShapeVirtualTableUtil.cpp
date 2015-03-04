/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/Util/ShapeVirtualTable/hkcdShapeVirtualTableUtil.h>


hkUlong hkcdShapeVirtualTableUtil::s_virtualTablePatches[] = 
{
	HK_NULL,			// hkcdShapeType::SPHERE
	HK_NULL,			// hkcdShapeType::CYLINDER
	HK_NULL,			// hkcdShapeType::TRIANGLE
	HK_NULL,			// hkcdShapeType::BOX
	HK_NULL,			// hkcdShapeType::CAPSULE
	HK_NULL,			// hkcdShapeType::CONVEX_VERTICES

	HK_NULL,			// hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION
	HK_NULL,			// hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE
	HK_NULL,			// hkcdShapeType::LIST
	HK_NULL,			// hkcdShapeType::MOPP

	HK_NULL,			// hkcdShapeType::CONVEX_TRANSLATE
	HK_NULL,			// hkcdShapeType::CONVEX_TRANSFORM
	HK_NULL,			// hkcdShapeType::SAMPLED_HEIGHT_FIELD

	HK_NULL,			// hkcdShapeType::EXTENDED_MESH
	HK_NULL,			// hkcdShapeType::TRANSFORM
	HK_NULL,			// hkcdShapeType::COMPRESSED_MESH
	HK_NULL,			// hkcdShapeType::STATIC_COMPOUND
	HK_NULL,			// hkcdShapeType::BV_COMPRESSED_MESH
	HK_NULL,			// hkcdShapeType::COLLECTION

	HK_NULL,			// hkcdShapeType::USER_0
	HK_NULL,			// hkcdShapeType::USER_1
	HK_NULL,			// hkcdShapeType::USER_2
};

HK_COMPILE_TIME_ASSERT( sizeof(hkcdShapeVirtualTableUtil::s_virtualTablePatches) == sizeof(hkUlong) * hkcdShapeType::MAX_SPU_SHAPE_TYPE );


#if	defined(HK_DEBUG) && !defined(HK_PLATFORM_SPU)
hkUint32 hkcdShapeVirtualTableUtil::s_unregisteredFunctions = 0;
#endif


void HK_CALL hkcdShapeVirtualTableUtil::patchVirtualTable( hkcdShape* shape )
{
	const hkcdShape::ShapeType shapeType = shape->getType();
	HK_ASSERT( 0x63e07b56, shapeType >= hkcdShapeType::FIRST_SHAPE_TYPE && shapeType < hkcdShapeType::MAX_SPU_SHAPE_TYPE );

	const hkUlong patchAddress = s_virtualTablePatches[ shapeType ];
	reinterpret_cast<hkUlong*>(shape)[0] = patchAddress;
}

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
