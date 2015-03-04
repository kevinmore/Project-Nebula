/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Destruction/BreakableMaterial/hkpBreakableMaterialUtil.h>

//
//	Compute the maximum material size

#ifdef HK_PLATFORM_HAS_SPU
	#define TEST_MATERIAL_SIZE(mtlType)	HK_COMPILE_TIME_ASSERT(sizeof(mtlType) <= hkpBreakableMaterialUtil::MAX_MATERIAL_SIZE)
#else
	#define TEST_MATERIAL_SIZE(mtlType)
#endif

TEST_MATERIAL_SIZE(hkpBreakableMaterial);
TEST_MATERIAL_SIZE(hkpSimpleBreakableMaterial);
TEST_MATERIAL_SIZE(hkpBreakableMultiMaterial);
TEST_MATERIAL_SIZE(hkpListShapeBreakableMaterial);
TEST_MATERIAL_SIZE(hkpExtendedMeshShapeBreakableMaterial);
TEST_MATERIAL_SIZE(hkpStaticCompoundShapeBreakableMaterial);

#undef TEST_MATERIAL_SIZE

//
//	Compute the maximum shape size for the case of extended mesh shape wrapped with a MOPP

#ifdef HK_PLATFORM_HAS_SPU
	#define TEST_SHAPE_SIZE(shapeType)	HK_COMPILE_TIME_ASSERT(sizeof(shapeType) <= hkpBreakableMaterialUtil::MAX_SHAPE_SIZE)
#else
	#define TEST_SHAPE_SIZE(shapeType)
#endif

TEST_SHAPE_SIZE(hkpMoppBvTreeShape);
TEST_SHAPE_SIZE(hkpListShape);
TEST_SHAPE_SIZE(hkpStaticCompoundShape);
TEST_SHAPE_SIZE(hkpExtendedMeshShape);

#undef TEST_SHAPE_SIZE

HK_COMPILE_TIME_ASSERT(HK_OFFSET_EQUALS(hkpExtendedMeshShape::Subpart, m_shapeInfo, hkpBreakableMaterialUtil::OFFSET_OF_EMS_SHAPE_INFO));

//
//	Creates a hkpSimpleBreakableMaterial clone of the given material

hkpSimpleBreakableMaterial* HK_CALL hkpBreakableMaterialUtil::cloneAsSimpleMaterial(const hkpBreakableMaterial* mtl)
{
	hkpSimpleBreakableMaterial* sm = new hkpSimpleBreakableMaterial(mtl->getMaterialStrength());
	sm->m_properties = mtl->m_properties;
	return sm;
}

//
//	Copies the behaviors and strength from the source to the destination material

void HK_CALL hkpBreakableMaterialUtil::copyMaterialData(hkpBreakableMaterial* dstMtl, const hkpBreakableMaterial* srcMtl)
{
	if ( srcMtl && dstMtl )
	{
		dstMtl->m_properties = srcMtl->m_properties;
		dstMtl->setMaterialStrength(srcMtl->getMaterialStrength());
	}
}

//
//	Collects all materials recursively, starting from the given material

void HK_CALL hkpBreakableMaterialUtil::collectMaterialsRecursive(const hkpBreakableMaterial* rootMtl, hkArray<const hkpBreakableMaterial*>& materialsOut)
{
	// Add current node
	if ( !rootMtl )
	{
		return;
	}
	if ( materialsOut.indexOf(rootMtl) < 0 )
	{
		materialsOut.pushBack(rootMtl);
	}

	// Recurse
	if ( rootMtl->getType() == hkpBreakableMaterial::MATERIAL_TYPE_MULTI )
	{
		const hkpBreakableMultiMaterial* multiMtl = reinterpret_cast<const hkpBreakableMultiMaterial*>(rootMtl);
		for (int k = multiMtl->getNumSubMaterials() - 1; k >= 0; k--)
		{
			collectMaterialsRecursive(multiMtl->getSubMaterial((MaterialId)k), materialsOut);
		}
	}
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
