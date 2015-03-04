/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Destruction/BreakableMaterial/MultiMaterial/ListShape/hkpListShapeBreakableMaterial.h>

//
//	Constructor

hkpListShapeBreakableMaterial::hkpListShapeBreakableMaterial(const hkArray<hkpBreakableMaterial*>& subMaterials, hkReal strength)
:	hkpBreakableMultiMaterial(subMaterials, strength)
{
	setMapping(hkpBreakableMaterial::MATERIAL_MAPPING_LIST_SHAPE);
}

//
//	Return the class types

const hkClass* hkpListShapeBreakableMaterial::getClassType() const
{
	return &hkpListShapeBreakableMaterialClass;
}

//
//	Serialization constructor

hkpListShapeBreakableMaterial::hkpListShapeBreakableMaterial(hkFinishLoadedObjectFlag flag)
:	hkpBreakableMultiMaterial(flag)
{}

//
//	Copy constructor

hkpListShapeBreakableMaterial::hkpListShapeBreakableMaterial(const hkpListShapeBreakableMaterial& other)
:	hkpBreakableMultiMaterial(other)
{}

//
//	Clones the given material

hkpBreakableMaterial* hkpListShapeBreakableMaterial::duplicate()
{
	return new hkpListShapeBreakableMaterial(*this);
}

//
//	Sets the default mapping

void hkpListShapeBreakableMaterial::setDefaultMapping()
{
	setMapping(hkpBreakableMaterial::MATERIAL_MAPPING_LIST_SHAPE);
}

//
//	Returns the index of the sub-material belonging to the given sub-shape.

hkpBreakableMaterial::MaterialId hkpListShapeBreakableMaterial::getSubShapeMaterialIndex(const hkcdShape* rootShape, hkUint32 subShapeId) const
{
	const hkpListShape* ls = getListShape(rootShape);
	return (MaterialId)ls->getShapeInfo(subShapeId);
}

//
//	Returns the material set on the given shape key

hkpBreakableMaterial* hkpListShapeBreakableMaterial::getShapeKeyMaterial(const hkcdShape* shapePpu, hkpShapeKey subShapeKey) const
{
	const hkcdShape* shapeSpu = getListShape(shapePpu);

	HK_ASSERT2(0x623113be, shapeSpu->getType() == hkcdShapeType::LIST, "Failed to access the hkpListShape");
	const hkpListShape* ls		= reinterpret_cast<const hkpListShape*>(shapeSpu);
	const MaterialId subMtlIdx	= (MaterialId)ls->getShapeInfo(subShapeKey);
	return const_cast<hkpBreakableMaterial*>(getSubMaterial(subMtlIdx));
}

//
//	Disables a set of sub-shapes based on their sub-material Id

void hkpListShapeBreakableMaterial::disableSubShapes(hkcdShape* rootShape, const MaterialId* subMaterialIndices, int numSubMaterialIndices)
{
	hkpListShape* listShape = const_cast<hkpListShape*>(getListShape(rootShape));

	// For each sub-material, disable the corresponding shape keys
	for (int i = numSubMaterialIndices - 1; i >= 0; i--)
	{
		const int subMaterialIndex = subMaterialIndices[i];
		const InverseMappingDescriptor& descriptor = m_inverseMapping->m_descriptors[subMaterialIndex];

		const hkpShapeKey* shapeKeyPtr = &m_inverseMapping->m_subShapeIds[descriptor.m_offset];
		for (int k = descriptor.m_numKeys - 1; k >= 0; k--)
		{
			listShape->disableChild(shapeKeyPtr[k]);	
		}
	}
}

//
//	Collects the shape keys belonging to the given sub-shape

void hkpListShapeBreakableMaterial::getShapeKeysForSubShapes(const hkcdShape* rootShape, const hkUint32* subShapeIdPtr, int numSubShapeIds, ShapeKeyCollector* collector) const
{
	// No conversion, there's only one shape key
	collector->addShapeKeyBatch(subShapeIdPtr, numSubShapeIds);
}

//
//	Appends the sub-material indices set on the given sub-shapes to the given array

void hkpListShapeBreakableMaterial::getSubShapeMaterialIndices(const hkcdShape* rootShape, const hkArray<hkUint32>& subShapeIdsIn, hkArray<MaterialId>& subMaterialsOut) const
{
	const hkpListShape* ls = getListShape(rootShape);
	
	const int numSubShapes = subShapeIdsIn.getSize();
	MaterialId* mtlPtr = subMaterialsOut.expandBy(numSubShapes);
	for(int i = numSubShapes - 1; i >= 0; i--)
	{
		mtlPtr[i] = (MaterialId)ls->getShapeInfo(subShapeIdsIn[i]);
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
