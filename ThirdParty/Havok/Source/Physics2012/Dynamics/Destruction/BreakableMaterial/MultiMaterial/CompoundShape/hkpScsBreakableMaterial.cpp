/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Destruction/BreakableMaterial/MultiMaterial/CompoundShape/hkpScsBreakableMaterial.h>

//
//	Constructor

hkpStaticCompoundShapeBreakableMaterial::hkpStaticCompoundShapeBreakableMaterial(const hkArray<hkpBreakableMaterial*>& subMaterials, hkReal strength)
:	hkpBreakableMultiMaterial(subMaterials, strength)
{
	setMapping(hkpBreakableMaterial::MATERIAL_MAPPING_STATIC_COMPOUND);
}

//
//	Return the class types

const hkClass* hkpStaticCompoundShapeBreakableMaterial::getClassType() const
{
	return &hkpStaticCompoundShapeBreakableMaterialClass;
}

//
//	Serialization constructor

hkpStaticCompoundShapeBreakableMaterial::hkpStaticCompoundShapeBreakableMaterial(hkFinishLoadedObjectFlag flag)
:	hkpBreakableMultiMaterial(flag)
{}

//
//	Copy constructor

hkpStaticCompoundShapeBreakableMaterial::hkpStaticCompoundShapeBreakableMaterial(const hkpStaticCompoundShapeBreakableMaterial& other)
:	hkpBreakableMultiMaterial(other)
{}

//
//	Clones the given material

hkpBreakableMaterial* hkpStaticCompoundShapeBreakableMaterial::duplicate()
{
	return new hkpStaticCompoundShapeBreakableMaterial(*this);
}

//
//	Sets the default mapping

void hkpStaticCompoundShapeBreakableMaterial::setDefaultMapping()
{
	setMapping(hkpBreakableMaterial::MATERIAL_MAPPING_STATIC_COMPOUND);
}

//
//	Returns the index of the sub-material belonging to the given sub-shape.

hkpBreakableMaterial::MaterialId hkpStaticCompoundShapeBreakableMaterial::getSubShapeMaterialIndex(const hkcdShape* rootShape, hkUint32 subShapeId) const
{
	// Get the compound shape
	HK_ASSERT2(0x623113be, rootShape->getType() == hkcdShapeType::STATIC_COMPOUND, "Failed to access the static compound shape");
	const StaticCompoundShape* cpShape = reinterpret_cast<const StaticCompoundShape*>(rootShape);
	return (MaterialId)cpShape->getInstanceExtraInfo(subShapeId);
}

//
//	Returns the material set on the given shape key

hkpBreakableMaterial* hkpStaticCompoundShapeBreakableMaterial::getShapeKeyMaterial(const hkcdShape* shapePpu, hkpShapeKey shapeKey) const
{
	HK_ASSERT2(0x623113be, shapePpu->getType() == hkcdShapeType::STATIC_COMPOUND, "Failed to access the static compound shape");
	const StaticCompoundShape* cp = reinterpret_cast<const StaticCompoundShape*>(shapePpu);

	// Must retrieve the sub-material id.
	const int numTerminalBits	= getExtraData();
	const hkUint32 instanceId	= (hkUint32)(shapeKey >> numTerminalBits);
	const MaterialId mtlIdx		= cp->getInstanceExtraInfo(instanceId);

	// Return sub-material on PPU
	return const_cast<hkpBreakableMaterial*>(getSubMaterial(mtlIdx));
}

//
//	Converts a shape key into a sub-shape id the material is working with.

hkUint32 hkpStaticCompoundShapeBreakableMaterial::convertShapeKeyToSubShapeId(hkpShapeKey shapeKey) const
{
	const int numTerminalBits	= getExtraData();
	const hkUint32 instanceId	= (hkUint32)(shapeKey >> numTerminalBits);
	return instanceId;
}

//
//	Converts a set of shape keys into their corresponding sub-shape ids in-place

void hkpStaticCompoundShapeBreakableMaterial::convertShapeKeysToSubShapeIds(hkArray<hkpShapeKey>& shapeKeysInOut) const
{
	const int numTerminalBits = getExtraData();
	for (int i = shapeKeysInOut.getSize() - 1; i >= 0; i--)
	{
		shapeKeysInOut[i] = (hkUint32)(shapeKeysInOut[i] >> numTerminalBits);
	}
}

//
//	Disables a set of sub-shapes based on their sub-material Id

void hkpStaticCompoundShapeBreakableMaterial::disableSubShapes(hkcdShape* rootShape, const MaterialId* subMaterialIndices, int numSubMaterialIndices)
{
	// Get the compound shape
	HK_ASSERT2(0x623113be, rootShape->getType() == hkcdShapeType::STATIC_COMPOUND, "Failed to access the static compound shape");
	StaticCompoundShape* cp = reinterpret_cast<StaticCompoundShape*>(rootShape);

	// For each sub-material, get the corresponding nodes and disable them
	for (int i = numSubMaterialIndices - 1; i >= 0; i--)
	{
		const int subMaterialIndex = subMaterialIndices[i];
		const InverseMappingDescriptor& descriptor = m_inverseMapping->m_descriptors[subMaterialIndex];

		// The sub-shape Ids stored in the inverse mapping are actually the physics shape's instance ids.
		const hkUint32* instanceIdPtr = &m_inverseMapping->m_subShapeIds[descriptor.m_offset];

		for (int k = descriptor.m_numKeys - 1; k >= 0; k--)
		{
			// Get the instance id and disable it
			const hkUint32 instanceId = (hkUint32)(instanceIdPtr[k]);
			cp->setInstanceEnabled(instanceId, false);
		}
	}
}

//
//	Collects the shape keys belonging to the given sub-shape

void hkpStaticCompoundShapeBreakableMaterial::getShapeKeysForSubShapes(const hkcdShape* rootShape, const hkUint32* subShapeIdPtr, int numSubShapeIds, ShapeKeyCollector* collector) const
{
	// Get the compound shape
	HK_ASSERT2(0x623113be, rootShape->getType() == hkcdShapeType::STATIC_COMPOUND, "Failed to access the static compound shape");
	const StaticCompoundShape* cpShape = reinterpret_cast<const StaticCompoundShape*>(rootShape);

	const int numTerminalBits = getExtraData();
	for (int i = 0; i < numSubShapeIds; i++)
	{
		const int instanceId			= subShapeIdPtr[i];
		const hkpShapeKey shapeKeyBase	= (instanceId << numTerminalBits);

		// See if we have a container
		const hkpShape* childShape = cpShape->getInstances()[instanceId].getShape();
		const hkpShapeContainer* container = childShape->getContainer();
		if ( container )
		{
			hkpShapeKey subKey = container->getFirstKey();
			while ( subKey != HK_INVALID_SHAPE_KEY )
			{
				collector->addShapeKey(shapeKeyBase | subKey);
				subKey = container->getNextKey(subKey);
			}
		}
		else
		{
			collector->addShapeKey(shapeKeyBase);
		}
	}
}

//
//	Appends the sub-material indices set on the given sub-shapes to the given array

void hkpStaticCompoundShapeBreakableMaterial::getSubShapeMaterialIndices(const hkcdShape* rootShape, const hkArray<hkUint32>& subShapeIdsIn, hkArray<MaterialId>& subMaterialsOut) const
{
	// Get the compound shape
	HK_ASSERT2(0x623113be, rootShape->getType() == hkcdShapeType::STATIC_COMPOUND, "Failed to access the static compound shape");
	const StaticCompoundShape* cpShape = reinterpret_cast<const StaticCompoundShape*>(rootShape);

	const int numSubShapes = subShapeIdsIn.getSize();
	MaterialId* mtlPtr = subMaterialsOut.expandBy(numSubShapes);
	for (int i = numSubShapes - 1; i >= 0; i--)
	{
		mtlPtr[i] = (MaterialId)cpShape->getInstanceExtraInfo(subShapeIdsIn[i]);
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
