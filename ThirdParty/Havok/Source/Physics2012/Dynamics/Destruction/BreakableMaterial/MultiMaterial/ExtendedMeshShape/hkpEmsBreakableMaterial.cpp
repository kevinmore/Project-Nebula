/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Destruction/BreakableMaterial/MultiMaterial/ExtendedMeshShape/hkpEmsBreakableMaterial.h>

//
//	Constructor

hkpExtendedMeshShapeBreakableMaterial::hkpExtendedMeshShapeBreakableMaterial(const hkArray<hkpBreakableMaterial*>& subMaterials, int numBitsPerSubPart, hkReal strength)
:	hkpBreakableMultiMaterial(subMaterials, strength)
{
	setExtraData((hkUint8)(32 - numBitsPerSubPart));
	setMapping(hkpBreakableMaterial::MATERIAL_MAPPING_EMS);
}

//
//	Return the class types

const hkClass* hkpExtendedMeshShapeBreakableMaterial::getClassType() const
{
	return &hkpExtendedMeshShapeBreakableMaterialClass;
}

//
//	Serialization constructor

hkpExtendedMeshShapeBreakableMaterial::hkpExtendedMeshShapeBreakableMaterial(hkFinishLoadedObjectFlag flag)
:	hkpBreakableMultiMaterial(flag)
{}

//
//	Copy constructor

hkpExtendedMeshShapeBreakableMaterial::hkpExtendedMeshShapeBreakableMaterial(const hkpExtendedMeshShapeBreakableMaterial& other)
:	hkpBreakableMultiMaterial(other)
{}

//
//	Creates the inverse mapping. Calls the base class and additionally encodes the maximum shape key for each sub-shape id

void hkpExtendedMeshShapeBreakableMaterial::createInverseMapping(const hkcdShape* shape)
{
	// Call base class
	hkpBreakableMultiMaterial::createInverseMapping(shape);

	// Get the hkpExtendedMeshShape. This can either be the root shape or if the root is a MOPP, its child
	if ( shape->getType() == hkcdShapeType::MOPP )
	{
		const hkpMoppBvTreeShape* mopp = reinterpret_cast<const hkpMoppBvTreeShape*>(shape);
		shape = mopp->getChild();
	}
	HK_ASSERT2(0x623113be, shape->getType() == hkcdShapeType::EXTENDED_MESH, "Failed to access the hkpExtendedMeshShape");
	const hkpExtendedMeshShape* ems = reinterpret_cast<const hkpExtendedMeshShape*>(shape);

	// Post-process the shape-keys by encoding the maximum shape key
	InverseMapping* imap = m_inverseMapping;
	const hkArray<InverseMappingDescriptor>& descriptors = imap->m_descriptors;
	hkpShapeKey* shapeKeys = imap->m_subShapeIds.begin();
	for (int di = descriptors.getSize() - 1; di >= 0; di--)
	{
		const InverseMappingDescriptor& descriptor = descriptors[di];
		for (int si = descriptor.m_numKeys - 1; si >= 0; si--)
		{
			hkUint32& subShapeId = shapeKeys[descriptor.m_offset + si];

			// Get number of primitives on the sub-part
			int numPrimitives;
			const hkpExtendedMeshShape::Subpart& subPart = ems->getSubPart((hkpShapeKey)subShapeId);
			if ( subPart.getType() == hkpExtendedMeshShape::SUBPART_TRIANGLES )
			{
				const hkpExtendedMeshShape::TrianglesSubpart& tris = reinterpret_cast<const hkpExtendedMeshShape::TrianglesSubpart&>(subPart);
				numPrimitives = tris.m_numTriangleShapes;		
			}
			else
			{
				const hkpExtendedMeshShape::ShapesSubpart& shapes = reinterpret_cast<const hkpExtendedMeshShape::ShapesSubpart&>(subPart);
				numPrimitives = shapes.m_childShapes.getSize();
			}

			subShapeId |= (numPrimitives - 1);
		}
	}
}

//
//	Clones the given material

hkpBreakableMaterial* hkpExtendedMeshShapeBreakableMaterial::duplicate()
{
	return new hkpExtendedMeshShapeBreakableMaterial(*this);
}

//
//	Sets the default mapping

void hkpExtendedMeshShapeBreakableMaterial::setDefaultMapping()
{
	setMapping(hkpBreakableMaterial::MATERIAL_MAPPING_EMS);
}

//
//	Returns the index of the sub-material belonging to the given sub-shape.

hkpBreakableMaterial::MaterialId hkpExtendedMeshShapeBreakableMaterial::getSubShapeMaterialIndex(const hkcdShape* rootShape, hkUint32 subShapeId) const
{
	const hkpExtendedMeshShape* ems = getExtendedMeshShape(rootShape);
	return (MaterialId)ems->getShapeInfo((hkpShapeKey)subShapeId);
}

//
//	Returns the material set on the given shape key

hkpBreakableMaterial* hkpExtendedMeshShapeBreakableMaterial::getShapeKeyMaterial(const hkcdShape* shapePpu, hkpShapeKey shapeKey) const
{
	const hkcdShape* shapeSpu = getExtendedMeshShape(shapePpu);
	HK_ASSERT2(0x623113be, shapeSpu->getType() == hkcdShapeType::EXTENDED_MESH, "Failed to access the hkpExtendedMeshShape");
	const hkpExtendedMeshShape* ems = reinterpret_cast<const hkpExtendedMeshShape*>(shapeSpu);

	// Must retrieve the sub-material id. Get sub-part, the sub-material id is stored in the userData.
	const MaterialId subMaterialIdx = ems->getShapeInfo(shapeKey);
	return const_cast<hkpBreakableMaterial*>(getSubMaterial(subMaterialIdx));
}

//
//	Converts a shape key into a sub-shape id the material is working with.

hkUint32 hkpExtendedMeshShapeBreakableMaterial::convertShapeKeyToSubShapeId(hkpShapeKey shapeKey) const
{
	// Remove the terminal bits
	const int numTerminalBits	= getExtraData();
	const int terminalMask		= (0xFFFFFFFF << numTerminalBits);
	const int subPartId			= (shapeKey & terminalMask);
	return (hkUint32)subPartId;
}

//
//	Converts a set of shape keys into their corresponding sub-shape ids in-place

void hkpExtendedMeshShapeBreakableMaterial::convertShapeKeysToSubShapeIds(hkArray<hkpShapeKey>& shapeKeysInOut) const
{
	// Remove the terminal bits
	const int numTerminalBits	= getExtraData();
	const int terminalMask		= (0xFFFFFFFF << numTerminalBits);

	for (int i = shapeKeysInOut.getSize() - 1; i >= 0; i--)
	{
		shapeKeysInOut[i] &= terminalMask;
	}
}

//
//	Collects the shape keys belonging to the given sub-shape

void hkpExtendedMeshShapeBreakableMaterial::getShapeKeysForSubShapes(const hkcdShape* /*rootShape*/, const hkUint32* subShapeIdPtr, int numSubShapeIds, ShapeKeyCollector* collector) const
{
	const int numTerminalBits	= getExtraData();
	const int shapeKeyMask		= (0xFFFFFFFF << numTerminalBits);
	const int terminalMask		= ~shapeKeyMask;

	// Get sub-part, we will return all its children
	for (int i = 0; i < numSubShapeIds; i++)
	{
		const hkUint32 subShapeId = subShapeIdPtr[i];
		const hkpShapeKey shapeKeyBase = subShapeId & shapeKeyMask;
		const int numPrimitives = (subShapeId & terminalMask) + 1;

		collector->addContiguousShapeKeyRange(shapeKeyBase, numPrimitives);
	}
}

//
//	Appends the sub-material indices set on the given sub-shapes to the given array

void hkpExtendedMeshShapeBreakableMaterial::getSubShapeMaterialIndices(const hkcdShape* rootShape, const hkArray<hkUint32>& subShapeIdsIn, hkArray<MaterialId>& subMaterialsOut) const
{
	const int numSubShapes = subShapeIdsIn.getSize();
	MaterialId* mtlPtr = subMaterialsOut.expandBy(numSubShapes);

	const hkpExtendedMeshShape* emsShape = getExtendedMeshShape(rootShape);
	for (int i = numSubShapes - 1; i >= 0; i--)
	{
		mtlPtr[i] = (MaterialId)emsShape->getShapeInfo((hkpShapeKey)subShapeIdsIn[i]);
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
