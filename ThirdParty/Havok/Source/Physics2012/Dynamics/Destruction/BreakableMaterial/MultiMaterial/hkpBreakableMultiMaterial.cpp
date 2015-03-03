/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Destruction/BreakableMaterial/MultiMaterial/hkpBreakableMultiMaterial.h>
#include <Physics2012/Dynamics/Destruction/BreakableMaterial/hkpBreakableMaterialUtil.h>

//
//	Breakable multi-material constructor

hkpBreakableMultiMaterial::hkpBreakableMultiMaterial(const hkArray<hkpBreakableMaterial*>& subMaterials, hkReal strength)
:	hkpBreakableMaterial(DEFAULT_FLAGS, strength)
{
	// Copy sub-materials
	const int numSubMtls = subMaterials.getSize();
	m_subMaterials.setSize(numSubMtls);
	for (int i = 0; i < numSubMtls; i++)
	{
		m_subMaterials[i] = subMaterials[i];
	}
}

//
//	Breakable multi-material copy constructor

hkpBreakableMultiMaterial::hkpBreakableMultiMaterial(const hkpBreakableMultiMaterial& other)
:	hkpBreakableMaterial(other)
,	m_inverseMapping(other.m_inverseMapping)
{
	const int numSubMaterials = other.m_subMaterials.getSize();
	m_subMaterials.setSize(numSubMaterials);
	for (int i = numSubMaterials - 1; i >= 0; i--)
	{
		m_subMaterials[i] = other.m_subMaterials[i];
	}
}

//
//	Inverse mapping constructor

hkpBreakableMultiMaterial::InverseMapping::InverseMapping()
:	hkReferencedObject()
{}

//
//	Breakable multi-material serialization constructor

hkpBreakableMultiMaterial::hkpBreakableMultiMaterial(hkFinishLoadedObjectFlag flag)
:	hkpBreakableMaterial(flag)
,	m_subMaterials(flag)
,	m_inverseMapping(flag)
{}

//
//	Inverse mapping serialization constructor

hkpBreakableMultiMaterial::InverseMapping::InverseMapping(hkFinishLoadedObjectFlag flag)
:	hkReferencedObject(flag)
,	m_descriptors(flag)
,	m_subShapeIds(flag)
{}

//
//	Returns the index of the given material in the m_subMaterials array. If the material is not found, the function returns -1.

int hkpBreakableMultiMaterial::findSubMaterial(const hkpBreakableMaterial* mtl) const
{
	for (int i = m_subMaterials.getSize() - 1; i >= 0; i--)
	{
		if ( m_subMaterials[i] == mtl )
		{
			return i;
		}
	}

	return -1;
}

//
//	Creates the inverse mapping

void hkpBreakableMultiMaterial::createInverseMapping(const hkcdShape* shape)
{
	// Validate input shape
	if ( !shape )
	{
		return;	// No shape!
	}
	const hkpShapeContainer* shapeContainer = reinterpret_cast<const hkpShape*>(shape)->getContainer();
	if ( !shapeContainer )
	{
		return;	// Can't create inverse mapping, the shape is not a container
	}

	// Allocate the mapping
	InverseMapping* im = new InverseMapping();
	m_inverseMapping.setAndDontIncrementRefCount(im);
	const int numSubMtls = m_subMaterials.getSize();
	im->m_descriptors.setSize(numSubMtls);

	hkpShapeKey shapeKey = shapeContainer->getFirstKey();
	while ( shapeKey != HK_INVALID_SHAPE_KEY )
	{
		// Get the material this shape key maps to
		const hkUint32 subShapeId = convertShapeKeyToSubShapeId(shapeKey);
		hkpBreakableMaterial* subMtl = getShapeKeyMaterial(shape, shapeKey);

		// Get the sub-material index
		const int subMaterialIdx = findSubMaterial(subMtl);
		if ( subMaterialIdx >= 0 )
		{
			// Found the material. See if we already stored this inverse mapping
			InverseMappingDescriptor& descriptor = im->m_descriptors[subMaterialIdx];
			int k = descriptor.m_numKeys - 1;
			for (; k >= 0; k--)
			{
				if ( im->m_subShapeIds[descriptor.m_offset + k] == subShapeId )
				{
					break;
				}
			}

			// Inverse mapping is new
			if ( k < 0 )
			{
				im->m_subShapeIds.insertAt(descriptor.m_offset + descriptor.m_numKeys, &subShapeId, 1);
				descriptor.m_numKeys++;

				// Shift all other offsets by one
				for (int di = numSubMtls - 1; di > subMaterialIdx; di--)
				{
					im->m_descriptors[di].m_offset++;
				}
			}
		}

		// Go to the next key
		shapeKey = shapeContainer->getNextKey(shapeKey);
	}
}

//
//	Returns the class type

const hkClass* hkpBreakableMultiMaterial::getClassType() const
{
	return &hkpBreakableMultiMaterialClass;
}

//
//	Returns the number of sub-materials or zero if the material does not support sub-materials

int hkpBreakableMultiMaterial::getNumSubMaterials() const
{
	return m_subMaterials.getSize();
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
