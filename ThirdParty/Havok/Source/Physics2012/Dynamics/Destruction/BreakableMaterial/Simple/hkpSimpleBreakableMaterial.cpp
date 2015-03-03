/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Destruction/BreakableMaterial/Simple/hkpSimpleBreakableMaterial.h>

//
//	Constructor

hkpSimpleBreakableMaterial::hkpSimpleBreakableMaterial(hkReal strength)
:	hkpBreakableMaterial(DEFAULT_FLAGS, strength)
{}

//
//	Returns the class type

const hkClass* hkpSimpleBreakableMaterial::getClassType() const
{
	return &hkpSimpleBreakableMaterialClass;
}

//
//	Serialization constructor

hkpSimpleBreakableMaterial::hkpSimpleBreakableMaterial(hkFinishLoadedObjectFlag flag)
:	hkpBreakableMaterial(flag)
{
	if ( flag.m_finishing )
	{
		m_typeAndFlags = DEFAULT_FLAGS;
	}
}

//
//	Copy constructor

hkpSimpleBreakableMaterial::hkpSimpleBreakableMaterial(const hkpSimpleBreakableMaterial& other)
:	hkpBreakableMaterial(other)
{}

//
//	Clones the given material

hkpBreakableMaterial* hkpSimpleBreakableMaterial::duplicate()
{
	return new hkpSimpleBreakableMaterial(*this);
}

//
//	Sets the default mapping

void hkpSimpleBreakableMaterial::setDefaultMapping()
{
	setMapping(hkpBreakableMaterial::MATERIAL_MAPPING_NONE);
}

//
//	Returns the material set on the given shape key

hkpBreakableMaterial* hkpSimpleBreakableMaterial::getShapeKeyMaterial(const hkcdShape* shapePpu, hkpShapeKey shapeKey) const
{
	return const_cast<hkpSimpleBreakableMaterial*>(this);
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
