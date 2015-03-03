/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Actions/Wind/hkpWindAction.h>	

hkpWindAction::hkpWindAction( hkpRigidBody* body, const hkpWind* wind, hkReal resistanceFactor, hkReal obbFactor )
:	hkpUnaryAction( body ), 
	m_wind( wind ),
	m_resistanceFactor( resistanceFactor ),
	m_obbFactor( obbFactor )
{
	m_wind->addReference();
}

hkpWindAction::~hkpWindAction()
{
	m_wind->removeReference();
}

hkpAction* hkpWindAction::clone( const hkArray<hkpEntity*>& newEntities, const hkArray<hkpPhantom*>& newPhantoms ) const
{
	HK_ASSERT2(0xf578efca, newEntities.getSize() == 1, "Wrong clone parameters given to an aerodynamics action (needs 1 body).");
	if (newEntities.getSize() != 1) return HK_NULL;

	HK_ASSERT2(0x277857f0, newPhantoms.getSize() == 0, "Wrong clone parameters given to an aerodynamics action (needs 0 phantoms).");
	// should have no phantoms.
	if (newPhantoms.getSize() != 0) return HK_NULL;

	hkpWindAction* aa = new hkpWindAction( (hkpRigidBody*)newEntities[0], m_wind, m_resistanceFactor, m_obbFactor );
	aa->m_userData = m_userData;

	return aa;
}

void hkpWindAction::applyAction( const hkStepInfo& stepInfo )
{
	hkpRigidBody* rb = getRigidBody();

	m_wind->applyWindAndResistance( rb, stepInfo.m_deltaTime, m_resistanceFactor, m_obbFactor );
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
