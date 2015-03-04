/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>


void hkpSymmetricAgentFlipCollector::addCdPoint( const hkpCdPoint& point )
{
	hkContactPoint contact; contact.setFlipped( point.getContact() );
	hkpCdPoint cdpoint( point.m_cdBodyB, point.m_cdBodyA, contact);
	m_collector.addCdPoint(cdpoint);
	this->m_earlyOutDistance = m_collector.getEarlyOutDistance();
}


//#if !defined(HK_PLATFORM_SPU)
void hkpSymmetricAgentFlipCastCollector::addCdPoint( const hkpCdPoint& point )
{
	hkContactPoint contact;
	hkVector4 cpPos; cpPos.setAddMul( point.getContact().getPosition(), m_path, point.getContact().getDistanceSimdReal());
	contact.setPosition(cpPos);
	hkVector4 sepNormal; sepNormal.setNeg<3>(point.getContact().getSeparatingNormal());
	contact.setSeparatingNormal(sepNormal);

	hkpCdPoint cdpoint( point.m_cdBodyB, point.m_cdBodyA, contact);
	m_collector.addCdPoint(cdpoint);
	this->m_earlyOutDistance = m_collector.getEarlyOutDistance();
}
//#endif


//#if !defined(HK_PLATFORM_SPU)
void hkpSymmetricAgentFlipBodyCollector::addCdBodyPair( const hkpCdBody& bodyA, const hkpCdBody& bodyB )
{
	m_collector.addCdBodyPair( bodyB, bodyA );
	m_earlyOut = m_collector.getEarlyOut();
}
//#endif

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
