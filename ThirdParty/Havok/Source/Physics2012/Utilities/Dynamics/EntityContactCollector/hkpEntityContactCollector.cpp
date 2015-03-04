/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Dynamics/EntityContactCollector/hkpEntityContactCollector.h>
#include <Physics2012/Collide/Agent/hkpProcessCdPoint.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Agent/Collidable/hkpCollidable.h>
#include <Physics2012/Dynamics/Entity/hkpEntity.h>
		

hkpEntityContactCollector::~hkpEntityContactCollector()
{
	while( m_entities.getSize() )
	{
		removeFromEntity( m_entities[0] );
	}
}

void hkpEntityContactCollector::flipContactPoints( hkpRigidBody* body )
{
	for (int i = 0; i < m_contactPoints.getSize(); i++ )
	{
		ContactPoint& cp = m_contactPoints[i];
		if ( cp.m_bodyB == body ) 
		{
			cp.m_point.flip();
			cp.m_bodyB = cp.m_bodyA;
			cp.m_bodyA = body;
		}
	}
}

void hkpEntityContactCollector::reset()
{
	m_contactPoints.clear();
}


void hkpEntityContactCollector::addToEntity( hkpEntity* entity )
{
	entity->addContactListener( this );
	entity->addEntityListener( this );
	m_entities.pushBack(entity);
}

void hkpEntityContactCollector::removeFromEntity( hkpEntity* entity )
{
	HK_ASSERT2(0x1d933c54,  m_entities.indexOf( entity ) != -1, "Trying to remove a contact collector from an entity to which it has not been added");
	m_entities.removeAt( m_entities.indexOf( entity ) );
	entity->removeContactListener( this );
	entity->removeEntityListener( this );
}


void hkpEntityContactCollector::contactPointCallback( const hkpContactPointEvent& event )
{
	if ( !event.isToi() )
	{
		HK_ASSERT2( 0x2cc4eae6, m_contactPoints.getSize() < 1023, "Warning: Too many contact points gathered, are you sure you called hkpEntityContactCollector::reset() every frame ??" );
		ContactPoint& cp = m_contactPoints.expandOne();
		cp.m_bodyA = event.m_bodies[0];
		cp.m_bodyB = event.m_bodies[1];
		cp.m_point = *event.m_contactPoint;
	}
}


void hkpEntityContactCollector::entityDeletedCallback( hkpEntity* entity )
{
	removeFromEntity( entity );
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
