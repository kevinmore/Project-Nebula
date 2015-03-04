/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Extensions/AutoLookAheadDistance/hknpAutoLookAheadDistanceUtil.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>


hknpAutoLookAheadDistanceUtil::hknpAutoLookAheadDistanceUtil( hknpWorld* world )
{
	HK_ASSERT( 0x3412bcd1, world );
	m_world = world;
	m_world->m_signals.m_preCollide.subscribe( this, &hknpAutoLookAheadDistanceUtil::onPreCollide, "hknpAutoLookAheadDistanceUtil" );
	m_world->m_signals.m_bodyDestroyed.subscribe( this, &hknpAutoLookAheadDistanceUtil::onBodyDestroyed, "hknpAutoLookAheadDistanceUtil" );
}

hknpAutoLookAheadDistanceUtil::~hknpAutoLookAheadDistanceUtil()
{
	m_world->m_signals.m_preCollide.unsubscribeAll( this );
	m_world->m_signals.m_bodyDestroyed.unsubscribeAll( this );
}

void hknpAutoLookAheadDistanceUtil::registerBody( hknpBodyId id )
{
	HK_ASSERT2( 0x3412bcd2, m_world->isBodyValid( id ), "Invalid body ID" );
	m_registeredBodies.pushBack( id );
}

void hknpAutoLookAheadDistanceUtil::unregisterBody( hknpBodyId id )
{
	int index = m_registeredBodies.lastIndexOf( id );
	if( index >= 0 )
	{
		m_registeredBodies.removeAt( index );
	}
}

void hknpAutoLookAheadDistanceUtil::onPreCollide( hknpWorld* world )
{
	HK_ASSERT( 0x3412bcd3, world == m_world );

	//
	//	Search all registered objects.
	//  For found dynamic bodies we set their temporary linear velocity cage to the velocity of the searched object
	//  to provide the collision system with a hint of the direction the bodies will be displaced in.
	//
	for( int i = 0; i < m_registeredBodies.getSize(); ++i )
	{
		hknpBodyId bodyId = m_registeredBodies[i];
		if( world->getBody(bodyId).isDynamic() )
		{
			const hkVector4& linVel = m_world->getBodyLinearVelocity( bodyId );

			// Query for overlapping expanded AABBs
			hkAabb aabb; m_world->getBodyAabb( bodyId, aabb );
			hkInplaceArray<hknpBodyId,32> hits;		
			m_world->queryAabb( aabb, hits );
			for( int k = 0; k < hits.getSize(); k++ )
			{
				const hknpBodyId hitId = hits[k];
				if( hitId == bodyId )	// ignore self
				{
					continue;
				}

				const hknpBody& hitBody = m_world->getBody( hitId );
				if( hitBody.isStaticOrKeyframed() )
				{
					continue;
				}

				// Set the look ahead distance, using the linear velocity
				m_world->setBodyCollisionLookAheadDistance( hitId, -1.0f, linVel );
			}
		}
	}
}

void hknpAutoLookAheadDistanceUtil::onBodyDestroyed( hknpWorld* world, hknpBodyId bodyId )
{
	HK_ASSERT( 0x3412bcd4, world == m_world );
	unregisterBody( bodyId );
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
