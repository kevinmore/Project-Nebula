/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/World/ShapeManager/hknpShapeManager.h>


hknpShapeManager::MutableShapeInfo::MutableShapeInfo( hknpShapeManager *shapeManager )
{
	m_shapeManager = shapeManager;
	m_shape = HK_NULL;
	m_mutations.clear();
}

hknpShapeManager::MutableShapeInfo::~MutableShapeInfo()
{
	if( m_shape )
	{
		deinit();
	}
}

void hknpShapeManager::MutableShapeInfo::init( const hknpShape* shape )
{
	m_shape = shape;
	m_mutations.clear();

	hknpShape::MutationSignals* signals = const_cast<hknpShape*>(m_shape)->getMutationSignals();
	HK_ASSERT( 0x12ab23c8, signals );
	signals->m_shapeMutated.subscribe( this, &hknpShapeManager::MutableShapeInfo::onShapeMutated, "hknpShapeManager" );
	signals->m_shapeDestroyed.subscribe( this, &hknpShapeManager::MutableShapeInfo::onShapeDestroyed, "hknpShapeManager" );
}

void hknpShapeManager::MutableShapeInfo::deinit()
{
	hknpShape::MutationSignals* signals = const_cast<hknpShape*>(m_shape)->getMutationSignals();
	signals->m_shapeMutated.unsubscribe( this, &hknpShapeManager::MutableShapeInfo::onShapeMutated );
	signals->m_shapeDestroyed.unsubscribe( this, &hknpShapeManager::MutableShapeInfo::onShapeDestroyed );

	m_shape = HK_NULL;
}

void hknpShapeManager::MutableShapeInfo::onShapeMutated( hkUint8 mutationFlags )
{
	m_mutations.orWith( mutationFlags );
	m_shapeManager->m_isAnyShapeMutated = true;
}

void hknpShapeManager::MutableShapeInfo::onShapeDestroyed()
{
	deinit();
	m_shapeManager->m_isAnyShapeMutated = true;
}


hknpShapeManager::hknpShapeManager()
	: m_isAnyShapeMutated(false)
{

}

hknpShapeManager::~hknpShapeManager()
{
	// Free all allocated mutable shape infos
	for( int i=0, ei=m_mutableShapeInfos.getSize(); i<ei; i++ )
	{
		delete m_mutableShapeInfos[i];
	}
	for( int i=0, ei=m_freeMutableShapeInfos.getSize(); i<ei; i++ )
	{
		delete m_freeMutableShapeInfos[i];
	}
}

void hknpShapeManager::registerBodyWithMutableShape( hknpBody& body )
{
	HK_ASSERT( 0x12bb23d8, body.m_shape->isMutable() );

	// Find the mutable shape info (linear search)
	int shapeIndex = -1;
	for( int i=0, ei=m_mutableShapeInfos.getSize(); i<ei; i++ )
	{
		if( m_mutableShapeInfos[i]->m_shape == body.m_shape )
		{
			shapeIndex = i;
			break;
		}
	}

	// If we didn't find it, create a new one
	if( shapeIndex == -1 )
	{
		shapeIndex = m_mutableShapeInfos.getSize();
		if( m_freeMutableShapeInfos.isEmpty() )
		{
			m_mutableShapeInfos.pushBack( new MutableShapeInfo(this) );
		}
		else
		{
			m_mutableShapeInfos.pushBack( m_freeMutableShapeInfos.back() );
			m_freeMutableShapeInfos.popBack();
		}
		m_mutableShapeInfos[shapeIndex]->init( body.m_shape );
		HK_ASSERT( 0x12bb23c8, m_mutableShapeInfos[shapeIndex]->m_bodyIds.isEmpty() );
	}

	// Add the body to the mutable shape info
	m_mutableShapeInfos[shapeIndex]->m_bodyIds.pushBack( body.m_id );
}

void hknpShapeManager::deregisterBodyWithMutableShape( hknpBody& body )
{
	HK_ASSERT( 0x12bb23d8, body.m_shape->isMutable() );

	// Find the shape
	int shapeIndex = -1;
	for ( int i=0, ei=m_mutableShapeInfos.getSize(); i<ei; i++ )
	{
		if ( m_mutableShapeInfos[i]->m_shape == body.m_shape )
		{
			shapeIndex = i;
			break;
		}
	}
	HK_ASSERT( 0x12bb23d9, shapeIndex != -1 );

	// Find the body
	int bodyIndex = -1;
	MutableShapeInfo* mutShapeInfo = m_mutableShapeInfos[shapeIndex];
	for ( int i=0, ei=mutShapeInfo->m_bodyIds.getSize(); i<ei; i++ )
	{
		if( mutShapeInfo->m_bodyIds[i] == body.m_id )
		{
			bodyIndex = i;
			break;
		}
	}
	HK_ASSERT( 0x12bb23d9, bodyIndex != -1 );

	// Remove the body
	mutShapeInfo->m_bodyIds.removeAt( bodyIndex );

	// If no bodies are left, free the shape info
	if( mutShapeInfo->m_bodyIds.isEmpty() )
	{
		mutShapeInfo->deinit();
		m_mutableShapeInfos.removeAt( shapeIndex );
		m_freeMutableShapeInfos.pushBack( mutShapeInfo );
	}
}


extern void hknpWorld_updateBodyAabbOfAcceleratedMotion( hknpWorld* world, hknpBodyId bodyId, const hknpMotion& motion );

void hknpShapeManager::processMutatedShapes( hknpWorld* world )
{
	// Early out of there is nothing to do
	if( !m_isAnyShapeMutated )
	{
		return;
	}

	for( int i=m_mutableShapeInfos.getSize()-1; i>=0; i-- )
	{
		MutableShapeInfo* shapeInfo = m_mutableShapeInfos[i];

		if( shapeInfo->m_shape == HK_NULL )
		{
			// Move to free list
			m_mutableShapeInfos.removeAt( i );
			m_freeMutableShapeInfos.pushBack( shapeInfo );
			continue;
		}

		if( shapeInfo->m_mutations.get() == 0 )
		{
			// No mutations
			continue;
		}

		for( int j=0, ej=shapeInfo->m_bodyIds.getSize(); j<ej; j++ )
		{
			hknpBodyId bodyId = shapeInfo->m_bodyIds[j];
			hknpBody& body = world->accessBody( bodyId );

			
			
			

			if( shapeInfo->m_mutations.anyIsSet(hknpShape::MUTATION_AABB_CHANGED) )
			{
				if( body.isDynamic() )
				{
					const hknpMotion& motion = world->getMotion(body.m_motionId );
					body.updateComCenteredBoundingRadius( motion );
					hknpWorld_updateBodyAabbOfAcceleratedMotion( world, bodyId, motion );
				}
				else
				{
					world->updateMotionAndAttachedBodiesAfterModifyingTransform(
						bodyId, HK_NULL, hknpWorld::PIVOT_CENTER_OF_MASS, hknpActivationBehavior::ACTIVATE_NEW_OVERLAPS );
				}
			}

			// We don't have to do the other updates if the body has not been added to the world
			if( !body.isAddedToWorld() )
			{
				continue;
			}

			if( shapeInfo->m_mutations.anyIsSet(hknpShape::MUTATION_REBUILD_COLLISION_CACHES) )
			{
				world->rebuildBodyCollisionCaches( bodyId );
			}

			if( shapeInfo->m_mutations.anyIsSet(hknpShape::MUTATION_DISCARD_CACHED_DISTANCES) )
			{
				// Kill any cached distances by setting the TIMs to max
				body.m_maxTimDistance = 0xffff;
				body.m_timAngle = 0xff;
				if( body.isStatic() )
				{
					world->m_bodyManager.setScheduledBodyFlags( bodyId, hknpBodyManager::MOVED_STATIC );
				}
			}

			if( body.isInactive() )
			{
				world->activateBody( bodyId );
			}

			if( body.isStatic() )
			{
				world->activateBodiesInAabb( body.m_aabb );
			}
		}
		shapeInfo->m_mutations.clear();
	}

	m_isAnyShapeMutated = false;
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
