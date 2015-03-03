/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Dispatcher/hknpCollisionDispatcher.h>

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSignedDistanceCollisionCache.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>


namespace
{
	// Dummy cache creation function
	static void HK_CALL nullCacheCreator(
		const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
		hknpCdCacheWriter* cacheWriterInOut )
	{
		HK_ERROR( 0x30ff763a, "Unhandled cache creator dispatch" );
	}
}


hknpCollisionDispatcher::hknpCollisionDispatcher()
{
	// Initialize force creation mask for collision detection events
	m_keyframeCacheCreationMask.setAll(
		hknpBody::RAISE_TRIGGER_VOLUME_EVENTS |
		hknpBody::RAISE_MANIFOLD_STATUS_EVENTS |
		hknpBody::RAISE_MANIFOLD_PROCESSED_EVENTS |
		hknpBody::RAISE_CONTACT_IMPULSE_EVENTS );

	// Initialize table to dummy function
	for( int i=0; i<NUM_DISPATCH_TYPES; ++i )
	{
		for( int j=0; j<NUM_DISPATCH_TYPES; ++j )
		{
			m_dispatchTable[i][j] = nullCacheCreator;
		}
	}

	typedef hknpCollisionDispatchType T;

	// Register convex vs convex
	registerCacheCreator( T::CONVEX,	T::CONVEX,			hknpConvexConvexCollisionCache::construct );

	// Register convex vs composite
	registerCacheCreator( T::CONVEX,	T::COMPOSITE,		hknpConvexCompositeCollisionCache::construct );

	// Register composite vs composite
	registerCacheCreator( T::COMPOSITE,	T::COMPOSITE,		hknpCompositeCompositeCollisionCache::construct );

	// Register convex,composite vs signed distance field
	registerCacheCreator( T::CONVEX,	T::DISTANCE_FIELD,	hknpSignedDistanceFieldCollisionCache::construct );
	registerCacheCreator( T::COMPOSITE,	T::DISTANCE_FIELD,	hknpSignedDistanceFieldCollisionCache::construct );

	// Note: We don't handle SDF vs SDF
}

void hknpCollisionDispatcher::dispatchBodyPairs(
	const hknpSimulationThreadContext& context, hkBlockStream<hknpBodyIdPair>::Reader* newPairsReader, int numPairs,
	hknpCdCacheWriter* cacheWriter ) const
{
	const hknpWorld* HK_RESTRICT world = context.m_world;

	// Pre-process the pairs
	hkLocalArray<hknpBodyIdPair> filteredPairs( numPairs );
	{
		for( const hknpBodyIdPair* pair = newPairsReader->access<hknpBodyIdPair>();
			 pair;
			 pair = newPairsReader->advanceAndAccessNext<hknpBodyIdPair>() )
		{
			hknpBodyIdPair iPair = *pair;

			HK_ASSERT( 0x7bf5a1e9, iPair.m_bodyA != iPair.m_bodyB );
			const hknpBody* HK_RESTRICT bodyA = &world->getSimulatedBody( iPair.m_bodyA );
			const hknpBody* HK_RESTRICT bodyB = &world->getSimulatedBody( iPair.m_bodyB );

			// Ignore overlaps between attached bodies.
			if( bodyA->m_motionId == bodyB->m_motionId )
			{
				continue;
			}

			const int combinedBodyFlags = bodyA->m_flags.get() | bodyB->m_flags.get();

			// Ignore overlaps if either body is flagged as never colliding.
			if( combinedBodyFlags & hknpBody::DONT_COLLIDE )
			{
				continue;
			}

			// Ignore overlaps involving only bodies with infinite mass,
			// unless either of them is using flags that should force cache creation.
			if( bodyA->isStaticOrKeyframed() && bodyB->isStaticOrKeyframed() )
			{
				if( m_keyframeCacheCreationMask.get( combinedBodyFlags ) == 0 )
				{
					continue;
				}
			}

			// Order such as A,B <= B,A and A.m_dispatchType <= B.m_dispatchType.
			// Collision caches expect this.
			const hknpCollisionDispatchType::Enum dispatchTypeA = bodyA->m_shape->m_dispatchType;
			const hknpCollisionDispatchType::Enum dispatchTypeB = bodyB->m_shape->m_dispatchType;
			if( dispatchTypeB < dispatchTypeA || ( (dispatchTypeB == dispatchTypeA) && (iPair.m_bodyB < iPair.m_bodyA) ) )
			{
				hknpBodyId tempId = iPair.m_bodyA;
				iPair.m_bodyA = iPair.m_bodyB;
				iPair.m_bodyB = tempId;
			}

			filteredPairs.pushBackUnchecked( iPair );
		}
	}

	// Apply the collision filter
	{
		hknpCollisionFilter* collisionFilter = world->m_modifierManager->getCollisionFilter();
		numPairs = collisionFilter->filterBodyPairs( context, filteredPairs.begin(), filteredPairs.getSize() );
	}

	// Dispatch the remaining pairs to their cache creation functions
	for( int i=0; i<numPairs; i++ )
	{
		const hknpBody& bodyA = world->getBodyUnchecked( filteredPairs[i].m_bodyA );
		const hknpBody& bodyB = world->getBodyUnchecked( filteredPairs[i].m_bodyB );
		HK_ASSERT( 0x7bf5a1ea, bodyA.m_shape && bodyB.m_shape );

		// Create a cache based on the dispatch types provided by the shapes
		createCollisionCache(
			*world, bodyA, bodyB,
			bodyA.m_shape->m_dispatchType, bodyB.m_shape->m_dispatchType,
			cacheWriter );
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
