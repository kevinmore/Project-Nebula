/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_DISPATCHER_H
#define HKNP_COLLISION_DISPATCHER_H

#include <Physics/Physics/hknpTypes.h>


/// Collision dispatcher.
/// Creates collision caches for pairs of bodies, based on a set of registered cache creation functions.
class hknpCollisionDispatcher
{
	public:

		/// A function to create a collision cache in a given cache writer.
		typedef void (HK_CALL *CreateFunc)(
			const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
			hknpCdCacheWriter* cacheWriterInOut );

		enum { NUM_DISPATCH_TYPES = hknpCollisionDispatchType::NUM_TYPES };

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollisionDispatcher );

		/// Constructor. Registers all default cache creation functions.
		hknpCollisionDispatcher();

		/// Get the body flags which force cache creation for keyframed vs static pairs.
		HK_FORCE_INLINE hknpBody::Flags getKeyframeCacheCreationMask() const;

		/// Set the body flags which force cache creation for keyframed vs static pairs.
		HK_FORCE_INLINE void setKeyframeCacheCreationMask( hknpBody::Flags flags );

		/// Register a cache creation function, overriding any existing registration of the same types.
		/// \a createFn cannot be null.
		HK_FORCE_INLINE void registerCacheCreator(
			hknpCollisionDispatchType::Enum typeA, hknpCollisionDispatchType::Enum typeB,
			CreateFunc createFn );

		/// Create a collision cache for a pair of bodies, using the given dispatch types.
		HK_FORCE_INLINE void createCollisionCache(
			const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
			hknpCollisionDispatchType::Enum typeA, hknpCollisionDispatchType::Enum typeB,
			hknpCdCacheWriter* cacheWriter ) const;

		/// Filter and create collision caches for the all the given body pairs.
		void dispatchBodyPairs(
			const hknpSimulationThreadContext& context,
			hkBlockStream<hknpBodyIdPair>::Reader* newPairsReader, int numPairs,
			hknpCdCacheWriter* cacheWriter ) const;

	protected:

		/// Body pairs are ignored if both are keyframed or static, since collisions between these have no effect.
		/// This mask of body flags allows users to force cache creation in these cases if a keyframed body is involved
		/// and either of the body's flags exists in the mask. This allows keyframed bodies to generate callbacks and
		/// events from the collision pipeline.
		/// Note that static vs static pairs are always ignored.
		hknpBody::Flags m_keyframeCacheCreationMask;

		/// Symmetric table of cache creation functions.
		CreateFunc m_dispatchTable[NUM_DISPATCH_TYPES][NUM_DISPATCH_TYPES];
};

#include <Physics/Physics/Collide/Dispatcher/hknpCollisionDispatcher.inl>

#endif // HKNP_COLLISION_DISPATCHER_H

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
