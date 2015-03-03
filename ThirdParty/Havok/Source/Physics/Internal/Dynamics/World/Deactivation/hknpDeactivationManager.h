/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_DEACTIVATION_MANAGER_H
#define HKNP_DEACTIVATION_MANAGER_H

#include <Common/Base/Container/BitField/hkBitField.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationState.h>
#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>

class hknpWorld;
class hknpCdCacheStream;
class hknpActivationListener;


/// This holds a list of interconnected bodies deactivated together.
/// When a body is activated, all other bodies in its island must be activated as well.
class hknpDeactivatedIsland
{
	public:

		/// A struct used to track activation callbacks
		struct ActivationInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, ActivationInfo);

			hknpActivationListener* m_activationListener;
			void* m_userData;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpDeactivatedIsland );

		/// Constructor.
		hknpDeactivatedIsland();

		/// Deletes all caches owned by the island (but not the m_bodyIds).
		void deleteAllCaches();

		/// Returns the index of a matching activation info
		int findActivationInfo( const ActivationInfo& ref );

		/// Set the island id
		HK_FORCE_INLINE void setIslandId( hknpIslandId id );

	public:

		/// All top level hknpCollisionCaches. This implicitly references the child caches (in case of hknpCompositeCollisionCache)
		hknpCdCacheRange  m_cdCaches;
		hknpCdCacheRange  m_cdChildCaches;

		/// An array of deactivated bodies.
		hkArray<hknpBodyId> m_bodyIds;

		/// An array of activation listeners.
		hkArray<ActivationInfo> m_activationListeners;

		/// A list of body pairs, which have to be turned into collision caches once this island activates.
		hkArray<hknpBodyIdPair> m_deletedCaches;

		/// The ID of this deactivated island.
		hknpIslandId m_islandId;

		/// The ID of the next connected deactivated island (next in linked list, hknpIslandId::invalid if last).
		hknpIslandId m_nextConnectedIslandId;

		/// The ID of first island in the list of connected islands (head of linked list)
		hknpIslandId m_headConnectedIslandId;

		/// Internal index used to track garbage collection.
		hkUint16 m_garbageIndex;

		/// Whether this island is a large island that uses exclusive block stream ranges.
		hkBool m_useExclusiveCdCacheRanges;

		/// Internal helper flag. This is set if the island is scheduled for reactivation.
		hkBool m_isMarkedForActivation;
};


/// A manager handling deactivation.
class hknpDeactivationManager
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpDeactivationManager );

		/// Constructor.
		hknpDeactivationManager( int largeIslandSize );

		/// Destructor.
		~hknpDeactivationManager();

		/// Set world pointer.
		HK_FORCE_INLINE void initDeactivationManager( hknpWorld* world );

		//
		// Manage body links
		//

		HK_FORCE_INLINE const hkPointerMap<hkUint64, int>& getBodyLinks() const;

		/// For deactivation, all links added between bodies must be tracked.
		/// That'd be all actions, constraints, and body compounds.
		void addBodyLink( hknpBodyId i, hknpBodyId d );

		/// Remove a link between bodies.
		void removeBodyLink( hknpBodyId i, hknpBodyId d );

		//
		// Access motion deactivation information
		//

		/// Ensures allocation of storage for deactivation state for the given hknpMotionId
		HK_FORCE_INLINE void reserveNumMotions( int numMotions );

		HK_FORCE_INLINE hknpDeactivationState* getAllDeactivationStates();

		//
		// Deactivation
		//

		/// Enables/disables deactivation of the specified body
		/// NOTE: only works if deactivation is enabled in the world
		HK_FORCE_INLINE void setBodyDeactivation( hknpBodyId bodyId, bool enableDeactivation );

		/// Adds island to world, marks all bodies as inactive; collision caches will be moved
		/// to inactive stream on the next collision processing, next frame.
		void deactivateIsland( const hknpSimulationThreadContext& tl, hknpDeactivatedIsland* island );

		/// This function is expected to be called exactly once during a frame, and it
		/// matches exactly one call to addDeactivationIsland().
		void clearAndTrackIslandForDeactivatingCaches();

		//
		// Reactivation
		//

		/// If the body is deactivated it will activate the island it belongs to.
		/// If active it will reset the deactivation frame counter.
		void markBodyForActivation( hknpBodyId bodyId );

		/// Force deactivation of the entire island the body belongs to.
		/// Since only one island is deactivated every frame it can take several frames before
		/// the island is deactivated.
		/// Note: Avoid using this function, it can produce very unintuitive behavior since
		/// an entire island of bodies (that would not normally deactivate) will be deactivated!
		void forceIslandOfBodyToDeactivate( hknpBodyId bodyId );

		/// Removes all bodies that are invalid from the list of bodies scheduled forced island deactivation.
		void removeInvalidBodiesScheduledForDeactivation();

		/// Activate an island.
		HK_FORCE_INLINE void markIslandForActivation( hknpIslandId island );

		/// Activate all islands.
		void makrAllAllIslandsForActivation();

		/// Examine the results from the broad phase and mark newly 'touched' islands as active.
		void markIslandsForActivationFromNewPairs( hkBlockStream<hknpBodyIdPair>::Reader* newPairsReader );

		/// Activate the marked island, should only be called by the simulation.
		/// If clearNewlyReactivatedIslands is set to false it must be called manually before the next frame.
		void activateMarkedIslands( bool clearNewlyReactivatedIslands = true );

		/// Move the newly activated cdcaches, should only be called by the simulation.
		void moveActivatedCaches(
			const hknpSimulationThreadContext& tl,
			hknpCdCacheStream* activatedCollideCachesOut,
			hknpCdCacheStream* activatedChildCdCachesOut,
			hkBlockStream<hknpBodyIdPair>* newPairsStream );

		//
		// Memory management
		//

		/// Incremental do garbage collection on deactivated islands.
		void garbageCollectInactiveCaches(
			const hknpSimulationThreadContext& tl,
			hknpCdCacheStream* inactiveStreamInOut, hknpCdCacheStream* inactiveChildStreamInOut,
			bool forceGarbageCollection = false );

		/// Force a complete garbage collection run.
		void garbageCollectAllInactiveCaches(
			const hknpSimulationThreadContext& tl,
			hknpCdCacheStream* inactiveStreamInOut, hknpCdCacheStream* inactiveChildStreamInOut );

		//
		// Helper functions
		//

		/// Reset the deactivation counters. This enforces that this motion will not deactivate for several frames.
		HK_FORCE_INLINE void resetDeactivationFrameCounter( hknpMotionId id );

		// Add an inactive body. Returns false if it fails (and body should be considered active).
		bool addInactiveBody( hknpBodyId bodyId );

		// Connect two deactivated islands.
		void connectDeactivatedIslands( hknpIslandId islandA, hknpIslandId islandB );

		// Check if two deactivated islands are connected.
		bool areIslandsConnected( hknpIslandId islandA, hknpIslandId islandB );

		// Sort the m_islandsMarkedForActivation array (needed by the constraint setup code).
		HK_FORCE_INLINE void sortIslandsMarkedForActivation();

		// Clear the m_islandsMarkedForActivation array (needed by the constraint setup code).
		HK_FORCE_INLINE void clearIslandsMarkedForDeactivation();

	public:

		//
		//	Internal public section
		//

		// Access to hknpDeactivationState implemented in hknpWorld
		friend class hknpWorld;

		hknpWorld* m_world;

		/// State of individual motions. This array matches 1-1 the array of motions in hknpMotionManager.
		hkArray<hknpDeactivationState> m_deactivationStates;

		/// An array of deactivated islands.
		hkArray<hknpDeactivatedIsland*> m_deactivatedIslands;

		/// An index into m_deactivatedIslands array of unused island ids.
		hkArray<hknpIslandId> m_freeIslandIds;

		/// List of links, additional to collision pairs, to be considered when finding deactivation islands.
		hkPointerMap<hkUint64, int> m_bodyLinks;

		/// list of bodies marked for deactivation.
		hkArray<hknpBodyId> m_bodiesMarkedForDeactivation;

		//
		// Temporary frame state
		//

		/// An array of islands to be activated when activateMarkedIslands() is called.
		hkArray<hknpIslandId> m_islandsMarkedForActivation;

		/// This points to an island which is just about to get deactivated.
		/// This is set at the end of the integration step and used by
		/// the collision detector the next step to move the caches.
		hknpDeactivatedIsland* m_newlyDeactivatedIsland;

		/// Track what is in the inactiveCdCache stream.
		hkArray<hknpIslandId> m_inactiveIslands; // if entries are invalid, it means they are no longer in use (and can be freed).

		/// Track the amount of garbage (to decide when to trigger the collector).
		int m_totalGarbage;

		/// Iterator index for the incremental garbage collector (-1 if garbage collector is inactive).
		int	m_garbageIterator;

		/// When the incremental GC is active this is end index (when to stop), otherwise 0.
		int	m_garbageIteratorEnd;

		/// Used to decide when to use exclusive ranges for the inactive islands.
		int	m_largeIslandSize;

		/// New activated body pairs.
		hkArray<hknpBodyIdPair> m_newActivatedPairs;

		/// Cache ranges of the newly activated islands.
		hkArray<hknpCdCacheRange> m_newActivatedCdCacheRanges;

		/// New activated bodies.
		hkArray<hknpBodyId> m_newActivatedBodyIds;
};



//
// Deactivation world state at runtime
//

class hkUnionFind;
template <typename T>
class hkFixedArray;


struct hknpDeactivationThreadData
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpDeactivationThreadData);

	// Aligned for SPU access.
	HK_ALIGN16(hkBitField m_solverVelOkToDeactivate);
};


/// Handles deactivation functionality for hknpWorld.
/// This includes creation and usage of hkUnionFind employed to find deactivation islands for the hknpWorld.
class hknpDeactivationStepInfo
{
		friend struct hknpDeactivationThreadData;

	public:

		enum { MAX_NUM_THREADS = 12 };

	private:

		// Disable default constructor
		hknpDeactivationStepInfo() {}

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, hknpDeactivationStepInfo);

		hknpDeactivationStepInfo(int sizeOfSolverVelocitiesArray, int numThreads);

		~hknpDeactivationStepInfo();

		hknpDeactivationThreadData& getThreadData(int threadIdx);

		/// Adds a single body link to the internal union find.
		void addBodyPair(const hknpWorld& world, hknpBodyId aIdx, hknpBodyId bIdx, const hknpIdxRangeGrid& cellIdxToGlobalSolverId);

		/// Adds a single body link to the internal union find. Single-threaded version.
		void addBodyPairSt(const hknpWorld& world, hknpBodyId aIdx, hknpBodyId bIdx);

		/// Populates the main world union find with all links provided in a hkPointerMap.
		void addAllBodyLinks(const hknpWorld& world, const hkPointerMap<hkUint64, int>& bodyLinks, const hknpIdxRangeGrid& cellIdxToGlobalSolverId); 

		/// Populates the main world union find with all links provided in a hkPointerMap. Single-threaded version.
		void addAllBodyLinksSt(const hknpWorld& world, const hkPointerMap<hkUint64, int>& bodyLinks);

		/// Call before the first call to addActiveBodyPairs().
		void addActiveBodyPairsBegin();

		/// Combines pairs stream into the main world union-find.
		void addActiveBodyPairs(const hknpCdPairStream& stream);

		/// Combines pairs stream into the main world union-find. Used when simulation has multiple cells.
		void addActiveBodyPairs(const hknpCdPairStream& stream, const hknpIdxRangeGrid& cellInfo);

		/// Call after the last call to addActiveBodyPairs.
		void addActiveBodyPairsEnd();

		/// Combines solverVelocity activity bitfields from all threads.
		void combineActivityBitFields(hkBitField& out);

		// Creates a deactivated island, if one is found.
		hknpDeactivatedIsland* createDeactivatedIsland(const hknpWorld& world, const hkBitField& solverVelOkToDeactivate, hknpIdxRangeGrid* cellIdxToGlobalSolverId, bool useIdxGrid);

		HK_FORCE_INLINE hkUnionFind* getUnionFind();

	private:

		hkUnionFind* m_unionFind;
		hkFixedArray<int>* m_buffer;

	public:

		hknpDeactivationThreadData m_data[MAX_NUM_THREADS];

		int m_numBodies;
		int m_numThreads;

		bool m_canAddEdges;
};

#if !defined(HK_PLATFORM_SPU)
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.inl>
#endif

#endif // HKNP_DEACTIVATION_MANAGER_H

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
