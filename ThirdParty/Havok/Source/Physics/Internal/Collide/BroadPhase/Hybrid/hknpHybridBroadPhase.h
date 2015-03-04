/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_HYBRID_BROAD_PHASE_H
#define HKNP_HYBRID_BROAD_PHASE_H

#include <Physics/Physics/Collide/BroadPhase/hknpBroadPhase.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>
#include <Common/Base/Container/BitField/hkBitField.h>

template<typename KeyType, typename hkIdxType> class hknpHybridAabbTree;

class hknpBroadPhaseConfig;
class hknpHybridBroadPhaseTaskContext;

/// A broad phase implementation which uses multiple hybrid AABB trees.
HK_ALIGN16(class) hknpHybridBroadPhase : public hknpBroadPhase
{
	public:

		typedef hkUint32 HybridTreeNodeIndex;
		typedef hknpHybridAabbTree<hknpBodyId::Type, HybridTreeNodeIndex> HybridTree;

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpHybridBroadPhase );

		/// Constructor.
#ifdef HK_PLATFORM_SPU
		hknpHybridBroadPhase( int maxNumBodies=0, const hknpBroadPhaseConfig* broadPhaseConfig=HK_NULL ) {}
#else
		hknpHybridBroadPhase( int maxNumBodies, const hknpBroadPhaseConfig* broadPhaseConfig );

		/// Destructor.
		virtual ~hknpHybridBroadPhase();
		void checkConsistency();

		//
		// hknpBroadPhase interface
		//

		virtual void addBodies( const hknpBodyId* idx, int numIdx, hknpBody* bodies ) HK_OVERRIDE;
		virtual void removeBodies( const hknpBodyId* idx, int numIdx, hknpBody* bodies ) HK_OVERRIDE;

		virtual void markBodiesDirty( hknpBodyId* ids, int numIds, hknpBody* bodies ) HK_OVERRIDE;
		virtual void update( hknpBody* bodies, UpdateMode mode ) HK_OVERRIDE;

		virtual void optimize( hknpBody* bodies ) HK_OVERRIDE;

		virtual void buildTaskGraph(
			hknpWorld* world, hknpSimulationContext* simulationContext, hkBlockStream<hknpBodyIdPair>* newPairsStream,
			hkTaskGraph* taskGraphOut )  HK_OVERRIDE;

		//
		// hknpBroadPhase query interface
		//

		virtual void getExtents( hkAabb16& extents ) const HK_OVERRIDE;

		virtual void findAllPairs( hkBlockStream<hknpBodyIdPair>::Writer *pairsWriter ) HK_OVERRIDE;

		virtual void findNewPairs(
			hknpBody* bodyBuffer, const hkAabb16* previousAabbs,
			hkBlockStream<hknpBodyIdPair>::Writer *newPairsWriter ) HK_OVERRIDE;

		virtual void queryAabb( const hkAabb16& aabb, const hknpBody* bodies,
			hkArray<hknpBodyId>& hits ) HK_OVERRIDE;

		virtual void queryAabbNmp( const hkAabb16& aabb, const hkAabb16& expandedAabb, hknpQueryAabbNmp* nmpOut,
			hkArray<hknpBodyId>& hits ) const HK_OVERRIDE;

		virtual void queryAabb( const hknpAabbQuery& query,
			const hknpBody* bodies, const hknpWorld& world, const hkIntSpaceUtil& intSpaceUtil,
			hknpCollisionQueryCollector* collector ) const HK_OVERRIDE;
#endif
		virtual void castRay( const hknpRayCastQuery& query,
			const hknpBody* bodies, const hknpWorld& world, const hkIntSpaceUtil& intSpaceUtil,
			hknpCollisionQueryCollector* collector ) const HK_OVERRIDE;

		virtual void castShape( const hknpShapeCastQuery& query, const hkRotation& queryShapeOrientation,
			const hknpBody* bodies, const hknpWorld& world, const hkIntSpaceUtil& intSpaceUtil,
			hknpCollisionQueryCollector* collector ) const HK_OVERRIDE;

		virtual void getClosestPoints( const hknpClosestPointsQuery& query, const hkTransform& queryShapeTransform,
			const hknpBody* bodies, const hknpWorld& world, const hkIntSpaceUtil& intSpaceUtil,
			hknpCollisionQueryCollector* collector) const HK_OVERRIDE;
	public:

		// For the multi threaded tasks we separate update into parts
		void updateDirtyBodies( hknpBody* bodies );
		void updateVolatileTree( hknpBody* bodies, int treeIndex );
		void updateNonVolatileTree( hknpBody* bodies, int treeIndex, bool updateAll );

		// Internal helpers
		static HK_FORCE_INLINE void setTreeLeafIdx( hknpBody& body, HybridTreeNodeIndex idx ) { body.m_broadPhaseId = ( idx | (body.m_broadPhaseId & s_treeIdxMask) ); }
		static HK_FORCE_INLINE HybridTreeNodeIndex getTreeLeafIdx( const hknpBody& body ) { return body.m_broadPhaseId & s_treeLeafIdxMask; }
		static HK_FORCE_INLINE void setTreeIdx( hknpBody& body, int idx ) { body.m_broadPhaseId = ( hknpBroadPhaseId( (idx << s_treeIdxShift) | getTreeLeafIdx(body) ) ); }
		static HK_FORCE_INLINE int getTreeIdx( const hknpBody& body ) { return body.m_broadPhaseId >> s_treeIdxShift; }
		static HK_FORCE_INLINE bool isInTree( const hknpBody& body ) { return body.m_broadPhaseId != HKNP_INVALID_BROAD_PHASE_ID; }

		// Internal constants
		static const HybridTreeNodeIndex s_treeLeafIdxMask = 0x1fffffff;
		static const HybridTreeNodeIndex s_treeIdxMask = 0xe0000000;
		static const int s_treeIdxShift = 29;
		static const int s_maxNumTrees = 8;

	public:

		/// The broad phase config. Used to map bodies to layers.
		hkRefPtr<const hknpBroadPhaseConfig> m_broadPhaseConfig;

		/// A tree for each layer.
		
		
		HybridTree* m_trees[s_maxNumTrees];

		/// Whether the tree should be updated each step (e.g. if it contains dynamic bodies).
		hkBool m_isTreeVolatile[s_maxNumTrees];

		/// Whether the tree needs to be updated (e.g. if a body has been added or removed).
		hkBool m_isTreeDirty[s_maxNumTrees];

		/// The number of layers (trees) in use.
		int m_numLayers;

		/// List of pairs of layers to collide. Build at construction time from m_broadPhaseConfig.
		hkArray<int> m_collideTreePairs;

		/// Bitfield to mark which bodies need to be updated (e.g. if it changed its layer or its AABB).
		/// This is initialized to the number of bodies passed to the constructor, but can grow over time.
		hkBitField m_dirtyBodies;

		/// The lowest m_dirtyBodies bit that is set.
		hkUint32 m_minDirtyBodiesBit;

		/// The highest m_dirtyBodies bit that is set.
		hkUint32 m_maxDirtyBodiesBit;

		/// Task context for multi threading.
		hknpHybridBroadPhaseTaskContext* m_taskContext;
};

#include <Physics/Internal/Collide/BroadPhase/Hybrid/hknpHybridBroadPhase.inl>


#endif // HKNP_HYBRID_BROAD_PHASE_H

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
