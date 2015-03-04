/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BRUTE_FORCE_BROAD_PHASE_H
#define HKNP_BRUTE_FORCE_BROAD_PHASE_H

#include <Physics/Physics/Collide/BroadPhase/hknpBroadPhase.h>


/// A simple broad phase implementation.
class hknpBruteForceBroadPhase : public hknpBroadPhase
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBruteForceBroadPhase );

		virtual ~hknpBruteForceBroadPhase() {}

		//
		// hknpBroadPhase implementation
		//

		virtual void addBodies( const hknpBodyId* idx, int numIdx, hknpBody* bodies ) HK_OVERRIDE;

		virtual void removeBodies( const hknpBodyId* ids, int numIds, hknpBody* bodies ) HK_OVERRIDE;

		virtual void findAllPairs( hkBlockStream<hknpBodyIdPair>::Writer *newPairWriter ) HK_OVERRIDE;

		virtual void findNewPairs( hknpBody* bodyBuffer, const hkAabb16* previousAabbs, hkBlockStream<hknpBodyIdPair>::Writer *newPairWriter ) HK_OVERRIDE;

		virtual void getExtents( hkAabb16& extents ) const HK_OVERRIDE;

		virtual void update( hknpBody* bodies, UpdateMode mode ) {}

		virtual void buildTaskGraph( hknpWorld* world, hknpSimulationContext* simulationContext, hkBlockStream<hknpBodyIdPair>* newPairsStream, hkTaskGraph* taskGraphOut ) HK_OVERRIDE;

		virtual void markBodiesDirty( hknpBodyId* ids, int numIds, hknpBody* bodies) HK_OVERRIDE;

		virtual void queryAabb( const hkAabb16& aabb, const hknpBody* bodies, hkArray<hknpBodyId>& hitsOut ) HK_OVERRIDE;

		// This broad phase does not support the query interface (yet)

		virtual void queryAabbNmp( const hkAabb16& aabb, const hkAabb16& expandedAabb, hknpQueryAabbNmp* nmpOut,
								   hkArray<hknpBodyId>& hitsOut ) const HK_OVERRIDE {}

		virtual void queryAabb(const hknpAabbQuery& query, const hknpBody* bodies, const hknpWorld& world,
							   const hkIntSpaceUtil& intSpaceUtil, hknpCollisionQueryCollector* collector ) const HK_OVERRIDE {}

		virtual void castRay( const hknpRayCastQuery& query, const hknpBody* bodies, const hknpWorld& world,
							  const hkIntSpaceUtil& intSpaceUtil, hknpCollisionQueryCollector* collector ) const HK_OVERRIDE {}

		virtual void castShape( const hknpShapeCastQuery& query, const hkRotation& queryShapeOrientation,
								const hknpBody* bodies, const hknpWorld& world,
								const hkIntSpaceUtil& intSpaceUtil, hknpCollisionQueryCollector* collector ) const HK_OVERRIDE {}

		virtual void getClosestPoints(const hknpClosestPointsQuery& query, const hkTransform& queryShapeTransform,
									  const hknpBody* bodies, const hknpWorld& world,
									  const hkIntSpaceUtil& intSpaceUtil, hknpCollisionQueryCollector* collector ) const HK_OVERRIDE {}

	public:

		hkArray<hknpBodyId> m_bodies;
		hkArray<hknpBroadPhaseId> m_freeList;
};

#endif // HKNP_BRUTE_FORCE_BROAD_PHASE_H

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
