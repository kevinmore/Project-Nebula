/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BROAD_PHASE_H
#define HKNP_BROAD_PHASE_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Dynamics/World/ModifierManager/hknpModifierManager.h>

class hknpWorld;
class hknpBody;
class hkIntSpaceUtil;
struct hknpRayCastQuery;
struct hknpCastRayOutput;
class hknpCollisionQueryCollector;
struct hkTaskGraph;

/// The broad phase interface.
/// Broad phases use hknpBodyIds to identify bodies, so many functions require a hknpBody array in order to
/// resolve the ID to a body.
/// Broad phases can store internal per-body data in hknpBody::m_broadPhaseId.
class hknpBroadPhase
{
	public:

		enum UpdateMode
		{
			UPDATE_DYNAMIC,
			UPDATE_ALL,
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBroadPhase );

#ifndef HK_PLATFORM_SPU
		/// Destructor.
		virtual ~hknpBroadPhase() {}

		//
		// Add/remove
		//

		/// Add a batch of bodies. (Modifies hknpBody::m_broadPhaseId)
		virtual void addBodies( const hknpBodyId* idx, int numIdx, hknpBody* bodies ) = 0;

		/// Remove a batch of bodies.
		virtual void removeBodies( const hknpBodyId* ids, int numIds, hknpBody* bodies ) = 0;

		//
		// Update
		//

		/// Refresh a batch of bodies. Should be called if a body is changed in a way that require broad phase updates.
		/// Necessary if a body is changed in a way that would cause it to move to a different layer. Or if a static body is moved.
		virtual void markBodiesDirty( hknpBodyId* ids, int numIds, hknpBody* bodies ) = 0;

		/// Update the broad phase. It will extract AABB from dynamic bodies (also fixed bodies if UPDATE_ALL is used)
		/// and update the internal data structures.
		virtual void update( hknpBody* bodies, UpdateMode mode ) = 0;

		/// Optimize internal data structures where possible.
		virtual void optimize( hknpBody* bodies ) {}

		/// Populate a task graph for multi threaded simulation (update() & findNewPairs()).
		/// When the tasks are completed, newPairsStream will contain the new pairs found.
		virtual void buildTaskGraph(
			hknpWorld* world, hknpSimulationContext* simulationContext, hkBlockStream<hknpBodyIdPair>* newPairsStream,
			hkTaskGraph* taskGraphOut ) = 0;

		//
		// Query
		//

		/// Get an AABB enclosing all objects currently in the broad phase.
		virtual void getExtents( hkAabb16& extents ) const = 0;

		/// Find all pairs of overlapping AABBs (except "static vs static").
		virtual void findAllPairs( hkBlockStream<hknpBodyIdPair>::Writer *pairsWriter ) = 0;

		/// Find all new pairs of overlapping AABBs (except "static vs static").
		virtual void findNewPairs(
			hknpBody* bodyBuffer, const hkAabb16* previousAabbs,
			hkBlockStream<hknpBodyIdPair>::Writer *newPairsWriter ) = 0;

		/// Query the broad phase using an AABB.
		virtual void queryAabb( const hkAabb16& aabb, const hknpBody* bodies, hkArray<hknpBodyId>& hitsOut ) = 0;

		/// Query the broad phase using an AABB, with NMP support.
		virtual void queryAabbNmp(
			const hkAabb16& aabb, const hkAabb16& expandedAabb, hknpQueryAabbNmp* nmpOut,
			hkArray<hknpBodyId>& hitsOut ) const = 0;

		/// Query the broad phase using an AABB. Forward all filtered hits to the collector.
		virtual void queryAabb(
			const hknpAabbQuery& query, const hknpBody* bodies, const hknpWorld& world,
			const hkIntSpaceUtil& intSpaceUtil, hknpCollisionQueryCollector* collector ) const = 0;
#endif
		/// Cast a ray. Forward all hits to the collector.
		virtual void castRay(
			const hknpRayCastQuery& query, const hknpBody* bodies, const hknpWorld& world,
			const hkIntSpaceUtil& intSpaceUtil, hknpCollisionQueryCollector* collector ) const = 0;

		/// Cast a shape. Forward all hits to the collector.
		virtual void castShape(
			const hknpShapeCastQuery& query, const hkRotation& queryShapeOrientation,
			const hknpBody* bodies, const hknpWorld& world,
			const hkIntSpaceUtil& intSpaceUtil, hknpCollisionQueryCollector* collector ) const = 0;

		/// Get the closest points between a shape and all objects in the world. Forward all hits to the collector.
		virtual void getClosestPoints(
			const hknpClosestPointsQuery& query, const hkTransform& queryShapeTransform,
			const hknpBody* bodies, const hknpWorld& world,
			const hkIntSpaceUtil& intSpaceUtil, hknpCollisionQueryCollector* collector ) const = 0;
};


#endif // HKNP_BROAD_PHASE_H

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
