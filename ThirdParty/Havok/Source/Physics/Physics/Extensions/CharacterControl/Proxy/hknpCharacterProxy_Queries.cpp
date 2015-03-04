/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxy.h>

#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>
#include <Physics/Physics/Collide/Shape/hknpShapeQueryInterface.h>
#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>
#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>

#define NP_CHARACTER_PROXY_USE_DYNAMIC_COLLECTOR  1


namespace
{
	// A more or less straight port of hkpIterativeLinearCastAgent::staticLinearCast()
	static HK_FORCE_INLINE void iterativeLinearCast(
		hknpCollisionQueryContext* queryContext, const hknpShape* queryShape, const hkTransform& queryShapeToWorld,
		const hknpShape* targetShape, const hkTransform& targetShapeToWorld, hkVector4Parameter castPath,
		hkSimdRealParameter startPointTolerance, hkSimdRealParameter earlyOutDistance, hkSimdRealParameter maxExtraPenetration,
		hknpCollisionResult* castHitOut, hknpCollisionResult* startHitOut, int maxIterations)
	{
		// Calculate the distance between query (character shape) and target with a maximum limit of maxDistance
		hkSimdReal pathLength = castPath.length<3>();
		hkSimdReal maxDistance; maxDistance.setMax(pathLength, startPointTolerance);
		hknpClosestPointsQuery query(*queryShape, maxDistance.getReal());
		hknpClosestHitCollector closestHitCollector;
		hknpShapeQueryInterface::getClosestPoints(queryContext, query, queryShapeToWorld, *targetShape, targetShapeToWorld, &closestHitCollector);
		if (!closestHitCollector.hasHit())
		{
			return;
		}

		hkSimdReal distance;
		hkSimdReal currentFraction; currentFraction.setZero();
		{
			// Store hit as start point if within start point tolerance
			const hknpClosestPointsQueryResult& hit = closestHitCollector.getHits()->asClosestPoints();
			distance.setFromFloat(hit.getDistance());
			if (distance.isLessEqual(startPointTolerance) && startHitOut)
			{
				*startHitOut = hit;
				startHitOut->m_fraction = 0;

				// Store distance in the W component of the normal
				startHitOut->m_normal.setW(distance);
			}

			{
				const hkSimdReal pathProjected = hit.getSeparatingDirection().dot<3>(castPath);
				const hkSimdReal endDistance = distance + pathProjected;

				//Check whether we could move the full distance
				if (endDistance.isGreaterZero())
				{
					return;
				}

				HK_ASSERT2(0x74a9aa26, maxExtraPenetration.isGreaterEqualZero(), "You have to set the maxExtraPenetration to something bigger than 0");

				// We are not moving closer than maxExtraPenetration
				if ((pathProjected + maxExtraPenetration).isGreaterEqualZero())
				{
					return;
				}

				// Check for early outs
				if (distance.isLessEqual(earlyOutDistance))
				{
					*castHitOut = hit;
					castHitOut->m_fraction = 0;

					// Store distance in the W component of the normal
					castHitOut->m_normal.setW(distance);

					return;
				}

				// Update current cast fraction
				currentFraction = distance / (distance - endDistance);
			}
		}

		hkTransform currentQueryShapeToWorld = queryShapeToWorld;
		for (int i = maxIterations - 1; i >= 0; i--)
		{
			closestHitCollector.reset();

			// Move the object along the path
			{
				const hkVector4& startPosition = queryShapeToWorld.getTranslation();
				hkVector4 currentPosition; currentPosition.setAddMul(startPosition, castPath, currentFraction);
				currentQueryShapeToWorld.setTranslation(currentPosition);
			}

			hknpShapeQueryInterface::getClosestPoints(
				queryContext, query, currentQueryShapeToWorld, *targetShape, targetShapeToWorld, &closestHitCollector);
			if (!closestHitCollector.hasHit())
			{
				return;
			}

			// Redo the checks
			{
				const hknpClosestPointsQueryResult& hit = closestHitCollector.getHits()->asClosestPoints();
				const hkVector4& normal = hit.getSeparatingDirection();

				// Normal points away
				hkSimdReal pathProjected = normal.dot<3>(castPath);
				if (pathProjected.isGreaterEqualZero())
				{
					return;
				}
				pathProjected = -pathProjected;

				// Pre distance is the negative already traveled distance relative to the new normal
				const hkSimdReal preDistance = pathProjected * currentFraction;
				HK_ASSERT2(0x573be33d,  preDistance.getReal() >= 0.0f, "Numerical accuracy problem in linearCast");

				distance.setFromFloat(hit.getDistance());
				if (distance + preDistance > pathProjected)
				{
					// endDistance + preDistance = realEndDistance;
					// if realEndDistance > 0, than endplane is not penetrated, so no hit
					return;
				}

				// Early out if we are already very close
				if (distance.isLessEqual(earlyOutDistance))
				{
					*castHitOut = hit;
					castHitOut->m_fraction = currentFraction.getReal();

					// Calculate the distance from the start position to the contact plane using the cast fraction
					// and store it in the W component of the normal
					hkSimdReal normalDistance = -currentFraction * castPath.dot<3>(hit.m_normal);
					castHitOut->m_normal.setW(normalDistance);

					break;
				}

				currentFraction = currentFraction + (distance / pathProjected);
			}
		}
	}

	// A collector used in hknpCharacterProxy::worldLinearCast()
	class hknpCompositeShapeKeysCollector : public hknpAllHitsCollector
	{
		public:

			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCompositeShapeKeysCollector );

			hknpCompositeShapeKeysCollector() {}

			hknpShapeKey operator[](int k) const
			{
				return m_hits[k];
			}

			int getSize() const
			{
				return m_hits.getSize();
			}

			void clearAndDeallocate()
			{
				m_hits.clear();
			}

			// hknpCollisionQueryCollector implementation
			virtual void addHit( const hknpCollisionResult& hit ) HK_OVERRIDE
			{
				m_hits.pushBack( hit.m_hitBodyInfo.m_shapeKey );
			}

		public:

			hkInplaceArray<hknpShapeKey, 32> m_hits;
	};

}	// anonymous namespace


void hknpCharacterProxy::worldLinearCast(
	hkVector4Parameter castPath,
	hknpAllHitsCollector& castCollector, hknpAllHitsCollector* startCollector,
	hkLocalArray<TriggerVolumeHit>* triggerHits ) const
{
	HK_TIME_CODE_BLOCK( "hknpCharacterProxy_worldLinearCast", HK_NULL );

	// Calculate the maximum distance at which we can consider a point in contact in the startCollector
	hkSimdReal startPointTolerance = hkSimdReal::fromFloat(startCollector ? m_keepDistance + m_keepContactTolerance : 0.f);

	// Calculate an aabb enclosing the proxy shape in the start and end positions
	hkAabb castAabb = m_aabb;
	{
		castAabb.expandBy(startPointTolerance + m_world->m_collisionTolerance);
		hkAabbUtil::calcAabb(m_transform, castAabb, castAabb);
		hkAabbUtil::expandAabbByMotion(castAabb, castPath, castAabb);
	}

	// Obtain all overlapping bodies
	hkLocalArray<hknpBodyId> targets(16);
	m_world->queryAabb(castAabb, targets);
	if (targets.isEmpty() || (targets.getSize() == 1 && m_bodyId == targets[0]))
	{
		return;
	}

	// Filter out results we are not interested in
	hknpQueryFilterData filterData;
	filterData.m_collisionFilterInfo = m_collisionFilterInfo;
	for (int i = 0; i < targets.getSize(); ++i)
	{
		// We don't want to hit ourselves, if present in the world.
		if (targets[i] == m_bodyId)
		{
			targets[i] = hknpBodyId::invalid();
			continue;
		}

		const hknpBody& targetBody = m_world->getSimulatedBody(targets[i]);

		// Filter away collision disabled bodies
		if( !m_world->m_modifierManager->getCollisionFilter()->isCollisionEnabled( hknpCollisionQueryType::UNDEFINED, filterData, targetBody ) )
		{
			targets[i] = hknpBodyId::invalid();
			continue;
		}
	}

	// Set up cast shape query context. We don't need pre-created triangle shapes as we will only be querying leaf shapes.
	hknpCollisionQueryContext queryContext(HK_NULL, HK_NULL);
	queryContext.m_dispatcher = m_world->m_collisionQueryDispatcher;
	ShapeInfo targetShapeInfo;

	// Perform a linear cast against all targets
	for (int i = 0; i < targets.getSize(); ++i)
	{
		if (!targets[i].isValid())
		{
			continue;
		}

		const hknpBody& targetBody = m_world->getSimulatedBody(targets[i]);
		const hknpShape* targetShape = targetBody.m_shape;
		targetShapeInfo.m_body = &targetBody;

		// Check if target body is a composite shape
		hknpCollisionDispatchType::Enum targetDispatchType = targetShape->m_dispatchType;
		if ((targetDispatchType == hknpCollisionDispatchType::COMPOSITE) ||
			(targetDispatchType == hknpCollisionDispatchType::DISTANCE_FIELD))
		{
			// Obtain all leaf shapes that overlap the cast aabb
			hknpAabbQuery aabbQuery;
			hkTransform worldToTarget; worldToTarget.setInverse(targetBody.getTransform());
			hkAabbUtil::calcAabb(worldToTarget, castAabb, aabbQuery.m_aabb);

#if NP_CHARACTER_PROXY_USE_DYNAMIC_COLLECTOR
			hknpCompositeShapeKeysCollector targetLeafs;
			hknpShapeQueryInterface::queryAabb( aabbQuery, *targetShape, &targetLeafs );
#else
			hkInplaceArray<hknpShapeKey, NP_CHARACTER_PROXY_MAX_SHAPE_KEYS> targetLeafs;
			hknpShapeQueryInterface::queryAabb( aabbQuery, *targetShape, &targetLeafs );
			
			
			HK_ASSERT2(0x2cd7ae9d, (targetLeafs.getNumOverflows() == 0), "Too many children shapes obtained in composite shape aabb query");
#endif

			// If the target body raises trigger events, prepare codec context to decode children materials
			hknpShapeTagCodec::Context codecContext;
			codecContext.m_queryType		= hknpCollisionQueryType::GET_CLOSEST_POINTS;
			codecContext.m_body				= &targetBody;
			codecContext.m_rootShape		= targetShape;
			codecContext.m_partnerBody		= HK_NULL;
			codecContext.m_partnerRootShape	= HK_NULL;
			codecContext.m_partnerShapeKey	= HKNP_INVALID_SHAPE_KEY;
			codecContext.m_parentShape		= HK_NULL;

			// Process all leaf shapes
			for (int j = 0; j < targetLeafs.getSize(); ++j)
			{
				// Get leaf shape
				const hknpShapeKey targetLeafShapeKey = targetLeafs[j];
				hknpShapeCollectorWithInplaceTriangle leafShapeCollector;
				leafShapeCollector.reset( targetBody.getTransform() );
				targetShape->getLeafShape( targetLeafShapeKey, &leafShapeCollector );

				// Setup shape info for cast query
				targetShapeInfo.m_shapeKey = targetLeafShapeKey;
				targetShapeInfo.m_shape = leafShapeCollector.m_shapeOut;
				targetShapeInfo.m_transform = &leafShapeCollector.m_transformOut;
				targetShapeInfo.m_materialId = targetBody.m_materialId;
				targetShapeInfo.m_collisionFilterInfo = targetBody.m_collisionFilterInfo;
				targetShapeInfo.m_userData = targetBody.m_userData;

				// Decode leaf material
				codecContext.m_partnerShapeKey = targetLeafShapeKey;
				codecContext.m_partnerShape = leafShapeCollector.m_shapeOut;
				codecContext.m_parentShape = leafShapeCollector.m_parentShape;
				m_world->getShapeTagCodec()->decode(
					leafShapeCollector.m_shapeTagPath.begin(), leafShapeCollector.m_shapeTagPath.getSize(),
					leafShapeCollector.m_shapeTagOut, &codecContext,
					&targetShapeInfo.m_collisionFilterInfo, &targetShapeInfo.m_materialId, &targetShapeInfo.m_userData);

				shapeCast(
					castPath, castCollector, startCollector, triggerHits, &queryContext, targetShapeInfo, startPointTolerance);
			}

			targetLeafs.clearAndDeallocate();
		}

		else
		{
			// Setup shape info for cast query
			targetShapeInfo.m_shapeKey = HKNP_INVALID_SHAPE_KEY;
			targetShapeInfo.m_shape = targetShape;
			targetShapeInfo.m_transform = &targetBody.getTransform();
			targetShapeInfo.m_materialId = targetBody.m_materialId;
			targetShapeInfo.m_collisionFilterInfo = targetBody.m_collisionFilterInfo;
			targetShapeInfo.m_userData = targetBody.m_userData;

			shapeCast(
				castPath, castCollector, startCollector, triggerHits, &queryContext, targetShapeInfo, startPointTolerance);
		}
	}
}


void hknpCharacterProxy::shapeCast(
	hkVector4Parameter castPath, hknpAllHitsCollector& castCollector, hknpAllHitsCollector* startCollector,
	hkLocalArray<TriggerVolumeHit>* triggerHits, hknpCollisionQueryContext* queryContext,
	const ShapeInfo& targetShapeInfo, hkSimdRealParameter startPointTolerance) const
{
	
	hkSimdReal earlyOutDistance; earlyOutDistance.setFromFloat(0.01f);
	hkSimdReal maxExtraPenetration; maxExtraPenetration.setFromFloat(0.01f);
	const int maxIterations = 20;

	// Check if the target shape is a trigger volume.
	bool isTargetATrigger = false;
	{
		// Note: This only considers the root shape.
		const hknpMaterial& targetMaterial = m_world->getMaterialLibrary()->getEntry(targetShapeInfo.m_materialId);
		if (targetMaterial.m_triggerVolumeType != hknpMaterial::TRIGGER_VOLUME_NONE)
		{
			if (!triggerHits)
			{
				return;
			}
			isTargetATrigger = true;
		}
	}

	// Cast proxy shape against target shape
	hknpCollisionResult castHit;
	hknpCollisionResult startHit;
	iterativeLinearCast(
		queryContext, m_shape, m_transform, targetShapeInfo.m_shape, *targetShapeInfo.m_transform, castPath,
		startPointTolerance, earlyOutDistance, maxExtraPenetration, &castHit, startCollector ? &startHit : HK_NULL,
		maxIterations);

	// Add hits to collectors
	hknpCollisionResult* hits[] = { &startHit, &castHit };
	hknpAllHitsCollector* collectors[] = { startCollector, &castCollector };
	for (int i = 0; i < 2; ++i)
	{
		hknpCollisionResult* hit = hits[i];
		if (hit->m_queryType != hknpCollisionQueryType::UNDEFINED)
		{
			if (isTargetATrigger)
			{
				TriggerVolumeHit* triggerHit = &triggerHits->expandOne();
				new ((void*)triggerHit) TriggerVolumeHit(targetShapeInfo.m_body->m_id, targetShapeInfo.m_shapeKey, hit->m_fraction);

				// Report trigger hits only once
				break;
			}
			else
			{
				hit->m_hitBodyInfo.m_bodyId = targetShapeInfo.m_body->m_id;
				hit->m_hitBodyInfo.m_shapeKey = targetShapeInfo.m_shapeKey;
				hit->m_hitBodyInfo.m_shapeMaterialId = targetShapeInfo.m_materialId;
				hit->m_hitBodyInfo.m_shapeCollisionFilterInfo = targetShapeInfo.m_collisionFilterInfo;
				hit->m_hitBodyInfo.m_shapeUserData = targetShapeInfo.m_userData;
				collectors[i]->addHit(*hit);
			}
		}
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
