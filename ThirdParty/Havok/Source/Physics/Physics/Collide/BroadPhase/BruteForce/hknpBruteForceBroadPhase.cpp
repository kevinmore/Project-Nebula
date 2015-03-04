/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/BroadPhase/BruteForce/hknpBruteForceBroadPhase.h>

void hknpBruteForceBroadPhase::addBodies( const hknpBodyId* idx, int numIdx, hknpBody* bodies )
{
	for (int i=0; i<numIdx; i++)
	{
		hknpBody& body = bodies[idx[i].value()];
		HK_ASSERT(0x32bc45dc, body.m_broadPhaseId == HKNP_INVALID_BROAD_PHASE_ID);
		if (m_freeList.isEmpty())
		{
			int index = m_bodies.getSize();
			m_bodies.pushBack(idx[i]);
			body.m_broadPhaseId = index;
		}
		else
		{
			hknpBroadPhaseId bpId = m_freeList.back();
			m_freeList.popBack();
			body.m_broadPhaseId = bpId;
			m_bodies[bpId] = idx[i];
		}
	}
}


void hknpBruteForceBroadPhase::removeBodies( const hknpBodyId* idx, int numIdx, hknpBody* bodies )
{
	for (int i=0; i<numIdx; i++)
	{
		hknpBody& body = bodies[idx[i].value()];
		HK_ASSERT(0x32bc45dc, body.m_broadPhaseId != HKNP_INVALID_BROAD_PHASE_ID);
		m_freeList.pushBack(body.m_broadPhaseId);
		m_bodies[body.m_broadPhaseId] = hknpBodyId::invalid();
		body.m_broadPhaseId = hknpBroadPhaseId(HKNP_INVALID_BROAD_PHASE_ID);
	}
}

void hknpBruteForceBroadPhase::markBodiesDirty( hknpBodyId* ids, int numIds, hknpBody* bodies)
{
	// Since we don't have any state (except added or not) we don't need to do anything.
}

static HK_FORCE_INLINE int checkPrevOverlap(hknpBodyIdPair& pair, const hkAabb16* aabbCache)
{
	hknpBodyId iIdx = pair.m_bodyA;
	hknpBodyId jIdx = pair.m_bodyB;

	const hkAabb16& iAabb = aabbCache[ iIdx.value() ];
	const hkAabb16& jAabb = aabbCache[ jIdx.value() ];

	return iAabb.disjoint( jAabb );
}

void hknpBruteForceBroadPhase::findNewPairs( hknpBody* bodies, const hkAabb16* previousAabbs, hkBlockStream<hknpBodyIdPair>::Writer *newPairWriter )
{
	HK_TIME_CODE_BLOCK("bruteForceNSquared", HK_NULL);

	int num = m_bodies.getSize();
	for (int iA=0; iA<num; iA++)
	{
		const hknpBodyId idA = m_bodies[iA];
		if (!idA.isValid())
		{
			continue;
		}
		const hknpBody& bodyA = bodies[idA.value()];
		const hkAabb16& aabbA = bodyA.m_aabb;
		for (int iB=iA+1; iB<num; iB++)
		{
			const hknpBodyId idB = m_bodies[iB];
			if (!idB.isValid())
			{
				continue;
			}
			const hkAabb16& aabbB = bodies[idB.value()].m_aabb;
			if( !aabbA.disjoint(aabbB) )
			{
				hknpBodyIdPair pair;
				pair.m_bodyA = idA;
				pair.m_bodyB = idB;
				if (checkPrevOverlap(pair, previousAabbs))
				{
					hknpBodyIdPair* spair = newPairWriter->reserve<hknpBodyIdPair>();
					*spair = pair;
					newPairWriter->advance(sizeof(hknpBodyIdPair));
				}
			}
		}
	}
}

void hknpBruteForceBroadPhase::queryAabb( const hkAabb16& aabb, const hknpBody* bodies, hkArray<hknpBodyId>& hitsOut )
{
	for (int i=0, ei=m_bodies.getSize(); i<ei; i++)
	{
		hknpBodyId id = m_bodies[i];
		if (!id.isValid())
		{
			continue;
		}
		const hkAabb16& bodyAabb = bodies[id.value()].m_aabb;
		if ( !bodyAabb.disjoint(aabb))
		{
			hitsOut.pushBack( id );
		}
	}
}

void hknpBruteForceBroadPhase::findAllPairs( hkBlockStream<hknpBodyIdPair>::Writer *newPairWriter )
{
	HK_ASSERT(0x76c800e0,0); //NIY
}

void hknpBruteForceBroadPhase::getExtents( hkAabb16& extents ) const
{
	HK_ASSERT(0x76c800e1,0);//NIY
	extents.setEmpty();
}

void hknpBruteForceBroadPhase::buildTaskGraph(
	hknpWorld* world, hknpSimulationContext* simulationContext,
	hkBlockStream<hknpBodyIdPair>* newPairsStream, hkTaskGraph* taskGraphOut )
{
	HK_ASSERT(0x76c800e0,0); //NIY
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
