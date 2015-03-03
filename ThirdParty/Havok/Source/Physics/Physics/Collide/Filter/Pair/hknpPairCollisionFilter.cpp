/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/Filter/Pair/hknpPairCollisionFilter.h>

#include <Common/Base/Container/PointerMap/hkMap.cxx>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>


// force explicit template instantiation (required by gcc release builds)
template class hkMapBase<hknpPairCollisionFilter::Key, hkUint32, hknpPairCollisionFilter::MapOperations>;


hknpPairCollisionFilter::hknpPairCollisionFilter( const hknpCollisionFilter* childFilter )
:	hknpCollisionFilter( hknpCollisionFilter::PAIR_FILTER )
,	m_childFilter( childFilter )
{
}


hknpPairCollisionFilter::hknpPairCollisionFilter( hkFinishLoadedObjectFlag flag )
:	hknpCollisionFilter( flag )
{
}


int hknpPairCollisionFilter::disableCollisionsBetween( hknpWorld* world, hknpBodyId bodyIdA, hknpBodyId bodyIdB )
{
	HK_ASSERT2( 0xaf25143f, bodyIdA.isValid() && bodyIdB.isValid(), "Both body ids have to be valid." );

	Key key;
	calcKey( bodyIdA, bodyIdB, key );

	hkUint32 numDisables = m_disabledPairs.getWithDefault( key, 0 );
	numDisables++;

	m_disabledPairs.insert( key, numDisables );

	// Rebuild the cd caches for the smaller dynamic body of the pair.
	{
		const hknpBody& bodyA = world->getBody(bodyIdA);
		const hknpBody& bodyB = world->getBody(bodyIdB);

		hknpBodyId smallerDynamicBodyId;

		if      ( !bodyA.isDynamic() ) { smallerDynamicBodyId = bodyIdB; }
		else if ( !bodyB.isDynamic() ) { smallerDynamicBodyId = bodyIdA; }
		else
		{
			if ( bodyB.m_radiusOfComCenteredBoundingSphere < bodyA.m_radiusOfComCenteredBoundingSphere ) { smallerDynamicBodyId = bodyIdB; }
			else                                                                                         { smallerDynamicBodyId = bodyIdA; }
		}

		world->rebuildBodyCollisionCaches(smallerDynamicBodyId);
	}

	return numDisables;
}


int hknpPairCollisionFilter::enableCollisionsBetween( hknpWorld* world, hknpBodyId bodyIdA, hknpBodyId bodyIdB )
{
	HK_ASSERT2( 0xaf25143e, bodyIdA.isValid() && bodyIdB.isValid(), "Both body ids have to be valid." );

	Key key;
	calcKey( bodyIdA, bodyIdB, key );

	hkUint32 numDisables = m_disabledPairs.getWithDefault( key, 0 );
	if ( numDisables == 0 )
	{
		return 0;
	}
	numDisables--;

	// If the counter is still positive, update it. Otherwise remove the entry.
	if ( numDisables > 0 )
	{
		m_disabledPairs.insert( key, numDisables );
	}
	else
	{
		m_disabledPairs.remove( key );

		// Rebuild cd caches for the body pair if pair is really supposed to collide again.
		{
			const hknpBody& bodyA = world->getBody(bodyIdA);
			const hknpBody& bodyB = world->getBody(bodyIdB);

			hknpQueryFilterData filterDataA(bodyA);

			if ( !m_childFilter || m_childFilter->isCollisionEnabled(hknpCollisionQueryType::UNDEFINED, filterDataA, bodyB) )
			{
				hknpBodyIdPair bodyPair;
				bodyPair.m_bodyA = bodyIdA;
				bodyPair.m_bodyB = bodyIdB;
				world->rebuildBodyPairCollisionCaches( &bodyPair, 1 );
			}
		}
	}

	return numDisables;
}


void hknpPairCollisionFilter::clearAll()
{
	m_disabledPairs.clear();
}


int hknpPairCollisionFilter::filterBodyPairs( const hknpSimulationThreadContext& context, hknpBodyIdPair* pairs, int numPairs ) const
{
	if ( m_childFilter )
	{
		numPairs = m_childFilter->filterBodyPairs( context, pairs, numPairs );
	}

	hknpBodyIdPair* dst = pairs;
	hknpBodyIdPair* src = pairs;
	for ( int i=0; i<numPairs; i++ )
	{
		if ( _isCollisionEnabled( src->m_bodyA, src->m_bodyB ) )
		{
			*dst = *src;
			dst++;
		}
		src++;
	}
	int newNumPairs = (int)(dst - pairs);
	return newNumPairs;
}



bool hknpPairCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	hknpBroadPhaseLayerIndex layerIndex ) const
{
	return !m_childFilter || m_childFilter->isCollisionEnabled( queryType, layerIndex );
}


bool hknpPairCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	hknpBodyId bodyIdA,
	hknpBodyId bodyIdB ) const
{
	if (m_childFilter && !m_childFilter->isCollisionEnabled(queryType, bodyIdA, bodyIdB))
	{
		return false;
	}

	if (!_isCollisionEnabled(bodyIdA, bodyIdB))
	{
		return false;
	}

	return true;
}


bool hknpPairCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	const hknpQueryFilterData& queryFilterData,
	const hknpBody& body ) const
{
	// As we only support filtering at BODY VS BODY level with this filter, we forward the test to a child filter
	// (if available) or otherwise report a valid collision.
	return !m_childFilter || m_childFilter->isCollisionEnabled(queryType, queryFilterData, body);
}


bool hknpPairCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	bool targetShapeIsB,
	const FilterInput& shapeInputA,
	const FilterInput& shapeInputB ) const
{
	// As we only support filtering at BODY VS BODY level with this filter, we forward the test to a child filter
	// (if available) or otherwise report a valid collision.
	return !m_childFilter || m_childFilter->isCollisionEnabled( queryType, targetShapeIsB, shapeInputA, shapeInputB );
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
