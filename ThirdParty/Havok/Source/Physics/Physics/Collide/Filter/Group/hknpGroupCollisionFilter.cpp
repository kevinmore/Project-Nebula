/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/Filter/Group/hknpGroupCollisionFilter.h>

#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>


#if !defined(HK_PLATFORM_SPU)

HK_COMPILE_TIME_ASSERT( hknpGroupCollisionFilter::COLLISION_LUT_SIZE >= (sizeof(hkUint32)*8) );

#else
//HK_COMPILE_TIME_ASSERT(sizeof(hknpGroupCollisionFilter)==256);
#endif

hknpGroupCollisionFilter::hknpGroupCollisionFilter()
:	hknpCollisionFilter( hknpCollisionFilter::GROUP_FILTER )
{
#if !defined(HK_PLATFORM_SPU)
	// Initially enable all collision groups
	for (int i=0; i<COLLISION_LUT_SIZE; i++)
	{
		m_collisionLookupTable[i] = 0xffffffff;
	}
	m_nextFreeSystemGroup = 0;
#endif
}

#if !defined(HK_PLATFORM_SPU)
int hknpGroupCollisionFilter::filterBodyPairs(const hknpSimulationThreadContext& context, hknpBodyIdPair* pairs, int numPairs ) const
{
	hknpBodyIdPair* dst = pairs;
	hknpBodyIdPair* src = pairs;
	hknpWorld* world = context.m_world;
	for( int i = 0; i < numPairs; i++ )
	{
		const hknpBody& bodyA = world->getBody( src->m_bodyA );
		const hknpBody& bodyB = world->getBody( src->m_bodyB );

		const hkUint32 collisionFilterInfoA = bodyA.m_collisionFilterInfo;
		const hkUint32 collisionFilterInfoB = bodyB.m_collisionFilterInfo;

		if( _isCollisionEnabled( collisionFilterInfoA, collisionFilterInfoB ) )
		{
			*dst = *src;
			dst++;
		}
		src++;
	}
	int newNumPairs = (int)(dst - pairs);
	return newNumPairs;
}
#endif


bool hknpGroupCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	hknpBroadPhaseLayerIndex layerIndex ) const
{
	return true;
}


bool hknpGroupCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	hknpBodyId bodyIdA,
	hknpBodyId bodyIdB ) const
{
	return true;
}


bool hknpGroupCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	const hknpQueryFilterData& queryFilterData,
	const hknpBody& body ) const
{
	return _isCollisionEnabled( queryFilterData.m_collisionFilterInfo, body.m_collisionFilterInfo );
}


bool hknpGroupCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	bool targetShapeIsB,
	const FilterInput& shapeInputA,
	const FilterInput& shapeInputB ) const
{
	return _isCollisionEnabled( shapeInputA.m_filterData.m_collisionFilterInfo, shapeInputB.m_filterData.m_collisionFilterInfo );
}


#if !defined(HK_PLATFORM_SPU)

void hknpGroupCollisionFilter::enableCollisionsUsingBitfield( hkUint32 layerBitsA, hkUint32 layerBitsB )
{
	HK_ASSERT2(0x3c3a0084,  (layerBitsA|layerBitsB) != 0, "layer bits not set");
	for (int i=0; i< COLLISION_LUT_SIZE; i++)
	{
		int b = 1<<i;
		if ( b & layerBitsA )
		{
			m_collisionLookupTable[i] |= layerBitsB;
		}
		if ( b & layerBitsB )
		{
			m_collisionLookupTable[i] |= layerBitsA;
		}
	}
}

void hknpGroupCollisionFilter::enableCollisionsBetween( int layerA, int layerB )
{
	HK_ASSERT(0x66c2b6fd,  0 <= layerA && layerA < COLLISION_LUT_SIZE );
	HK_ASSERT(0x5a285631,  0 <= layerB && layerB < COLLISION_LUT_SIZE );

	m_collisionLookupTable[layerA] |= hkUint32(1 << layerB);
	m_collisionLookupTable[layerB] |= hkUint32(1 << layerA);
}

void hknpGroupCollisionFilter::disableCollisionsBetween( int layerA, int layerB )
{
	HK_ASSERT(0x2a168aec,  0 <= layerA && layerA < COLLISION_LUT_SIZE );
	HK_ASSERT(0x234fb60b,  0 <= layerB && layerB < COLLISION_LUT_SIZE );
	HK_ASSERT2(0x4ab45935,  layerA > 0, "You are not allowed to disable collision of layer 0");
	HK_ASSERT2(0x358c7ccd,  layerB > 0, "You are not allowed to disable collision of layer 0");

	m_collisionLookupTable[layerA] &= hkUint32(~(1 << layerB));
	m_collisionLookupTable[layerB] &= hkUint32(~(1 << layerA));
}

void hknpGroupCollisionFilter::disableCollisionsUsingBitfield( hkUint32 layerBitsA, hkUint32 layerBitsB )
{
	HK_ASSERT2(0x41c4fad2,  (layerBitsA|layerBitsB) != 0, "layer bits not set");
	HK_ASSERT2(0x49059b77,  (layerBitsA&1) == 0, "You are not allowed to disable collision of layer 0");
	HK_ASSERT2(0x371ca278,  (layerBitsB&1) == 0, "You are not allowed to disable collision of layer 0");
	for (int i=0; i< COLLISION_LUT_SIZE; i++)
	{
		int b = 1<<i;
		if ( b & layerBitsA )
		{
			m_collisionLookupTable[i] &= ~layerBitsB;
		}
		if ( b & layerBitsB )
		{
			m_collisionLookupTable[i] &= ~layerBitsA;
		}
	}
}

#endif

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
