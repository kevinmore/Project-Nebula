/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Query/CastUtil/hkpSimpleWorldRayCaster.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>


hkReal 	hkpSimpleWorldRayCaster::addBroadPhaseHandle( const hkpBroadPhaseHandle* broadPhaseHandle, int castIndex )
{
	const hkpCollidable* col = static_cast<hkpCollidable*>( static_cast<const hkpTypedBroadPhaseHandle*>(broadPhaseHandle)->getOwner() );
	const hkpShape* shape = col->getShape();
	hkpWorldRayCastOutput* output = m_result + castIndex;

	// phantoms do not have shapes
	if (shape)
	{
		if ( m_filter->isCollisionEnabled( m_input[castIndex], *col ) )
		{
			const hkTransform& trans = col->getTransform();
			m_shapeInput.m_from._setTransformedInversePos( trans, m_input[castIndex].m_from);
			m_shapeInput.m_to.  _setTransformedInversePos( trans, m_input[castIndex].m_to);
			m_shapeInput.m_filterInfo = m_input[castIndex].m_filterInfo;
			m_shapeInput.m_collidable = col;
			m_shapeInput.m_userData = m_input[castIndex].m_userData;
			if ( shape->castRay( m_shapeInput, *output) )
			{
				output->m_rootCollidable = col;
				output->m_normal._setRotatedDir( trans.getRotation(), output->m_normal );
			}
		}
	}
	return output->m_hitFraction;
}



void hkpSimpleWorldRayCaster::castRay( const hkpBroadPhase& broadphase, const hkpWorldRayCastInput& input, const hkpCollisionFilter* filter, hkpWorldRayCastOutput& output )
{
	HK_ASSERT2(0x13ddc6c1,  filter, "You need to specify a valid filter");
	HK_ASSERT2(0x41fd5db8,  output.hasHit() == false, "Your output has not been reset");

	HK_TIMER_BEGIN("RayCastSimpl", HK_NULL);
	m_input = &input;
	m_result = &output;
	m_filter = filter;

	if ( input.m_enableShapeCollectionFilter )
	{
		m_shapeInput.m_rayShapeCollectionFilter = filter;
	}
	else
	{
		m_shapeInput.m_rayShapeCollectionFilter = HK_NULL;
	}

	hkpBroadPhase::hkpCastRayInput rayInput;
	rayInput.m_from = input.m_from;
	rayInput.m_toBase = &input.m_to;
		
	broadphase.castRay( rayInput, this, 0 );
	HK_TIMER_END();
}

void hkpSimpleWorldRayCaster::castRay( const hkpBroadPhase& broadphase, const hkpWorldRayCastInput& input, const hkpCollisionFilter* filter, const hkpBroadPhaseAabbCache* cache, hkpWorldRayCastOutput& output )
{
	HK_ASSERT2(0x2d198b3b,  filter, "You need to specify a valid filter");
	HK_ASSERT2(0x3f9759a2,  output.hasHit() == false, "Your output has not been reset");

	HK_TIMER_BEGIN("RayCstCchSim", HK_NULL);
	m_input = &input;
	m_result = &output;
	m_filter = filter;

	if ( input.m_enableShapeCollectionFilter )
	{
		m_shapeInput.m_rayShapeCollectionFilter = filter;
	}
	else
	{
		m_shapeInput.m_rayShapeCollectionFilter = HK_NULL;
	}
	hkpBroadPhase::hkpCastRayInput rayInput;
	rayInput.m_from = input.m_from;
	rayInput.m_toBase = &input.m_to;
	rayInput.m_aabbCacheInfo = cache;
		
	broadphase.castRay( rayInput, this, 0 );
	HK_TIMER_END();
}

void hkpSimpleWorldRayCaster::castRayGroup( const hkpBroadPhase& broadphase, const hkpWorldRayCastInput* inputArray, int numRays, const hkpCollisionFilter* filter, hkpWorldRayCastOutput* outputs ) 
{
	HK_TIMER_BEGIN("RayCstGrpSim", HK_NULL);
	hkAabb aabb;
	{
		aabb.m_min.setMin( inputArray->m_from, inputArray->m_to );
		aabb.m_max.setMax( inputArray->m_from, inputArray->m_to );

		const hkpWorldRayCastInput* in = inputArray;
		in++;
		for ( int x = numRays-2; x>=0; x--)
		{
			aabb.m_min.setMin( aabb.m_min, in->m_to );
			aabb.m_min.setMin( aabb.m_min, in->m_from );

			aabb.m_max.setMax( aabb.m_max, in->m_to );
			aabb.m_max.setMax( aabb.m_max, in->m_from );
			in++;
		}
	}
	int cacheSize = broadphase.getAabbCacheSize();
	hkpBroadPhaseAabbCache* cache = hkAllocateStack<hkpBroadPhaseAabbCache>(cacheSize);

	broadphase.calcAabbCache( aabb, cache );
	//
	//	Cast rays
	//
	{
		const hkpWorldRayCastInput* in = inputArray;
		hkpWorldRayCastOutput* out = outputs;
		for ( int x = numRays-1; x>=0; x--)
		{
			castRay( broadphase, *in, filter, cache, *out);
			in++;
			out++;
		}
	}
	hkDeallocateStack(cache, cacheSize);
	HK_TIMER_END();
}


void hkpSimpleWorldRayCaster::castRaysFromSinglePoint( const hkpBroadPhase& broadphase, const hkpWorldRayCastInput* inputArray, int numRays, const hkpCollisionFilter* filter, const hkpBroadPhaseAabbCache* cache, hkpWorldRayCastOutput* outputs )
{
	HK_ASSERT2(0x4791b6a4,  filter, "You need to specify a valid filter");
	HK_ASSERT2(0x3299aa1e,  outputs->hasHit() == false, "Your output has not been reset");

	HK_TIMER_BEGIN("RayCstFSPSim", HK_NULL);
	m_input = inputArray;
	m_result = outputs;
	m_filter = filter;

	if ( inputArray->m_enableShapeCollectionFilter )
	{
		m_shapeInput.m_rayShapeCollectionFilter = filter;
	}
	else
	{
		m_shapeInput.m_rayShapeCollectionFilter = HK_NULL;
	}

	hkpBroadPhase::hkpCastRayInput rayInput;
	rayInput.m_from = inputArray->m_from;
	rayInput.m_toBase = &inputArray->m_to;
	rayInput.m_toStriding = hkSizeOf( hkpWorldRayCastInput );
	rayInput.m_aabbCacheInfo = cache;
	rayInput.m_numCasts = numRays;

	broadphase.castRay( rayInput, this, 0 );
	HK_TIMER_END();
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
