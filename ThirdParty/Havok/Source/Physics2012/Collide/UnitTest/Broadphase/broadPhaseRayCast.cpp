/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Physics2012/Collide/Agent/Collidable/hkpCollidable.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastOutput.h>
#include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseHandlePair.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseCastCollector.h>


class BroadPhaseFlagCollector: public hkpBroadPhaseCastCollector
{
public:
	BroadPhaseFlagCollector(){ m_hasHit = false; }
	virtual	hkReal addBroadPhaseHandle( const hkpBroadPhaseHandle* broadphaseHandle, int castIndex )
	{
		m_hasHit = true;
		return 0.0f;
	}

	hkBool m_hasHit;

};

static void HK_CALL testRayPatch( hkPseudoRandomGenerator& rndGen, hkpBroadPhase& broadPhase, hkReal worldSize )
{
	broadPhase.lock();

	//
	// add a single random object to the broadphase
	//

	hkAabb aabb;

	hkReal size = hkMath::min2( hkReal(1.0f), worldSize );

#define RL rndGen.getRandRange( -size, 0 )
#define RH(i) rndGen.getRandRange( aabb.m_min(i), size )

	aabb.m_min.set( RL, RL, RL );
	aabb.m_max.set( RH(0), RH(1), RH(2) );

	aabb.m_max.setNeg<4>( aabb.m_min );

	hkpBroadPhaseHandle object;

	hkArray<hkpBroadPhaseHandlePair> newPairsOut;

	broadPhase.addObject( &object, aabb, newPairsOut );

	//
	// create a shape with the same dimensions
	//
	hkVector4 extents; extents.setSub( aabb.m_max, aabb.m_min );
	extents.mul( hkSimdReal::getConstant(HK_QUADREAL_INV_2) );

	hkpBoxShape boxShape( extents, 0.0f );
	hkVector4 center; center.setInterpolate( aabb.m_max, aabb.m_min, hkSimdReal::getConstant(HK_QUADREAL_INV_2) );


	//
	//	Perform a series of random raycasts
	//
	for (int i = 0; i < 20; i++ )
	{
		hkVector4 from;  rndGen.getRandomVector11( from );
		hkVector4 to;	 rndGen.getRandomVector11( to );

		const hkReal scaledWorldSize = worldSize * 0.99f;
		for (int c = 0; c < 3; c++)
		{
			from(c) = hkMath::clamp(from(c), -scaledWorldSize, scaledWorldSize);
			to(c) = hkMath::clamp(to(c), -scaledWorldSize, scaledWorldSize);
		}

		if ( worldSize < 0.5f )
		{
			to.setNeg<4>( from );	// make sure we always hit this object
		}

		//
		// check the shape raycast
		//
		hkBool shapeHits;
		{
			hkpShapeRayCastInput input;
			input.m_from.setSub( from, center );
			input.m_to.setSub( to, center );

			hkpShapeRayCastOutput output;

			shapeHits = boxShape.castRay( input, output );
		}

		static int ci = 0;
		if ( ++ci == 23081 )
		{
			ci = ci;
		}

		if ( !shapeHits )
		{
			continue;
		}

		//
		//	Now our broadphase also must return a hit
		//


		BroadPhaseFlagCollector collector;

		hkpBroadPhase::hkpCastRayInput rayInput;
		rayInput.m_from = from;
		rayInput.m_toBase = &to;
		broadPhase.castRay( rayInput, &collector, 0 );

		HK_TEST2( collector.m_hasHit, "At iteration: " << ci );
	}

	broadPhase.unlock();
}

int broadphaseRaycast_main()
{
	hkDisableError disable0xf034de45( 0xf034de45 );// Your broadphase extents is bigger than 12k
	{
		hkPseudoRandomGenerator rndGen(101);
		for ( hkReal worldSize  = 100000.001f; worldSize > 0.001f; worldSize *= 0.99f )
		{
			hkVector4 worldMax; worldMax.setAll( worldSize );
			hkVector4 worldMin; worldMin.setNeg<4>( worldMax );

			hkVector4 scale;
			hkVector4 offsetLow;
			hkVector4 offsetHigh;
			{
				hkVector4 span, spanInv;
				span.setSub( worldMax, worldMin);
				spanInv.set( 1.0f/span(0), 1.0f/span(1), 1.0f/span(2), 0.0f );

				scale.setMul( hkSimdReal::fromFloat(hkReal(hkAabbUtil::AABB_UINT32_MAX_FVALUE)), spanInv );
				offsetLow.setNeg<4>( worldMin );
				hkVector4 rounding; rounding.setMul( hkSimdReal::fromFloat(1.0f/hkAabbUtil::AABB_UINT32_MAX_FVALUE), span);
				offsetHigh.setAdd(offsetLow, rounding);

				scale.zeroComponent<3>();
				offsetLow .zeroComponent<3>();
				offsetHigh.zeroComponent<3>();
			}

			{
				hkpBroadPhase* broadPhase = hk3AxisSweep16CreateBroadPhase( worldMin, worldMax, 4 );
				broadPhase->set32BitOffsetAndScale(offsetLow, offsetHigh, scale);
				testRayPatch( rndGen, *broadPhase, worldSize );
				broadPhase->markForWrite();
				delete broadPhase;
			}
		}
	}
	{
		hkPseudoRandomGenerator rndGen(101);
		for ( hkReal worldSize  = 1000.001f; worldSize > 0.001f; worldSize *= 0.99f )
		{
			hkVector4 worldMax; worldMax.setAll( worldSize );
			hkVector4 worldMin; worldMin.setNeg<4>( worldMax );

			hkVector4 scale;
			hkVector4 offsetLow;
			hkVector4 offsetHigh;
			{
				hkVector4 span, spanInv;
				span.setSub( worldMax, worldMin);
				spanInv.set( 1.0f/span(0), 1.0f/span(1), 1.0f/span(2), 0.0f );

				scale.setMul( hkSimdReal::fromFloat(hkReal(hkAabbUtil::AABB_UINT32_MAX_FVALUE)), spanInv );
				offsetLow.setNeg<4>( worldMin );
				hkVector4 rounding; rounding.setMul( hkSimdReal::fromFloat(1.0f/hkAabbUtil::AABB_UINT32_MAX_FVALUE), span);
				offsetHigh.setAdd(offsetLow, rounding);

				scale.zeroComponent<3>();
				offsetLow .zeroComponent<3>();
				offsetHigh.zeroComponent<3>();
			}

			{
				hkpBroadPhase* broadPhase = hk3AxisSweep32CreateBroadPhase( worldMin, worldMax, 4 );
				broadPhase->set32BitOffsetAndScale(offsetLow, offsetHigh, scale);
				testRayPatch( rndGen, *broadPhase, worldSize );
				broadPhase->markForWrite();
				delete broadPhase;
			}
		}
	}
	return 0;
}



#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(broadphaseRaycast_main, "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__     );

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
