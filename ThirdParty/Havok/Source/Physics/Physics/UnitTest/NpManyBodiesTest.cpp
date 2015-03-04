/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/Container/BlockStream/Allocator/Fixed/hkFixedBlockStreamAllocator.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>


int NpManyBodiesTest_main()
{
	// Create >64K static and >64K dynamic bodies, then destroy them.
	// Collisions are completely disabled, to speed things up.

	const unsigned int numStaticBodies = 75000;
	const unsigned int numDynamicBodies = 75000;

	// These can be small since there will be no collisions
	hkFixedBlockStreamAllocator persistentStreamAllocator( 1024 * 1024 );
	hkFixedBlockStreamAllocator stepLocalStreamAllocator( 1024 * 1024 );

	// Create world
	hknpWorldCinfo worldCinfo;
	{
		worldCinfo.m_bodyBufferCapacity = numStaticBodies + numDynamicBodies + 1;		// +1 for the special body 0
		worldCinfo.m_motionBufferCapacity = numDynamicBodies + 1;						// +1 for the special motion 0
		worldCinfo.setBroadPhaseSize( 1000.0f );
		worldCinfo.m_persistentStreamAllocator = &persistentStreamAllocator;
	}
	hknpWorld world( worldCinfo );

	HK_TEST( world.getBodyCapacity() == hkUint32(worldCinfo.m_bodyBufferCapacity) );
	HK_TEST( world.getNumFreeStaticBodies() >= numStaticBodies );
	HK_TEST( world.getNumFreeDynamicBodies() == numDynamicBodies );

	// Create shape
	hknpShape* shape = hknpSphereShape::createSphereShape( hkVector4::getZero(), 0.5f );

	// Create bodies
	hkLocalArray<hknpBodyId> bodyIds( numStaticBodies + numDynamicBodies );
	{
		hknpBodyCinfo bodyCinfo;
		{
			bodyCinfo.m_shape = shape;
			bodyCinfo.m_flags = hknpBody::DONT_COLLIDE;
		}

		hkPseudoRandomGenerator random(1337);
		for( unsigned int i = 0; i<numStaticBodies; ++i )
		{
			random.getRandomVectorRange( worldCinfo.m_broadPhaseAabb.m_min, worldCinfo.m_broadPhaseAabb.m_max, bodyCinfo.m_position );
			bodyIds.pushBackUnchecked( world.createStaticBody( bodyCinfo ) );
		}
		for( unsigned int i = 0; i<numDynamicBodies; ++i )
		{
			random.getRandomVectorRange( worldCinfo.m_broadPhaseAabb.m_min, worldCinfo.m_broadPhaseAabb.m_max, bodyCinfo.m_position );

			hknpMotionCinfo motionCinfo;
			{
				motionCinfo.initializeWithDensity( &bodyCinfo, 1, 1.0f );
				random.getRandomVector11( motionCinfo.m_linearVelocity );
			}

			bodyIds.pushBackUnchecked( world.createDynamicBody( bodyCinfo, motionCinfo ) );
		}
	}

	HK_TEST( world.getNumBodies() == numStaticBodies + numDynamicBodies + 1 );

	world.commitAddBodies();

#ifndef HK_DEBUG
	world.destroyBodies( bodyIds.begin(), bodyIds.getSize() );	// this is slow in debug

	world.freeDestroyedBodiesAndMotions();

	HK_TEST( world.getNumBodies() == 1 );	// the world's body
#endif

	shape->removeReference();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER( NpManyBodiesTest_main, "Fast", "Physics/Test/UnitTest/Physics/", __FILE__ );

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
