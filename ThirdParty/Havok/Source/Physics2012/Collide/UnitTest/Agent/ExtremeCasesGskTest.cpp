/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

// This checks the sphere-triangle agent, both the linearCast() and getPenetrations() methods.
#include <Physics2012/Collide/hkpCollide.h>


#include <Common/Base/UnitTest/hkUnitTest.h>
 // Large include
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskfAgent.h>
#include <Physics2012/Collide/Agent/hkpCollisionInput.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#include <Physics2012/Collide/Query/Collector/BodyPairCollector/hkpFlagCdBodyPairCollector.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>

#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>


void _checkGsk( const hkpCollidable& a, const hkpCollidable& b, const hkVector4& resultNormal, hkReal maxDistance )
{
	hkpCollisionInput input;
	input.m_tolerance = 1.f;

	hkpClosestCdPointCollector collector;
	hkpGskfAgent::staticGetClosestPoints( a, b, input, collector );
	HK_TEST( collector.hasHit() );
	if ( collector.hasHit() )
	{
		const hkpRootCdPoint& hit = collector.getHit();
		HK_TEST( hit.m_contact.getDistance() <= maxDistance + HK_REAL_EPSILON );
		if ( resultNormal.lengthSquared<3>().getReal() > 0 )
		{
			hkVector4 negNormal; negNormal.setNeg<3>( resultNormal );
			HK_TEST( resultNormal.allEqual<3>( hit.m_contact.getSeparatingNormal(), hkSimdReal::fromFloat(1e-3f)) || negNormal.allEqual<3>( hit.m_contact.getSeparatingNormal(), hkSimdReal::fromFloat(1e-3f) ));
		}
	}
}


	// Check various configurations, both penetrating and non-penetrating.
int ExtremeCasesGskTest()
{
	//
	//	Create a sphere in the middle of a triangle
	//
	hkMotionState ms;	ms.getTransform().setIdentity();
	hkpTriangleShape triangleShape;
	hkpSphereShape sphereShape( 1.0f );

	{
		hkVector4 va; va.set( 0.f,-1.f, 0.f);
		hkVector4 vb; vb.set( 0.f, 1.f, 1.f );
		hkVector4 vc; vc.set( 0.f, 1.f,-1.f);

		triangleShape.setVertex( 0, va );
		triangleShape.setVertex( 1, vb );
		triangleShape.setVertex( 2, vc );

		_checkGsk( hkpCollidable( &sphereShape, &ms ), hkpCollidable ( &triangleShape, &ms ), hkVector4::getZero(), 0.0f);
	}

	// two capsules intersecting with 90 degree angle and at 0 degrees
	{
		hkVector4 a0; a0.set( -1.0f, 0.0f, 0.0f  );
		hkVector4 a1; a1.set(  1.0f, 0.0f, 0.0f  );
		hkpCapsuleShape capsA( a0, a1, 0.0f );

		hkVector4 b0; b0.set( 0.0f, -1.0f, 0.0f  );
		hkVector4 b1; b1.set( 0.0f,  1.0f, 0.0f  );
		hkpCapsuleShape capsB( b0, b1, 0.0f );

		_checkGsk( hkpCollidable( &capsA, &ms ), hkpCollidable ( &capsB, &ms ), hkTransform::getIdentity().getColumn(2), 0.0f);

		// lift second capsule
		hkVector4 c0; c0.set( 0.0f, -1.0f, 0.00001f  );
		hkVector4 c1; c1.set( 0.0f,  1.0f, 0.00001f  );
		hkpCapsuleShape capsC( c0, c1, 0.0f );
		_checkGsk( hkpCollidable( &capsA, &ms ), hkpCollidable ( &capsC, &ms ), hkTransform::getIdentity().getColumn(2), 0.00001f);

		// embedded second
		hkVector4 d0; d0.set( -0.5f, 0.0f, 0.0f  );
		hkVector4 d1; d1.set(  0.5f, 0.0f, 0.0f  );
		hkpCapsuleShape capsD( d0, d1, 0.0f );
		_checkGsk( hkpCollidable( &capsA, &ms ), hkpCollidable ( &capsD, &ms ), hkVector4::getZero(), 0.0f);
	}

	// sphere on the middle of a capsule
	{
		hkpSphereShape A( 0.0f );

		hkVector4 b0; b0.set( 0.0f, -1.0f, 0.0f  );
		hkVector4 b1; b1.set( 0.0f,  1.0f, 0.0f  );
		hkpCapsuleShape capsB( b0, b1, 0.0f );
		_checkGsk( hkpCollidable( &A, &ms ), hkpCollidable ( &capsB, &ms ), hkVector4::getZero(), 0.0f);
	}

	// sphere on the end of a capsule
	{
		hkpSphereShape A( 0.0f );

		hkVector4 b0; b0.set( 0.0f,  0.0f, 0.0f  );
		hkVector4 b1; b1.set( 0.0f,  1.0f, 0.0f  );
		hkpCapsuleShape capsB( b0, b1, 0.0f );
		_checkGsk( hkpCollidable( &A, &ms ), hkpCollidable ( &capsB, &ms ), hkVector4::getZero(), 0.0f);
	}

	// 2 spheres
	{
		hkpSphereShape A( 0.0f );
		_checkGsk( hkpCollidable( &A, &ms ), hkpCollidable ( &A, &ms ), hkVector4::getZero(), 0.0f);
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(ExtremeCasesGskTest, "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__     );

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
