/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>

namespace
{
	static void validate( const hknpShape* shape, const hkAabb& aabb, hkReal volume, const hkVector4& com, hkReal eps = 0.001f )
	{
		const hkSimdReal epsSr = hkSimdReal::fromFloat(eps);

		// AABB
		{
			hkAabb shapeAabb;
			shape->calcAabb( hkTransform::getIdentity(), shapeAabb );
			HK_TEST2(	aabb.m_max.allEqual<3>( shapeAabb.m_max, epsSr ) &&
						aabb.m_min.allEqual<3>( shapeAabb.m_min, epsSr ),
						"Incorrect AABB" );
		}

		// Mass properties
		{
			const hknpShapeMassProperties* props = (const hknpShapeMassProperties*)shape->getProperty( hknpShapePropertyKeys::MASS_PROPERTIES );
			hkVector4 shapeCom; props->m_compressedMassProperties.m_centerOfMass.unpack( shapeCom );
			HK_TEST2( hkMath::equal( props->m_compressedMassProperties.m_volume, volume, eps ), "Incorrect volume" );
			HK_TEST2( com.allEqual<3>( shapeCom, epsSr ), "Incorrect COM" );
		}
	}
}


int ConvexShapeConstructionTest_main()
{
	// Test a "box" shape
	{
		hkPseudoRandomGenerator rng(0);

		hkVector4 halfExtents;
		halfExtents.set( 1.0f, 2.0f, 3.0f );

		hkAabb aabb;
		aabb.m_max = halfExtents;
		aabb.m_min.setNeg<4>( halfExtents );
		const hkReal volume = halfExtents(0) * halfExtents(1) * halfExtents(2) * 8.0f;

		const hkReal radius = 0.25f;
		hkAabb expandedAabb = aabb; expandedAabb.expandBy( hkSimdReal::fromFloat(radius) );
		hkReal expandedVolume = (halfExtents(0)+radius) * (halfExtents(1)+radius) * (halfExtents(2)+radius) * 8.0f;

		// Maximum length of the random transform.
		hkVector4 maxTransform;
		maxTransform.setMul(hkSimdReal::fromFloat(10.f), halfExtents);

		// From AABB, zero radius
		{
			const hknpConvexShape* shape = hknpConvexShape::createFromHalfExtents( halfExtents, 0.0f );
			validate( shape, aabb, volume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From AABB, zero radius, no faces
		{
			hknpConvexShape::BuildConfig buildConfig;
			buildConfig.m_buildFaces = false;
			const hknpConvexShape* shape = hknpConvexShape::createFromHalfExtents( halfExtents, 0.0f, buildConfig );
			validate( shape, aabb, volume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From AABB, shrunk by radius
		{
			const hknpConvexShape* shape = hknpConvexShape::createFromHalfExtents( halfExtents, radius );
			validate( shape, aabb, volume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From AABB, shrunk by radius, no faces
		{
			hknpConvexShape::BuildConfig buildConfig;
			buildConfig.m_buildFaces = false;
			const hknpConvexShape* shape = hknpConvexShape::createFromHalfExtents( halfExtents, radius, buildConfig );
			validate( shape, aabb, volume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From AABB, expanded by radius
		{
			hknpConvexShape::BuildConfig buildConfig;
			buildConfig.m_shrinkByRadius = false;
			const hknpConvexShape* shape = hknpConvexShape::createFromHalfExtents( halfExtents, radius, buildConfig );
			validate( shape, expandedAabb, expandedVolume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From AABB, expanded by radius, no faces
		{
			hknpConvexShape::BuildConfig buildConfig;
			buildConfig.m_shrinkByRadius = false;
			buildConfig.m_buildFaces = false;
			const hknpConvexShape* shape = hknpConvexShape::createFromHalfExtents( halfExtents, radius, buildConfig );
			validate( shape, expandedAabb, expandedVolume, hkVector4::getZero() );
			shape->removeReference();
		}

		hkVector4 vertices[8];
		hkAabbUtil::get8Vertices( aabb, vertices );
		hkStridedVertices stridedVertices( vertices, 8 );

		// From vertices, zero radius
		{
			const hknpConvexShape* shape = hknpConvexShape::createFromVertices( stridedVertices, 0.0f );
			validate( shape, aabb, volume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From vertices, zero radius, no faces
		{
			hknpConvexShape::BuildConfig buildConfig;
			buildConfig.m_buildFaces = false;
			const hknpConvexShape* shape = hknpConvexShape::createFromVertices( stridedVertices, 0.0f, buildConfig );
			validate( shape, aabb, volume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From vertices, shrunk by radius
		{
			const hknpConvexShape* shape = hknpConvexShape::createFromVertices( stridedVertices, radius );
			validate( shape, aabb, volume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From vertices, shrunk by radius, no faces
		
		if(0)
		{
			hknpConvexShape::BuildConfig buildConfig;
			buildConfig.m_buildFaces = false;
			const hknpConvexShape* shape = hknpConvexShape::createFromVertices( stridedVertices, radius, buildConfig );
			validate( shape, aabb, volume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From vertices, expanded by radius
		{
			hknpConvexShape::BuildConfig buildConfig;
			buildConfig.m_shrinkByRadius = false;
			const hknpConvexShape* shape = hknpConvexShape::createFromVertices( stridedVertices, radius, buildConfig );
			validate( shape, expandedAabb, expandedVolume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From vertices, expanded by radius, no faces
		{
			hknpConvexShape::BuildConfig buildConfig;
			buildConfig.m_shrinkByRadius = false;
			buildConfig.m_buildFaces = false;
			const hknpConvexShape* shape = hknpConvexShape::createFromVertices( stridedVertices, radius, buildConfig );
			validate( shape, expandedAabb, expandedVolume, hkVector4::getZero() );
			shape->removeReference();
		}

		// From vertices, shrunk by radius with extra transform
		{
			hknpConvexShape::BuildConfig buildConfig;

			// Set a random transform.
			hkTransform transform;
			{
				rng.getRandomRotation(transform.getRotation());
				rng.getRandomVector11(transform.getTranslation());
				transform.getTranslation().mul(maxTransform);
			}
			buildConfig.m_extraTransform = &transform;

			const hknpConvexShape* shape = hknpConvexShape::createFromVertices( stridedVertices, radius, buildConfig );

			// Compute the new AABB.
			hkAabb transformedAabb;
			hkAabbUtil::calcAabb(transform, halfExtents, hkVector4::getZero(),transformedAabb);

			validate( shape, transformedAabb, volume, transform.getTranslation(), 0.13f);
			shape->removeReference();
		}

		// From AABB, shrunk by radius with extra transform
		{
			hknpConvexShape::BuildConfig buildConfig;

			// Set a random transform.
			hkTransform transform;
			{
				rng.getRandomRotation(transform.getRotation());
				rng.getRandomVector11(transform.getTranslation());
				transform.getTranslation().mul(maxTransform);
			}

			buildConfig.m_extraTransform = &transform;
			buildConfig.m_massConfig.m_inertiaFactor = 1.0f;
			const hknpConvexShape* shape = hknpConvexShape::createFromHalfExtents( halfExtents, radius, buildConfig );

			// Compute the new AABB.
			hkAabb transformedAabb;
			hkAabbUtil::calcAabb(transform,aabb,transformedAabb);

			validate( shape, transformedAabb, volume, transform.getTranslation(),0.13f);

			// Also test the inertia tensor
			{
				const hknpShapeMassProperties* props = (const hknpShapeMassProperties*)shape->getProperty( hknpShapePropertyKeys::MASS_PROPERTIES );
				hkMassProperties mp;
				props->m_compressedMassProperties.unpack(mp);

				// Apply the inverse transform to the tensor to see if it matches
				hkMatrix3 originalTensor;
				originalTensor.setMulInverseMul(transform.getRotation(),mp.m_inertiaTensor);
				originalTensor.mul(transform.getRotation());

				hkInertiaTensorComputer::computeBoxVolumeMassProperties(halfExtents, mp.m_mass, mp);

				HK_TEST( mp.m_inertiaTensor.isApproximatelyEqual(originalTensor,0.01f));
			}

			shape->removeReference();
		}

	}

	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER( ConvexShapeConstructionTest_main, "Fast", "Physics/Test/UnitTest/Physics/", __FILE__ );

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
