/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQueryDispatcher.h>

#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>
#include <Physics/Physics/Collide/Shape/hknpShapeQueryInterface.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>


namespace NpShapeCastTest
{

	hknpShape* createSphereShape(hkReal radius)
	{
		hknpShape* shape = hknpSphereShape::createSphereShape(hkVector4::getZero(), radius);
		return shape;
	}

	hknpShape* createConvexShape(hkPseudoRandomGenerator* rng, hkReal scaleIn = 0.25f)
	{
		hkArray<hkVector4>vertices;
		vertices.setSize(64);

		hkSimdReal scale;
		scale.setFromFloat(scaleIn);

		for (int i = 0; i < vertices.getSize(); i++)
		{
			rng->getRandomVector11(vertices[i]);
			vertices[i].mul(scale);
		}

		hknpShape* shape = hknpConvexShape::createFromVertices(vertices);

		return shape;
	}

	void distanceToHitTest_normalizedDirectionVsDistanceVector(const hknpCollisionQueryDispatcher &collisionQueryDispatcher)
	{
		//hkPseudoRandomGenerator rng(12041973);

		hknpShape* queryShape = createSphereShape(0.5f);
		hknpShape* targetShape = createSphereShape(0.5f);

		hkVector4 queryShapePosition;
		queryShapePosition.set(-5.5f,0,0);
		hkTransform queryShapeTransform;
		queryShapeTransform.setRotation(hkQuaternion::getIdentity());
		queryShapeTransform.setTranslation(queryShapePosition);

		hkVector4 targetShapePosition;
		targetShapePosition.set(+5.5f,0,0);
		hkTransform targetShapeTransform;
		targetShapeTransform.setRotation(hkQuaternion::getIdentity());
		targetShapeTransform.setTranslation(targetShapePosition);

		hkSimdReal castDistance;
		castDistance.setFromFloat(20.0f);

		bool hasHit_normalizedDirectionCast = false;
		hkSimdReal distanceToHit_normalizedDirectionCast;	distanceToHit_normalizedDirectionCast.setZero();
		{
			hkVector4 castStartingPositionInTargetSpace;
			castStartingPositionInTargetSpace._setTransformedInversePos(targetShapeTransform, queryShapeTransform.getTranslation());
			const hkVector4 castDirectionNormalized = hkVector4::getConstant<HK_QUADREAL_1000>();
			hkVector4 castDirectionNormalizedInTargetSpace;
			castDirectionNormalizedInTargetSpace._setRotatedInverseDir(targetShapeTransform.getRotation(), castDirectionNormalized);

			hknpCollisionQueryContext queryContext;
			queryContext.m_dispatcher = &collisionQueryDispatcher;
			hknpShapeCastQuery query(*queryShape, castStartingPositionInTargetSpace, castDirectionNormalizedInTargetSpace, castDistance);
			hknpAllHitsCollector collector;
			hknpShapeQueryInterface::castShape(&queryContext, query, queryShapeTransform.getRotation(), *targetShape, targetShapeTransform, &collector);

			hasHit_normalizedDirectionCast = collector.hasHit();

			if (hasHit_normalizedDirectionCast)
			{
				hkVector4 distanceVector = castDirectionNormalized;
				hkSimdReal fraction = hkSimdReal::fromFloat(collector.getHits()[0].m_fraction);
				distanceVector.mul(fraction);
				distanceToHit_normalizedDirectionCast = distanceVector.length<3>();
			}
		}

		bool hasHit_distanceVectorCast = false;
		hkSimdReal distanceToHit_distanceVectorCast;	distanceToHit_distanceVectorCast.setZero();
		{
			hkVector4 castStartingPositionInTargetSpace;
			castStartingPositionInTargetSpace._setTransformedInversePos(targetShapeTransform, queryShapeTransform.getTranslation());
			hkVector4 castDistanceVector = hkVector4::getConstant<HK_QUADREAL_1000>();
			castDistanceVector.mul(castDistance);
			hkVector4 castDistanceVectorInTargetSpace;
			castDistanceVectorInTargetSpace._setRotatedInverseDir(targetShapeTransform.getRotation(), castDistanceVector);

			hknpCollisionQueryContext queryContext;
			queryContext.m_dispatcher = &collisionQueryDispatcher;
			hknpShapeCastQuery query(*queryShape, castStartingPositionInTargetSpace, castDistanceVectorInTargetSpace, hkSimdReal_1);
			hknpAllHitsCollector collector;
			hknpShapeQueryInterface::castShape(&queryContext, query, queryShapeTransform.getRotation(), *targetShape, targetShapeTransform, &collector);

			hasHit_distanceVectorCast = collector.hasHit();

			if (hasHit_distanceVectorCast)
			{
				hkVector4 distanceVector = castDistanceVector;
				hkSimdReal fraction = hkSimdReal::fromFloat(collector.getHits()[0].m_fraction);
				distanceVector.mul(fraction);
				distanceToHit_distanceVectorCast = distanceVector.length<3>();
			}
		}

		HK_TEST(hasHit_normalizedDirectionCast == hasHit_distanceVectorCast);

		if (hasHit_normalizedDirectionCast == hasHit_distanceVectorCast)
		{
			HK_TEST(distanceToHit_normalizedDirectionCast.approxEqual(distanceToHit_distanceVectorCast, hkSimdReal::fromFloat(0.001f)));
		}

		targetShape->removeReference();
		queryShape->removeReference();
	}


	void zeroLengthCastValidFractionTest(const hknpCollisionQueryDispatcher &collisionQueryDispatcher)
	{
		hkPseudoRandomGenerator rng(12041973);

		hknpShape* queryShape = createSphereShape(0.5f);
		hknpShape* targetShape = createSphereShape(0.5f);

		hkVector4 queryShapePosition;
		queryShapePosition.set(-0.5f,0,0);
		hkTransform queryShapeTransform;
		queryShapeTransform.setRotation(hkQuaternion::getIdentity());
		queryShapeTransform.setTranslation(queryShapePosition);

		hkVector4 targetShapePosition;
		targetShapePosition.set(+0.5f,0,0);
		hkTransform targetShapeTransform;
		targetShapeTransform.setRotation(hkQuaternion::getIdentity());
		targetShapeTransform.setTranslation(targetShapePosition);

		hkSimdReal castDistance;
		castDistance.setFromFloat(0.0f);

		{
			hkVector4 castStartingPositionInTargetSpace;
			castStartingPositionInTargetSpace._setTransformedInversePos(targetShapeTransform, queryShapeTransform.getTranslation());
			const hkVector4 castDirectionNormalized = hkVector4::getConstant<HK_QUADREAL_1000>();
			hkVector4 castDirectionNormalizedInTargetSpace;
			castDirectionNormalizedInTargetSpace._setRotatedInverseDir(targetShapeTransform.getRotation(), castDirectionNormalized);

			hknpCollisionQueryContext queryContext;
			queryContext.m_dispatcher = &collisionQueryDispatcher;
			hknpShapeCastQuery query(*queryShape, castStartingPositionInTargetSpace, castDirectionNormalizedInTargetSpace, castDistance);
			hknpAllHitsCollector collector;
			hknpShapeQueryInterface::castShape(&queryContext, query, queryShapeTransform.getRotation(), *targetShape, targetShapeTransform, &collector);

			if (collector.hasHit())
			{
				hkSimdReal fraction = hkSimdReal::fromFloat(collector.getHits()[0].m_fraction);
				HK_TEST(fraction.isOk());
				if (fraction.isOk())
				{
					HK_TEST(fraction.approxEqual(hkSimdReal_0, hkSimdReal_Eps));
				}
			}
		}

		{
			hkVector4 castStartingPositionInTargetSpace;
			castStartingPositionInTargetSpace._setTransformedInversePos(targetShapeTransform, queryShapeTransform.getTranslation());
			hkVector4 castDistanceVector = hkVector4::getConstant<HK_QUADREAL_1000>();
			castDistanceVector.mul(castDistance);
			hkVector4 castDistanceVectorInTargetSpace;
			castDistanceVectorInTargetSpace._setRotatedInverseDir(targetShapeTransform.getRotation(), castDistanceVector);

			hknpCollisionQueryContext queryContext;
			queryContext.m_dispatcher = &collisionQueryDispatcher;
			hknpShapeCastQuery query(*queryShape, castStartingPositionInTargetSpace, castDistanceVectorInTargetSpace, hkSimdReal_1);
			hknpAllHitsCollector collector;
			hknpShapeQueryInterface::castShape(&queryContext, query, queryShapeTransform.getRotation(), *targetShape, targetShapeTransform, &collector);

			if (collector.hasHit())
			{
				hkSimdReal fraction = hkSimdReal::fromFloat(collector.getHits()[0].m_fraction);
				HK_TEST(fraction.isOk());
				if (fraction.isOk())
				{
					HK_TEST(fraction.approxEqual(hkSimdReal_0, hkSimdReal_Eps));
				}
			}
		}

		targetShape->removeReference();
		queryShape->removeReference();
	}
}


int NpShapeCastTest_main()
{
	hknpCollisionQueryDispatcher collisionQueryDispatcher;
	NpShapeCastTest::distanceToHitTest_normalizedDirectionVsDistanceVector(collisionQueryDispatcher);
	NpShapeCastTest::zeroLengthCastValidFractionTest(collisionQueryDispatcher);

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(NpShapeCastTest_main, "Fast", "Physics/Test/UnitTest/Physics/", __FILE__);

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
