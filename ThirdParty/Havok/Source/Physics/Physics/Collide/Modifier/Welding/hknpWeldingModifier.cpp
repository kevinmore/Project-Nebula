/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Modifier/Welding/hknpWeldingModifier.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Geometry/Internal/Algorithms/Welding/hkcdWeldingUtil.h>
#include <Geometry/Internal/Algorithms/Welding/hkcdWeldingUtil.inl>
#include <Geometry/Internal/Algorithms/Gsk/hkcdGsk.h>

#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>
#include <Physics/Physics/Collide/hknpCollideSharedData.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>
#include <Physics/Physics/Dynamics/World/Commands/hknpInternalCommands.h>
#include <Physics/Internal/Collide/NarrowPhase/Welding/hknpWeldingUtil.h>


namespace
{
	static const hkUint8 LOOKUP_EDGE_BITS_FROM_SINGLE_VERTEX[2][4] =
		{{5,3,6,0}, {9,3,6,12}}; // First if triangle, second if quad

	static const hkUint8 LOOKUP_EDGE_BITS_FROM_TWO_VERTICES[2][4][4] =
		{{{0,1,4,0},{1,0,2,0},{4,2,0,0},{0,0,0,0}},		// If triangle
		 {{0,1,0,8},{1,0,2,0},{0,2,0,4},{8,0,4,0}}};	// If quad

	HK_DISABLE_OPTIMIZATION_VS2008_X64
	hkResult setupMotionWeldConfigBodyA(
		const hknpSolverInfo* HK_RESTRICT solverInfo, const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
		hkSimdRealParameter accuracy, hkLocalBuffer<hkcdVertex>* vertexBufferA,
		hkcdWeldingUtil::MotionWeldConfig &configOut )
	{
		configOut.m_transformA = *cdBodyA.m_transform;
		configOut.m_allowedApproachingVelocity = hkSimdReal_Inv_15 * solverInfo->m_unitScale;

		{
			hkVector4 relVel; relVel.setSub(cdBodyA.m_motion->m_linearVelocity, cdBodyB.m_motion->m_linearVelocity);
			hkSimdReal relVelLen2 = relVel.lengthSquared<3>();

			if ( relVelLen2 < configOut.m_allowedApproachingVelocity * configOut.m_allowedApproachingVelocity )	// if we are too slow, forget it, just wasting CPU cycles.
			{
				return HK_FAILURE;
			}
			// note the next lines are written to allow for maximum parallel execution of floating point instructions.
			hkSimdReal relVelLenInv = relVelLen2.sqrtInverse<HK_ACC_23_BIT, HK_SQRT_IGNORE>();
			hkVector4 relVelA; relVelA._setRotatedInverseDir(configOut.m_transformA.getRotation(),relVel);
			configOut.m_velocityDirection. setMul( relVel,  relVelLenInv );
			configOut.m_velocityDirectionA.setMul( relVelA, relVelLenInv );
			configOut.m_relLinearVelocity = relVelLen2 * relVelLenInv;
		}

#if defined(HK_PLATFORM_HAS_SPU)
		configOut.m_shapeA = cdBodyA.m_leafShape;
#else
		int numVerticesA = cdBodyA.m_leafShape->getNumberOfSupportVertices();
		const hkcdVertex* verticesA = cdBodyA.m_leafShape->getSupportVertices( vertexBufferA->begin(), numVerticesA );
		{
			configOut.m_verticesA	  = verticesA;
			configOut.m_numVerticesA = numVerticesA;
		}
#endif

		// we are adding the object radius just 2 be 100% sure to get the right normal
		{
			hkSimdReal maxAllowedTotalCastDistance = configOut.m_relLinearVelocity * solverInfo->m_deltaTime + accuracy;
			hkcdVertex best0, best1;
			hkcdVertex negDirectionA; negDirectionA.setNeg<4>(configOut.m_velocityDirectionA);
#if defined(HK_PLATFORM_HAS_SPU)
			cdBodyA.m_leafShape->getSupportingVertex( configOut.m_velocityDirectionA, &best0 );
			cdBodyA.m_leafShape->getSupportingVertex( negDirectionA, &best1 );
#else
			hknpConvexShapeUtil::getSupportingVertices2(
				verticesA, numVerticesA, verticesA, numVerticesA,
				configOut.m_velocityDirectionA, negDirectionA, &best0, &best1 );
#endif
			hkVector4 diff; diff.setSub( best0, best1 );
			hkSimdReal objectDiameter = diff.dot<3>( configOut.m_velocityDirectionA );
			configOut.m_maxAllowedTotalCastDistance = maxAllowedTotalCastDistance + hkSimdReal_Inv2* objectDiameter + hkSimdReal::fromFloat(cdBodyA.m_leafShape->m_convexRadius);
		}
		return HK_SUCCESS;
	}
	HK_RESTORE_OPTIMIZATION_VS2008_X64

	// An experimental simple motion method. Uses the triangle normal for welding.
	static HK_FORCE_INLINE void _triNormalWeld(
		const hknpSimulationThreadContext& tl,
		const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
		hknpManifold* HK_RESTRICT manifolds, int numManifolds,
		hkSimdRealParameter accuracy, hkSimdRealParameter deltaTime)
	{
		const hknpShape* compShape = cdBodyB.m_rootShape;

		hkVector4 relVel;
		relVel.setSub(cdBodyA.m_motion->m_linearVelocity, cdBodyB.m_motion->m_linearVelocity);

		hknpShapeCollector leafShapeCollector( tl.m_triangleShapePrototypes[0] );

		for (int i = 0; i < numManifolds; i++)
		{
			hknpManifold& manifold = manifolds[i];

			// Only weld if we are moving towards the triangle.
			if (relVel.dot<3>(manifold.m_normal).isLess(-hkSimdReal_Inv_255))
			{
				hkVector4 triangleNormal;
				{
					hknpShapeKey childKey = manifold.m_collisionCache->getShapeKey();
					leafShapeCollector.reset( cdBodyB.m_body->getTransform() );

					compShape->getLeafShape( childKey, &leafShapeCollector );

					const hknpConvexPolytopeShape* polytopeShape = leafShapeCollector.m_shapeOut->asConvexPolytopeShape();
					if (!polytopeShape || polytopeShape->getNumberOfVertices()>4)
					{
						// Welding only works on triangles.
						return;
					}
					triangleNormal._setRotatedDir( leafShapeCollector.m_transformOut.getRotation(), polytopeShape->getPlane(0) );
				}

				// flip sign of triangle normal, we should not do this as we assume oriented triangles
				// if ( triangleNormal.dot<3>(manifold.m_normal).isLessEqualZero())
				// {
				// 		triangleNormal.setNeg<4>(triangleNormal);
				// 	}

				hkSimdReal d = relVel.dot<3>(triangleNormal);

				// Only weld if the angle between velocity and triangle plane is less than 45 degrees.
				if( (d*d).isLessEqual (relVel.lengthSquared<3>() * hkSimdReal_Inv2) )
				{
					hkSimdReal radiusA; radiusA.load<1>( &cdBodyA.m_leafShape->m_convexRadius );
					hkcdWeldingUtil::_applyModifiedNormal(triangleNormal, radiusA, &manifold );
				}
			}
		}
	}

	// A welding method which looks at neighboring triangles
	static HK_FORCE_INLINE void _neighborWeld(
		const hknpSimulationThreadContext& tl, const hknpSolverInfo* HK_RESTRICT solverInfo,
		const hknpWeldingModifier::WeldingInfo& wInfo, const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
		hknpManifold* HK_RESTRICT manifolds, int numManifolds, hkSimdRealParameter accuracy )
	{
		HK_ASSERT(0xf034df1c, numManifolds <= HKNP_MAX_NUM_MANIFOLDS_PER_BATCH);

		const hknpShape* HK_RESTRICT compositeShape = cdBodyB.m_body->m_shape;
		hkLocalBuffer<hknpWeldingUtil::ManifoldData> manifoldsData(HKNP_MAX_NUM_MANIFOLDS_PER_BATCH, "hkcdWeldingUtil::ManifoldData");
		hknpShapeCollector leafShapeCollector(tl.m_triangleShapePrototypes[0]);

		const bool allowConvexWeld = false;
		bool someChildShapesAreRealCvx = false;

		// Compute manifolds data
		for (int i = 0; i < numManifolds; i++)
		{
			const hknpManifold& manifold = manifolds[i];
			hknpWeldingUtil::ManifoldData* HK_RESTRICT manifoldData = &manifoldsData[i];

			// Initialize from manifold
			manifoldData->m_normal = manifold.m_normal;
			manifoldData->m_gskPosition = manifold.m_gskPosition;
			int dimB = manifold.m_collisionCache->m_gskCache.getDimB();
			manifoldData->m_dimBInverse = (hkUint8)(dimB ^ 0x3);

			// Obtain contacted triangle normal
			{
				// Obtain contacted triangle
				hknpShapeKey childKey = manifold.m_collisionCache->getShapeKey();
				leafShapeCollector.reset( cdBodyB.m_body->getTransform() );
				compositeShape->getLeafShape(childKey, &leafShapeCollector);
				const hknpConvexPolytopeShape* polytopeShape = leafShapeCollector.m_shapeOut->asConvexPolytopeShape();
				if (!polytopeShape || polytopeShape->getNumberOfVertices()>4)
				{
					if ( allowConvexWeld )
					{
						hkVector4 planeBinB;
						int minAngleB;
						hkVector4 nearestInB; nearestInB.setTransformedInversePos( leafShapeCollector.m_transformOut, manifoldData->m_gskPosition);
						hkUint32 prevFaceIdB = (hkUint32) -1;
						leafShapeCollector.m_shapeOut->getSupportingFace(nearestInB, &manifold.m_collisionCache->m_gskCache, true, planeBinB, minAngleB, prevFaceIdB);
						manifoldData->m_triangleNormal._setRotatedDir( leafShapeCollector.m_transformOut.getRotation(), planeBinB );
						someChildShapesAreRealCvx = true;
					}
					else
					{
						manifoldData->m_triangleNormal.setZero();
					}
					manifoldData->m_isTriangleOrQuad = false;
				}
				else
				{
					// Obtain normal in world space
					manifoldData->m_triangleNormal._setRotatedDir( leafShapeCollector.m_transformOut.getRotation(), polytopeShape->getPlane(0) );
					manifoldData->m_isTriangleOrQuad = true;

					// Consider edge welding...
					int edgeWeldingInfo = manifold.m_collisionCache->m_edgeWeldingInfo;
					if ( edgeWeldingInfo && dimB<3)
					{
						int isQuad = (polytopeShape->getVertex(3).getInt16W()==3);
						if (dimB == 2)
						{
							const hkcdGsk::Cache::VertexId* vertices = manifold.m_collisionCache->m_gskCache.getVertexIdsB();
							if (LOOKUP_EDGE_BITS_FROM_TWO_VERTICES[isQuad][vertices[0]][vertices[1]] & edgeWeldingInfo)
							{
								manifoldData->m_normal = manifoldData->m_triangleNormal;
							}
						}
						else //(dimB == 1)
						{
							const hkcdGsk::Cache::VertexId* vertices = manifold.m_collisionCache->m_gskCache.getVertexIdsB();
							if (LOOKUP_EDGE_BITS_FROM_SINGLE_VERTEX[isQuad][vertices[0]] & edgeWeldingInfo)
							{
								manifoldData->m_normal = manifoldData->m_triangleNormal;
							}
						}
					}
				}
			}

			// If the triangle normal is very close to the gsk normal we consider it a 3-manifold for welding purposes and set it to be the triangle normal
			hkSimdReal normalDotTriNormal = manifoldData->m_triangleNormal.dot<3>(manifoldData->m_normal);
			if (normalDotTriNormal.isGreater(hkSimdReal_1-hkSimdReal_Inv_255))
			{
				manifoldData->m_dimBInverse = 0;
				manifoldData->m_normal = manifoldData->m_triangleNormal;
			}
			else if (manifoldData->m_dimBInverse==0 && normalDotTriNormal.isLess(hkSimdReal_0))
			{
				manifoldData->m_dimBInverse = 3;
			}

			hkSimdReal distance = manifold.m_gskPosition.getW();

			// Modify distance slightly based on how close the normal is to the triangle normal.
			// The manifolds with GSK normals pointing in the triangle normal config.m_velocityDirection are better
			// candidates for welding so we will report smaller distances. The distance is used when
			// sorting them in hkcdWeldingUtil::weld.
			distance = distance - normalDotTriNormal * (accuracy*hkSimdReal_Inv7);
			distance.store<1>(&manifoldData->m_distance.m_asFloat);
		}

		hkLocalBuffer<hkcdVertex> vertexBufferA( 256 );

		hkBool32 useMotionWeld = int(someChildShapesAreRealCvx) | wInfo.m_qualityFlags.anyIsSet( hknpBodyQuality::ENABLE_MOTION_WELDING);
		hkcdWeldingUtil::MotionWeldConfig motionWeldConfig;
		if ( useMotionWeld )
		{
			hkResult result = setupMotionWeldConfigBodyA(
				solverInfo, cdBodyA, cdBodyB, accuracy, &vertexBufferA, motionWeldConfig);
			if ( result == HK_FAILURE )
			{
				useMotionWeld = false;
			}
		}


		// Compute welded contact normals
		hkcdWeldingUtil::Config config(cdBodyA.m_leafShape->m_convexRadius);
		accuracy.store<1>( &(HK_PADSPU_REF(config.m_tolerance)) );

		if (someChildShapesAreRealCvx && allowConvexWeld)
		{
			hknpWeldingUtil::neighborWeldConvex(
				config, cdBodyB, manifolds, manifoldsData.begin(), sizeof(hknpManifold), numManifolds, &leafShapeCollector );
		}
		else
		{
			hknpWeldingUtil::neighborWeld(
				manifolds, manifoldsData.begin(), sizeof(hknpManifold), numManifolds, cdBodyA, cdBodyB, &leafShapeCollector, config,
				((useMotionWeld) ? &motionWeldConfig : HK_NULL) );
		}
	}

}	// anonymous namespace


void hknpTriangleWeldingModifier::postMeshCollideCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const WeldingInfo& wInfo,
	const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
	hknpManifold* HK_RESTRICT manifolds, int numManifolds
	)
{
	HK_TIME_CODE_BLOCK( "TriangleWelding", HK_NULL );

	const hknpMaterial& materialA = *cdBodyA.m_material;
	const hknpMaterial& materialB = *cdBodyB.m_material;
	const hknpMaterial& bodyMaterialB = tl.m_materials[cdBodyB.m_body->m_materialId.value()];

	hkSimdReal accuracyA; accuracyA.setFromHalf( materialA.m_weldingTolerance );
	hkSimdReal accuracyB; accuracyB.setFromHalf( materialB.m_weldingTolerance );
	hkSimdReal accuracyBB; accuracyBB.setFromHalf( bodyMaterialB.m_weldingTolerance );

	hkSimdReal accuracy;
	accuracy.setMax(accuracyA, accuracyB );
	accuracy.setMax( accuracy, accuracyBB );

	hkSimdRealParameter deltaTime = sharedData.m_solverInfo->m_deltaTime;
	_triNormalWeld( tl, cdBodyA, cdBodyB, manifolds, numManifolds, accuracy, deltaTime );
}

void hknpNeighborWeldingModifier::postMeshCollideCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const WeldingInfo& wInfo,
	const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
	hknpManifold* HK_RESTRICT manifolds, int numManifolds
	)
{
	HK_TIME_CODE_BLOCK( "NeighborWelding", HK_NULL );

	const hknpMaterial& materialA = *cdBodyA.m_material;
	const hknpMaterial& materialB = *cdBodyB.m_material;
	const hknpMaterial& bodyMaterialB = tl.m_materials[cdBodyB.m_body->m_materialId.value()];

	hkSimdReal accuracyA; accuracyA.setFromHalf( materialA.m_weldingTolerance );
	hkSimdReal accuracyB; accuracyB.setFromHalf( materialB.m_weldingTolerance );
	hkSimdReal accuracyBB; accuracyBB.setFromHalf( bodyMaterialB.m_weldingTolerance );

	hkSimdReal accuracy;
	accuracy.setMax( accuracyA, accuracyB );
	accuracy.setMax( accuracy, accuracyBB );

	_neighborWeld( tl, sharedData.m_solverInfo, wInfo, cdBodyA, cdBodyB, manifolds, numManifolds, accuracy );
}

hkcdGsk::GetClosestPointStatus HK_CALL hknpMotionWeldingModifier_getClosestPoint(
	const void* shapeA, const hkcdVertex* vertsA, int numVertsA,
	const void* shapeB, const hkcdVertex* vertsB, int numVertsB,
	const hkcdGsk::GetClosestPointInput& input, hkcdGsk::Cache* HK_RESTRICT cache, hkcdGsk::GetClosestPointOutput& output )
{
	extern hkcdGsk::GetClosestPointStatus hknpConvexConvexShapeBaseInterfaceGetClosestPoint(
		const void* shapeA, const void* shapeB, const hkcdGsk::GetClosestPointInput& input, hkcdGsk::Cache* HK_RESTRICT cache,
		hkcdGsk::GetClosestPointOutput& output );

	return hknpConvexConvexShapeBaseInterfaceGetClosestPoint( shapeA, shapeB, input, cache, output );
}

void hknpMotionWeldingModifier::postMeshCollideCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const WeldingInfo& wInfo,
	const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
	hknpManifold* HK_RESTRICT manifolds, int numManifolds
	)
{
	/*
	Tolerances used:
		- allowedPenetration:	This indicates how much we could allow a body to penetrate
			without risking a really bad normal. Basically the system will on purpose let
			the object penetrate by this amount to get a better normal.

	Algorithm:
		-	if currentDistance <= -allowedPenetration, do nothing		// already a bad case
		-	if currentDistance < allowedPenetration						// normal case
				calculate normal by moving object forward by currentDistance+allowedPenetration, accept point
		-   else (currentDistance>allowedPenetration)
				conservativeAdvancement: produces hitNormal or normalOfLargestMovement	(normal orthogonal to config.m_velocityDirection)
					generate ghost plane (no friction, forces set to 0, only last solver iterations)
	*/

	HK_TIME_CODE_BLOCK( "MotionWelding", HK_NULL );

	const hknpMaterial& materialA = *cdBodyA.m_material;
	const hknpMaterial& materialB = *cdBodyB.m_material;
	const hknpMaterial& bodyMaterialB = tl.m_materials[cdBodyB.m_body->m_materialId.value()];

	hkSimdReal accuracyA; accuracyA.setFromHalf( materialA.m_weldingTolerance );
	hkSimdReal accuracyB; accuracyB.setFromHalf( materialB.m_weldingTolerance );
	hkSimdReal accuracyBB; accuracyBB.setFromHalf( bodyMaterialB.m_weldingTolerance );

	hkSimdReal accuracy;
	accuracy.setMax( accuracyA, accuracyB );
	accuracy.setMax( accuracy, accuracyBB );

	HK_ASSERT( 0xf03456de, numManifolds > 0 );

	//
	// Extract verticesA
	//
#if defined(HK_PLATFORM_HAS_SPU)
	hkLocalBuffer<hkcdVertex>* vertexBufferAPtr = HK_NULL; // Vertex buffer is not used on SPU
#else
	hkLocalBuffer<hkcdVertex> vertexBufferA( 256 );
	hkLocalBuffer<hkcdVertex>* vertexBufferAPtr = &vertexBufferA;
#endif

	hkcdWeldingUtil::MotionWeldConfig config;
	hkResult configResult;

	configResult = setupMotionWeldConfigBodyA(sharedData.m_solverInfo, cdBodyA, cdBodyB, accuracy, vertexBufferAPtr, config);
	if( configResult == HK_SUCCESS )
	{
#if defined(HK_PLATFORM_HAS_SPU)
		hkcdVertex* vertexBufferBPtr = HK_NULL; // Vertex buffer is not used on SPU
#else
		hkLocalBuffer<hkcdVertex> vertexBufferB( 256 );
		hkcdVertex* vertexBufferBPtr = vertexBufferB.begin();
#endif
		hknpShapeCollector leafShapeCollector( tl.m_triangleShapePrototypes[1] );

		for (int i =0; i < numManifolds; i++)
		{
			hknpManifold* HK_RESTRICT manifold = &manifolds[i];
			hkSimdReal unweldedDistance = manifold->m_distances.horizontalMin<4>();

			configResult = hknpWeldingUtil::setupMotionWeldConfigBodyB( cdBodyA, cdBodyB, *manifold, accuracy, &config, vertexBufferBPtr, &leafShapeCollector );
			if ( configResult == HK_SUCCESS )
			{
				hkVector4 weldedNormal = manifold->m_normal;
#if defined(HK_PLATFORM_HAS_SPU)
				hkResult result = hkcdWeldingUtil::motionWeldCvxVsCvx( &hknpMotionWeldingModifier_getClosestPoint, config, manifold->m_collisionCache->m_gskCache, &weldedNormal);
#else
				hkResult result = hkcdWeldingUtil::motionWeldCvxVsCvx( config, manifold->m_collisionCache->m_gskCache, &weldedNormal);
#endif
				if ( result == HK_SUCCESS )
				{
					hkSimdReal radiusA; radiusA.load<1>( &cdBodyA.m_rootShape->m_convexRadius );
					hkcdWeldingUtil::_applyModifiedNormal(weldedNormal, radiusA, manifold );
				}
			}

			// check for cast distance even if we have not welded the normal
			if ( unweldedDistance > accuracy * hkSimdReal_Inv2)
			{
				// set the maxForce to 0, disable friction and create TOI Jacobians
				manifold->m_manifoldType = hknpManifold::TYPE_TOI;
			}
		}
	}
}

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
