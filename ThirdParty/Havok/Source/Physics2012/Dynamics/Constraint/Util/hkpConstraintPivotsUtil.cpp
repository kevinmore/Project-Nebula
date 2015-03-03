/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintPivotsUtil.h>

#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics/Constraint/Data/Fixed/hkpFixedConstraintData.h>
#include <Physics/Constraint/Data/DeformableFixed/hkpDeformableFixedConstraintData.h>
#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/StiffSpring/hkpStiffSpringConstraintData.h>

#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Malleable/hkpMalleableConstraintData.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Filter/Null/hkpNullCollisionFilter.h>

// This is the function used for finding the hkpShapeKey attached to a constraint pivot and which should be optimized.
//                
hkpShapeKey HK_CALL hkpConstraintPivotsUtil::findClosestShapeKey(const hkpWorld* world, const hkpShape* shape, const hkVector4& pivotInBodySpace)
{
	// Create a temporary shape and hkpCdBody for the query.
	hkpSphereShape sphere(0.01f);
	hkTransform transform; 
	transform.setTranslation(pivotInBodySpace);
	transform.getRotation().setIdentity();
	hkpCdBody cdBodyA(&sphere, &transform);

	hkpCdBody cdBodyB(shape, &hkTransform::getIdentity());


	hkpClosestCdPointCollector collector;
	hkpCollisionDispatcher* dispatcher = world->getCollisionDispatcher();
	hkpCollisionDispatcher::GetClosestPointsFunc getClosestPointsFunc = dispatcher->getGetClosestPointsFunc(sphere.getType(), shape->getType());

	hkpCollisionInput input = *world->getCollisionInput();
	//if (maxDistanceToReport != HK_REAL_MAX)
	//{
	//	input.m_tolerance = maxDistanceToReport;
	//}
	input.m_tolerance = HK_REAL_MAX * 0.5f;
	hkpNullCollisionFilter filter;
	input.m_filter = &filter;

	getClosestPointsFunc(cdBodyA, cdBodyB, input, collector);

	if (collector.hasHit())
	{
		return collector.getHit().m_shapeKeyB;
	}
	else
	{
		return HK_NULL;
	}
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
