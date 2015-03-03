/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/World/hknpWorldShiftUtil.h>

#include <Physics/Physics/Dynamics/Body/hknpBodyManager.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>

#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.h>

#include <Physics/Constraint/Data/hkpConstraintData.h>
#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>
#include <Physics/Constraint/Data/CogWheel/hkpCogWheelConstraintData.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Data/PointToPlane/hkpPointToPlaneConstraintData.h>
#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>
#include <Physics/Constraint/Data/RackAndPinion/hkpRackAndPinionConstraintData.h>
#include <Physics/Constraint/Data/StiffSpring/hkpStiffSpringConstraintData.h>


#define DISPLAY_OLD_NEW_BOXES 0

static void shiftConstraint( hkpConstraintData* ctData, hkVector4Parameter offset, const hknpBodyId& bodyA, const hknpBodyId& bodyB );

void HK_CALL hknpWorldShiftUtil::shiftWorld( hknpWorld* world, hkVector4Parameter offset )
{
	HK_TIMER_BEGIN( "ShiftWorld", HK_NULL );

	world->checkNotInSimulation();
	world->commitAddBodies();	// commit all bodies so that we do not have to shift them with extra code.

	// Adjust broad phase
	{
		hkAabb aabb = world->m_intSpaceUtil.m_aabb;
		aabb.m_min.add( offset );
		aabb.m_max.add( offset );
		world->m_intSpaceUtil.set( aabb );
	}

	// Shift motions
	{
		hknpMotion* motions = world->m_motionManager.accessMotionBuffer();
		for( hknpMotionIterator it(world->m_motionManager); it.isValid(); it.next() )
		{
			hknpMotion& motion = motions[ it.getMotionId().value() ];
			hkVector4 com;
			com.setAdd( motion.getCenterOfMassInWorld(), offset );
			motion.setCenterOfMassInWorld( com );
		}
	}

	// Shift bodies
	{
		hknpBody* bodies = world->m_bodyManager.accessBodyBuffer();
		for( hknpBodyIterator it = world->getBodyIterator(); it.isValid(); it.next() )
		{
			// Get write access
			hknpBody& body = bodies[ it.getBodyId().value() ];
			hkTransform transform = body.getTransform();
			transform.getTranslation().add( offset );
			body.setTransform( transform );
		}
	}

	// Shift constraints
	{
		hknpConstraint** constraints = world->m_constraintAtomSolver->getConstraints();
		const int numConstraints = world->m_constraintAtomSolver->getNumConstraints();

		for (int i = 0; i < numConstraints; ++i)
		{
			hknpConstraint* constraint = constraints[i];
			if (constraint->m_bodyIdA.value() == 0 || constraint->m_bodyIdB.value() == 0)
			{
				shiftConstraint( const_cast<hkpConstraintData*>(constraint->m_data.val()), offset, constraint->m_bodyIdA, constraint->m_bodyIdB );
			}
		}
	}

	// Shift space splitter
	{
		world->m_spaceSplitter->onShiftWorld( offset );
	}

	// Shift external objects
	{
		world->m_signals.m_worldShifted.fire( world, offset );
	}

	HK_TIMER_END();
}


#define CASE_SHIFT_CONSTRAINT_V1(ENUM, TYPE)					\
case hkpConstraintData::ENUM:									\
{																\
	TYPE* data = (TYPE*) ctData;								\
	if ( bodyA.value() == 0 )									\
		data->m_atoms.m_pivots.m_translationA.add( offset );	\
	if ( bodyB.value() == 0 )									\
		data->m_atoms.m_pivots.m_translationB.add( offset );	\
	break;														\
}


#define CASE_SHIFT_CONSTRAINT_V2(ENUM, TYPE)					\
case hkpConstraintData::ENUM:									\
	{															\
	TYPE* data = (TYPE*) ctData;								\
	if ( bodyA.value() == 0 )									\
		data->m_atoms.m_transforms.m_transformA.getTranslation().add( offset );		\
	if ( bodyB.value() == 0 )									\
		data->m_atoms.m_transforms.m_transformB.getTranslation().add( offset );		\
	break;														\
}


static void shiftConstraint( hkpConstraintData* ctData, hkVector4Parameter offset, const hknpBodyId& bodyA, const hknpBodyId& bodyB )
{
	switch ( ctData->getType() )
	{
		CASE_SHIFT_CONSTRAINT_V1(CONSTRAINT_TYPE_BALLANDSOCKET, hkpBallAndSocketConstraintData);
		CASE_SHIFT_CONSTRAINT_V2(CONSTRAINT_TYPE_HINGE, hkpHingeConstraintData);
		CASE_SHIFT_CONSTRAINT_V2(CONSTRAINT_TYPE_COG_WHEEL, hkpCogWheelConstraintData);
		CASE_SHIFT_CONSTRAINT_V2(CONSTRAINT_TYPE_LIMITEDHINGE, hkpLimitedHingeConstraintData);
		CASE_SHIFT_CONSTRAINT_V2(CONSTRAINT_TYPE_POINTTOPLANE, hkpPointToPlaneConstraintData);
		CASE_SHIFT_CONSTRAINT_V2(CONSTRAINT_TYPE_PRISMATIC, hkpPrismaticConstraintData);
		CASE_SHIFT_CONSTRAINT_V2(CONSTRAINT_TYPE_RACK_AND_PINION, hkpRackAndPinionConstraintData);
		CASE_SHIFT_CONSTRAINT_V1(CONSTRAINT_TYPE_STIFFSPRING, hkpStiffSpringConstraintData);
	}
}


HK_FORCE_INLINE static void shiftBroadPhaseAabb( hkInt32 intShift[3], hkAabb16& aabb )
{
	for (int i = 0; i < 3; ++i)
	{
		aabb.m_min[i] = (hkUint16) hkMath::clamp<hkInt32>( hkInt32(aabb.m_min[i] + intShift[i]), (hkInt32)0, (hkInt32)hkIntSpaceUtil::AABB_UINT16_MAX_VALUE);
		aabb.m_max[i] = (hkUint16) hkMath::clamp<hkInt32>( hkInt32(aabb.m_max[i] + intShift[i]), (hkInt32)0, (hkInt32)hkIntSpaceUtil::AABB_UINT16_MAX_VALUE);
	}
}

HK_FORCE_INLINE static bool aabbIsOnBorder( const hkAabb16& aabb )
{
	for (int i = 0; i < 3; ++i)
	{
		if (aabb.m_min[i] == 0 || aabb.m_max[i] == hkIntSpaceUtil::AABB_UINT16_MAX_VALUE)
		{
			return true;
		}
	}
	return false;
}


#if 0
	#include <Common/Base/Types/Color/hkColor.h>
	#include <Common/Visualize/hkDebugDisplay.h>
#endif

void HK_CALL hknpWorldShiftUtil::shiftBroadPhase( hknpWorld* world, hkVector4Parameter requestedCenter, hkVector4& effectiveCenter, hkArray<hknpBodyId>* bodiesOutsideTheBroadPhase )
{
	HK_TIMER_BEGIN( "ShiftBroadPhase", HK_NULL );

	hkInt32 intShift[3];

#if DISPLAY_OLD_NEW_BOXES
	hkIntSpaceUtil oldIntSpaceUtil = world->m_intSpaceUtil;
#endif

	hkIntSpaceUtil& intSpaceUtil = world->m_intSpaceUtil;

	// Calculate the effective shift
	{
		hkVector4 currCenter; intSpaceUtil.m_aabb.getCenter( currCenter );

		hkVector4 reqShift; reqShift.setSub( currCenter, requestedCenter );

		for (int i = 0; i < 3; ++i)
		{
			intShift[i] = (int) (reqShift(i) * intSpaceUtil.m_bitScale(i));
			effectiveCenter(i) = currCenter(i) - (intShift[i] * intSpaceUtil.m_bitScaleInv(i));
		}
		effectiveCenter(3) = 0.0f;

		hkVector4 halfExtents; intSpaceUtil.m_aabb.getHalfExtents( halfExtents );

		hkAabb newAabb;
		newAabb.m_min.setSub( effectiveCenter, halfExtents);
		newAabb.m_max.setAdd( effectiveCenter, halfExtents);

		intSpaceUtil.set( newAabb );
	}

	// Shift body AABBs
	{
		hknpBodyManager& bodyManager = world->m_bodyManager;
		hknpBody* bodies = bodyManager.accessBodyBuffer();
		for( hknpBodyIterator it = bodyManager.getBodyIterator(); it.isValid(); it.next() )
		{
			// Get write access
			hknpBody& body = bodies[ it.getBodyId().value() ];

			if( !body.isAddedToWorld() )
			{
				continue;
			}

			// Copy the AABB
			HK_ALIGN16(hkAabb16 aabb);
			aabb = body.m_aabb;

			// If the body was intersecting the border of the broad phase we cannot just move it, we need to recalculate the AABB.
			if (aabbIsOnBorder(aabb))
			{
				if (body.isDynamic())
				{
					const hknpBodyQuality& quality = world->getBodyQualityLibrary()->getEntry( body.m_qualityId );
					hknpMotionUtil::calcSweptBodyAabb(&body,  world->getMotion(body.m_motionId), quality, world->m_collisionTolerance, world->m_solverInfo, world->m_intSpaceUtil );
					aabb = body.m_aabb;
				}
				else
				{
					hkAabb aabbF; hknpMotionUtil::calcStaticBodyAabb(body, world->m_collisionTolerance, &aabbF);
					world->m_intSpaceUtil.convertAabb( aabbF, aabb );
					body.m_aabb = aabb;
				}
			}
			else
			{
				shiftBroadPhaseAabb( intShift, aabb );
			}

			// If the previous aabb of a static body overlaps the border we need to treat is as a moved static body
			// so that the previousAabb gets reset after next step.
			if (body.isStatic() && aabbIsOnBorder(bodyManager.getPreviousAabbs()[body.m_id.value()]))
			{
				bodyManager.setScheduledBodyFlags( body.m_id, hknpBodyManager::MOVED_STATIC );
			}
			shiftBroadPhaseAabb( intShift, bodyManager.accessPreviousAabbs()[body.m_id.value()] );

			if ( bodiesOutsideTheBroadPhase != HK_NULL )
			{
				for (int i = 0; i < 3; ++i)
				{
					if ( (aabb.m_min[i] == aabb.m_max[i] )
						&& ( aabb.m_min[i] == 0 || aabb.m_min[i] == hkIntSpaceUtil::AABB_UINT16_MAX_VALUE ) )
					{
						bodiesOutsideTheBroadPhase->pushBack( body.m_id );
						break;
					}
				}
			}

#if DISPLAY_OLD_NEW_BOXES
			hkAabb oldBox; oldIntSpaceUtil.restoreAabb( body.getAabb(), oldBox );
			hkAabb newBox; intSpaceUtil.restoreAabb( aabb, newBox );

			hkVector4 v; v.set(0.1f, 0.1f, 0.1f);
			oldBox.m_min.sub( v );
			oldBox.m_max.add( v );

			HK_DISPLAY_BOUNDING_BOX(oldBox, hkColor::WHITE);
			HK_DISPLAY_BOUNDING_BOX(newBox, hkColor::BLACK);
#endif

			body.m_aabb = aabb;
		}
	}

	// Update the broad phase
	{
		world->m_broadPhase->update( world->m_bodyManager.accessBodyBuffer(), hknpBroadPhase::UPDATE_ALL );

#if 0
		HK_ALIGN16(hkAabb16 aabb16); world->m_broadPhase->getExtents( aabb16 );
		hkAabb aabb;	intSpaceUtil.restoreAabb( aabb16, aabb );

		HK_DISPLAY_BOUNDING_BOX( aabb, hkColor::RED );
		HK_DISPLAY_BOUNDING_BOX( intSpaceUtil.m_aabb, hkColor::RED );

#endif
	}

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
