/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpBoxMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpFixedRigidMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpKeyframedRigidMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/ThinBoxMotion/hkpThinBoxMotion.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Physics/ConstraintSolver/Solve/hkpSolverInfo.h>

	// simply calculates the velocity of the object
static HK_FORCE_INLINE hkSimdReal calcVelocityForDeactivation( const hkpMotion& motion, hkSimdRealParameter objectRadius)
{
	const hkSimdReal linVelSqrd = motion.m_linearVelocity.lengthSquared<3>();
	const hkSimdReal angVelSqrd = motion.m_angularVelocity.lengthSquared<3>();
	hkSimdReal vel = objectRadius * objectRadius * angVelSqrd + linVelSqrd;
	return vel;
}



	// returns the number of inactive frames
	// Every 4th/16th frame it checks the transform of the motion against the reference positions
	// See hkpWorldCinfo::m_deactivationReferenceDistance

HK_FORCE_INLINE int hkRigidMotionUtilCheckDeactivation( const struct hkpSolverInfo& si, hkpMotion& motion )
{
	hkCheckDeterminismUtil::checkMt(0xf0000030, motion.getNumInactiveFrames(0));
	hkCheckDeterminismUtil::checkMt(0xf0000031, motion.getNumInactiveFrames(1));

	hkUint32 c = motion.m_deactivationIntegrateCounter;
	hkCheckDeterminismUtil::checkMt(0xf0000032, c);
	c++;
	motion.m_deactivationIntegrateCounter = hkUchar(c);
	if ( (c&3) == 0 )
	{
		// select high or low frequency check
		int select;
		if ( (c&15)!=0)
		{
			// high frequency checks
			select = 0;
		}
		else
		{
				// check for no deactivation
			if ( c == 0x100 )
			{
				motion.m_deactivationIntegrateCounter = 0xff;
				goto END_OF_FUNCTION;
			}
				// select low frequency checks
			motion.m_deactivationIntegrateCounter = 0;
			select = 1;
		}

		// we have to clip the radius. The reason is jitter because we store the quaternion in 32 bit
		// as a result we get 4% jitter. Without clipping the radius, the jitter could be enough to keep the object alive
		hkReal     radius          = hkMath::min2( 1.0f, motion.m_motionState.m_objectRadius);
		HK_ASSERT2(0x5c457c12,  radius > 0, "Radius was not set correctly for entity ");

		hkSimdReal velSqrd = calcVelocityForDeactivation( motion,hkSimdReal::fromFloat(radius) );
		{
			// we remember our maximum velocity. This is used to as a reference velocity for the final
			// deactivation
			hkSimdReal maxD; maxD.setMax(motion.m_deactivationRefPosition[select].getW(), velSqrd);
			motion.m_deactivationRefPosition[select].setW(maxD);
		}
		hkCheckDeterminismUtil::checkMt(0xf0000033, velSqrd.getReal());


		hkVector4& refPosition    = motion.m_deactivationRefPosition[select];
		hkCheckDeterminismUtil::checkMt(0xf0000034, refPosition);
		hkUint32&  refOrientation = motion.m_deactivationRefOrientation[select];


		const hkSweptTransform& sweptTransform = motion.m_motionState.getSweptTransform();

		const hkVector4&    currentPosition = sweptTransform.m_centerOfMass1;
		const hkQuaternion& currentRotation = sweptTransform.m_rotation1;

		while(1)	// dummy loop to improve code layout
		{
			const hkpSolverInfo::DeactivationInfo& di = si.m_deactivationInfo[motion.m_motionState.m_deactivationClass];
				// distance check
			{
				const hkSimdReal maxSqrd = hkSimdReal::fromFloat(di.m_maxDistSqrd[select]);

				hkVector4 transDist;	transDist.setSub( refPosition, currentPosition );
				const hkSimdReal distSqrd = transDist.lengthSquared<3>();
				if ( distSqrd > maxSqrd)
				{
					break;
				}
			}

				// orientation check
			{
				hkQuaternion refQ; hkVector4Util::unPackInt32IntoQuaternion( refOrientation, refQ.m_vec );

				const hkSimdReal maxSqrd = hkSimdReal::fromFloat(di.m_maxRotSqrd[select]);

				hkVector4 dist;	dist.setSub( refQ.m_vec, currentRotation.m_vec );
				const hkSimdReal distSqrd = dist.lengthSquared<4>();
				if ( distSqrd > maxSqrd)
				{
					break;
				}

			}

				// deactivate. Increment counter, but clip value at 64
			{
				motion.incrementNumInactiveFramesMt(select, si.m_deactivationNumInactiveFramesSelectFlag[select]);
				hkCheckDeterminismUtil::checkMt(0xf0000035, select);
				hkCheckDeterminismUtil::checkMt(0xf0000036, motion.getNumInactiveFrames(select));
				goto END_OF_FUNCTION;
			}
		}


		motion.zeroNumInactiveFramesMt(select, si.m_deactivationNumInactiveFramesSelectFlag[select]); // <ag.todo.z> ask oli: we still use the old value for splitting ... and what about deactivation later ?

		// reset reference position and orientation and max velocity
		{
			hkVector4 pos = currentPosition;
			pos.zeroComponent<3>();	// thats the max velocity
			refPosition = pos;
			hkCheckDeterminismUtil::checkMt(0xf0000037, refPosition);
			refOrientation = hkVector4Util::packQuaternionIntoInt32(currentRotation.m_vec);
		}
	}
END_OF_FUNCTION:
	return hkMath::max2( motion.getNumInactiveFrames(0), motion.getNumInactiveFrames(1) );
}

#if !defined(HK_PLATFORM_SPU)

bool HK_CALL hkRigidMotionUtilCanDeactivateFinal( const hkStepInfo& info, hkpMotion*const* motions, int numMotions, int motionOffset )
{
	for (int i = numMotions-1; i>=0; motions++, i--)
	{
		hkpMotion& motion = *hkAddByteOffset(motions[0], motionOffset);
		HK_ASSERT2( 0xf03245df, motion.m_type != hkpMotion::MOTION_FIXED, "Internal error, checking fixed motion is not allowed" );

		// we have to clip the radius. The reason is jitter because we store the quaternion in 32 bit
		// as a result we get 4% jitter. Without clipping the radius, the jitter could be enough to keep the object alive
		hkReal     radius          = hkMath::min2( 1.0f, motion.m_motionState.m_objectRadius);

			// we allow 2 times the max velocity as well as an extra sleepVel
		const hkSimdReal sleepVel = hkSimdReal::fromFloat(0.1f);	
		hkSimdReal velSqrd = hkSimdReal_Inv4 * calcVelocityForDeactivation( motion, hkSimdReal::fromFloat(radius) ) - sleepVel*sleepVel;

		hkCheckDeterminismUtil::checkMt(0xf0000041, velSqrd.getReal());
		hkCheckDeterminismUtil::checkMt(0xf0000042, motion.getNumInactiveFrames(0));
		hkCheckDeterminismUtil::checkMt(0xf0000043, motion.getNumInactiveFrames(1));

			// check the velocity where we have the higher counter; if counters are equal, use the low frequency check
		if ( motion.getNumInactiveFrames(0) > motion.getNumInactiveFrames(1))
		{
			if ( velSqrd > motion.m_deactivationRefPosition[0].getW())
			{
				return false;
			}
		}
		else
		{
			if ( velSqrd > motion.m_deactivationRefPosition[1].getW())
			{
				return false;
			}
		}
	}
	return true;
}


void HK_CALL hkRigidMotionUtilStep( const hkStepInfo& info, hkpMotion*const* motions, int numMotions, int motionOffset )
{
	for (int i = numMotions-1; i>=0; motions++, i--)
	{
		hkpMotion* motion = hkAddByteOffset(motions[0], motionOffset);
		if ( motion->m_type != hkpMotion::MOTION_FIXED )
		{
			hkSweptTransformUtil::_stepMotionState( info, motion->m_linearVelocity, motion->m_angularVelocity, motion->m_motionState);
		}
	}
}
#endif


int HK_CALL hkRigidMotionUtilApplyForcesAndStep( const struct hkpSolverInfo& solverInfo, const hkStepInfo& info, const hkVector4& deltaVel, hkpMotion*const* motions, int numMotions, int motionOffset )
{
	int numInactiveFrames = 0x7fffffff;
	for (int i = numMotions-1; i>=0; motions++, i--)
	{
		hkpMotion* motion = hkAddByteOffset(motions[0], motionOffset);
		hkSimdReal gravityFactor; gravityFactor.setFromHalf( motion->m_gravityFactor );

		switch( motion->m_type )
		{
			case hkpMotion::MOTION_FIXED:
				{
					continue;
				}

			case hkpMotion::MOTION_THIN_BOX_INERTIA:
				{
					hkpThinBoxMotion* tbm = reinterpret_cast<hkpThinBoxMotion*>( motion );
					tbm->m_linearVelocity.addMul( gravityFactor, deltaVel );
					const hkSimdReal deltaTime = hkSimdReal::fromFloat(info.m_deltaTime);
					{
						hkSimdReal m; m.setFromHalf(tbm->m_motionState.m_linearDamping); m.mul(deltaTime);
						hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
						tbm->m_linearVelocity.mul(mmax);
					}
					{
						hkSimdReal m; m.setFromHalf(tbm->m_motionState.m_angularDamping); m.mul(deltaTime);
						hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
						tbm->m_angularVelocity.mul(mmax);
					}

					// get the angular momentum in world space
					hkVector4 angularMomentum;
					{
						hkVector4 h; h.setRotatedInverseDir( tbm->getTransform().getRotation(), tbm->m_angularVelocity );
						h.div( tbm->m_inertiaAndMassInv );
						angularMomentum.setRotatedDir( tbm->getTransform().getRotation(), h );
					}

					hkSweptTransformUtil::_stepMotionState( info, tbm->m_linearVelocity, tbm->m_angularVelocity, tbm->m_motionState);

					// get the new angular velocity from the world momentum
					{
						hkVector4 h; h.setRotatedInverseDir( tbm->getTransform().getRotation(), angularMomentum );
						h.mul( tbm->m_inertiaAndMassInv );
						tbm->m_angularVelocity.setRotatedDir( tbm->getTransform().getRotation(), h );
					}
					// we have to reclip the velocities as the angular velocity was overwritten by the angular momentum
					hkSweptTransformUtil::_clipVelocities( motion->m_motionState, motion->m_linearVelocity, motion->m_angularVelocity );
					goto CHECK_DEACTIVATION;
				}

			case hkpMotion::MOTION_KEYFRAMED:
				{
					break;
				}
			default:
				{
					motion->m_linearVelocity.addMul( gravityFactor, deltaVel );
				}
			case hkpMotion::MOTION_CHARACTER:
				{
					const hkSimdReal deltaTime = hkSimdReal::fromFloat(info.m_deltaTime);
					{
						hkSimdReal m; m.setFromHalf(motion->m_motionState.m_linearDamping); m.mul(deltaTime);
						hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
						motion->m_linearVelocity.mul(mmax);
					}
					{
						hkSimdReal m; m.setFromHalf(motion->m_motionState.m_angularDamping); m.mul(deltaTime);
						hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
						motion->m_angularVelocity.mul(mmax);
					}
					break;
				}
		}
		hkSweptTransformUtil::_stepMotionState( info, motion->m_linearVelocity, motion->m_angularVelocity, motion->m_motionState);

CHECK_DEACTIVATION:
		numInactiveFrames = hkMath::min2( numInactiveFrames, hkRigidMotionUtilCheckDeactivation( solverInfo, *motion ));
	}
	return numInactiveFrames;
}

hkpVelocityAccumulator* HK_CALL hkRigidMotionUtilBuildAccumulators(const hkStepInfo& info, hkpMotion*const* motions, int numMotions, int motionOffset, hkpVelocityAccumulator* accumulatorsOut )
{
	for (int i = numMotions-1; i>=0; motions++, accumulatorsOut++, i--)
	{
		hkpMotion* motion = hkAddByteOffset(motions[0], motionOffset);
#if (defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_XBOX360))
		if ( i > 1)
		{
			hkpMotion* next = hkAddByteOffset(motions[2], motionOffset);
			void* p2 = hkAddByteOffset(accumulatorsOut,256);
			hkMath::forcePrefetch<256>( next );
			hkMath::prefetch128( p2 );
		}
#endif

		switch( motion->m_type )
		{
		case hkpMotion::MOTION_INVALID:
			{
				HK_ASSERT2( 0xf03243ee, 0, "hkpMotion::MOTION_INVALID detected");
			}

		case hkpMotion::MOTION_THIN_BOX_INERTIA:
		case hkpMotion::MOTION_BOX_INERTIA:
			{
				hkpVelocityAccumulator* accumulator = accumulatorsOut;

				accumulator->m_type		= hkpVelocityAccumulator::HK_RIGID_BODY;
				accumulator->m_context	= hkpVelocityAccumulator::ACCUMULATOR_CONTEXT_BUILD_JACOBIANS;

				hkMatrix3 coreFromWorldTransform;
				coreFromWorldTransform._setTranspose(motion->getTransform().getRotation());
				accumulator->setCoreFromWorldMatrix(coreFromWorldTransform);

				const hkpBoxMotion* bm = reinterpret_cast<hkpBoxMotion*>( motion );
				accumulator->m_invMasses = bm->m_inertiaAndMassInv;
				accumulator->m_linearVel = motion->m_linearVelocity;

				hkMotionState& ms = motion->m_motionState;
				hkVector4 worldCenterOfMass;
				hkSweptTransformUtil::calcCenterOfMassAt( ms, info.m_startTime, worldCenterOfMass);
				accumulator->setCenterOfMassInWorld(worldCenterOfMass);

				accumulator->m_angularVel = motion->m_angularVelocity;

				accumulator->m_deactivationClass   = motion->getDeactivationClass();
				accumulator->m_gravityFactor	   = motion->m_gravityFactor;

				break;
			}

		case hkpMotion::MOTION_CHARACTER:
			{
				accumulatorsOut->m_type = hkpVelocityAccumulator::HK_NO_GRAVITY_RIGID_BODY;
				goto buildRigidBodyAccumulator;
			}

		case hkpMotion::MOTION_SPHERE_INERTIA:
			{
				accumulatorsOut->m_type		= hkpVelocityAccumulator::HK_RIGID_BODY;

buildRigidBodyAccumulator:
				accumulatorsOut->m_context	= hkpVelocityAccumulator::ACCUMULATOR_CONTEXT_BUILD_JACOBIANS;

				const hkSimdReal deltaTime = hkSimdReal::fromFloat(info.m_deltaTime);
				//motion->m_linearVelocity.mul4(  hkMath::max2( 0.0f, one - deltaTime * motion->m_motionState.m_linearDamping ) );
				//motion->m_angularVelocity.mul4( hkMath::max2( 0.0f, one - deltaTime * motion->m_motionState.m_angularDamping) );

				hkpVelocityAccumulator* accumulator = accumulatorsOut;

				accumulator->m_invMasses = motion->m_inertiaAndMassInv;

				const hkMotionState& ms = motion->m_motionState;

				hkVector4 worldCenterOfMass;
				hkSweptTransformUtil::calcCenterOfMassAt( ms, info.m_startTime, worldCenterOfMass);
				accumulator->setCenterOfMassInWorld(worldCenterOfMass);

				accumulator->m_angularVel.setXYZ_0( motion->m_angularVelocity );
				{
					hkSimdReal m; m.setFromHalf(motion->m_motionState.m_angularDamping); m.mul(deltaTime);
					hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
					accumulator->m_angularVel.mul(mmax);
				}
				accumulator->m_linearVel.setXYZ_0( motion->m_linearVelocity );
				{
					hkSimdReal m; m.setFromHalf(motion->m_motionState.m_linearDamping); m.mul(deltaTime);
					hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
					accumulator->m_linearVel.mul(mmax);
				}

				accumulator->setCoreFromWorldMatrixToIdentity();
				accumulator->m_deactivationClass    = motion->getDeactivationClass();
				accumulator->m_gravityFactor	    = motion->m_gravityFactor;

				break;
			}

		case hkpMotion::MOTION_FIXED:
			{
				HK_ASSERT2(0x7a3053cc, 0, "fixed rigid bodies cannot go into the solver");

				hkpVelocityAccumulator* accumulator = accumulatorsOut;
				accumulator->setFixed();

				break;
			}

		case hkpMotion::MOTION_KEYFRAMED:
			{
				hkpVelocityAccumulator* accumulator = accumulatorsOut;

				accumulator->m_type		= hkpVelocityAccumulator::HK_KEYFRAMED_RIGID_BODY;
				accumulator->m_context	= hkpVelocityAccumulator::ACCUMULATOR_CONTEXT_BUILD_JACOBIANS;

				accumulator->m_invMasses.               setZero();
				accumulator->setCenterOfMassInWorld(motion->getCenterOfMassInWorld());
				accumulator->m_angularVel             = motion->m_angularVelocity;
				accumulator->m_linearVel              = motion->m_linearVelocity;
				accumulator->setCoreFromWorldMatrixToIdentity();
				accumulator->m_gravityFactor	      = motion->m_gravityFactor;

				break;
			}

		default:
			{
				HK_ASSERT2( 0xf0323456, 0, "Unknown motion" );
			}
		}
	}

	return accumulatorsOut;
}

hkpVelocityAccumulator* HK_CALL hkRigidMotionUtilApplyForcesAndBuildAccumulators(const hkStepInfo& info, hkpMotion*const* motions, int numMotions, int motionOffset, hkpVelocityAccumulator* accumulatorsOut )
{
	for (int i = numMotions-1; i>=0; motions++, accumulatorsOut++, i--)
	{
		hkpMotion* motion = hkAddByteOffset(motions[0], motionOffset);
#if (defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_XBOX360))
		if ( i > 1)
		{
			hkpMotion* next = hkAddByteOffset(motions[2], motionOffset);
			void* p2 = hkAddByteOffset(accumulatorsOut,256);
			hkMath::forcePrefetch<256>( next );
			hkMath::prefetch128( p2 );
		}
#endif

		switch( motion->m_type )
		{
			case hkpMotion::MOTION_INVALID:
				{
					HK_ASSERT2( 0xf03243ee, 0, "hkpMotion::MOTION_INVALID detected");
				}

			case hkpMotion::MOTION_THIN_BOX_INERTIA:
			case hkpMotion::MOTION_BOX_INERTIA:
				{
					const hkSimdReal deltaTime = hkSimdReal::fromFloat(info.m_deltaTime);
					{
						hkSimdReal m; m.setFromHalf(motion->m_motionState.m_angularDamping); m.mul(deltaTime);
						hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
						motion->m_angularVelocity.mul(mmax);
					}
					{
						hkSimdReal m; m.setFromHalf(motion->m_motionState.m_linearDamping); m.mul(deltaTime);
						hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
						motion->m_linearVelocity.mul(mmax);
					}

					hkpVelocityAccumulator* accumulator = accumulatorsOut;

					accumulator->m_type		= hkpVelocityAccumulator::HK_RIGID_BODY;
					accumulator->m_context	= hkpVelocityAccumulator::ACCUMULATOR_CONTEXT_BUILD_JACOBIANS;

					hkMatrix3 coreFromWorldTransform;
					coreFromWorldTransform._setTranspose(motion->getTransform().getRotation());
					accumulator->setCoreFromWorldMatrix(coreFromWorldTransform);

					const hkpBoxMotion* bm = reinterpret_cast<hkpBoxMotion*>( motion );
					accumulator->m_invMasses = bm->m_inertiaAndMassInv;
					accumulator->m_linearVel = motion->m_linearVelocity;

					hkMotionState& ms = motion->m_motionState;
					hkVector4 worldCenterOfMass;
					hkSweptTransformUtil::calcCenterOfMassAt( ms, info.m_startTime, worldCenterOfMass);
					accumulator->setCenterOfMassInWorld(worldCenterOfMass);

					accumulator->m_angularVel.           _setRotatedDir( accumulator->getCoreFromWorldMatrix(), motion->m_angularVelocity );

					accumulator->m_deactivationClass   = motion->getDeactivationClass();
					accumulator->m_gravityFactor	   = motion->m_gravityFactor;

					break;
				}

			case hkpMotion::MOTION_CHARACTER:
				{
					accumulatorsOut->m_type = hkpVelocityAccumulator::HK_NO_GRAVITY_RIGID_BODY;
					goto buildRigidBodyAccumulator;
				}

			case hkpMotion::MOTION_SPHERE_INERTIA:
				{
					accumulatorsOut->m_type		= hkpVelocityAccumulator::HK_RIGID_BODY;

buildRigidBodyAccumulator:
					accumulatorsOut->m_context	= hkpVelocityAccumulator::ACCUMULATOR_CONTEXT_BUILD_JACOBIANS;

					const hkSimdReal deltaTime = hkSimdReal::fromFloat(info.m_deltaTime);
					{
						hkSimdReal m; m.setFromHalf(motion->m_motionState.m_angularDamping); m.mul(deltaTime);
						hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
						motion->m_angularVelocity.mul(mmax);
					}
					{
						hkSimdReal m; m.setFromHalf(motion->m_motionState.m_linearDamping); m.mul(deltaTime);
						hkSimdReal mmax; mmax.setMax(hkSimdReal_0, hkSimdReal_1 - m);
						motion->m_linearVelocity.mul(mmax);
					}

					hkpVelocityAccumulator* accumulator = accumulatorsOut;

					accumulator->m_invMasses = motion->m_inertiaAndMassInv;

					const hkMotionState& ms = motion->m_motionState;

					hkVector4 worldCenterOfMass;
					hkSweptTransformUtil::calcCenterOfMassAt( ms, info.m_startTime, worldCenterOfMass);
					accumulator->setCenterOfMassInWorld(worldCenterOfMass);

					accumulator->m_angularVel           = motion->m_angularVelocity;
					accumulator->m_linearVel            = motion->m_linearVelocity;
					accumulator->setCoreFromWorldMatrixToIdentity();
					accumulator->m_deactivationClass    = motion->getDeactivationClass();
					accumulator->m_gravityFactor	    = motion->m_gravityFactor;
				
					break;
				}

			case hkpMotion::MOTION_FIXED:
				{
					HK_ASSERT2(0x7a3053cc, 0, "fixed rigid bodies cannot go into the solver");

					hkpVelocityAccumulator* accumulator = accumulatorsOut;
					accumulator->setFixed();

					break;
				}

			case hkpMotion::MOTION_KEYFRAMED:
				{
					hkpVelocityAccumulator* accumulator = accumulatorsOut;

					accumulator->m_type		= hkpVelocityAccumulator::HK_KEYFRAMED_RIGID_BODY;
					accumulator->m_context	= hkpVelocityAccumulator::ACCUMULATOR_CONTEXT_BUILD_JACOBIANS;

					accumulator->m_invMasses.               setZero();
					accumulator->setCenterOfMassInWorld(motion->getCenterOfMassInWorld());
					accumulator->m_angularVel             = motion->m_angularVelocity;
					accumulator->m_linearVel              = motion->m_linearVelocity;
					accumulator->setCoreFromWorldMatrixToIdentity();
					accumulator->m_gravityFactor	      = motion->m_gravityFactor;

					break;
				}

			default:
				{
					HK_ASSERT2( 0xf0323456, 0, "Unknown motion" );
				}
		}
	}

	return accumulatorsOut;
}

int HK_CALL hkRigidMotionUtilApplyAccumulators(const struct hkpSolverInfo& solverInfo, const hkStepInfo& info, const hkpVelocityAccumulator* accumulators, hkpMotion*const* motions, int numMotions, int motionOffset )
{
	const hkpVelocityAccumulator* accu = accumulators;

	int numInactiveFrames = 0x7fffffff;
	hkVector4 integrationLinearVelocity;
	hkVector4 integrationAngularVelocity;

	// Initialize to zero to get rid of "...used uninitialized..." warning
	hkVector4 angularMomentum; angularMomentum.setZero();

	for (int i = numMotions-1; i>=0; motions++, accu++, i--)
	{
		hkpMotion* motion = hkAddByteOffset(motions[0], motionOffset);

#if (defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_XBOX360))
		if ( i > 1)
		{
			hkpMotion* next = hkAddByteOffset(motions[2], motionOffset);
			const void* p2 = hkAddByteOffsetConst(accu,256);
			hkMath::forcePrefetch<256>( next );
			hkMath::prefetch128( p2 );
		}
#endif

		switch( motion->m_type )
		{
			case hkpMotion::MOTION_INVALID:
				{
					HK_ASSERT2( 0xf03243ed, 0, "hkpMotion::MOTION_INVALID detected");
				}

			case hkpMotion::MOTION_THIN_BOX_INERTIA:
				{
					// get the angular momentum in world space
					hkpThinBoxMotion* tbm = reinterpret_cast<hkpThinBoxMotion*>( motion );
					hkVector4 angVelLocalSpace = accu->m_angularVel;
					angVelLocalSpace.div( tbm->m_inertiaAndMassInv );
					angularMomentum._setRotatedDir( tbm->getTransform().getRotation(), angVelLocalSpace );
					// no break here
				}
			case hkpMotion::MOTION_BOX_INERTIA:
				{
					motion->m_linearVelocity	= accu->m_linearVel;
 					motion->m_angularVelocity	. _setRotatedDir( motion->getTransform().getRotation(), accu->m_angularVel );
					integrationLinearVelocity	= accu->getSumLinearVel();
 					integrationAngularVelocity	. _setRotatedDir( motion->getTransform().getRotation(), accu->getSumAngularVel() );
					goto sphereInertia;
				}

			case hkpMotion::MOTION_KEYFRAMED:
				{
					motion->m_linearVelocity	= motion->m_linearVelocity;
					motion->m_angularVelocity	= motion->m_angularVelocity;
					integrationLinearVelocity	= motion->m_linearVelocity;
					integrationAngularVelocity	= motion->m_angularVelocity;
					goto sphereInertiaNoDeactivationCount;
				}



			case hkpMotion::MOTION_CHARACTER:
			case hkpMotion::MOTION_SPHERE_INERTIA:
				{
					motion->m_linearVelocity	= accu->m_linearVel;
					motion->m_angularVelocity	= accu->m_angularVel;
					integrationLinearVelocity	= accu->getSumLinearVel();
					integrationAngularVelocity	= accu->getSumAngularVel();
sphereInertia:
sphereInertiaNoDeactivationCount:
					hkSweptTransformUtil::_stepMotionState( info, integrationLinearVelocity, integrationAngularVelocity, motion->m_motionState);
					break;
				}

			case hkpMotion::MOTION_FIXED:
			default:
				{
					break;
				}
		}

		if ( motion->m_type == hkpMotion::MOTION_THIN_BOX_INERTIA )
		{
			hkpThinBoxMotion* tbm = reinterpret_cast<hkpThinBoxMotion*>( motion );
			tbm->m_linearVelocity = accu->m_linearVel;
			// get the new angular velocity from the world momentum
			{
				hkVector4 h; h.setRotatedInverseDir( tbm->getTransform().getRotation(), angularMomentum );
				h.mul( tbm->m_inertiaAndMassInv );
				tbm->m_angularVelocity.setRotatedDir( tbm->getTransform().getRotation(), h );
			}
		}
		hkSweptTransformUtil::_clipVelocities( motion->m_motionState, motion->m_linearVelocity, motion->m_angularVelocity );
		numInactiveFrames = hkMath::min2( numInactiveFrames, hkRigidMotionUtilCheckDeactivation( solverInfo, *motion ));
	}
	return numInactiveFrames;
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
