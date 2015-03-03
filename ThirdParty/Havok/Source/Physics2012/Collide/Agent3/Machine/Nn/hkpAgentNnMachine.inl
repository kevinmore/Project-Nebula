/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
// all those needed for the determinism checks only
#if defined HK_ENABLE_DETERMINISM_CHECKS
#	include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#	include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#endif

struct hkAgentNnMachineBodyTemp
{
	hkpCdBody m_bodyA;
	hkpCdBody m_bodyB;
	hkTransform m_transA;
	hkTransform m_transB;
};

HK_FORCE_INLINE static void HK_CALL hkAgentNnMachine_initInputAtTime( hkpAgent3Input& in, hkAgentNnMachineBodyTemp& temp, hkpAgent3Input& out)
{
	hkSweptTransformUtil::lerp2( in.m_bodyA->getMotionState()->getSweptTransform(), in.m_input->m_stepInfo.m_startTime, temp.m_transA );
	hkSweptTransformUtil::lerp2( in.m_bodyB->getMotionState()->getSweptTransform(), in.m_input->m_stepInfo.m_startTime, temp.m_transB );
	
	out.m_bodyA = &temp.m_bodyA;
	out.m_bodyB = &temp.m_bodyB;
	out.m_contactMgr = in.m_contactMgr;
	out.m_input = in.m_input;

	temp.m_bodyA.setShape( in.m_bodyA->getShape(), in.m_bodyA->getShapeKey());
	temp.m_bodyB.setShape( in.m_bodyB->getShape(), in.m_bodyB->getShapeKey());
	
	new (&temp.m_bodyA) hkpCdBody( in.m_bodyA, &temp.m_transA );
	new (&temp.m_bodyB) hkpCdBody( in.m_bodyB, &temp.m_transB );
	out.m_aTb.setMulInverseMul(temp.m_transA, temp.m_transB);
}

//
// Processing
//
#include <Physics2012/Collide/Agent/CompoundAgent/List/hkpListAgent.h>
#ifdef HK_PLATFORM_CTR
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#endif
static HK_FORCE_INLINE void HK_CALL
hkAgentNnMachine_InlineProcessAgent( hkpAgentNnEntry* entry, const hkpProcessCollisionInput& input, int& numTim, hkpProcessCollisionOutput& output, hkpContactMgr* mgr  )
{
	HK_TIMER_BEGIN( "ProcessNnEntry", HK_NULL );

#if defined HK_ENABLE_DETERMINISM_CHECKS
	hkCheckDeterminismUtil::checkMt( 0xf00001b0, hkpGetRigidBody(entry->getCollidableA())->getSimulationIsland()->m_uTag);
	hkCheckDeterminismUtil::checkMt( 0xf00001b1, hkpGetRigidBody(entry->getCollidableB())->getSimulationIsland()->m_uTag);
	hkCheckDeterminismUtil::checkMt( 0xf00001b2, hkpGetRigidBody(entry->getCollidableA())->m_storageIndex);
	hkCheckDeterminismUtil::checkMt( 0xf00001b3, hkpGetRigidBody(entry->getCollidableB())->m_storageIndex);
	hkCheckDeterminismUtil::checkMt( 0xf00001b4, hkpGetRigidBody(entry->getCollidableA())->m_uid);
	hkCheckDeterminismUtil::checkMt( 0xf00001b5, hkpGetRigidBody(entry->getCollidableB())->m_uid);
#endif

#if !defined(HK_PLATFORM_SPU)
	const hkpCdBody* cdBodyA = entry->getCollidableA();
	const hkpCdBody* cdBodyB = entry->getCollidableB();
#else
	const hkpCdBody* cdBodyA = input.m_collisionObjects->m_collidableA;
	const hkpCdBody* cdBodyB = input.m_collisionObjects->m_collidableB;
#endif

#ifndef HK_PLATFORM_CTR
	hkMotionState transformForTransformedShapeB[4]; 
	hkMotionState transformForTransformedShapeA[4];
	hkpCdBody cdBodyForTransformedShapeB[4];
	hkpCdBody cdBodyForTransformedShapeA[4];
#else
	hkLocalArray<hkMotionState> _transformForTransformedShapeB (4); _transformForTransformedShapeB.setSize(4);
	hkMotionState* transformForTransformedShapeB = &_transformForTransformedShapeB[0];
	
	hkLocalArray<hkMotionState> _transformForTransformedShapeA (4); _transformForTransformedShapeA.setSize(4);
	hkMotionState* transformForTransformedShapeA = &_transformForTransformedShapeA[0];

	hkLocalArray<hkpCdBody> _cdBodyForTransformedShapeB (4); _cdBodyForTransformedShapeB.setSize(4);
	hkpCdBody* cdBodyForTransformedShapeB = &_cdBodyForTransformedShapeB[0];

	hkLocalArray<hkpCdBody> _cdBodyForTransformedShapeA (4); _cdBodyForTransformedShapeA.setSize(4);
	hkpCdBody* cdBodyForTransformedShapeA = &_cdBodyForTransformedShapeA[0];

#endif

#	if defined (HK_ENABLE_DETERMINISM_CHECKS)
	{
		const hkMotionState* msA = cdBodyA->getMotionState();
		const hkMotionState* msB = cdBodyB->getMotionState();
		hkCheckDeterminismUtil::checkMtCrc( 0xf0000372, &msA->getTransform(), 1 );
		hkCheckDeterminismUtil::checkMtCrc( 0xf0000373, &msB->getTransform(), 1 );
	}
#endif



	hkAgent3::StreamCommand command = static_cast<hkAgent3::StreamCommand>(entry->m_streamCommand);
commandSwitch:
	switch ( command )
	{
		case hkAgent3::STREAM_CALL_AGENT_WITH_TRANSFORM:
		case hkAgent3::STREAM_CALL_WITH_TRANSFORM:
		case hkAgent3::STREAM_CALL_WITH_TIM_WITH_TRANSFORM:
			{
				// create new cdBodies, calculate new hkMotionStates
				hkPadSpu<hkUchar> dummyCdBodyHasTransformFlag = 0;
				cdBodyA = hkAgentMachine_processTransformedShapes(cdBodyA, cdBodyForTransformedShapeA, transformForTransformedShapeA, 4, dummyCdBodyHasTransformFlag);
				cdBodyB = hkAgentMachine_processTransformedShapes(cdBodyB, cdBodyForTransformedShapeB, transformForTransformedShapeB, 4, dummyCdBodyHasTransformFlag);

				command = static_cast<hkAgent3::StreamCommand> ( static_cast<hkUchar>(command) & static_cast<hkUchar>(~hkAgent3::TRANSFORM_FLAG) );
				goto commandSwitch;
			}

		case hkAgent3::STREAM_CALL_AGENT:
			{
#if !defined(HK_PLATFORM_SPU)
				hkpAgentNnMachinePaddedEntry* paddedEntry = static_cast<hkpAgentNnMachinePaddedEntry*>(entry);
				hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(paddedEntry+1);
				hkpCollisionAgent* agent = hkAgent3Bridge::getChildAgent(agentData);
				agent->processCollision( *cdBodyA, *cdBodyB, input, output );
#else
				HK_ASSERT2( 0xf0345465, 0, "This type of collision is not handled on spu, consider setting m_forceCollideOntoPpu to true on one of the entities involved" );
#endif
				break;
			}

		case hkAgent3::STREAM_CALL:
			{
				hkpAgentNnMachinePaddedEntry* e = reinterpret_cast<hkpAgentNnMachinePaddedEntry*>(entry);

				hkpAgent3ProcessInput in3;
				{
					in3.m_bodyA			= cdBodyA;
					in3.m_bodyB			= cdBodyB;
#if !defined(HK_PLATFORM_SPU)
					in3.m_contactMgr	= entry->m_contactMgr;
#else
					in3.m_contactMgr	= input.m_collisionObjects->m_contactMgr;
#endif
					in3.m_input			= &input;

					const hkMotionState* msA = in3.m_bodyA->getMotionState();
					const hkMotionState* msB = in3.m_bodyB->getMotionState();
					hkSweptTransformUtil::calcTimInfo( *msA, *msB, in3.m_input->m_stepInfo.m_deltaTime, in3.m_linearTimInfo);

					in3.m_aTb.setMulInverseMul(msA->getTransform(), msB->getTransform());
				}

				//
				//	call agent
				//
				hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(e+1);
				HK_ON_DEBUG(hkpAgentData* agentEnd =)
					input.m_dispatcher->getAgent3ProcessFunc( entry->m_agentType )( in3, entry, agentData, HK_NULL, output );
				HK_ON_DEBUG(hkUlong size = hkGetByteOffset( entry, agentEnd ));
				HK_ASSERT2(0xf0436def, size <= hkpAgentNnTrack::getAgentSize( entry->m_nnTrackType ), "Agent exceeding its size");
				break;
			}
		case hkAgent3::STREAM_CALL_WITH_TIM:
		{
			hkpAgentNnMachineTimEntry* e = reinterpret_cast<hkpAgentNnMachineTimEntry*>(entry);

			const hkMotionState* msA;
			const hkMotionState* msB;
			hkpAgent3ProcessInput in3;
			{
				in3.m_bodyA			= cdBodyA;
				in3.m_bodyB			= cdBodyB;
#if !defined(HK_PLATFORM_SPU)
				in3.m_contactMgr	= entry->m_contactMgr;
#else
				in3.m_contactMgr	= input.m_collisionObjects->m_contactMgr;
#endif
				in3.m_input			= &input;

				msA = in3.m_bodyA->getMotionState();
				msB = in3.m_bodyB->getMotionState();

				hkSweptTransformUtil::calcTimInfo( *msA, *msB, in3.m_input->m_stepInfo.m_deltaTime, in3.m_linearTimInfo );
			}
			hkpAgentData* agentData = reinterpret_cast<hkpAgentData*>(e+1);
			
			//
			//	validate separating plane
			//
			hkSimdReal distAtT1;
			if ( ! (e->m_timeOfSeparatingNormal == input.m_stepInfo.m_startTime.val() ) )
			{
				const hkpCollisionQualityInfo& ci = *input.m_collisionQualityInfo;
				if ( !ci.m_useContinuousPhysics.val() )
				{
					e->m_timeOfSeparatingNormal = input.m_stepInfo.m_endTime;
					distAtT1 = hkSimdReal_MinusMax;
					e->m_separatingNormal.setXYZ_W(hkVector4::getZero(), distAtT1);
					goto PROCESS_AT_T1;
				}

				HK_INTERNAL_TIMER_BEGIN("recalcT0", HK_NULL);
				hkAgentNnMachineBodyTemp prevTemp;
				hkpAgent3Input prevInput;
				hkAgentNnMachine_initInputAtTime( in3, prevTemp, prevInput );
				input.m_dispatcher->getAgent3SepNormalFunc( entry->m_agentType )( prevInput, e, agentData, e->m_separatingNormal );
				HK_INTERNAL_TIMER_END();
			}

				// optimistically set the separatingNormal time to the end of the step
			e->m_timeOfSeparatingNormal = input.m_stepInfo.m_endTime;

			{
				hkSimdReal worstCaseApproachingDelta  = in3.m_linearTimInfo.dot4xyz1( e->m_separatingNormal );
				distAtT1 = e->m_separatingNormal.getW() - worstCaseApproachingDelta;
			}
			//
			//	Check if traditional tims work
			//  Check if the worst case projected distance is still greater than the tolerance
			//
			if ( distAtT1.isGreaterEqual(hkSimdReal::fromFloat(input.getTolerance())) )
			{
				e->m_separatingNormal.setW(distAtT1);

				if ( entry->m_numContactPoints )
				{
					HK_ON_DEBUG(hkpAgentData* agentEnd  =)
						input.m_dispatcher->getAgent3CleanupFunc( entry->m_agentType )( entry, agentData, in3.m_contactMgr, *output.m_constraintOwner.val() );
					HK_ON_DEBUG(hkUlong size = hkGetByteOffset( entry, agentEnd ));
					HK_ASSERT2(0xf0436df0, size <= hkpAgentNnTrack::getAgentSize( entry->m_nnTrackType ),"Agent exceeding its size");
				}
				numTim++;
				break;
			}
PROCESS_AT_T1:
			distAtT1.store<1>((hkReal*)&in3.m_distAtT1);
			in3.m_aTb.setMulInverseMul(msA->getTransform(), msB->getTransform());

			HK_ON_DEBUG(hkpAgentData* agentEnd  =)
				input.m_dispatcher->getAgent3ProcessFunc( entry->m_agentType )( in3, entry, agentData, &e->m_separatingNormal, output );

			HK_ON_DEBUG(hkUlong size = hkGetByteOffset( entry, agentEnd ));
			HK_ASSERT2(0xf0436df0, size <= hkpAgentNnTrack::getAgentSize( entry->m_nnTrackType ),"Agent exceeding its size");
			break;
		}

	case hkAgent3::STREAM_NULL:
		{
			break;
		}
	default:
		HK_ASSERT2( 0xf0de82ea, 0, "Unknown Stream Command in hkAgentNnMachine_ProcessAgent" );
	}

#	if defined (HK_ENABLE_DETERMINISM_CHECKS)
	{
		hkpProcessCdPoint* points = output.getFirstContactPoint();
		int numPoints = output.getNumContactPoints();
		for (int i =0; i < numPoints; i++)
		{
			hkCheckDeterminismUtil::checkMtCrc( 0xf0000376, &points[i].m_contact, 1 );
		}

		if ( output.hasToi() )
		{
			hkCheckDeterminismUtil::checkMt( 0xf0000374, output.getToi() );
			hkCheckDeterminismUtil::checkMtCrc( 0xf0000375, &output.getToiContactPoint(), 1 );
		}
	}
#endif

	HK_TIMER_END();
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
