/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/World/hkpWorldCinfo.h>


#if defined (HK_PLATFORM_HAS_SPU)
#define HK_DEFAULT_TOI_MIDPHASE_COLLIDE_TASKS 4
#define HK_DEFAULT_TOI_NARROWPHASE_COLLIDE_TASKS 6
#else
#define HK_DEFAULT_TOI_MIDPHASE_COLLIDE_TASKS 4
#define HK_DEFAULT_TOI_NARROWPHASE_COLLIDE_TASKS 12
#endif

hkpWorldCinfo::hkpWorldCinfo()
{
	m_gravity.set(0.0f, -9.8f, 0.0f);
	m_enableSimulationIslands = true;
	m_broadPhaseQuerySize = 1024;
	m_mtPostponeAndSortBroadPhaseBorderCallbacks = false;
	m_broadPhaseWorldAabb.m_min.set(-500.0f, -500.0f, -500.0f);
	m_broadPhaseWorldAabb.m_max.set(500.0f, 500.0f, 500.0f);
	m_collisionFilter = HK_NULL;
	m_convexListFilter = HK_NULL; 
	m_broadPhaseNumMarkers = 0;
	m_sizeOfToiEventQueue = 250;
	m_solverTau = 0.6f;
	m_solverDamp = 1.0f;
	m_contactRestingVelocity = 1.0f;
	m_solverIterations = 4;
	m_solverMicrosteps = 1; 
	m_maxConstraintViolation = HK_REAL_HIGH;
	m_forceCoherentConstraintOrderingInSolver = false;
	m_snapCollisionToConvexEdgeThreshold = 0.524f;
	m_snapCollisionToConcaveEdgeThreshold = 0.698f;
	m_enableToiWeldRejection = false;
	m_collisionTolerance = 0.1f;
	m_broadPhaseType = BROADPHASE_TYPE_SAP;
	m_broadPhaseBorderBehaviour = BROADPHASE_BORDER_ASSERT;
	m_toiCollisionResponseRotateNormal = 0.2f;
	
	//m_enableForceLimitBreachedSecondaryEventsFromToiSolver = false;
	m_useCompoundSpuElf = false;
	m_processToisMultithreaded = true;	
	m_maxSectorsPerMidphaseCollideTask = 2;
	m_maxSectorsPerNarrowphaseCollideTask = 4;
	m_maxEntriesPerToiMidphaseCollideTask = HK_DEFAULT_TOI_MIDPHASE_COLLIDE_TASKS;
	m_maxEntriesPerToiNarrowphaseCollideTask = HK_DEFAULT_TOI_NARROWPHASE_COLLIDE_TASKS;
	m_maxNumToiCollisionPairsSinglethreaded = 0;
	m_numToisTillAllowedPenetrationSimplifiedToi =  3.0f;
	m_numToisTillAllowedPenetrationToi           =  3.0f;
	m_numToisTillAllowedPenetrationToiHigher     =  4.0f;
	m_numToisTillAllowedPenetrationToiForced     = 20.0f;
	m_deactivationReferenceDistance = 0.02f;
	m_expectedMaxLinearVelocity = 200.0f;
	m_expectedMinPsiDeltaTime = 1.0f / 30.0f;
	m_iterativeLinearCastEarlyOutDistance = 0.01f;
	m_enableDeprecatedWelding = false;
	m_iterativeLinearCastMaxIterations = 20;
	m_enableDeactivation = true;
	m_shouldActivateOnRigidBodyTransformChange = true;
	m_minDesiredIslandSize = 64;
	m_deactivationNumInactiveFramesSelectFlag0 = 0;
	m_deactivationNumInactiveFramesSelectFlag1 = 0;
	m_deactivationIntegrateCounter = 0;
	m_contactPointGeneration = CONTACT_POINT_REJECT_MANY;
	m_allowToSkipConfirmedCallbacks = false;
	m_simulationType = SimulationType(SIMULATION_TYPE_CONTINUOUS);
	m_frameMarkerPsiSnap = .0001f;
	m_memoryWatchDog = HK_NULL;
	m_processActionsInSingleThread = true;
#if ! defined (HK_PLATFORM_HAS_SPU)
	m_allowIntegrationOfIslandsWithoutConstraintsInASeparateJob = false;
#else
	m_allowIntegrationOfIslandsWithoutConstraintsInASeparateJob = true;
#endif
	m_fireCollisionCallbacks = false;
}

void hkpWorldCinfo::setBroadPhaseWorldSize(hkReal sideLength)
{
	m_broadPhaseWorldAabb.m_min.setAll( -0.5f * sideLength );
	m_broadPhaseWorldAabb.m_max.setAll(  0.5f * sideLength );
}

void hkpWorldCinfo::setupSolverInfo( enum hkpWorldCinfo::SolverType st)
{
	switch ( st )
	{
        case SOLVER_TYPE_2ITERS_SOFT:
                                    m_solverTau = 0.3f;
                                    m_solverDamp = 0.9f;
                                    m_solverIterations = 2;
                                    break;
        case SOLVER_TYPE_2ITERS_MEDIUM:
                                    m_solverTau = 0.6f;
                                    m_solverDamp = 1.0f;
                                    m_solverIterations = 2;
                                    break;
        case SOLVER_TYPE_2ITERS_HARD:
                                    m_solverTau = 0.9f;
                                    m_solverDamp = 1.1f;
                                    m_solverIterations = 2;
                                    break;
        case SOLVER_TYPE_4ITERS_SOFT:
                                    m_solverTau = 0.3f;
                                    m_solverDamp = 0.9f;
                                    m_solverIterations = 4;
                                    break;
        case SOLVER_TYPE_4ITERS_MEDIUM:
                                    m_solverTau = 0.6f;
                                    m_solverDamp = 1.0f;
                                    m_solverIterations = 4;
                                    break;
        case SOLVER_TYPE_4ITERS_HARD:
                                    m_solverTau = 0.9f;
                                    m_solverDamp = 1.1f;
                                    m_solverIterations = 4;
                                    break;
        case SOLVER_TYPE_8ITERS_SOFT:
                                    m_solverTau = 0.3f;
                                    m_solverDamp = 0.9f;
                                    m_solverIterations = 8;
                                    break;
        case SOLVER_TYPE_8ITERS_MEDIUM:
                                    m_solverTau = 0.6f;
                                    m_solverDamp = 1.0f;
                                    m_solverIterations = 8;
                                    break;
        case SOLVER_TYPE_8ITERS_HARD:
                                    m_solverTau = 0.9f;
                                    m_solverDamp = 1.1f;
                                    m_solverIterations = 8;
                                    break;
        default:
            HK_ASSERT2(0x32ba3a5b, 0, "Unknown solver type" );
	}
}

hkpWorldCinfo::hkpWorldCinfo( hkFinishLoadedObjectFlag flag ) :
	hkReferencedObject(flag), m_collisionFilter(flag), m_convexListFilter(flag), m_memoryWatchDog(flag)
{
	if( flag.m_finishing )
	{
		if ( 0.0f == m_contactRestingVelocity )
		{
			HK_WARN( 0xf03243ed, "m_contactRestingVelocity not set, setting it to 1.0f, so that the new collision restitution code will be disabled" );
			m_contactRestingVelocity = 1.0f;
		} 

		if (-1 == m_maxEntriesPerToiMidphaseCollideTask)
		{
			//HK_WARN( 0xf13243ed, "m_maxEntriesPerToiMidphaseCollideTask not set, setting it to " << HK_DEFAULT_TOI_MIDPHASE_COLLIDE_TASKS << "." );
			m_maxEntriesPerToiMidphaseCollideTask = HK_DEFAULT_TOI_MIDPHASE_COLLIDE_TASKS;
		}

		if (-1 == m_maxEntriesPerToiNarrowphaseCollideTask)
		{
			//HK_WARN( 0xf13243ed, "m_maxEntriesPerToiNarrowphaseCollideTask not set, setting it to " << HK_DEFAULT_TOI_NARROWPHASE_COLLIDE_TASKS << "." );
			m_maxEntriesPerToiNarrowphaseCollideTask = HK_DEFAULT_TOI_NARROWPHASE_COLLIDE_TASKS;
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
