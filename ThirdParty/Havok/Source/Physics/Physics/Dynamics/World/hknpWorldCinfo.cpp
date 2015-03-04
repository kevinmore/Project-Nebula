/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/World/hknpWorldCinfo.h>

HK_COMPILE_TIME_ASSERT( (sizeof(hknpWorldCinfo) % 16) == 0 );


hknpWorldCinfo::hknpWorldCinfo()
{
	hkString::memClear16( this, sizeof(*this)/16 );

#if (HK_CONFIG_THREAD == HK_CONFIG_SINGLE_THREADED)
	m_simulationType = SIMULATION_TYPE_SINGLE_THREADED;
#else
	m_simulationType = SIMULATION_TYPE_MULTI_THREADED;
#endif

	m_numSplitterCells = (m_simulationType == SIMULATION_TYPE_SINGLE_THREADED) ? 1 : 16;
	m_solverMicrosteps = 1;
	m_defaultSolverTimestep = 1.0f / 30.0f;
	m_unitScale = 1.0f;
	m_relativeCollisionAccuracy = 0.005f;	// 0.5 cm accuracy
	m_bodyBufferCapacity = 4096;
	m_userBodyBuffer = HK_NULL;
	m_motionBufferCapacity = 4096;
	m_userMotionBuffer = HK_NULL;
	m_materialLibrary = HK_NULL;
	m_motionPropertiesLibrary = HK_NULL;
	m_qualityLibrary = HK_NULL;
	m_enableSolverDynamicScheduling = false;
	m_collisionFilter = HK_NULL;
	m_collisionQueryFilter = HK_NULL;
	m_shapeTagCodec = HK_NULL;
	m_persistentStreamAllocator = HK_NULL;
	m_largeIslandSize = 100;
	m_mergeEventsBeforeDispatch = true;
	m_maxApproachSpeedForHighQualitySolver = 1.0f;
	m_enableDeactivation = true;
	m_broadPhaseConfig = 0;
	m_leavingBroadPhaseBehavior = ON_LEAVING_BROAD_PHASE_FREEZE_BODY;

	m_collisionTolerance = 0.05f;

	m_gravity.set( 0.0f,-9.81f, 0.0f, 0.0f );
	setSolverType( SOLVER_TYPE_4ITERS_MEDIUM );
	setBroadPhaseSize( 2000.0f );
}

hknpWorldCinfo::hknpWorldCinfo( hkFinishLoadedObjectFlag flag )
:	m_materialLibrary(flag),
	m_motionPropertiesLibrary(flag),
	m_qualityLibrary(flag),
	m_broadPhaseConfig(flag),
	m_collisionFilter(flag),
	m_collisionQueryFilter(flag),
	m_shapeTagCodec(flag)
{

}

void hknpWorldCinfo::setBroadPhaseSize( hkReal side )
{
	hkReal extent = side * 0.5f;
	m_broadPhaseAabb.m_min.set(-extent,-extent,-extent, 0.0f);
	m_broadPhaseAabb.m_max.set( extent, extent, extent, 0.0f);
}

void hknpWorldCinfo::setSolverType( hknpWorldCinfo::SolverType st )
{
	switch( st )
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
