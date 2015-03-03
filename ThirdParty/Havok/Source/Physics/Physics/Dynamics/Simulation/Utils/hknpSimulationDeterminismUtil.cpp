/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Simulation/Utils/hknpSimulationDeterminismUtil.h>

#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
	#include <Physics/Physics/Dynamics/Solver/Event/hknpSolverEvents.h>
	#include <Physics/Physics/Dynamics/World/Commands/hknpInternalCommands.h>
#endif


/*static*/ void hknpSimulationDeterminismUtil::check(const hkCommand* com)
{
#if defined (HK_ENABLE_DETERMINISM_CHECKS) && !defined(HK_PLATFORM_SPU)
	if (com->m_primaryType == hkCommand::TYPE_PHYSICS_EVENTS && com->m_secondaryType == hknpEventType::MANIFOLD_PROCESSED)
	{
		int exclude[] = { HK_OFFSET_OF(hknpManifoldProcessedEvent, m_manifoldCache), sizeof(hknpManifoldCollisionCache*), -1 };
		hkCheckDeterminismUtil::checkMt<hkUchar>(0xad343424, (hkUchar*) com, com->determinismGetSizeInBytes(), exclude );
	}
	else if (com->m_primaryType == hkCommand::TYPE_PHYSICS_EVENTS && com->m_secondaryType == hknpEventType::MANIFOLD_STATUS)
	{
		int exclude[] = { HK_OFFSET_OF(hknpManifoldStatusEvent, m_manifoldCache), sizeof(hknpManifoldCollisionCache*), -1 };
		hkCheckDeterminismUtil::checkMt<hkUchar>(0xad343425, (hkUchar*) com, com->determinismGetSizeInBytes(), exclude );
	}
	else if (com->m_primaryType == hkCommand::TYPE_PHYSICS_EVENTS &&
		(com->m_secondaryType == hknpEventType::MANIFOLD_PROCESSED || com->m_secondaryType == hknpEventType::CONTACT_IMPULSE || com->m_secondaryType == hknpEventType::CONTACT_IMPULSE_CLIPPED  || com->m_secondaryType == hknpEventType::MANIFOLD_STATUS) )
	{
		int exclude[] = { HK_OFFSET_OF(hknpContactSolverEvent, m_contactJacobian), sizeof(hknpMxContactJacobian*), -1 };
		hkCheckDeterminismUtil::checkMt<hkUchar>(0xad343425, (hkUchar*) com, com->determinismGetSizeInBytes(), exclude );
	}
	else if (com->m_primaryType == hkCommand::TYPE_PHYSICS_EVENTS && com->m_secondaryType == hknpEventType::CONSTRAINT_FORCE_EXCEEDED)
	{
		int exclude[] = { HK_OFFSET_OF(hknpConstraintForceExceededEvent, m_constraint), sizeof(hknpConstraint*), -1 };
		hkCheckDeterminismUtil::checkMt<hkUchar>(0xad343425, (hkUchar*) com, com->determinismGetSizeInBytes(), exclude );
	}
	else if (com->m_primaryType == hkCommand::TYPE_PHYSICS_EVENTS && com->m_secondaryType == hknpEventType::CONSTRAINT_FORCE)
	{
		int exclude[] = { HK_OFFSET_OF(hknpConstraintForceEvent, m_constraint), sizeof(hknpConstraint*), -1 };
		hkCheckDeterminismUtil::checkMt<hkUchar>(0xad343425, (hkUchar*) com, com->determinismGetSizeInBytes(), exclude );
	}
	else if (com->m_primaryType == hkCommand::TYPE_PHYSICS_INTERNAL &&
		com->m_secondaryType == hknpInternalCommand::CMD_VALIDATE_TRIGGER_VOLUME_EVENT || com->m_secondaryType == hknpInternalCommand::CMD_MOTION_WELD_TOI )
	{
		int exclude[] = { HK_OFFSET_OF(hknpContactSolverEvent, m_contactJacobian), sizeof(hknpMxContactJacobian*), -1 };
		hkCheckDeterminismUtil::checkMt<hkUchar>(0xad343425, (hkUchar*) com, com->determinismGetSizeInBytes(), exclude );
	}
	else if (com->m_primaryType == hkCommand::TYPE_PHYSICS_INTERNAL
		&& com->m_secondaryType == hknpInternalCommand::CMD_ADD_CONSTRAINT_RANGE)
	{
		int exclude[] = { HK_OFFSET_OF(hknpAddConstraintRangeCommand, m_range), sizeof(hknpConstraintSolverJacobianRange2),
			HK_OFFSET_OF(hknpAddConstraintRangeCommand, m_grid), sizeof(hknpConstraintSolverJacobianGrid*), -1 };
		hkCheckDeterminismUtil::checkMt<hkUchar>(0xad343425, (hkUchar*) com, com->determinismGetSizeInBytes(), exclude );
	}
	else
	{
		hkCheckDeterminismUtil::checkMt<hkUchar>(0xad343426, (hkUchar*) com, com->determinismGetSizeInBytes() );
	}
#endif
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
