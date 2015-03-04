/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>

#if defined (HK_PLATFORM_HAS_SPU)
#	include <Physics2012/Collide/Query/Multithreaded/Spu/hkpSpuConfig.h>
#	include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>
#endif


// ===============================================================================================================================================================================================
//  PAIR LINEAR CAST
// ===============================================================================================================================================================================================

// Size of hkpPairLinearCastCommand has to be a multiple of 16.
#if !defined(HK_COMPILER_MWERKS)
HK_COMPILE_TIME_ASSERT( (sizeof(hkpPairLinearCastCommand) & 0xf) == 0 );
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpPairLinearCastJob::setRunsOnSpuOrPpu()
{
#if defined (HK_PLATFORM_HAS_SPU)
	{
		for (int commandIdx = 0; commandIdx < m_numCommands; commandIdx++)
		{
			const hkpPairLinearCastCommand* command = &m_commandArray[commandIdx];

			const hkpCollidable* collidables[2] = {command->m_collidableA, command->m_collidableB};
			for (int collIdx = 0; collIdx < 2; collIdx++)
			{
				hkpCollidable* collidable = const_cast<hkpCollidable*>(collidables[collIdx]);
				if (collidable->m_forceCollideOntoPpu == hkpCollidable::FORCE_PPU_SHAPE_UNCHECKED)
				{
					collidable->setShapeSizeForSpu();
				}
			}

			if ((command->m_collidableA->m_forceCollideOntoPpu != 0) || (command->m_collidableB->m_forceCollideOntoPpu != 0))
			{
				HK_WARN_ONCE(0xaf15e142, "There's at least one shape in the job that is not supported on the SPU. The job has to therefore move to the PPU. Performance loss likely. Consider moving the unsupported shape(s) into a separate job.");
				m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
				return;
			}
		}
	}

	m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
	return;
#endif
}
#endif


// ===============================================================================================================================================================================================
//  WORLD LINEAR CAST
// ===============================================================================================================================================================================================

// Size of hkpWorldLinearCastCommand seems to have increased. Think about reducing the maximum job granularity to avoid having to use getFromMainMemoryLarge() when bringing in the commands on SPU.
HK_COMPILE_TIME_ASSERT( (int)hkpWorldLinearCastJob::MAXIMUM_NUMBER_OF_COMMANDS_PER_TASK * sizeof(hkpWorldLinearCastCommand) <= 0x4000 );

#if !defined(HK_PLATFORM_SPU)
void hkpWorldLinearCastJob::setRunsOnSpuOrPpu()
{
#if defined (HK_PLATFORM_HAS_SPU)
	HK_ASSERT(0x74bcfa34, m_broadphase );
	const hkpBroadPhase* implBp = m_broadphase->getCapabilityDelegate(hkpBroadPhase::SUPPORT_SPU_LINEAR_CAST);
	if ( !implBp )
	{
		HK_WARN_ONCE(0xaf35e144, "This broadphase query is not supported on SPU. Moving job onto PPU.");
		m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
		return;
	}

	m_broadphase = implBp;

	{
		for (int commandIdx = 0; commandIdx < m_numCommands; commandIdx++)
		{
			const hkpWorldLinearCastCommand* command = &m_commandArray[commandIdx];

			hkpCollidable* collidable = const_cast<hkpCollidable*>(command->m_collidable);
			if (collidable->m_forceCollideOntoPpu == hkpCollidable::FORCE_PPU_SHAPE_UNCHECKED)
			{
				collidable->setShapeSizeForSpu();
			}

			if (collidable->m_forceCollideOntoPpu != 0)
			{
				HK_WARN_ONCE(0xaf15e143, "There's at least one shape in the job that is not supported on the SPU. The job has to therefore move to the PPU. Performance loss likely. Consider moving the unsupported shape(s) into a separate job.");
				m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
				return;
			}
		}
	}

	m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
#endif
}
#endif


// ===============================================================================================================================================================================================
//  MOPP AABB
// ===============================================================================================================================================================================================


// ===============================================================================================================================================================================================
//  PAIR GET CLOSEST POINTS
// ===============================================================================================================================================================================================

// Size of hkpPairGetClosestPointsCommand has to be a multiple of 16.
#if !defined(HK_COMPILER_MWERKS) && (HK_NATIVE_ALIGNMENT==16)
HK_COMPILE_TIME_ASSERT( (sizeof(hkpPairGetClosestPointsCommand) & 0xf) == 0 );
#endif

#if !defined(HK_PLATFORM_SPU)
void hkpPairGetClosestPointsJob::setRunsOnSpuOrPpu()
{
#if defined (HK_PLATFORM_HAS_SPU)
	{
		for (int commandIdx = 0; commandIdx < m_numCommands; commandIdx++)
		{
			const hkpPairGetClosestPointsCommand* command = &m_commandArray[commandIdx];

			const hkpCollidable* collidables[2] = {command->m_collidableA, command->m_collidableB};
			for (int collIdx = 0; collIdx < 2; collIdx++)
			{
				hkpCollidable* collidable = const_cast<hkpCollidable*>(collidables[collIdx]);
				if (collidable->m_forceCollideOntoPpu == hkpCollidable::FORCE_PPU_SHAPE_UNCHECKED)
				{
					collidable->setShapeSizeForSpu();
				}
			}

			if ((command->m_collidableA->m_forceCollideOntoPpu != 0) || (command->m_collidableB->m_forceCollideOntoPpu != 0))
			{
				HK_WARN_ONCE(0xaf15e141, "There's at least one shape in the job that is not supported on the SPU. The job has to therefore move to the PPU. Performance loss likely. Consider moving the unsupported shape(s) into a separate job.");
				
				m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
				return;
			}
		}
	}
	m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
#endif
}
#endif

// ===============================================================================================================================================================================================
//  WORLD GET CLOSEST POINTS
// ===============================================================================================================================================================================================

#if !defined(HK_PLATFORM_SPU)
void hkpWorldGetClosestPointsJob::setRunsOnSpuOrPpu()
{
#if defined (HK_PLATFORM_HAS_SPU)
	HK_ASSERT(0xc9e51c3, m_broadphase );
	const hkpBroadPhase* implBp = m_broadphase->getCapabilityDelegate(hkpBroadPhase::SUPPORT_SPU_CLOSEST_POINTS);
	if ( !implBp )
	{
		HK_WARN_ONCE(0xaf35e145, "The broadphase do not have support for this query on SPU. Moving job onto PPU.");
		m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
		return;
	}

	m_broadphase = implBp;

	{
		for (int commandIdx = 0; commandIdx < m_numCommands; commandIdx++)
		{
			const hkpWorldGetClosestPointsCommand* command = &m_commandArray[commandIdx];

			hkpCollidable* collidable = const_cast<hkpCollidable*>(command->m_collidable);
			if (collidable->m_forceCollideOntoPpu == hkpCollidable::FORCE_PPU_SHAPE_UNCHECKED)
			{
				collidable->setShapeSizeForSpu();
			}

			if (collidable->m_forceCollideOntoPpu != 0)
			{
				HK_WARN_ONCE(0xaf15e143, "There's at least one shape in the job that is not supported on the SPU. The job has to therefore move to the PPU. Performance loss likely. Consider moving the unsupported shape(s) into a separate job.");
				m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
				return;
			}
		}
	}

	m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
#endif
}
#endif

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
