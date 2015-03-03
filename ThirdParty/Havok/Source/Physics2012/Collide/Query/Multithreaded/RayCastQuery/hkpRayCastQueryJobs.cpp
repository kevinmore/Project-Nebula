/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>

#if defined (HK_PLATFORM_HAS_SPU)
#	include <Physics2012/Collide/Query/Multithreaded/Spu/hkpSpuConfig.h>
#	include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>
#endif


// ===============================================================================================================================================================================================
//  SHAPE RAYCAST
// ===============================================================================================================================================================================================

// Size of hkpShapeRayCastCommand seems to have increased. Think about reducing the maximum job granularity to avoid having to use getFromMainMemoryLarge() when bringing in the commands on SPU.
HK_COMPILE_TIME_ASSERT( (int)hkpShapeRayCastJob::MAXIMUM_NUMBER_OF_COMMANDS_PER_TASK * sizeof(hkpShapeRayCastCommand) <= 0x4000 );


#if !defined(HK_PLATFORM_SPU)
void hkpShapeRayCastJob::setRunsOnSpuOrPpu()
{
#if defined (HK_PLATFORM_HAS_SPU)
	{
		for (int commandIdx = 0; commandIdx < m_numCommands; commandIdx++)
		{
			const hkpShapeRayCastCommand* command = &m_commandArray[commandIdx];
			{
				for (int collidableIdx = 0; collidableIdx < command->m_numCollidables; collidableIdx++)
				{
					hkpCollidable* collidable = const_cast<hkpCollidable*>(command->m_collidables[collidableIdx]);
					if (collidable->m_forceCollideOntoPpu == hkpCollidable::FORCE_PPU_SHAPE_UNCHECKED)
					{
						collidable->setShapeSizeForSpu();
					}

					if (collidable->m_forceCollideOntoPpu != 0)
					{
						HK_WARN(0xaf15e144, "There's at least one shape in the job that is not supported on the SPU. The job has to therefore move to the PPU. Performance loss likely. Consider moving the unsupported shape(s) into a separate job.");
						m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
						return;
					}
				}
			}
		}
		m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
	}
#endif
}
#endif


// ===============================================================================================================================================================================================
//  WORLD RAYCAST
// ===============================================================================================================================================================================================

// Size of hkpWorldRayCastCommand seems to have increased. Think about reducing the maximum job granularity to avoid having to use getFromMainMemoryLarge() when bringing in the commands on SPU.
HK_COMPILE_TIME_ASSERT( (int)hkpWorldRayCastJob::MAXIMUM_NUMBER_OF_COMMANDS_PER_TASK * sizeof(hkpWorldRayCastCommand) <= 0x4000 );


#if !defined(HK_PLATFORM_SPU)
void hkpWorldRayCastJob::setRunsOnSpuOrPpu()
{
#if defined (HK_PLATFORM_HAS_SPU)
	if( m_broadphase )
	{
		const hkpBroadPhase* implBp = m_broadphase->getCapabilityDelegate(hkpBroadPhase::SUPPORT_SPU_RAY_CAST);
		if ( !implBp )
		{
			HK_WARN(0xaf35e144, "This broadphase query is not supported on SPU. Moving job onto PPU.");
			m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
		}

		m_broadphase = implBp;
	}
	else
	{
		m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
	}
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
