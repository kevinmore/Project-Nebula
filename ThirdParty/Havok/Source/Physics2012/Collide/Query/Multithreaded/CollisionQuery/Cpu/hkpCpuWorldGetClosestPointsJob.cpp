/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpFixedBufferCdPointCollector.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuWorldGetClosestPointsJob.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuPairGetClosestPointsJob.h>

#include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseCastCollector.h>
#include <Physics2012/Collide/Filter/Group/hkpGroupFilter.h>


hkReal hkCpuWorldGetClosestPointsCollector::addBroadPhaseHandle( const hkpBroadPhaseHandle* broadphaseHandle, int castIndex )
{
	const hkpCollidable* collidable	= static_cast<const hkpCollidable*>( static_cast<const hkpTypedBroadPhaseHandle*>(broadphaseHandle)->getOwner() );			
	const hkpShape* shape = collidable->getShape();

	if ( shape && (m_collidable != collidable) )
	{
		if ( m_filter->isCollisionEnabled( *m_collidable, *collidable ) )
		{
			hkpShapeType shapeType = shape->getType();
			hkpCollisionDispatcher::GetClosestPointsFunc getClosestPointsFunc = m_input.m_dispatcher->getGetClosestPointsFunc( m_shapeType, shapeType );
			getClosestPointsFunc( *m_collidable, *collidable, m_input, *m_castCollector);
		}
	}

	return m_castCollector->getEarlyOutDistance();
}


hkJobQueue::JobStatus HK_CALL hkCpuWorldGetClosestPointsJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& nextJobOut )
{
	HK_CHECK_FLUSH_DENORMALS();
	HK_TIMER_BEGIN("CollQueryWorldGetClosestPoints", HK_NULL);

	const hkpWorldGetClosestPointsJob& worldGetClosestPointsJob = reinterpret_cast<hkpWorldGetClosestPointsJob&>(nextJobOut);

	// Create cast collector.
	// Note: we will properly initialize the collector's buffer and capacity individually for each command right before calling castAabb().
	hkpFixedBufferCdPointCollector castCollector(HK_NULL, 1);

	// Create broadphase collector and init some persistent and/or command-independent values.
	hkCpuWorldGetClosestPointsCollector broadPhaseCollector;
	{
		broadPhaseCollector.m_filter = reinterpret_cast<const hkpGroupFilter*>( static_cast<const hkpCollisionFilter*>( worldGetClosestPointsJob.m_collisionInput->m_filter.val() ) );
		broadPhaseCollector.m_castCollector					= &castCollector;

		// Transfer the 'hkpCollisionInput' part of the input.
		*((hkpCollisionInput*)&broadPhaseCollector.m_input)	= *worldGetClosestPointsJob.m_collisionInput;

		broadPhaseCollector.m_input.m_config				= worldGetClosestPointsJob.m_collisionInput->m_config;
	}

	const hkpBroadPhase* broadphase	= worldGetClosestPointsJob.m_broadphase;
	HK_ASSERT( 0x70ccbb3c, broadphase );


	// Process all commands.
	for ( int i = 0; i < worldGetClosestPointsJob.m_numCommands; i++ )
	{
		hkpWorldGetClosestPointsCommand& command = const_cast<hkpWorldGetClosestPointsCommand&>( worldGetClosestPointsJob.m_commandArray[i] );

		// Properly initialize the cast collector's buffer and capacity. This will also reset the collector's 'earlyOutDistance'.
		new (&castCollector) hkpFixedBufferCdPointCollector(command.m_results, command.m_resultsCapacity);

		// Query broadphase for any overlapping objects
		{
			hkAabb aabb;

			const hkpShape*			shape		= command.m_collidable->getShape();
			const hkTransform&		transform	= command.m_collidable->getTransform();

			// the AABBs in the broadphase are already expanded by getCollisionInput()->getTolerance() * 0.5f, so we only have to
			// increase our AABB by the restTolerance
			hkReal restTolerance = worldGetClosestPointsJob.m_tolerance - worldGetClosestPointsJob.m_collisionInput->getTolerance() * 0.5f;
			shape->getAabb(transform, restTolerance, aabb);

			// set commend dependent data and query the broadphase
			{
				broadPhaseCollector.m_collidable					= command.m_collidable;
				broadPhaseCollector.m_shapeType						= shape->getType();
				broadPhaseCollector.m_input.setTolerance(worldGetClosestPointsJob.m_tolerance);

				broadphase->querySingleAabbWithCollector( aabb, &broadPhaseCollector );
			}
		}

		// Write back the # of results (the actual results have already been written directly to m_results by the cast collector).
		command.m_numResultsOut = castCollector.m_numPoints;
	}

	HK_TIMER_END();

	return jobQueue.finishJobAndGetNextJob( &nextJobOut, nextJobOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
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
