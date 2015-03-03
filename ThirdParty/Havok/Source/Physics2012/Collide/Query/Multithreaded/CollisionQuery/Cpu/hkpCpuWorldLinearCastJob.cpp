/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpFixedBufferCdPointCollector.h>
#include <Physics2012/Collide/Filter/Group/hkpGroupFilter.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>

#include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseCastCollector.h>

#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuWorldLinearCastJob.h>


class hkCpuWorldLinearCastCollector : public hkpBroadPhaseCastCollector
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_COLLIDE, hkCpuWorldLinearCastCollector );

	protected:

		hkReal addBroadPhaseHandle( const hkpBroadPhaseHandle* broadPhaseHandle, int castIndex )
		{
			const hkpCollidable*	collidable	= static_cast<hkpCollidable*>( static_cast<const hkpTypedBroadPhaseHandle*>(broadPhaseHandle)->getOwner() );
			const hkpShape*			shape		= collidable->getShape();

			if ( shape && (m_collidable != collidable) )
			{
				if ( m_filter->isCollisionEnabled( *m_collidable, *collidable ) )
				{
					hkpShapeType shapeType = shape->getType();
					hkpCollisionDispatcher::LinearCastFunc linearCastFunc = m_input.m_dispatcher->getLinearCastFunc( m_shapeType, shapeType );
					linearCastFunc( *m_collidable, *collidable, m_input, *m_castCollector, HK_NULL ); 
				}
			}

			return m_castCollector->getEarlyOutDistance();
		}

	public:

		// global, command-independent data
		const hkpGroupFilter*			m_filter;
		hkpFixedBufferCdPointCollector*	m_castCollector;

		// command-dependent data
		const hkpCollidable*			m_collidable;	// the collidable to be cast
		hkpShapeType					m_shapeType;
		hkpLinearCastCollisionInput		m_input;
};


hkJobQueue::JobStatus HK_CALL hkCpuWorldLinearCastJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& nextJobOut )
{
	HK_CHECK_FLUSH_DENORMALS();
	HK_TIMER_BEGIN("CollQueryWorldLinearCast", HK_NULL);

	const hkpWorldLinearCastJob& worldLinearCastJob = reinterpret_cast<hkpWorldLinearCastJob&>(nextJobOut);
	HK_ASSERT( 0x5e198b8f, worldLinearCastJob.m_broadphase );
	// Create cast collector.
	// Note: we will properly initialize the collector's buffer and capacity individually for each command right before calling castAabb().
	hkpFixedBufferCdPointCollector castCollector(HK_NULL, 1);

	// Create broad phase collector and init some persistent and/or command-independent values.
	hkCpuWorldLinearCastCollector broadPhaseCollector;
	{
		broadPhaseCollector.m_filter						= reinterpret_cast<const hkpGroupFilter*>( static_cast<const hkpCollisionFilter*>( worldLinearCastJob.m_collisionInput->m_filter.val() ) );
		broadPhaseCollector.m_castCollector					= &castCollector;

		// Transfer the 'hkpCollisionInput' part of the input.
		*((hkpCollisionInput*)&broadPhaseCollector.m_input)	= *worldLinearCastJob.m_collisionInput;

		broadPhaseCollector.m_input.m_config				= worldLinearCastJob.m_collisionInput->m_config;
	}

	// Process all commands.
	{
		for (int i = 0; i < worldLinearCastJob.m_numCommands; i++)
		{
			hkpWorldLinearCastCommand& command = const_cast<hkpWorldLinearCastCommand&>( worldLinearCastJob.m_commandArray[i] );

			// Properly initialize the cast collector's buffer and capacity. This will also reset the collector's 'earlyOutDistance'.
			new (&castCollector) hkpFixedBufferCdPointCollector(command.m_results, command.m_resultsCapacity);

			// Calculate the cast's path.
			hkVector4 castPath;
			castPath.setSub( command.m_input.m_to, command.m_collidable->getTransform().getTranslation() );
			
			// Set the command-dependent values that are needed by the collector during the addBroadPhaseHandle() callback.
			{
				broadPhaseCollector.m_collidable					= command.m_collidable;
				broadPhaseCollector.m_shapeType						= command.m_collidable->getShape()->getType();
				broadPhaseCollector.m_input.m_maxExtraPenetration	= command.m_input.m_maxExtraPenetration;
				broadPhaseCollector.m_input							. setPathAndTolerance( castPath, command.m_input.m_startPointTolerance );
			}


			{
				// Create cast input
				hkpBroadPhase::hkpCastAabbInput castInput;
				{
					hkAabb aabb;
					command.m_collidable->getShape()->getAabb( command.m_collidable->getTransform(), command.m_input.m_startPointTolerance, aabb );
					
					castInput.m_from		  . setInterpolate( aabb.m_min, aabb.m_max, hkSimdReal_Inv2 );
					castInput.m_to			  . setAdd( castInput.m_from, castPath );
					castInput.m_halfExtents	  . setSub( aabb.m_max, aabb.m_min );
					castInput.m_halfExtents	  . mul( hkSimdReal_Inv2 );
					castInput.m_aabbCacheInfo = HK_NULL;
				}

				// Query the broad phase
				worldLinearCastJob.m_broadphase->castAabb( castInput, broadPhaseCollector );
			}

			// Write back the # of results (the actual results have already been written directly to m_results by the cast collector).
			command.m_numResultsOut = castCollector.m_numPoints;
		}
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
