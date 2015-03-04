/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Cpu/hkpCpuShapeRaycastJob.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Util/hkpCollisionQueryUtil.h>

#include <Physics2012/Collide/Query/Collector/RayCollector/hkpFixedBufferRayHitCollector.h>

#if !defined(HK_PLATFORM_CTR) && !defined(HK_COMPILER_HAS_INTRINSICS_NEON)
HK_COMPILE_TIME_ASSERT((sizeof(hkpWorldRayCastOutput) & 0x0f) == 0); // we're using hkString::memCpy16NonEmpty below
#endif

HK_FORCE_INLINE void HK_CALL hkCpuShapeRayCastJobProcessCommand(const hkpShapeRayCastCommand& raycastCommand)
{
	hkpShapeRayCastCommand& command = const_cast<hkpShapeRayCastCommand&>( raycastCommand );

	hkpShapeRayCastInput	input = command.m_rayInput;	// make a copy of rayInput as we will modify its from/to vectors!
	hkpWorldRayCastOutput	output;


	hkpWorldRayCastOutput*	nextFreeResult = command.m_results;
	{
		hkpFixedBufferRayHitCollector fixedBufferRayHitCollector(command.m_results, command.m_resultsCapacity);
		registerFixedBufferRayHitCollectorAddRayHitCallbackFunction(hkpFixedBufferRayHitCollector::addRayHitImplementation);

		for (int i = 0; i < command.m_numCollidables; i++)
		{
#if (defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_XBOX360))
			// prefetch next collidable
			if ( i < command.m_numCollidables-1 )
			{
				hkMath::forcePrefetch<sizeof(hkpCollidable)>( command.m_collidables[i+1] );
			}
#endif

			const hkpCollidable* collidable = command.m_collidables[i];

			const hkpShape*		shape		= collidable->getShape();

			// Ignore AABB phantoms and other shapeless creatures.
			if( shape )
			{
				const hkTransform&	transform	= collidable->getTransform();

				// Transform ray's start/end points to shape-space
				input.m_to  ._setTransformedInversePos( transform, command.m_rayInput.m_to   );
				input.m_from._setTransformedInversePos( transform, command.m_rayInput.m_from );

				// Prepare output for raycasting.
				output.reset();

				if (!command.m_useCollector)
				{
					// Raycast.
					hkBool hit = shape->castRay( input, output );

					// If we have a hit, we try to insert the result into the array. This might drop the furthest hit from the array.
					if ( hit )
					{
						hkpWorldRayCastOutput* insertAt = hkpShapeRayCastJobUtil::getNextFreeResult(&command, command.m_results, nextFreeResult, output.m_hitFraction);

						// Insert if there's still room left in the array OR our current hit is closer than the furthest hit in the array.
						if ( insertAt )
						{
							// Transform results back to world space
							output.m_normal._setRotatedDir( transform.getRotation(), output.m_normal );

							// transfer results into result array;
							*insertAt = output;

							// not assigned yet
							insertAt->m_rootCollidable = collidable;
						}
					}
				}
				else
				{
					shape->castRayWithCollector(input, *collidable, fixedBufferRayHitCollector);
				}
			}
		}

		if (command.m_useCollector)
		{
			command.m_numResultsOut = fixedBufferRayHitCollector.m_numOutputs;
		}
	}
}


hkJobQueue::JobStatus HK_CALL hkCpuShapeRayCastJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& nextJobOut )
{
	HK_CHECK_FLUSH_DENORMALS();
	HK_TIMER_BEGIN("CollQueryShapeRayCast", HK_NULL);

	const hkpShapeRayCastJob& raycastJob = reinterpret_cast<hkpShapeRayCastJob&>(nextJobOut);

	{
		for (int i=0; i < raycastJob.m_numCommands; i++)
		{
			hkCpuShapeRayCastJobProcessCommand(raycastJob.m_commandArray[i]);
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
