/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_DYNAMICS_MULTITHREADED_SIMULATION_H
#define HK_DYNAMICS_MULTITHREADED_SIMULATION_H

#include <Physics2012/Dynamics/World/Simulation/Continuous/hkpContinuousSimulation.h>
#include <Common/Base/Thread/JobQueue/hkJobQueue.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpBroadPhaseListener.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>
#include <Common/Base/Thread/Semaphore/hkSemaphore.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

class hkpPostCollideJob;

	// A class which supports multithreaded physics 
	// Class hkpMultiThreadedSimulation
class hkpMultiThreadedSimulation : public hkpContinuousSimulation
{
	public: 
		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
			// Initializes locks and changes the world's broadphase dispatcher
		hkpMultiThreadedSimulation( hkpWorld* world );

			// 
		~hkpMultiThreadedSimulation();

		static hkpSimulation* HK_CALL create( hkpWorld* world );

		//
		// Internal
		//

			// Checks that there is no TOI Events relating to the entities.  
			// This overrides the hkContinuousSimuation's version as it must use a critical section to access the m_toiEvents array.
		virtual void assertThereIsNoCollisionInformationForEntities( hkpEntity** entities, int numEntities, hkpWorld* world );

			// Removes TOI Events created by the agent.
			// This overrides the hkContinuousSimuation's version as it must use a critical section to access the m_toiEvents array.
		virtual void assertThereIsNoCollisionInformationForAgent( hkpAgentNnEntry* agent );


		//
		// Additional multithreading interface.
		//

		virtual hkpStepResult stepBeginSt( hkJobQueue* queue, hkReal physicsDeltaTime );

			/// ###ACCESS_CHECKS###( [m_world,HK_ACCESS_RW] );
		virtual hkpStepResult finishMtStep( hkJobQueue* queue, hkJobThreadPool* threadPool );


		//
		// Interface implementation
		//

		virtual void getMultithreadConfig( hkpMultithreadConfig& config );

		virtual void setMultithreadConfig( const hkpMultithreadConfig& config, hkJobQueue* queue );

			// All subsequent threads to call step delta time execute this function
		static hkJobQueue::JobStatus HK_CALL processNextJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& jobInOut );

#if !defined(HK_PLATFORM_SPU)
		
			/// ###ACCESS_CHECKS###( [m_world,HK_ACCESS_RW] );
		void processAgentNnEntries( hkpAgentNnEntry** entries, int numEntries, const hkpProcessCollisionInput& collisionInput, struct hkpIslandsAgentEntriesInfo& info, CollisionQualityOverride collisionQualityOverride );

			/// ###ACCESS_CHECKS###( [m_world,HK_ACCESS_RW] );
		void processAgentNnEntriesFromMultipleIslands( hkpAgentNnEntry** allEntries, int numAllEntries, struct hkpIslandsAgentEntriesInfo* islandEntriesInfos, int numInfos, const hkpProcessCollisionInput& collisionInput, CollisionQualityOverride collisionQualityOverride );

		void processAgentNnEntries_oneInfo( hkpAgentNnEntry** entries, const hkpProcessCollisionInput& collisionInput, struct hkpIslandsAgentEntriesInfo& info, hkpContinuousSimulation::CollisionQualityOverride collisionQualityOverride, hkpPostCollideJob* postCollideJobOut );
#endif

			// Constructs a TOI-event information struct and appends it to the hkpContinuousSimulation::m_toievents list. 
			// This is a multithreading-safe version.
		void addToiEventWithCriticalSectionLock(const hkpProcessCollisionOutput& result, const hkpAgentNnEntry& entry, hkCriticalSection* section );

	protected:

		virtual void collideEntitiesOfOneIslandNarrowPhaseContinuous_toiOnly( hkpEntity** entities, int numEntities, const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection );		

		virtual void collideEntitiesNeedingPsiCollisionDetectionNarrowPhase_toiOnly( const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection );

	public:

				// This class is used internally by hkpWorld to dispatch broadphase pairs to the relevant phantoms.
		class MtEntityEntityBroadPhaseListener : public hkpBroadPhaseListener
		{
			public:
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkpMultiThreadedSimulation::MtEntityEntityBroadPhaseListener);
				MtEntityEntityBroadPhaseListener(  )
					: m_simulation(HK_NULL) {}

					// Delays addition of pairs between islands if the world is locked
				virtual void addCollisionPair( hkpTypedBroadPhaseHandlePair& pair );

					// Delays removal of pairs between islands if the world is locked
				virtual void removeCollisionPair( hkpTypedBroadPhaseHandlePair& pair );

			public:
				hkpMultiThreadedSimulation* m_simulation;
		};

		class MtPhantomBroadPhaseListener : public hkpBroadPhaseListener
		{
			public:
				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkpMultiThreadedSimulation::MtPhantomBroadPhaseListener);
				MtPhantomBroadPhaseListener(  ): m_criticalSection(HK_NULL) {}

					// Adds the collision pair elements A and B if they are phantoms
				virtual void addCollisionPair( hkpTypedBroadPhaseHandlePair& pair );

					// Removes the collision pair elements A and B if they are phantoms
				virtual void removeCollisionPair( hkpTypedBroadPhaseHandlePair& pair );

			public:
				hkCriticalSection* m_criticalSection;
		};
		class MtBroadPhaseBorderListener : public hkpBroadPhaseListener
		{
			public:
				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkpMultiThreadedSimulation::MtBroadPhaseBorderListener);
				MtBroadPhaseBorderListener(  ): m_criticalSection(HK_NULL) {}

				// Adds the collision pair elements A and B if they are phantoms
				virtual void addCollisionPair( hkpTypedBroadPhaseHandlePair& pair );

				// Removes the collision pair elements A and B if they are phantoms
				virtual void removeCollisionPair( hkpTypedBroadPhaseHandlePair& pair );

			public:
				hkCriticalSection* m_criticalSection;
		};

		//
		// Helper functions
		//
	public:

		//
		// Data shared between all threads during a step
		//

		MtEntityEntityBroadPhaseListener m_entityEntityBroadPhaseListener;
		MtPhantomBroadPhaseListener      m_phantomBroadPhaseListener;
		MtBroadPhaseBorderListener       m_broadPhaseBorderListener;
		
			// if this flag is set to true, new and deleted collidablepairs between different
			// islands are added to the m_addedCrossIslandPairs and m_removedCrossIslandPairs arrays
		hkBool							 m_crossIslandPairsCollectingActive;

		hkArray<hkpTypedBroadPhaseHandlePair> m_addedCrossIslandPairs;
		hkCriticalSection m_addCrossIslandPairCriticalSection;

		hkArray<hkpTypedBroadPhaseHandlePair> m_removedCrossIslandPairs;
		hkCriticalSection m_removeCrossIslandPairCriticalSection;

		hkpMultithreadConfig m_multithreadConfig;

		int m_numActiveIslandsAtBeginningOfStep;
		int m_numInactiveIslandsAtBeginningOfStep;

		hkJobQueue* m_jobQueueHandleForToiSolve;

		HK_ALIGN(hkCriticalSection m_toiQueueCriticalSection, 64);

		HK_ALIGN(hkCriticalSection m_phantomCriticalSection, 64 );

};


#endif // HK_DYNAMICS_MULTITHREADED_SIMULATION_H

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
