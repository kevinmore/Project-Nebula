/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_DYNAMICS_CONTINUOUS_SIMULATION_H
#define HK_DYNAMICS_CONTINUOUS_SIMULATION_H

#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Physics2012/Dynamics/World/Simulation/hkpSimulation.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Types/Physics/ContactPoint/hkContactPointMaterial.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionData.h>

#include <Common/Base/Container/PointerMap/hkPointerMap.h>

#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/hkpToiEvent.h>

struct hkpConstraintSchemaInfo;
struct hkpConstraintSolverResources;

class hkpToiResourceMgr;
struct hkpProcessCollisionInput;
class hkpBroadPhase;
class hkStepInfo;
struct hkpSolverInfo;
class hkpConstraintQueryIn;
class hkpDynamicsContactMgr;

struct hkpProcessCollisionOutput;
class hkWorldTimeEvent;
struct hkpToiResources;

	// A class which supports continuous physics
	// Class hkpContinuousSimulation
class hkpContinuousSimulation : public hkpSimulation
{
	public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
		hkpContinuousSimulation( hkpWorld* world );

		~hkpContinuousSimulation();

		static hkpSimulation* HK_CALL create( hkpWorld* world );

			// TOI events are reported by predictive collision detection performed during a PSI step or at the end of
			// code handling TOI events themselves.
			//
			// Handling TOIs:
			// Each new TOI event is appended onto an unsorted list of events. When handling events, the earliest one is chosen to
			// be handled. Handling a TOI event consists of backstepping an appropriate set of bodies, calculating proper collision
			// response, reintegrating bodies transforms and re-running collision detection. Therefore, this may result in removal
			// of some of the previously reported events, and addition of new ones.
			//
			// Executing simulation:
			// Current frame time and frameDeltaTime define the 'current time' or 'result' for the simulation. All TOI events which
			// have a time-stamp earlier than that time must be handled.
			/// ###ACCESS_CHECKS###( [m_world,HK_ACCESS_RW] );
		virtual hkpStepResult advanceTime();


			// implementation of hkpSimulation::reintegrateAndRecollideEntities
			/// ###ACCESS_CHECKS###( [world,HK_ACCESS_RW] );
		virtual void reintegrateAndRecollideEntities( hkpEntity** entityBatch, int numEntities, hkpWorld* world, int reintegrateRecollideMode );

		//
		// Internal
		//

			// Warps internal time variables by the specified value.
			// Here: all TOI-events are processed.
		virtual void warpTime( hkReal warpDeltaTime );

			// Calls hkpSimulation's version (invalidate TIM's + clear Manifolds) + removes TOI events.
		virtual void resetCollisionInformationForEntities( hkpEntity** entities, int numEntities, hkpWorld* world, enum ResetCollisionInformation resetInfo = hkpSimulation::RESET_ALL);

			// Checks that there is no TOI Events relating to the entities.
		virtual void assertThereIsNoCollisionInformationForEntities( hkpEntity** entities, int numEntities, hkpWorld* world );

			// Removes TOI Events created by the agent.
			/// ###ACCESS_CHECKS###( [hkpGetRigidBody(agent->m_collidable[0])->getWorld(),HK_ACCESS_RW] );
		virtual void removeCollisionInformationForAgent( hkpAgentNnEntry* agent );

			// Asserts when TOI Events created by the agent are found.
		virtual void assertThereIsNoCollisionInformationForAgent( hkpAgentNnEntry* agent );

	public:

			// Broadphase continuous collision detection for a set of hkEntities
	    void collideEntitiesBroadPhaseContinuous ( hkpEntity** entities, int numEntities, hkpWorld* world, hkChar* exportFinished = HK_NULL);

			// Broadphase continuous collision detection for a set of hkEntities, which only reports the pairs
		void collideEntitiesBroadPhaseContinuousFindPairs( hkpEntity** entities, int numEntities, hkpWorld* world, hkArray<hkpBroadPhaseHandlePair>& delPairs, hkArray<hkpBroadPhaseHandlePair>& newPairs );

			// Narrowphase continuous collide detection an entire hkpSimulationIsland
		void collideIslandNarrowPhaseContinuous  ( hkpSimulationIsland* isle, const hkpProcessCollisionInput& input);

			// Used by reintegrateAndRecollideEntityBatchImmediately
		void collideEntitiesNarrowPhaseContinuous( hkpEntity** entities, int numEntities, const hkpProcessCollisionInput& input);

	protected:
			// Callback function. Runs continuous collision detection for an agent entry.
		void processAgentCollideContinuous(hkpAgentNnEntry* entry, const hkpProcessCollisionInput& processInput, hkpProcessCollisionOutput& processOutput);

			// Performs continuous collision detection.
			// The only difference (relative to hkpSimulation version) is usage of
			// predictive/continuous implementations of Broad-/Narrowphase functions.
		virtual void collideInternal( const hkStepInfo& stepInfoIn );

			/// ###ACCESS_CHECKS###( [m_world,HK_ACCESS_RW] );
		hkpStepResult advanceTimeInternal();

	public:
			//
			// TOI events adding/removing
			//

			// Constructs a TOI-event information struct and appends it to the hkpContinuousSimulation::m_toievents list.
		void addToiEvent(const hkpProcessCollisionOutput& result, const hkpAgentNnEntry& entry);

	protected:
			// XXX called by remove entity batch - remove ToiEventsOfEntity is in worldOperationUtil
			/// ###ACCESS_CHECKS###( [entities[0]->getWorld(),HK_ACCESS_RW] );
		void removeToiEventsOfEntities( hkpEntity** entities, int numEntities );

			/// ###ACCESS_CHECKS###( [entity->getWorld(),HK_ACCESS_RW] );
		void removeToiEventsOfEntity( hkpEntity* entity);

	protected:

		HK_FORCE_INLINE void fireToiEventRemoved( hkpToiEvent& event );

			// Objects of this class lock the world for critical operations for their lifetime.
		class LockCriticalOperationsScope;

			// handle all TOIs till a given time
		hkpStepResult handleAllToisTill( hkTime minTime );

	protected:
			//
			// (Re) Colliding chosen entities of an island -- used in simulateToi only (might be slightly faster than hkpBackstepSimulation::collideEntitiesContinuous)
			//
			
#if ! defined (HK_PLATFORM_SPU)
			// Note: hkpEntities containing non-continuous agent entries, and simplified TOI agent entries are put on the entitiesNeedingPsiCollisionDetection list, and the corresponding agents are not processed by this call.
			// Internal: Calls broad and narrowphase for selected entities, which are assumed to all belong to the same hkpSimulationIsland.
			// This performs island merging in between broad-phase and narrow-phase collision detection.
		HK_FORCE_INLINE void collideEntitiesOfOneIslandContinuous_toiOnly( hkpEntity** entities, int numEntities, hkpWorld* world, const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection );

			// Internal: Narrowphase continuous collision detection of a set of hkEntities belonging to one hkpSimulationIsland.
			// Implementation info: It uses an internal array of flags to make sure that each agent is processed once only.
			// Note: we keep that in addition to collideEntitiesNarrowPhaseContinuous, as this doesn't use a pointerMap and is therefore slightly faster for our TOIs
			// Note: it's only used for handling TOI events.
			// Note: hkpEntities containing non-continuous agent entries, and simplified TOI agent entries are put on the entitiesNeedingPsiCollisionDetection list, and the corresponding agents are not processed by this call.
			// <todo> this should take the entityState table as input. Be aware of the hack, which prevents backsteppedAndFrozen bodies form being recollided.
		virtual void collideEntitiesOfOneIslandNarrowPhaseContinuous_toiOnly( hkpEntity** entities, int numEntities, const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection );

			// Internal: narrowphase collide of non-continuous collision entries connected to the specified bodies.
		virtual void collideEntitiesNeedingPsiCollisionDetectionNarrowPhase_toiOnly( const hkpProcessCollisionInput& input, hkPointerMap<hkUint32, hkpEntity*>& entitiesNeedingPsiCollisionDetection );


			// Internal: Removes TOI events.
			// Removes those TOI events from m_toiEvents list, which are related to one of the activated bodies.
		HK_FORCE_INLINE void deleteToiEventsForActiveEntitiesInIsland( const hkpSimulationIsland* island, const hkFixedArray<hkUchar>& entityState, hkArray<hkpToiEvent>& toiEvents );
#endif

	public:

#if !defined(HK_PLATFORM_SPU)
		HK_FORCE_INLINE void waitForSolverExport(hkChar* exportFinished);
		HK_FORCE_INLINE void removeAndAddPairs(hkpWorld* world, hkpEntity** entities, hkArray<hkpBroadPhaseHandlePair>& delPairs, hkArray<hkpBroadPhaseHandlePair>& newPairs);
#endif

	protected:

			// This performs simple backstepping for debris-quality object involved in the TOI.
			// Object of higher quality are left intact, and don't require reintegrating & recolliding.
			// Agents need to have TIM's invalidated and need to be processed at the end of TOI handling.
			// \param rotateNormal set by the contactPointCallback for this TOI.
		void handleSimpleToi( hkpWorld* world, hkpToiEvent& event, hkReal physicsDeltaTime, hkReal rotateNormal );

			// This function queries the toiResourceMgr for resources, conducts setup of resource and solver variables.
			// It calls localizedSolveToi, which returns a list of activeEntities. For those entities collision detection is then run,
			// potentially generating new TOI events. Finally the toiResourcesMgr is called to free allocated resources. And
			// any pending operations are run.
			// \param rotateNormal set by the contactPointCallback for this TOI.
		virtual void simulateToi( hkpWorld* world, hkpToiEvent& event, hkReal physicsDeltaTime, hkReal rotateNormal );

	public:

		enum CollisionQualityOverride
		{
			PROCESS_NORMALLY,
			DISABLE_TOIS,
			DISABLE_TOIS_AND_EXPAND_MANIFOLD
		};

		void processAgentNnEntries( hkpAgentNnEntry *const * entries, int numEntries, const hkpProcessCollisionInput& collisionInput, hkpSimulationIsland* island, CollisionQualityOverride collisionQualityOverride );

		//
		//	Predictive Collision Data
		//

	public:
			// Unsorted list of TOI events to be handled. Alignment because this array is transferred to spu.
		HK_ALIGN16( hkArray<hkpToiEvent>    m_toiEvents );

			// This pointer map uses Entity UID as the key to keep things deterministic.
		hkPointerMap<hkUint32, hkpEntity*> m_entitiesNeedingPsiCollisionDetection;

		hkpToiResourceMgr*    m_toiResourceMgr;


		// debug only
		int	m_toiCounter;

};

#include <Physics2012/Dynamics/World/Simulation/Continuous/hkpContinuousSimulation.inl>


#endif // HK_DYNAMICS_CONTINUOUS_SIMULATION_H

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
