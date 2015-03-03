/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>

#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>
#include <Common/Base/Types/Properties/hkRefCountedProperties.h>
#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>
#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#include <Common/Visualize/Container/CommandStream/DebugCommands/hkDebugCommands.h>
#include <Common/Base/Reflection/hkClass.h>

#include <Physics/Physics/Collide/BroadPhase/hknpBroadPhase.h>
#include <Physics/Physics/Collide/BroadPhase/hknpBroadPhaseConfig.h>
#include <Physics/Physics/Collide/Dispatcher/hknpCollisionDispatcher.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQueryDispatcher.h>
#include <Physics/Physics/Collide/Filter/Constraint/hknpConstraintCollisionFilter.h>
#include <Physics/Physics/Dynamics/Action/Manager/hknpActionManager.h>
#include <Physics/Physics/Dynamics/World/CacheManager/hknpCollisionCacheManager.h>
#include <Physics/Physics/Dynamics/World/Commands/hknpCommands.h>
#include <Physics/Physics/Dynamics/World/Commands/hknpInternalCommands.h>
#include <Physics/Physics/Dynamics/World/Deactivation/CdCacheFilter/hknpDeactiveCdCacheFilter.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>
#include <Physics/Physics/Dynamics/World/Events/Dispatchers/hknpEventMergeAndDispatcher.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/Simulation/SingleThreaded/hknpSingleThreadedSimulation.h>
#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpMultithreadedSimulation.h>
#include <Physics/Physics/Dynamics/Modifier/DefaultModifierSet/hknpDefaultModifierSet.h>
#include <Physics/Physics/Dynamics/World/hknpWorldShiftUtil.h>

#include <Physics/Internal/Collide/BroadPhase/Hybrid/hknpHybridBroadPhase.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.h>


#if defined(HKNP_ENABLE_SOLVER_LOG)
hkOstream* g_npDebugOstream = HK_NULL;
#endif

// Definition of the world global debug instance. See hknpTypes.h.
const hknpWorld* g_hknpWorldDebugInstance = HK_NULL;


// Extended hknpWorld functionality which doesn't need to be in the public API.
class hknpWorldEx : public hknpWorld
{
	public:

		// Allocate a body, firing a signal first if the buffer is full.
		HK_FORCE_INLINE hknpBodyId _allocateBody();

		// Allocate a motion, firing a signal first if the buffer is full.
		HK_FORCE_INLINE hknpMotionId _allocateMotion();

		// Attach a body to a motion.
		HK_FORCE_INLINE void _attachBodyToMotion(
			hknpBodyId bodyId, hknpBody& body, hknpMotionId motionId, hknpMotion& motion );

		// Detach a body from a motion.
		HK_FORCE_INLINE void _detachBodyFromMotion(
			hknpBody* HK_RESTRICT body, const hknpBodyId bodyId, hknpMotion* HK_RESTRICT motion );

		// Check consistency of active bodies.
		HK_FORCE_INLINE void _checkConsistencyOfActiveBodies();

		// Handle the activation mode of a constraint when it is awaken or enabled.
		void handleConstraintActivationMode( hknpConstraint* constraint, hknpActivationMode::Enum activationMode );

		// Remove the motion from the active list if the motion has no added bodies.
		void removeActiveMotionIfNoAddedBodies( hknpMotionId motionId );

		//
		// Simulation tasks
		//

		void preCollide( const hknpStepInput& stepInput );
		void postCollide( hknpSolverData* solverData );

		void preSolve( hknpSolverData* solverData );
		void postSolve( hknpSolverData* solverData );

		//
		// Signal/event handlers
		//

		/// hknpBodyExitedBroadPhaseEvent handler.
		/// Removes the bodies from the world.
		void removeBodiesWhichLeftTheBroadPhase( const hknpEventHandlerInput& input, const hknpEvent& event );

		/// hknpBodyExitedBroadPhaseEvent handler.
		/// Sets the linear velocity to zero and the motion properties to hknpMotionPropertiesId::FROZEN.
		void freezeBodiesWhichLeftTheBroadPhase( const hknpEventHandlerInput& input, const hknpEvent& event );

		/// Listener for additions to the material library.
		void onMaterialAddedSignal( hknpMaterialId materialId );

		/// Listener for modifications to the material library.
		/// Used to mark materials as dirty.
		void onMaterialModifiedSignal( hknpMaterialId materialId );

		/// Listener for removals from the material library.
		/// Used in debug to assert that the material is not in use by a body.
		void onMaterialRemovedSignal( hknpMaterialId materialId );

		/// Listener for additions to the motion properties library.
		/// Used in debug to make sure that the storage buffer does not get resized during simulation.
		void onMotionPropertiesAddedSignal( hknpMotionPropertiesId motionPropertiesId );

		/// Listener for removals from the motion properties library.
		/// Used in debug to assert that the motion properties is not in use by a motion.
		void onMotionPropertiesRemovedSignal( hknpMotionPropertiesId motionPropertiesId );

		/// Listener for quality modifications in the quality library.
		/// Used to mark qualities as dirty.
		void onQualityModifiedSignal( hknpBodyQualityId qualityId );

	private:

		hknpWorldEx( hknpWorldCinfo& cinfo ) : hknpWorld( cinfo ) {}
};


// A task which must be processed at the start of every collide step
class hknpPreCollideTask : public hkTask
{
	public:

		// hkTask implementation
		virtual void process() HK_OVERRIDE
		{
			hknpWorldEx* self = (hknpWorldEx*)m_world;
			self->preCollide( m_stepInput );
		}

	public:

		const hknpWorld* m_world;
		hknpStepInput m_stepInput;
};


// A task which must be processed at the end of every collide step
class hknpPostCollideTask : public hkTask
{
	public:

		// hkTask implementation
		virtual void process() HK_OVERRIDE
		{
			hknpWorldEx* self = (hknpWorldEx*)m_world;
			self->postCollide( m_solverData );
		}

	public:

		const hknpWorld* m_world;
		hknpSolverData* m_solverData;
};


// A task which must be processed at the end of every solve step
class hknpPostSolveTask : public hkTask
{
	public:

		// hkTask implementation
		virtual void process() HK_OVERRIDE
		{
			hknpWorldEx* self = (hknpWorldEx*)m_world;
			self->postSolve( m_solverData );
		}

	public:

		const hknpWorld* m_world;
		hknpSolverData* m_solverData;
};


// Helper functions
namespace
{
	// Set up a modifier manager, given a default modifier set, world info and solvers.
	static void setupModifierManager(
		const hknpWorldCinfo &cinfo,
		hknpDefaultModifierSet* defaultModifierSet,
		hknpConstraintSolver* contactSolver,
		hknpConstraintSolver* constraintAtomSolver,
		hknpModifierManager* modifierManager )
	{
		defaultModifierSet->removeReference();

		// Modifiers
		{
			modifierManager->addModifier(
				hknpBody::RAISE_MANIFOLD_PROCESSED_EVENTS | hknpBody::RAISE_MANIFOLD_STATUS_EVENTS,
				&defaultModifierSet->m_manifoldEventCreator );

			modifierManager->addModifier( hknpBody::RAISE_CONTACT_IMPULSE_EVENTS, &defaultModifierSet->m_contactImpulseEventCreator );

			modifierManager->addModifier( hknpMaterial::ENABLE_TRIGGER_VOLUME, &defaultModifierSet->m_triggerVolumeModifier );
			modifierManager->addModifier( hknpMaterial::ENABLE_RESTITUTION, &defaultModifierSet->m_restitutionModifier );
			modifierManager->addModifier( hknpMaterial::ENABLE_IMPULSE_CLIPPING, &defaultModifierSet->m_clippedImpulseEventCreator );
			modifierManager->addModifier( hknpMaterial::ENABLE_MASS_CHANGER, &defaultModifierSet->m_massChangerModifier );
			modifierManager->addModifier( hknpMaterial::ENABLE_SOFT_CONTACTS, &defaultModifierSet->m_softContactModifier );
			modifierManager->addModifier( hknpMaterial::ENABLE_SURFACE_VELOCITY, &defaultModifierSet->m_surfaceVelocityModifier );

			modifierManager->addModifier( hknpConstraint::RAISE_CONSTRAINT_FORCE_EVENTS, &defaultModifierSet->m_constraintForceEventCreator );
			modifierManager->addModifier( hknpConstraint::RAISE_CONSTRAINT_FORCE_EXCEEDED_EVENTS, &defaultModifierSet->m_constraintForceExceededEventCreator );
		}

		// Welding
		{
			modifierManager->m_neighborWeldingModifier = &defaultModifierSet->m_neighborWeldingModifier;
			modifierManager->m_motionWeldingModifier   = &defaultModifierSet->m_motionWeldingModifier;
			modifierManager->m_triangleWeldingModifier = &defaultModifierSet->m_triangleWeldingModifier;
		}

		// Collision filters
		{
			if( cinfo.m_collisionFilter )
			{
				modifierManager->setCollisionFilter( cinfo.m_collisionFilter );
			}
			if( cinfo.m_collisionQueryFilter )
			{
				modifierManager->setCollisionQueryFilter( cinfo.m_collisionQueryFilter );
			}
		}

		// Collision Detectors
		{
			modifierManager->setCollisionDetector( hknpCollisionCacheType::CONVEX_CONVEX, &defaultModifierSet->m_cvxCvxCdDetector ); // this one is implicit
			modifierManager->setCollisionDetector( hknpCollisionCacheType::SET_SHAPE_KEY_A, &defaultModifierSet->m_setShakeKeyACdDetector ); // this one is implicit
			modifierManager->setCollisionDetector( hknpCollisionCacheType::CONVEX_COMPOSITE, &defaultModifierSet->m_cvxCompositeCdDetector );
			modifierManager->setCollisionDetector( hknpCollisionCacheType::COMPOSITE_COMPOSITE, &defaultModifierSet->m_compositeCompositeCdDetector );
			modifierManager->setCollisionDetector( hknpCollisionCacheType::DISTANCE_FIELD, &defaultModifierSet->m_signedDistanceFieldCdDetector );
		}

		// Solvers
		{
			modifierManager->setSolver( hknpConstraintSolverType::CONTACT_CONSTRAINT_SOLVER, contactSolver );
			modifierManager->setSolver( hknpConstraintSolverType::ATOM_CONSTRAINT_SOLVER, constraintAtomSolver );
		}
	}

}	// anonymous namespace


HK_FORCE_INLINE hknpBodyId hknpWorldEx::_allocateBody()
{
	if( m_bodyManager.getNumAllocatedBodies() == m_bodyManager.getCapacity() )
	{
		// Give the user an opportunity to reallocate the buffer, free bodies etc.
		// If nothing is done, allocateBody() will fail and return an invalid ID.
		m_signals.m_bodyBufferFull.fire( this, &m_bodyManager );
	}
	return m_bodyManager.allocateBody();
}

HK_FORCE_INLINE hknpMotionId hknpWorldEx::_allocateMotion()
{
	if( m_motionManager.getNumAllocatedMotions() == m_motionManager.getCapacity() )
	{
		// Give the user an opportunity to reallocate the buffer, free motions etc.
		// If nothing is done, allocateMotion() will fail and return an invalid ID.
		m_signals.m_motionBufferFull.fire( this, &m_motionManager );
	}
	return m_motionManager.allocateMotion();
}

HK_FORCE_INLINE void hknpWorldEx::_attachBodyToMotion(
	hknpBodyId bodyId, hknpBody& body, hknpMotionId motionId, hknpMotion& motion )
{
	body.m_motionId = motionId;

	if( !motion.m_firstAttachedBodyId.isValid() )
	{
		motion.m_firstAttachedBodyId = bodyId;
		body.m_nextAttachedBodyId = bodyId;
	}
	else
	{
		hknpBody& firstBodyOfMotion = m_bodyManager.accessBody( motion.m_firstAttachedBodyId );
		body.m_nextAttachedBodyId = firstBodyOfMotion.m_nextAttachedBodyId;
		firstBodyOfMotion.m_nextAttachedBodyId = bodyId;
	}
}

HK_FORCE_INLINE void hknpWorldEx::_detachBodyFromMotion(
	hknpBody* HK_RESTRICT body, const hknpBodyId bodyId, hknpMotion* HK_RESTRICT motion )
{
	// search for the *previous* attached body
	hknpBodyId p = body->m_nextAttachedBodyId;
	while (1)
	{
		const hknpBody& pBody = m_bodyManager.getBody(p);
		if ( pBody.m_nextAttachedBodyId == bodyId )
		{
			break;
		}
		p = pBody.m_nextAttachedBodyId;
	}

	hknpBody& pBody = m_bodyManager.accessBody(p);

	if ( &pBody == body ) // If the previous body is the same there is only one body attached
	{
		motion->m_firstAttachedBodyId = hknpBodyId::invalid();
	}
	else
	{
		pBody.m_nextAttachedBodyId = body->m_nextAttachedBodyId;
		motion->m_firstAttachedBodyId = p;
	}
	body->m_nextAttachedBodyId = bodyId;
}

HK_FORCE_INLINE void hknpWorldEx::_checkConsistencyOfActiveBodies()
{
#ifdef HK_DEBUG
	if( areConsistencyChecksEnabled() )
	{
		const hkArray<hknpBodyId>& bodies = getActiveBodies();
		for( int i=0, ei=bodies.getSize(); i<ei; i++ )
		{
			checkBodyConsistency( this, bodies[i] );
		}
	}
#endif
}


hknpWorld::hknpWorld( const hknpWorldCinfo& cinfo )
	: m_bodyManager(HK_NULL, cinfo.m_userBodyBuffer, cinfo.m_bodyBufferCapacity )
	, m_motionManager()
{
	hknpWorldEx* self = (hknpWorldEx*)this;

	m_bodyManager.m_world = this;
	m_simulationStage = SIMULATION_DONE;

	m_gravity = cinfo.m_gravity;
	m_collisionTolerance.setFromFloat( cinfo.m_collisionTolerance );

	//
	//	Stream allocator
	//
	{
		HK_ASSERT2( 0x13bdc47a, cinfo.m_persistentStreamAllocator, "A stream allocator must be provided." );
		m_persistentStreamAllocator = cinfo.m_persistentStreamAllocator;
	}

	//
	//	Solver info
	//
	{
		m_solverInfo.m_unitScale.setFromFloat( cinfo.m_unitScale );
		m_solverInfo.m_collisionAccuracy.setFromFloat( cinfo.m_unitScale * cinfo.m_relativeCollisionAccuracy);
		m_solverInfo.setTauAndDamping( cinfo.m_solverTau, cinfo.m_solverDamp );
		m_solverInfo.setStepInfo( m_gravity, cinfo.m_collisionTolerance, cinfo.m_defaultSolverTimestep, cinfo.m_solverIterations, cinfo.m_solverMicrosteps );
		m_solverInfo.m_maxApproachSpeedForHighQualitySolver = cinfo.m_maxApproachSpeedForHighQualitySolver;
	}

	//
	//	Material library
	//
	{
		if (!cinfo.m_materialLibrary)
		{
			// Create a library if one was not provided
			m_materialLibrary.setAndDontIncrementRefCount( new hknpMaterialLibrary() );
		}
		else
		{
			// Otherwise use the user one making sure it does not exceed maximum capacity for SPU
			HK_ON_PLATFORM_HAS_SPU( HK_ASSERT( 0x43bdc47b, cinfo.m_materialLibrary->getCapacity() <= HKNP_MAX_NUM_MATERIALS_ON_SPU ) );
			m_materialLibrary = cinfo.m_materialLibrary;
		}

		HK_SUBSCRIBE_TO_SIGNAL( m_materialLibrary->m_materialAddedSignal, self, hknpWorldEx ) ;
		HK_SUBSCRIBE_TO_SIGNAL( m_materialLibrary->m_materialModifiedSignal, self, hknpWorldEx );
		HK_ON_DEBUG( HK_SUBSCRIBE_TO_SIGNAL( m_materialLibrary->m_materialRemovedSignal, self, hknpWorldEx ) );

		m_dirtyMaterials.setSizeAndFill( 0, m_materialLibrary->getCapacity(), 0 );
	}

	//
	//	Motion properties library
	//
	{
		if (!cinfo.m_motionPropertiesLibrary)
		{
			// Create a library if one was not provided
			m_motionPropertiesLibrary.setAndDontIncrementRefCount( new hknpMotionPropertiesLibrary() );
			m_motionPropertiesLibrary->initialize(this);
		}
		else
		{
			// Otherwise use the user one making sure it does not exceed maximum capacity for SPU
			HK_ON_PLATFORM_HAS_SPU( HK_ASSERT( 0x43bdc47c, cinfo.m_motionPropertiesLibrary->getCapacity() <= HKNP_MAX_NUM_MOTION_PROPERTIES_ON_SPU ) );
			m_motionPropertiesLibrary = cinfo.m_motionPropertiesLibrary;
		}

		HK_ON_DEBUG( HK_SUBSCRIBE_TO_SIGNAL( m_motionPropertiesLibrary->m_entryAddedSignal, self, hknpWorldEx ) );
		HK_ON_DEBUG( HK_SUBSCRIBE_TO_SIGNAL( m_motionPropertiesLibrary->m_entryRemovedSignal, self, hknpWorldEx ) );
	}

	//
	//	Quality library
	//
	{
		if (!cinfo.m_qualityLibrary)
		{
			// Create a quality library if one was not provided
			m_qualityLibrary.setAndDontIncrementRefCount( new hknpBodyQualityLibrary() );

			hknpBodyQualityLibraryCinfo bqtCinfo;
			bqtCinfo.m_unitScale = cinfo.m_unitScale;
			m_qualityLibrary->initialize( bqtCinfo );
		}
		else
		{
			m_qualityLibrary = cinfo.m_qualityLibrary;
		}
		HK_SUBSCRIBE_TO_SIGNAL( m_qualityLibrary->m_qualityModifiedSignal, self, hknpWorldEx );

		m_dirtyQualities.setSizeAndFill( 0, m_qualityLibrary->getCapacity(), 0 );
	}

	//
	//	Command queues
	//
	{
		m_commandDispatcher = new hkPrimaryCommandDispatcher();
		m_mergeEventsBeforeDispatch = cinfo.m_mergeEventsBeforeDispatch;
		if( cinfo.m_mergeEventsBeforeDispatch )
		{
			m_eventDispatcher = new hknpEventMergeAndDispatcher(this);
		}
		else
		{
			m_eventDispatcher = new hknpEventDispatcher(this);
		}

		hkSecondaryCommandDispatcher* physics = new hknpApiCommandProcessor(this);
		hknpInternalCommandProcessor* intern  = new hknpInternalCommandProcessor(this);
		hkSecondaryCommandDispatcher* debug   = new hkDebugCommandProcessor();
		m_internalPhysicsCommandDispatcher = intern;

		m_commandDispatcher->registerDispatcher( hkCommand::TYPE_DEBUG_DISPLAY, debug );
		m_commandDispatcher->registerDispatcher( hkCommand::TYPE_PHYSICS_API, physics );
		m_commandDispatcher->registerDispatcher( hkCommand::TYPE_PHYSICS_EVENTS, m_eventDispatcher );
		m_commandDispatcher->registerDispatcher( hkCommand::TYPE_PHYSICS_INTERNAL, intern );

		physics->removeReference();
		intern->removeReference();
		debug->removeReference();
		m_eventDispatcher->removeReference();
	}

	//
	//	Space splitter
	//
	{
		m_simulationType = cinfo.m_simulationType;
		if ( cinfo.m_simulationType == hknpWorldCinfo::SIMULATION_TYPE_SINGLE_THREADED )
		{
			m_spaceSplitter = new hknpSingleCellSpaceSplitter();
		}
		else
		{
			m_spaceSplitter = new hknpDynamicSpaceSplitter(cinfo.m_numSplitterCells);
		}
	}

	//
	//	Broad phase
	//
	{
		m_intSpaceUtil.set( cinfo.m_broadPhaseAabb );

		if (cinfo.m_broadPhaseConfig)
		{
			m_broadPhase = new hknpHybridBroadPhase( cinfo.m_bodyBufferCapacity, cinfo.m_broadPhaseConfig );
		}
		else
		{
			hknpDefaultBroadPhaseConfig* broadPhaseConfig = new hknpDefaultBroadPhaseConfig;
			m_broadPhase = new hknpHybridBroadPhase( cinfo.m_bodyBufferCapacity, broadPhaseConfig );
			broadPhaseConfig->removeReference();
		}
	}

	//
	//	Collide
	//
	{
		m_collisionDispatcher = new hknpCollisionDispatcher;
	}

	//
	//	Deactivation
	//
	{
		m_deactivationManager = new hknpDeactivationManager(cinfo.m_largeIslandSize);
		m_deactivationManager->initDeactivationManager(this);
		m_deactivationEnabled = cinfo.m_enableDeactivation;
	}

	//
	//	Bodies and Motions
	//
	{
		m_motionManager.initialize( cinfo.m_userMotionBuffer, cinfo.m_motionBufferCapacity, &m_bodyManager, *m_spaceSplitter );
		m_bodyManager.m_motionManager = &m_motionManager;

		int padding = 0;
		if( cinfo.m_simulationType != hknpWorldCinfo::SIMULATION_TYPE_SINGLE_THREADED )
		{
			padding = 132 * 4;
		}

		m_solverVelocities.reserve   ( cinfo.m_motionBufferCapacity + padding );	// 132 because of padding of multithreaded simulation
		m_solverSumVelocities.reserve( cinfo.m_motionBufferCapacity + padding );

		// create a dummy deactivation state for the static motion at position 0
		{
			m_deactivationManager->reserveNumMotions( cinfo.m_motionBufferCapacity );
			hknpDeactivationState& deactivationState = m_deactivationManager->m_deactivationStates[0];
			deactivationState.initDeactivationState(0);
		}
	}

	//
	//	Constraints/Contacts
	//
	{
		m_constraintAtomSolver = new hknpConstraintAtomSolver;
		m_contactSolver = new hknpContactSolver;
	}

	//
	//	Actions
	//
	{
		m_actionManager = new hknpActionManager;
		m_signals.m_worldShifted.subscribe( m_actionManager, &hknpActionManager::onWorldShifted, "hknpActionManager" );
	}

	//
	//	Collision cache manager
	//
	{
		hkThreadLocalBlockStreamAllocator alloc( m_persistentStreamAllocator, -1 );
		m_collisionCacheManager = new hknpCollisionCacheManager( &alloc, m_spaceSplitter->getNumLinks() );
		alloc.clear();
	}

	//
	//	Modifiers
	//
	{
		m_modifierManager = new hknpModifierManager;
		hknpDefaultModifierSet* df = new hknpDefaultModifierSet;
		m_defaultModifierSet = df;
		setupModifierManager( cinfo, df, m_contactSolver, m_constraintAtomSolver, m_modifierManager );
	}

	//
	//	Shape Tag Codec
	//
	{
		setShapeTagCodec( cinfo.m_shapeTagCodec );
	}

	//
	//	Deactivation policies
	//
	{
		hknpDeactiveCdCacheFilter* deactivationPolicy = new hknpDeactiveCdCacheFilter();
		setDeactiveCdCacheFilter( deactivationPolicy );
		deactivationPolicy->removeReference();
	}

	//
	//	Broad phase policies
	//
	{
		m_leavingBroadPhaseBehavior = cinfo.m_leavingBroadPhaseBehavior;
		if ( cinfo.m_leavingBroadPhaseBehavior == hknpWorldCinfo::ON_LEAVING_BROAD_PHASE_FREEZE_BODY )
		{
			getEventDispatcher()->getSignal( hknpEventType::BODY_EXITED_BROAD_PHASE).subscribe(
				self, &hknpWorldEx::freezeBodiesWhichLeftTheBroadPhase, "hknpWorld::freezeBodiesWhichLeftTheBroadPhase" );
		}
		else if ( cinfo.m_leavingBroadPhaseBehavior == hknpWorldCinfo::ON_LEAVING_BROAD_PHASE_REMOVE_BODY )
		{
			getEventDispatcher()->getSignal( hknpEventType::BODY_EXITED_BROAD_PHASE).subscribe(
				self, &hknpWorldEx::removeBodiesWhichLeftTheBroadPhase, "hknpWorld::removeBodiesWhichLeftTheBroadPhase" );
		}
	}

	//
	//	Simulation
	//
	{
		m_enableSolverDynamicScheduling = cinfo.m_enableSolverDynamicScheduling;

		if ( cinfo.m_simulationType == hknpWorldCinfo::SIMULATION_TYPE_SINGLE_THREADED )
		{
			m_simulation = new hknpSingleThreadedSimulation;
		}
		else
		{
			HK_ASSERT(0x266a2b1, HK_CONFIG_THREAD != HK_CONFIG_SINGLE_THREADED);
			m_simulation = new hknpMultithreadedSimulation;
		}

		// Reusable tasks
		m_preCollideTask = new hknpPreCollideTask();
		m_postCollideTask = new hknpPostCollideTask();
		m_postSolveTask = new hknpPostSolveTask();
	}

	//
	//	Query dispatcher
	//
	{
		m_collisionQueryDispatcher = new hknpCollisionQueryDispatcher;
	}

	//
	//	Collision filter
	//
	if ( cinfo.m_collisionFilter )
	{
		hknpCollisionFilter* filter = cinfo.m_collisionFilter;
		if ( filter->m_type == hknpCollisionFilter::CONSTRAINT_FILTER )
		{
			static_cast<hknpConstraintCollisionFilter*>(filter)->subscribeToWorld(this);
		}
	}

	// Work around a potential function level static thread safety issue by initializing it here (HNP-400)
	hknpTriangleShape::getReferenceTriangleShape();

	//
	//	Debugging
	//
	{
#if defined(HK_DEBUG_SLOW) && (HAVOK_BUILD_NUMBER == 0)	// Internal debug build
		m_consistencyChecksEnabled = true;
#else
		m_consistencyChecksEnabled = false;
#endif

#if defined(HK_DEBUG)
		// Check every change to the bodies
		m_signals.m_bodyChanged.subscribe( this, &hknpWorld::checkBodyConsistency, "World" );
		setAsDebugInstance();
#endif
	}
}

void hknpWorld::getCinfo( hknpWorldCinfo& cinfo ) const
{
	//
	// Memory
	//
	cinfo.m_bodyBufferCapacity = m_bodyManager.getCapacity();
	cinfo.m_motionBufferCapacity = m_motionManager.getCapacity();
	cinfo.m_persistentStreamAllocator = m_persistentStreamAllocator;

	//
	//	Libraries
	//
	cinfo.m_materialLibrary = m_materialLibrary;
	cinfo.m_motionPropertiesLibrary = m_motionPropertiesLibrary;
	cinfo.m_qualityLibrary = m_qualityLibrary;

	//
	//	Command queues
	//
	cinfo.m_mergeEventsBeforeDispatch = m_mergeEventsBeforeDispatch;

	//
	//	Space splitter
	//
	cinfo.m_simulationType = m_simulationType;

	//
	//	Broad phase
	//
	cinfo.m_broadPhaseAabb = m_intSpaceUtil.m_aabb;

	//
	//	Collide
	//
	cinfo.m_gravity = m_gravity;
	cinfo.m_collisionTolerance = m_collisionTolerance.getReal();

	//
	//	Deactivation
	//
	cinfo.m_largeIslandSize = m_deactivationManager->m_largeIslandSize;
	cinfo.m_enableDeactivation = m_deactivationEnabled;

	//
	//	Solver info
	//
	cinfo.m_solverTau = m_solverInfo.m_tau;
	cinfo.m_solverDamp = m_solverInfo.m_damping.getReal();
	cinfo.m_defaultSolverTimestep = m_solverInfo.m_expectedDeltaTime;
	cinfo.m_solverIterations = m_solverInfo.m_numSteps;
	cinfo.m_solverMicrosteps = m_solverInfo.m_numMicroSteps;

	//
	//	Modifiers, the order does matter
	//
	cinfo.m_collisionFilter = m_modifierManager->getCollisionFilter();
	cinfo.m_collisionQueryFilter = m_modifierManager->getCollisionQueryFilter();

	if (m_shapeTagCodec.val() == &m_nullShapeTagCodec)
	{
		cinfo.m_shapeTagCodec = HK_NULL;
	}
	else
	{
		cinfo.m_shapeTagCodec = m_shapeTagCodec;
	}

	//
	//	Special policies
	//
	cinfo.m_leavingBroadPhaseBehavior = m_leavingBroadPhaseBehavior;
}

hknpWorld::~hknpWorld()
{
	checkNotInSimulation();
	hknpWorldEx* self = (hknpWorldEx*)this;

	getEventDispatcher()->getSignal( hknpEventType::BODY_EXITED_BROAD_PHASE ).unsubscribe( self, &hknpWorldEx::freezeBodiesWhichLeftTheBroadPhase );
	getEventDispatcher()->getSignal( hknpEventType::BODY_EXITED_BROAD_PHASE ).unsubscribe( self, &hknpWorldEx::removeBodiesWhichLeftTheBroadPhase );

	HK_ON_DEBUG( m_motionPropertiesLibrary->m_entryAddedSignal.unsubscribe(self, &hknpWorldEx::onMotionPropertiesAddedSignal) );
	HK_ON_DEBUG( m_motionPropertiesLibrary->m_entryRemovedSignal.unsubscribe(self, &hknpWorldEx::onMotionPropertiesRemovedSignal) );

	m_materialLibrary->m_materialModifiedSignal.unsubscribe(self, &hknpWorldEx::onMaterialModifiedSignal);
	m_materialLibrary->m_materialAddedSignal.unsubscribe(self, &hknpWorldEx::onMaterialAddedSignal);
	HK_ON_DEBUG(m_materialLibrary->m_materialRemovedSignal.unsubscribe(self, &hknpWorldEx::onMaterialRemovedSignal));

	m_qualityLibrary->m_qualityModifiedSignal.unsubscribe(self, &hknpWorldEx::onQualityModifiedSignal);

	m_signals.m_worldDestroyed.fire(this);

	//	Query dispatcher
	{
		delete m_collisionQueryDispatcher;
		m_collisionQueryDispatcher = HK_NULL;
	}

	//	Actions
	{
		m_signals.m_worldShifted.unsubscribe( m_actionManager, &hknpActionManager::onWorldShifted );
		delete m_actionManager;
		m_actionManager = HK_NULL;
	}

	//	Tasks
	{
		delete m_preCollideTask;	m_preCollideTask = HK_NULL;
		delete m_postCollideTask;	m_postCollideTask = HK_NULL;
		delete m_postSolveTask;		m_postSolveTask = HK_NULL;
	}

	{
		hkThreadLocalBlockStreamAllocator tlAllocator( m_persistentStreamAllocator, -1 );

#if defined(HKNP_ENABLE_SOLVER_LOG)
		if (g_npDebugOstream)
		{
			g_npDebugOstream->flush();
			delete g_npDebugOstream;
			g_npDebugOstream = HK_NULL;
		}
#endif

		if( m_collisionCacheManager )
		{
			m_collisionCacheManager->clear( &tlAllocator );
			delete m_collisionCacheManager;
		}

		m_defaultModifierSet = HK_NULL;

		delete m_simulation; m_simulation = HK_NULL;
		delete m_broadPhase;
		delete m_collisionDispatcher;
		delete m_spaceSplitter;
		delete m_modifierManager;
		delete m_constraintAtomSolver;
		delete m_contactSolver;

		tlAllocator.clear();
	}

	m_solverVelocities.clearAndDeallocate();
	m_solverSumVelocities.clearAndDeallocate();

	delete m_deactivationManager;
	delete m_commandDispatcher;

	m_persistentStreamAllocator = HK_NULL;
}

void hknpWorld::rebuildMotionMassProperties( hknpMotionId motionId, RebuildMassPropertiesMode rebuildMode )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );

	hknpMotion& motion = accessMotionUnchecked( motionId );
	const hknpMotionProperties& motionProperties = m_motionPropertiesLibrary->getEntry( motion.m_motionPropertiesId );

	if ( motionProperties.m_flags.anyIsSet( hknpMotionProperties::NEVER_REBUILD_MASS_PROPERTIES ) )
	{
		return;
	}

	const hknpBodyId firstBodyId = motion.m_firstAttachedBodyId;
	if ( !firstBodyId.isValid() )
	{
		return;
	}

	if ( rebuildMode == REBUILD_DELAYED )
	{
		m_bodyManager.setScheduledBodyFlags(firstBodyId, hknpBodyManager::REBUILD_MASS_PROPERTIES);
		return;
	}

	// Remember angular velocity in world space
	hkVector4 angVel;
	motion.getAngularVelocity( angVel );

	// Gather mass elements from attached bodies
	hkArray<hkMassElement> massElements;
	hknpBodyId attachedBodyId = firstBodyId;
	do
	{
		const hknpBody& body = getBody( attachedBodyId );

		hkDiagonalizedMassProperties dmp;
		if( body.m_shape->getMassProperties( dmp ) == HK_FAILURE )
		{
			HK_WARN( 0x3eb23f0c, "A shape has no mass properties. Approximating them instead." );
			hknpShape::MassConfig massConfig;
			massConfig.m_quality = hknpShape::MassConfig::QUALITY_LOW;
			body.m_shape->buildMassProperties( massConfig, dmp );
		}

		hkSimdReal massFactor = motion.getMassFactor();
		dmp.m_mass *= massFactor.getReal();
		dmp.m_inertiaTensor.mul( massFactor );

		hkMassElement& me = massElements.expandOne();
		dmp.unpack( &me.m_properties );
		me.m_transform = body.getTransform();

		attachedBodyId = getBody(attachedBodyId).m_nextAttachedBodyId;
	} while (attachedBodyId != firstBodyId);

	// Combine them
	hkDiagonalizedMassProperties dmp;
	{
		hkMassProperties massProperties;
		hkInertiaTensorComputer::combineMassProperties( massElements, massProperties );
		dmp.pack( massProperties );
		dmp.m_inertiaTensor(3) = massProperties.m_mass;
	}

	// Store them in the motion
	hkVector4 invInertiaTensor;
	invInertiaTensor.setReciprocal( dmp.m_inertiaTensor );
	invInertiaTensor.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>( motion.m_inverseInertia );
	motion.setCenterOfMassInWorld( dmp.m_centerOfMass );
	motion.m_orientation = dmp.m_majorAxisSpace;

	// Update the attached bodies
	attachedBodyId = firstBodyId;
	do
	{
		hknpBody& body = m_bodyManager.m_bodies[ attachedBodyId.value() ];
		body.updateMotionToBodyTransform( motion );
		body.updateComCenteredBoundingRadius( motion );
		attachedBodyId = getBody( attachedBodyId ).m_nextAttachedBodyId;
	} while (attachedBodyId != firstBodyId);

	// Reapply angular velocity
	motion.setAngularVelocity( angVel );
}


void hknpWorld::addBodyToMotion( hknpBodyId bodyId, hknpMotionId motionId, const hkQuaternion* worldQbody )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	hknpWorldEx* self = (hknpWorldEx*)this;

	hknpBody* HK_RESTRICT body = &m_bodyManager.accessBody(bodyId);

	HK_ASSERT( 0xf03467dc, !body->m_motionId.isValid() );
	if( !body->m_shape )
	{
		HK_ASSERT2( 0xf0453445, 0, "Setup a dynamic body with no shape" );
		HK_WARN_ALWAYS( 0xf0453445, "Setup a dynamic body with no shape" );
	}

	hknpMotion* HK_RESTRICT	motion = &accessMotionUnchecked(motionId);
	self->_attachBodyToMotion(bodyId, *body, motionId, *motion);

	// Inherit flags
	body->m_motionId = motionId;
	if( motion->hasInfiniteMass() && !motion->isStatic() )
	{
		body->m_flags.orWith( hknpBody::IS_KEYFRAMED );
	}
	else
	{
		body->m_flags.clear( hknpBody::IS_KEYFRAMED );
	}

	// Calculate motion's angular velocity clipping if necessary
	const hknpBodyQuality& quality = getBodyQualityLibrary()->getEntry( body->m_qualityId );
	if( quality.m_requestedFlags.anyIsSet(hknpBodyQuality::CLIP_ANGULAR_VELOCITY) )
	{
		const hknpShape* shape = body->m_shape;
		if( shape->asConvexShape() )
		{
			hkReal maxAng = motion->m_maxRotationToPreventTunneling;
			const hknpConvexShape*	cvxShape = static_cast<const hknpConvexShape*>(shape);
			const hkReal			current = motion->m_maxRotationToPreventTunneling;
			hkReal					minAngle = 0.5f * cvxShape->calcMinAngleBetweenFaces();
			minAngle = hkMath::max2( minAngle, hkReal(0.05f) );		// make sure to not get zero.
			minAngle = hkMath::min2(current, hkMath::min2( maxAng, minAngle ));
			motion->m_maxRotationToPreventTunneling.setReal<false>(minAngle);
		}
		else
		{
			HK_WARN( 0xf0dfcf45, "hknpBodyQuality::CLIP_ANGULAR_VELOCITY is not implemented for non-convex shapes" );
		}
	}

	// Set motion's look ahead distance (if body's is set and smaller)
	if( body->getCollisionLookAheadDistance() > 0.0f )
	{
		hkSimdReal bodyLookAhead; bodyLookAhead.setFromFloat( body->getCollisionLookAheadDistance() );
		hkSimdReal motionLookAhead; motionLookAhead.load<1>( &motion->m_maxLinearAccelerationDistancePerStep );
		motionLookAhead.setMin( motionLookAhead, bodyLookAhead );
		motionLookAhead.store<1>( &motion->m_maxLinearAccelerationDistancePerStep );
	}

	// Set body's relative transform from the motion
	{
		hkQuaternion q;
		if( worldQbody )
		{
			q = *worldQbody;
		}
		else
		{
			q.set( body->getTransform().getRotation() );
		}
		body->updateMotionToBodyTransform( *motion, &q );
	}

	// Calculate body's swept AABB
	{
		hknpMotionUtil::calcSweptBodyAabb( body, *motion, quality, m_collisionTolerance, m_solverInfo, m_intSpaceUtil );
	}

	// Update the COM bounding radius
	{
		body->updateComCenteredBoundingRadius( *motion );
	}

	// Set body's linear TIM
	hknpMotionUtil::convertVelocityToLinearTIM( m_solverInfo, motion->m_linearVelocity, body->m_maxTimDistance );
}

void hknpWorld::removeBodyFromMotion( hknpBody* HK_RESTRICT body )
{
	hknpWorldEx* self = (hknpWorldEx*)this;
	if( body->isAddedToWorld() )
	{
		// We need to be outside the simulation look if the body is 'live'.
		checkNotInSimulation();
	}

	hknpBodyId bodyId = body->m_id;
	hknpMotionId motionId = body->m_motionId;
	hknpMotion* HK_RESTRICT motion = &accessMotionUnchecked( motionId );
	if ( body->m_nextAttachedBodyId == bodyId )
	{
		// This is the last body using this motion, free it
		m_signals.m_motionDestroyed.fire( this, motionId );
		m_motionManager.markMotionForDeletion( motionId );
	}
	else
	{
		// Other bodies are attached, detach this one
		self->_detachBodyFromMotion( body, bodyId, motion );
		self->removeActiveMotionIfNoAddedBodies( body->m_motionId );
	}
	body->m_motionId = hknpMotionId::invalid();
}

void hknpWorld::registerBodyAtActiveList( hknpBodyId bodyId, hknpMotionId motionId )
{
	// Reset deactivation
	hknpDeactivationState& deactivationState = m_deactivationManager->m_deactivationStates[motionId.value()];
	deactivationState.resetCounters( bodyId.value() );

	// If the motion is already registered, simply register the body only,
	// otherwise register all bodies attached to this motion and the motion.
	hknpMotion& motion = accessMotionUnchecked( motionId );
	if( !motion.m_solverId.isValid() )
	{
		m_motionManager.addActiveMotion( motion, motionId );
	}
	m_bodyManager.addSingleBodyToActiveGroup( bodyId );
}

void hknpWorld::unregisterBodyAtActiveList( hknpBody* HK_RESTRICT body )
{
	checkNotInSimulation();

	hknpBodyId bodyId = body->m_id;
	hknpMotionId motionId = body->m_motionId;
	if( body->isActive() )
	{
		if ( body->m_nextAttachedBodyId == bodyId )
		{
			m_bodyManager.removeActiveBodyGroup( bodyId );
		}
		else
		{
			m_bodyManager.removeSingleBodyFromActiveList( bodyId );
			m_bodyManager.m_bodies[bodyId.value()].m_indexIntoActiveListOrDeactivatedIslandId = hknpBodyId::InvalidValue;
		}
	}
}

hknpMotionId hknpWorld::createMotion( const hknpMotionCinfo& motionCinfo )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	hknpWorldEx* self = (hknpWorldEx*)this;

	// Get a motion ID
	const hknpMotionId motionId = self->_allocateMotion();

	// Initialize motion
	hknpMotion& motion = accessMotionUnchecked( motionId );
	m_motionManager.initializeMotion( &motion, motionCinfo, *m_spaceSplitter );
	HK_ON_DEBUG( motion.checkConsistency() );

	// Initialize deactivation
	m_deactivationManager->reserveNumMotions( motionId.value() + 1 );
	hknpDeactivationState& deactivationState = m_deactivationManager->m_deactivationStates[motionId.value()];
	deactivationState.initDeactivationState( motionId.value(), motionCinfo.m_enableDeactivation );

	// Fire signal
	m_signals.m_motionCreated.fire( this, &motionCinfo, motionId );

	return motionId;
}

void hknpWorld::addBodies( const hknpBodyId* ids, int numIds, AdditionFlags additionFlags )
{
	if( numIds <= 0 )
	{
		return;
	}

	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );

	if( !m_deactivationEnabled )
	{
		additionFlags.clear( START_DEACTIVATED );
	}

	if( additionFlags.get(ADD_BODY_NOW) == 0 )
	{
		if( additionFlags.get(START_DEACTIVATED) )
		{
			m_bodyManager.appendToPendingAddList( ids, numIds, hknpBodyManager::ADD_INACTIVE );
		}
		else
		{
			m_bodyManager.appendToPendingAddList( ids, numIds, hknpBodyManager::ADD_ACTIVE );
		}
		return;
	}


	HK_TIMER_BEGIN_LIST("AddBodies", "Setup");
	{
		for (int i = 0; i < numIds; i++ )
		{
			hknpBodyId bodyId = ids[i];
			addTrace( hknpAddBodyCommand( bodyId, additionFlags, i == numIds-1 ) );

			hknpBody* HK_RESTRICT body = &m_bodyManager.accessBody(bodyId);
			HK_ASSERT2( 0xf0456033, !body->isAddedToWorld(), "You can only add bodies that were not added yet" );
			if ( body->isDynamic() )
			{
				HK_WARN_ON_DEBUG_IF(
					body->getTransform().isApproximatelyEqualSimd( hkTransform::getIdentity(), hkSimdReal_Eps ), 0xf0456535,
					"Adding a dynamic body with identity transform. Is this intended? Body transforms should be set before adding to the world." );

				// If body was just removed while being deactivated AND we try to add it again within the same frame
				// we need to force the activation of the island to happen now.
				// Or if a body is added that is part of an inactive compound.
				if( body->isInactive() && body->m_indexIntoActiveListOrDeactivatedIslandId != hknpBodyId::InvalidValue )
				{
					m_deactivationManager->markIslandForActivation(hknpIslandId(body->m_indexIntoActiveListOrDeactivatedIslandId));
					m_deactivationManager->activateMarkedIslands();
				}

				body->m_flags.orWith( hknpBody::TEMP_REBUILD_COLLISION_CACHES );

				if( additionFlags.get( START_DEACTIVATED ) )
				{
					body->m_flags.clear( hknpBody::IS_ACTIVE );
				}
				else
				{
					registerBodyAtActiveList( bodyId, body->m_motionId );
				}
			}
			else
			{
				m_bodyManager.setScheduledBodyFlags( bodyId, hknpBodyManager::MOVED_STATIC );	
			}

			// Make sure any broad phase overlaps are treated as new ones
			m_bodyManager.m_previousAabbs[ bodyId.value() ].setEmpty();
		}
	}


	HK_TIMER_SPLIT_LIST("AddToBroadPhase");
	if ( additionFlags.get(START_DEACTIVATED) == 0 )
	{
		m_broadPhase->addBodies( ids, numIds, m_bodyManager.accessBodyBuffer() );
	}
	else
	{
		// start inactive
		for( int i = 0; i < numIds; i++ )
		{
			const hknpBody& body = getBody( ids[i]);
			if ( body.isDynamic() )
			{
				bool startInactive = m_deactivationManager->addInactiveBody( ids[i] );
				if (!startInactive)
				{
					registerBodyAtActiveList( ids[i], body.m_motionId );
				}
			}
			m_broadPhase->addBodies( &ids[i], 1, m_bodyManager.accessBodyBuffer() );
		}
	}

	HK_TIMER_SPLIT_LIST("FireCallbacks");
	if( m_signals.m_bodyAdded.hasSubscriptions() )
	{
		for (int i = 0; i < numIds; i++ )
		{
			hknpBodyId bodyId = ids[i];
			m_signals.m_bodyAdded.fire(this, bodyId);
		}
	}

	if( areConsistencyChecksEnabled() )
	{
		checkConsistency();
	}

	HK_TIMER_END_LIST();
}

void hknpWorld::commitAddBodies()
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );

	if( !m_bodyManager.m_bodiesToAddAsActive.isEmpty() )
	{
		addBodies( m_bodyManager.m_bodiesToAddAsActive.begin(), m_bodyManager.m_bodiesToAddAsActive.getSize(), ADD_BODY_NOW );
	}

	if( !m_bodyManager.m_bodiesToAddAsInactive.isEmpty() )
	{
		addBodies( m_bodyManager.m_bodiesToAddAsInactive.begin(), m_bodyManager.m_bodiesToAddAsInactive.getSize(), ADD_BODY_NOW | START_DEACTIVATED );
	}

	m_bodyManager.clearPendingAddLists();

	if( areConsistencyChecksEnabled() )
	{
		checkConsistency();
	}
}

hknpBodyId hknpWorld::reserveBodyId()
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	hknpWorldEx* self = (hknpWorldEx*)this;
	return self->_allocateBody();
}

hknpBodyId hknpWorld::createBody( const hknpBodyCinfo& bodyCinfo, AdditionFlags additionFlags )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	hknpWorldEx* self = (hknpWorldEx*)this;

	HK_ASSERT2(0x6eeac709, m_motionManager.isMotionValid(bodyCinfo.m_motionId), "Invalid motion ID" );

	if( !bodyCinfo.m_shape )
	{
		HK_WARN_ALWAYS( 0xf0453446, "Creating a body without a shape" );
	}

	// Get a body ID
	hknpBodyId bodyId;
	if( bodyCinfo.m_reservedBodyId.isValid() )
	{
		HK_ASSERT2( 0xf0dedffe, !isBodyValid(bodyCinfo.m_reservedBodyId), "Reserved body ID is already in use" );
		bodyId = bodyCinfo.m_reservedBodyId;
	}
	else
	{
		bodyId = self->_allocateBody();
	}

	addTrace( hknpCreateBodyCommand( bodyId, bodyCinfo.m_motionId ) );

	// Initialize the body
	hknpBody* HK_RESTRICT body = &m_bodyManager.accessBody(bodyId);
	if( bodyCinfo.m_motionId != hknpMotionId::STATIC )
	{
		// dynamic
		m_bodyManager.initializeDynamicBody( body, bodyId, bodyCinfo );
		addBodyToMotion( bodyId, bodyCinfo.m_motionId, &bodyCinfo.m_orientation );

		// fire signal
		m_signals.m_bodyCreated._fire( this, &bodyCinfo, bodyId );

		// add to world
		if( additionFlags.get(DO_NOT_ADD_BODY) == 0 )
		{
			addBodies( &bodyId, 1, additionFlags );
		}
	}
	else
	{
		// static
		m_bodyManager.initializeStaticBody( body, bodyId, bodyCinfo );
		body->syncStaticMotionToBodyTransform();

		// fire signal
		m_signals.m_bodyCreated._fire( this, &bodyCinfo, bodyId );

		if( bodyCinfo.m_shape )
		{
			// set AABB
			hkAabb aabb;
			hknpMotionUtil::calcStaticBodyAabb( *body, m_collisionTolerance, &aabb );
			m_intSpaceUtil.convertAabb( aabb, body->m_aabb );

			// add to world
			if( additionFlags.get(DO_NOT_ADD_BODY) == 0 )
			{
				addBodies( &bodyId, 1, additionFlags );
			}
		}
		else
		{
			m_bodyManager.m_previousAabbs[ bodyId.value() ].setEmpty();
		}
	}

	// If the body has a mutable shape, tell the shape manager
	if( bodyCinfo.m_shape && bodyCinfo.m_shape->isMutable() )
	{
		m_shapeManager.registerBodyWithMutableShape( *body );
	}

	return bodyId;
}

hknpBodyId hknpWorld::createAttachedBodies(
	const hknpBodyCinfo* bodyCinfos, int numBodies, const hknpMotionCinfo& motionCinfo, AdditionFlags additionFlags )
{
	HK_ASSERT2( 0x339a3122, numBodies > 0, "No bodies to create in compound body" );
	hknpMotionId motionId = createMotion( motionCinfo );
	for( int i = 0 ; i < numBodies ; ++i )
	{
		hknpBodyCinfo bodyCinfoCopy = bodyCinfos[i];
		bodyCinfoCopy.m_motionId = motionId;
		createBody( bodyCinfoCopy, additionFlags );
	}
	return getMotion( motionId ).m_firstAttachedBodyId;
}

void hknpWorld::removeBodies( const hknpBodyId* bodyIds, int numBodies, hknpActivationMode::Enum activationMode )
{
	checkNotInSimulation();
	hknpWorldEx* self = (hknpWorldEx*)this;

	if( areConsistencyChecksEnabled() )
	{
		checkConsistency();
	}

	int numAdded = 0;
	for( int i = 0; i < numBodies; i++ )
	{
		const hknpBodyId bodyId = bodyIds[i];
		HK_ASSERT2( 0xf034dfe4, bodyId.value() >= hknpBodyId::NUM_PRESETS, "Cannot remove preset bodies" );

		// Try first to remove the body from the lists of bodies pending to be added
		if( m_bodyManager.removeSingleBodyFromPendingAddLists(bodyId) )
		{
			continue;
		}

		hknpBody& body = m_bodyManager.accessBody( bodyId );

		if( body.isAddedToWorld() )
		{
			addTrace( hknpRemoveBodyCommand( bodyId, i == numBodies-1 ) );

			++numAdded;

			if( m_signals.m_bodyRemoved.hasSubscriptions() )
			{
				HK_TIME_CODE_BLOCK( "FireBodyRemoved", HK_NULL);
				m_signals.m_bodyRemoved._fire( this, bodyId );
			}

			// This is needed so that collision caches are moved to the active stream,
			// and properly destroyed on the next frame
			if( body.isInactive() )
			{
				if( m_deactivationEnabled )
				{
					m_deactivationManager->markBodyForActivation(bodyId);
				}
			}
			else if( body.isStatic() )
			{
				if( activationMode == hknpActivationMode::ACTIVATE )
				{
					// if a body is fixed we have to activate all bodies near it
					activateBodiesInAabb( body.m_aabb );
				}
			}
			else
			{
				HK_ASSERT( 0xf03e4545, body.isActive() );
				unregisterBodyAtActiveList( &body );
			}
		}
	}

	// removeBodies() can handle bodies that are not in the broad phase anymore (have an invalid body.m_broadPhaseId)
	// but we still avoid calling it if we know all the bodies have already been removed.
	if( numAdded > 0 )
	{
		m_broadPhase->removeBodies( bodyIds, numBodies, m_bodyManager.accessBodyBuffer() );
	}

	// Mark bodies as removed
	for( int i=0; i<numBodies; i++ )
	{
		const hknpBodyId bodyId = bodyIds[i];
		hknpBody& body = m_bodyManager.accessBody( bodyId );
		body.m_broadPhaseId = hknpBroadPhaseId(HKNP_INVALID_BROAD_PHASE_ID);
		body.m_flags.orWith( hknpBody::TEMP_REBUILD_COLLISION_CACHES );	// remove all old stale caches in the next step
		self->removeActiveMotionIfNoAddedBodies( body.m_motionId );

		//hkAabb16* HK_RESTRICT prevAabb = &m_bodyManager.m_prevAabbs[ bodyId.value() ];
		//prevAabb->setEmpty();
	}

	if( areConsistencyChecksEnabled() )
	{
		checkConsistency();
	}
}

void hknpWorld::destroyBodies( const hknpBodyId* bodyIds, int numBodies, hknpActivationMode::Enum activationMode )
{
	if( m_traceDispatcher )
	{
		for( int i=0; i<numBodies; i++ )
		{
			const hknpBodyId bodyId = bodyIds[i];
			addTrace( hknpDestroyBodyCommand(bodyId) );
		}
	}

	// removeBodies knows how to handle bodies that are already removed.
	removeBodies( bodyIds, numBodies, activationMode );

	// Delete the bodies
	for( int i = 0; i < numBodies; ++i )
	{
		const hknpBodyId bodyId = bodyIds[i];
		HK_ASSERT2( 0xf034dfe4, bodyId.value() >= hknpBodyId::NUM_PRESETS, "Cannot destroy preset bodies" );

		hknpBody& body = m_bodyManager.accessBody( bodyId );

		// If the body has a mutable shape, tell the shape manager
		if( body.m_shape->isMutable() )
		{
			m_shapeManager.deregisterBodyWithMutableShape( body );
		}

		// Fire destroyed signal (before changing the motion)
		if( m_signals.m_bodyDestroyed.hasSubscriptions() )
		{
			HK_TIME_CODE_BLOCK( "FireBodyDestroyed", HK_NULL);
			m_signals.m_bodyDestroyed._fire( this, bodyId );
		}

		// Remove it from its motion
		hknpMotionId motionId = body.m_motionId;
		if( motionId != hknpMotionId::STATIC )
		{
			hknpMotion& motion = m_motionManager.m_motions[motionId.value()];
			// If no other bodies are attached to this motion and it has a valid solver ID (i.e. is an active motion),
			// we need to tell motion manager remove it from the active list
			if( body.m_nextAttachedBodyId == bodyId && motion.m_solverId.isValid() )
			{
				m_motionManager.removeActiveMotion( motion, motionId );
			}
			removeBodyFromMotion( &body );
		}

		// Mark the ID for freeing
		m_bodyManager.markBodyForDeletion( bodyId );	
	}
}

void hknpWorld::destroyMotions( hknpMotionId* ids, int numIds )
{
	checkNotInSimulation();

	for( int i = 0; i < numIds; ++i )
	{
		hknpMotionId id = ids[i];

		// Sanity checks
		HK_ASSERT(0x107d71ec, id.isValid());
		HK_ASSERT2(0x49d11cab, id.value() >= hknpMotionId::NUM_PRESETS, "Cannot destroy preset motions");
		HK_ON_DEBUG(hknpMotion* HK_RESTRICT motion = &accessMotionUnchecked(id));
		HK_ASSERT2(0x107d71ec, !motion->m_firstAttachedBodyId.isValid() , "There are bodies still attached to this motion!");

		m_signals.m_motionDestroyed.fire( this, id );
		m_motionManager.markMotionForDeletion( id );	
	}
}

void hknpWorld::freeDestroyedBodiesAndMotions()
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE | SIMULATION_PRE_COLLIDE );
	m_deactivationManager->removeInvalidBodiesScheduledForDeactivation();
	m_bodyManager.deleteMarkedBodies();
	m_motionManager.deleteMarkedMotions();
}

void hknpWorld::predictBodyTransform( hknpBodyId bodyId, hkReal deltaTime, hkTransform* transformOut ) const
{
	const hknpBody& body = getBody( bodyId );
	const hknpMotion& motion = getMotion( body.m_motionId );

	hkVector4 motionCom;
	hkQuaternion motionOrientation;
	{
		hkSimdReal deltaTimeSr; deltaTimeSr.setFromFloat( deltaTime );
		hknpMotionUtil::_predictMotionTransform( deltaTimeSr, motion, &motionCom, &motionOrientation );
	}

	hknpMotionUtil::_calculateBodyTransform( body, motionCom, motionOrientation, transformOut );
}

void hknpWorld::activateBodiesInAabb( const hkAabb16& aabb )
{
	checkNotInSimulation();

	hkInplaceArray<hknpBodyId,512> bodyIds;		
	m_broadPhase->queryAabb( aabb, m_bodyManager.getBodyBuffer(), bodyIds );
	for( int i = 0; i < bodyIds.getSize(); ++i )
	{
		hknpBodyId bodyId = bodyIds[i];
		const hknpBody& body = m_bodyManager.getBody( bodyId );
		// We need to check the deactivated island index, because we might be called from
		// removeBodies(), resulting in a dynamic body getting removed from the active list
		// (not because of deactivation but because of removal), followed by a static body
		// getting removed and triggering activateBodiesInAabb(),
		// which then finds the dynamic body that is inactive but has no island.
		if ( body.isInactive() && hknpIslandId( body.getDeactivatedIslandIndex()).isValid() )
		{
			m_deactivationManager->markBodyForActivation(bodyId);
		}
	}
}

void hknpWorld::activateBodiesInAabb( const hkAabb& aabb )
{
	HK_ALIGN16( hkAabb16 aabb16 );
	m_intSpaceUtil.convertAabb( aabb, aabb16 );
	activateBodiesInAabb( aabb16 );
}

void hknpWorld::rebuildDynamicBodyCollisionCachesInAabb( const hkAabb& aabb )
{
	checkNotInSimulation();

	HK_ALIGN16( hkAabb16 aabb16 );
	m_intSpaceUtil.convertAabb( aabb, aabb16 );

	hkInplaceArray<hknpBodyId,512> bodyIds;		
	m_broadPhase->queryAabb( aabb16, m_bodyManager.getBodyBuffer(), bodyIds );
	for( int i = 0; i < bodyIds.getSize(); ++i )
	{
		hknpBodyId bodyId = bodyIds[i];
		const hknpBody& body = m_bodyManager.getBody( bodyId );

		// We need to check the deactivated island index, because we might be called from
		// removeBodies(), resulting in a dynamic body getting removed from the active list
		// (not because of deactivation but because of removal), followed by a static body
		// getting removed and triggering activateBodiesInAabb(),
		// which then finds the dynamic body that is inactive but has no island.
		if( body.isInactive() && hknpIslandId( body.getDeactivatedIslandIndex()).isValid() )
		{
			m_deactivationManager->markBodyForActivation(bodyId);
		}
		if( !body.isStatic() )
		{
			rebuildBodyCollisionCaches( bodyId );
		}
	}
}


void hknpWorld::linkBodyDeactivation( hknpBodyId bodyIdA, hknpBodyId bodyIdB )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );

	m_deactivationManager->addBodyLink(bodyIdA, bodyIdB);
}

void hknpWorld::unlinkBodyDeactivation( hknpBodyId bodyIdA, hknpBodyId bodyIdB )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );

	m_deactivationManager->removeBodyLink(bodyIdA, bodyIdB);
}

void hknpWorld::activateBody( hknpBodyId bodyId )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );

	const hknpBody& body = getBody( bodyId );
	if( body.isAddedToWorld() )
	{
		m_deactivationManager->markBodyForActivation( bodyId );
	}
}


void hknpWorld::attachBodies(
	hknpBodyId targetBodyId, const hknpBodyId* bodyIds, int numIds, UpdateMassPropertiesMode updateMassProperties )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
#if defined(HK_DEBUG)
	if ( m_simulationStage == SIMULATION_POST_COLLIDE )
	{
		HK_WARN_ONCE( 0xf0e60192, "Calling attachBodies() from post collide does not update existing contact Jacobians, this can lead to wrong behavior");
	}
#endif

	const hknpBody& targetBody = getBody( targetBodyId );
	const hknpMotionId targetMotionId = targetBody.m_motionId;
	HK_ASSERT2( 0xf034dfec, targetMotionId.isValid() && targetMotionId!=hknpMotionId::STATIC, "Target body must be dynamic or keyframed" );

	bool rebuildMotionProperties = (updateMassProperties != KEEP_MASS_PROPERTIES) && !targetBody.isStaticOrKeyframed();

	for( int i=0; i<numIds; ++i )
	{
		hknpBodyId bodyId = bodyIds[i];

		addTrace( hknpAttachBodyCommand( bodyId, targetBodyId, updateMassProperties ) );

		hknpBody& body = accessBody( bodyId );
		if( body.m_motionId == targetBody.m_motionId )
		{
			// already attached
			continue;
		}

		m_signals.m_bodyAttached.fire( this, bodyId, targetBodyId );

		// We need to make sure that neither the body nor the target is deactivated
		{
			bool activateNow = false;

			if( body.isInactive() )
			{
				activateBody( bodyId );
				activateNow = true;
			}

			if( targetBody.isInactive() )
			{
				activateBody( targetBodyId );
				activateNow = true;
			}

			if( activateNow )
			{
				m_deactivationManager->activateMarkedIslands();
			}
		}

		//
		// Switch motion
		//

		const hkBool32 isAddedToWorld = body.isAddedToWorld();

		if( body.m_motionId != hknpMotionId::STATIC )
		{
			if( isAddedToWorld )
			{
				unregisterBodyAtActiveList( &body );
			}
			removeBodyFromMotion( &body );
		}

		addBodyToMotion( bodyId, targetMotionId, HK_NULL );
		if( isAddedToWorld )
		{
			registerBodyAtActiveList( bodyId, targetMotionId );
			rebuildBodyCollisionCaches( bodyId );
		}

		if( !rebuildMotionProperties )
		{
			// We are not rebuilding mass properties so we need to update the COM radius.
			body.updateComCenteredBoundingRadius( getMotion(body.m_motionId) );
		}

		m_signals.m_bodyChanged.fire( this, bodyId );
	}

	if( rebuildMotionProperties )
	{
		rebuildMotionMassProperties( targetMotionId, REBUILD_DELAYED );
	}

	m_signals.m_bodyChanged.fire( this, targetBodyId );
}


void hknpWorld::detachBodies(
	const hknpBodyId* bodyIds, int numIds, UpdateMassPropertiesMode updateMassProperties )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	hknpWorldEx* self = (hknpWorldEx*)this;

#if defined(HK_DEBUG)
	if ( m_simulationStage == SIMULATION_POST_COLLIDE )
	{
		HK_WARN_ONCE( 0xf0e60192, "Calling detachBodies() from post collide does not update existing contact Jacobians, this can lead to wrong behavior");
	}
#endif

	for( int i=0; i<numIds; ++i )
	{
		hknpBodyId bodyId = bodyIds[i];

		addTrace( hknpDetachBodyCommand( bodyId, updateMassProperties ) );

		hknpBody& body = accessBody(bodyId);
		const hknpMotionId motionId = body.m_motionId;
		HK_ASSERT( 0xf034dfed, motionId.isValid() );

		hknpMotion& motion = accessMotionUnchecked( motionId );
		if( body.m_nextAttachedBodyId == bodyId )
		{
			// This is already the only attached body
			continue;
		}

		if( body.isInactive() )
		{
			activateBody( bodyId );
			m_deactivationManager->activateMarkedIslands();
		}

		// Detach from exiting motion
		hknpMotionCinfo motionCinfo;
		getMotionCinfo( motionId, motionCinfo );
		self->_detachBodyFromMotion( &body, bodyId, &motion );
		self->removeActiveMotionIfNoAddedBodies( motionId );

		// Get/build the body's mass properties in body space
		hkMassProperties newMassPropertiesBodySpace;
		if( body.m_shape->getMassProperties( newMassPropertiesBodySpace ) == HK_FAILURE )
		{
			HK_WARN( 0x7eb21f2c, "A detached shape has no mass properties. Approximating them instead." );
			hknpShape::MassConfig massConfig;
			massConfig.m_quality = hknpShape::MassConfig::QUALITY_LOW;
			hkDiagonalizedMassProperties dmp;
			body.m_shape->buildMassProperties( massConfig, dmp );
			dmp.unpack( &newMassPropertiesBodySpace );
		}

		// Note the angular velocity in world space
		hkVector4 angVelWorld; motion.getAngularVelocity(angVelWorld);

		//
		//	Create new motion
		//
		hknpMotionId newMotionId;
		hknpMotion* HK_RESTRICT newMotion;
		{
			
			newMotionId = createMotion( motionCinfo );
			newMotion = &accessMotionUnchecked( newMotionId );
			m_motionManager.initializeMotion( newMotion, motionCinfo, *m_spaceSplitter );

			newMassPropertiesBodySpace.m_mass *= motionCinfo.m_massFactor;
			newMassPropertiesBodySpace.m_inertiaTensor.mul( hkSimdReal::fromFloat( motionCinfo.m_massFactor ) );

			if( body.m_flags.anyIsSet( hknpBody::IS_KEYFRAMED ) )
			{
				newMotion->m_motionPropertiesId = hknpMotionPropertiesId::DYNAMIC;
				body.m_flags.clear( hknpBody::IS_KEYFRAMED );
			}

			newMotion->setFromMassProperties(newMassPropertiesBodySpace,body.getTransform());
			body.updateMotionToBodyTransform(*newMotion);
			{
				// Add the linear velocity component due to the angular velocity of the compound
				hkVector4 relPos; relPos.setSub(newMotion->getCenterOfMassInWorld(), motion.getCenterOfMassInWorld());
				hkVector4 linVelAdd; linVelAdd.setCross( angVelWorld, relPos );
				newMotion->m_linearVelocity.add( linVelAdd );
			}
			newMotion->m_firstAttachedBodyId = bodyId;
			body.m_motionId = newMotionId;
			body.updateComCenteredBoundingRadius( *newMotion );
		}

		//
		//	Update the remaining mass properties
		//
		if( (updateMassProperties == UPDATE_MASS_PROPERTIES) && !motion.hasInfiniteMass() )
		{
			hkInplaceArray<hkMassElement,2> massElements;	massElements.setSizeUnchecked(2);
			massElements[0].m_transform.setInverse(body.getTransform());
			hkMatrix3 inertiaWorld; motion.getInverseInertiaWorld(inertiaWorld); inertiaWorld.invertSymmetric();
			massElements[0].m_properties.m_inertiaTensor = inertiaWorld;
			massElements[0].m_properties.m_mass = 1.0f / motion.getInverseMass().getReal();
			massElements[0].m_properties.m_centerOfMass = motion.getCenterOfMassInWorld();

			massElements[1].m_transform.setIdentity();
			massElements[1].m_properties = newMassPropertiesBodySpace;	// subtract the new massOfDetachedBody properties
			massElements[1].m_properties.m_mass *= -1.0f;
			massElements[1].m_properties.m_inertiaTensor.mul(hkSimdReal_Minus1);
			body.getCenterOfMassLocal(massElements[1].m_properties.m_centerOfMass);

			hkMassProperties remainingMassProperties;
			hkInertiaTensorComputer::combineMassProperties(massElements, remainingMassProperties);


			hkVector4 prevCOMWorld = motion.getCenterOfMassInWorld();
			motion.setFromMassProperties(remainingMassProperties, body.getTransform());
			{
				// Add the linear velocity component due to the angular velocity of the compound
				hkVector4 relPos; relPos.setSub(motion.getCenterOfMassInWorld(), prevCOMWorld);
				hkVector4 linVelAdd; linVelAdd.setCross( angVelWorld, relPos );
				motion.m_linearVelocity.add(linVelAdd);
			}
			// fix up all attached bodies
			hknpBodyId firstBodyId = motion.m_firstAttachedBodyId;
			hknpBodyId attachedBodyId = firstBodyId;
			do
			{
				hknpBody& attachedBody = m_bodyManager.accessBody(attachedBodyId);
				attachedBody.updateMotionToBodyTransform(motion);
				attachedBody.updateComCenteredBoundingRadius(motion);

				attachedBodyId = attachedBody.m_nextAttachedBodyId;
			} while (attachedBodyId != firstBodyId);
		}

		if( (updateMassProperties == REBUILD_MASS_PROPERTIES) && !motion.hasInfiniteMass() )
		{
			hkVector4 prevCOMWorld = motion.getCenterOfMassInWorld();
			rebuildMotionMassProperties( motionId, REBUILD_NOW );
			{
				// Add the linear velocity component due to the angular velocity of the compound
				hkVector4 relPos; relPos.setSub(motion.getCenterOfMassInWorld(), prevCOMWorld);
				hkVector4 linVelAdd; linVelAdd.setCross( angVelWorld, relPos );
				motion.m_linearVelocity.add(linVelAdd);
			}
		}


		// Make our motion active. We do this at the very end because only now we have the centerOfMass in world needed
		// for the m_cellIndex. Note that we have to tell our new motion that the body is still using the old cell,
		// or updateCellIdx() might do the wrong thing.
		hknpMotionManager::overrideCellIndexInternal( *newMotion, motion.m_cellIndex );
		hknpMotionUtil::updateCellIdx( this, newMotion, newMotionId );
		if( body.isActive() )
		{
			m_motionManager.addActiveMotion( *newMotion, newMotionId );
		}

		// re-enable all collision between the new body and the previous
		{
			hkInplaceArray<hknpBodyIdPair,128> pairs;

			// fix up all attached bodies
			hknpBodyId firstBodyId = motion.m_firstAttachedBodyId;
			hknpBodyId attachedBodyId = firstBodyId;
			const hkArray<hkAabb16>& prevAabbs = m_bodyManager.m_previousAabbs;
			hkAabb16 prevAabb = prevAabbs[ bodyId.value() ];
			do
			{
				const hknpBody& attachedBody = m_bodyManager.getBody(attachedBodyId);

				if( ! prevAabbs[ attachedBodyId.value() ].disjoint( prevAabb ) )
				{
					hknpBodyIdPair* HK_RESTRICT pair = pairs.expandBy(1);
					pair->m_bodyA = bodyId;
					pair->m_bodyB = attachedBodyId;
				}

				attachedBodyId = attachedBody.m_nextAttachedBodyId;
			} while (attachedBodyId != firstBodyId);

			if( pairs.getSize() )
			{
				rebuildBodyPairCollisionCaches( pairs.begin(), pairs.getSize() );
			}
		}

		m_signals.m_bodyDetached.fire( this, bodyId, motionId );

		// Fire body changed signals for all relevant bodies
		if( m_signals.m_bodyChanged.hasSubscriptions() )
		{
			m_signals.m_bodyChanged.fire( this, bodyId );
			hknpBodyId firstBodyId = motion.m_firstAttachedBodyId;
			hknpBodyId attachedBodyId = firstBodyId;
			do
			{
				m_signals.m_bodyChanged.fire( this, attachedBodyId );
				const hknpBody& attachedBody = getBodyUnchecked( attachedBodyId );
				attachedBodyId = attachedBody.m_nextAttachedBodyId;
			} while (attachedBodyId != firstBodyId);
		}
	}

	if( areConsistencyChecksEnabled() )
	{
		checkConsistency();
	}
}


void hknpWorld::addAction( hknpAction* action, hknpActivationMode::Enum activationMode )
{
	checkNotInSimulation( ~SIMULATION_COLLIDE );
	m_actionManager->addAction( this, action, activationMode );
}

void hknpWorld::removeAction( hknpAction* action, hknpActivationMode::Enum activationMode )
{
	checkNotInSimulation( ~SIMULATION_COLLIDE );
	m_actionManager->removeAction( this, action, activationMode );
}

void hknpWorld::setDeactiveCdCacheFilter( hknpDeactiveCdCacheFilter* filter )
{
	m_deactiveCdCacheFilter = filter;
}

hknpEventSignal& hknpWorld::getEventSignal( hknpEventType::Enum eventType, hknpBodyId id )
{
	return m_eventDispatcher->getSignal( eventType, id );
}

hknpEventSignal& hknpWorld::getEventSignal( hknpEventType::Enum eventType )
{
	return m_eventDispatcher->getSignal( eventType );
}

void hknpWorld::optimizeBroadPhase()
{
	checkNotInSimulation( ~SIMULATION_COLLIDE );
	m_broadPhase->optimize( m_bodyManager.accessBodyBuffer() );
}


void hknpWorld::addConstraint( hknpConstraint* constraint, hknpActivationMode::Enum activationMode )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	hknpWorldEx* self = (hknpWorldEx*)this;

	m_constraintAtomSolver->addConstraint( constraint );
	self->handleConstraintActivationMode( constraint, activationMode );

	// Fire signal
	m_signals.m_constraintAdded.fire( this, constraint );
}


void hknpWorld::removeConstraint( hknpConstraint* constraint )
{
	checkNotInSimulation( SIMULATION_POST_COLLIDE );
	m_constraintAtomSolver->removeConstraint( constraint );

	// Fire signal
	m_signals.m_constraintRemoved.fire( this, constraint );
}

void hknpWorldEx::removeActiveMotionIfNoAddedBodies( hknpMotionId motionId )
{
	// Ignore the static motion
	if( motionId == hknpMotionId::STATIC )
	{
		return;
	}

	// Check if the motion is currently active
	hknpMotion& motion = m_bodyManager.m_motionManager->m_motions[motionId.value()];
	if( !motion.m_solverId.isValid() )
	{
		return;
	}

	// Look for attached bodies that are added to the world
	bool allBodiesNotInWorld = true;
	hknpBodyId firstBodyId = motion.m_firstAttachedBodyId;
	hknpBodyId attachedBodyId = firstBodyId;
	do
	{
		hknpBody& attachedBody = m_bodyManager.accessBody(attachedBodyId);
		if(attachedBody.isAddedToWorld())
		{
			allBodiesNotInWorld = false;
			break;
		}
		attachedBodyId = attachedBody.m_nextAttachedBodyId;
	} while (attachedBodyId != firstBodyId);

	if(allBodiesNotInWorld)
	{
		// Deactivate motion
		m_bodyManager.m_motionManager->removeActiveMotion(motion, motionId);
	}
}

void hknpWorldEx::handleConstraintActivationMode( hknpConstraint* constraint, hknpActivationMode::Enum activationMode )
{
	// Note we have to remove the assert since isAddedToWorld is broken, it indicates only if the bodies are added to the broadphase not to the world
	// HK_ASSERT2( 0xf0156b3b, getBody(instance->m_bodyA).isAddedToWorld() && getBody(instance->m_bodyB).isAddedToWorld(),
	// 		"You can only add constraints between bodies that have been added to the world" );

	if (isDeactivationEnabled())
	{
		const hknpBody& bodyA = getBody(constraint->m_bodyIdA);
		const hknpBody& bodyB = getBody(constraint->m_bodyIdB);

		bool inactiveA = bodyA.isInactive() && bodyA.isAddedToWorld();
		bool inactiveB = bodyB.isInactive() && bodyB.isAddedToWorld();
		if ( (activationMode == hknpActivationMode::KEEP_DEACTIVATED) && inactiveA && inactiveB )
		{
			// We need to make sure they are part of the same deactivated island
			hknpIslandId islandA = hknpIslandId(bodyA.getDeactivatedIslandIndex());
			hknpIslandId islandB = hknpIslandId(bodyB.getDeactivatedIslandIndex());
			if (islandA!=islandB)
			{
				m_deactivationManager->connectDeactivatedIslands(islandA, islandB);
			}
		}
		else
		{
			// Activate a body if the other body is active or if activationMode is hknpActivationMode::ACTIVATE.
			if (inactiveA && (bodyB.isActive() || ( activationMode == hknpActivationMode::ACTIVATE )) )
			{
				m_deactivationManager->markIslandForActivation( hknpIslandId(bodyA.getDeactivatedIslandIndex()) ); // Wake up A
			}
			if (inactiveB && (bodyA.isActive() || ( activationMode == hknpActivationMode::ACTIVATE )) )
			{
				m_deactivationManager->markIslandForActivation( hknpIslandId(bodyB.getDeactivatedIslandIndex()) ); // Wake up B
			}
		}
	}
}


void hknpWorld::enableConstraint( hknpConstraint* constraint, hknpActivationMode::Enum activationMode )
{
	checkNotInSimulation();
	hknpWorldEx* self = (hknpWorldEx*)this;

	constraint->m_flags.clear( hknpConstraint::IS_DISABLED );
	self->handleConstraintActivationMode( constraint, activationMode );
}

void hknpWorld::disableConstraint( hknpConstraint* constraint )
{
	checkNotInSimulation();

	constraint->m_flags.orWith(hknpConstraint::IS_DISABLED);
}


#if defined(HK_DEBUG)
void hknpWorld::addTrace( const hknpApiCommand& command )
{
	if ( m_traceDispatcher )
	{
		m_traceDispatcher->exec( command );
	}
}
#endif

hkSecondaryCommandDispatcher* hknpWorld::setTraceDispatcher( hkSecondaryCommandDispatcher* dispatcher )
{
	m_traceDispatcher = dispatcher;
	return dispatcher;
}

void hknpWorldEx::removeBodiesWhichLeftTheBroadPhase( const hknpEventHandlerInput& input, const hknpEvent& event )
{
	const hknpBodyExitedBroadPhaseEvent& bebpEvent = event.asBodyExitedBroadPhaseEvent();
	removeBodies( &bebpEvent.m_bodyId, 1 );
}

void hknpWorldEx::freezeBodiesWhichLeftTheBroadPhase( const hknpEventHandlerInput& input, const hknpEvent& event )
{
	const hknpBodyExitedBroadPhaseEvent& bebpEvent = event.asBodyExitedBroadPhaseEvent();
	setBodyVelocity( bebpEvent.m_bodyId, hkVector4::getZero(), hkVector4::getZero() );
	setBodyMotionProperties( bebpEvent.m_bodyId, hknpMotionPropertiesId::FROZEN );
}

void hknpWorld::deleteAllCaches()
{
	checkNotInSimulation();

	//
	//	iterate over all islands and mark the for activation, and delete all caches
	//
	{
		for( int i =0; i < m_deactivationManager->m_deactivatedIslands.getSize(); i++ )
		{
			hknpDeactivatedIsland* island = m_deactivationManager->m_deactivatedIslands[i];
			if( !island )
			{
				continue;
			}
			island->deleteAllCaches();

			// Record the index of the island to be activated at the beginning of the next step.
			if( !island->m_isMarkedForActivation )
			{
				island->m_isMarkedForActivation = true;
				m_deactivationManager->m_islandsMarkedForActivation.pushBack( hknpIslandId(i) );
			}
		}

		// Need to clear the caches that are scheduled to be moved over...
		m_deactivationManager->m_newActivatedCdCacheRanges.clear();
		m_deactivationManager->m_newActivatedPairs.clear();
	}

	//
	//	Delete all caches in the agents
	//
	hkThreadLocalBlockStreamAllocator heapAllocator( m_persistentStreamAllocator, -1 );
	{
		m_collisionCacheManager->m_cdCacheGrid.clearGrid();
		m_collisionCacheManager->m_cdCacheStream.reset( &heapAllocator );
		m_collisionCacheManager->m_childCdCacheStream.reset( &heapAllocator );

		m_collisionCacheManager->m_inactiveCdCacheStream.reset( &heapAllocator );
		m_collisionCacheManager->m_inactiveChildCdCacheStream.reset( &heapAllocator );

		m_collisionCacheManager->m_newCdCacheGrid.clearGrid();
		m_collisionCacheManager->m_newCdCacheStream.reset( &heapAllocator );
		m_collisionCacheManager->m_newChildCdCacheStream.reset( &heapAllocator );
	}

	// set the previous AABB of all non fixed bodies to invalid
	{
		for (int i =0 ; i < m_bodyManager.m_bodies.getSize();i++)
		{
			const hknpBody& body = m_bodyManager.m_bodies[i];
			if ( body.isStatic() )
			{
				continue;
			}
			m_bodyManager.m_previousAabbs[i].setEmptyKeyUnchanged();
		}
	}

	//
	//	Activate simulation islands now (the activated caches are already deleted)
	//
	{
		// Create a context for command dispatch during activation.
		hknpSimulationThreadContext threadContext;
		hknpCommandGrid grid; grid.setSize( 1 );
		threadContext.init( this, &grid, m_persistentStreamAllocator, -1, true );

		hknpCdCacheStream activatedCdCaches; activatedCdCaches.initBlockStream( threadContext.m_heapAllocator );
		hknpCdCacheStream activatedChildCdCaches; activatedChildCdCaches.initBlockStream( threadContext.m_heapAllocator );
		hkBlockStream<hknpBodyIdPair> newPairs; newPairs.initBlockStream( threadContext.m_heapAllocator );

		threadContext.beginCommands( 0 );
		m_deactivationManager->activateMarkedIslands();
		m_deactivationManager->moveActivatedCaches( threadContext, &activatedCdCaches, &activatedChildCdCaches, &newPairs );
		threadContext.endCommands( 0 );

		HK_ASSERT( 0xf034e545, activatedCdCaches.isEmpty() );
		HK_ASSERT( 0xf034e545, activatedChildCdCaches.isEmpty() );
		HK_ASSERT( 0xf034e545, newPairs.isEmpty() );

		newPairs.clear( threadContext.m_heapAllocator );
		activatedCdCaches.clear( threadContext.m_heapAllocator );
		activatedChildCdCaches.clear( threadContext.m_heapAllocator );

		// Dispatch any commands generated.
		threadContext.dispatchCommands();
		threadContext.shutdownThreadContext( threadContext.m_tempAllocator, threadContext.m_heapAllocator );
	}

	m_bodyManager.rebuildActiveBodyArray();
	m_bodyManager.rebuildBodyIdToCellIndexMap();

	m_motionManager.rebuildMotionHistogram( *m_spaceSplitter );

	if( areConsistencyChecksEnabled() )
	{
		checkConsistency();
	}
}

void hknpWorld::garbageCollectInactiveCacheStreams()
{
	checkNotInSimulation();

	// Create a thread context
	hknpSimulationThreadContext threadContext;
	hknpCommandGrid grid; grid.setSize( 1 );
	threadContext.init( this, &grid, m_persistentStreamAllocator, -1, true );

	// Do the garbage collection
	m_deactivationManager->garbageCollectAllInactiveCaches( threadContext,
		&m_collisionCacheManager->m_inactiveCdCacheStream, &m_collisionCacheManager->m_inactiveChildCdCacheStream );

	// Dispatch any commands generated.
	threadContext.dispatchCommands();
	threadContext.shutdownThreadContext( threadContext.m_tempAllocator, threadContext.m_heapAllocator );
}

void hknpWorld::checkConsistency()
{
#ifdef HK_DEBUG
	HK_TIMER_BEGIN( "CheckConsistency", HK_NULL );
	m_bodyManager.checkConsistency();
	m_simulation->checkConsistency( this, hknpSimulation::CHECK_ALL & (~hknpSimulation::CHECK_CACHE_CELL_INDEX) & (~hknpSimulation::CHECK_FORCE_BODIES_IN_SIMULATION) );
	HK_TIMER_END();
#endif
}


void hknpWorldEx::preCollide( const hknpStepInput& stepInput )
{
	if( m_simulationStage != SIMULATION_DONE && m_simulationStage != SIMULATION_POST_SOLVE )
	{
		HK_ASSERT2(0x331415da, 0, "PreCollide cannot be executed in the current simulation state");
		return;
	}

#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	//
	// Refresh any stale simulation data
	//

	HK_TIMER_BEGIN_LIST2( timerStream, "PreCollide", "RefreshStaleData" );
	{
		// Dirty materials or qualities
		if( m_dirtyMaterials.anyIsSet() || m_dirtyQualities.anyIsSet() )
		{
			HK_TIMER_BEGIN2( timerStream, "RebuildCachesOfDirtyMaterialsOrQualities", HK_NULL );
			for( hknpBodyIterator it = m_bodyManager.getBodyIterator(); it.isValid(); it.next() )
			{
				const hknpBody& body = it.getBody();
				if( body.isAddedToWorld() && (
					m_dirtyMaterials.get( body.m_materialId.value() ) ||
					m_dirtyQualities.get( body.m_qualityId.value() ) ) )
				{
					rebuildBodyCollisionCaches( body.m_id );
				}
			}
			m_dirtyMaterials.assignAll(0);
			m_dirtyQualities.assignAll(0);
			HK_TIMER_END2( timerStream );
		}

		// Mutated shapes
		
		if( m_shapeManager.isAnyShapeMutated() )
		{
			HK_TIMER_BEGIN2( timerStream, "ProcessMutatedShapes", HK_NULL );
			m_shapeManager.processMutatedShapes(this);
			HK_TIMER_END2( timerStream );
		}

		// Mass property changes
		if( !m_bodyManager.m_scheduledBodyChanges.isEmpty() )
		{
			HK_TIMER_BEGIN2( timerStream, "RebuildMotionMassProperties", HK_NULL );
			for( int i=0, ei=m_bodyManager.m_scheduledBodyChanges.getSize(); i<ei; i++ )
			{
				hkUint32 scehduledFlags = m_bodyManager.m_scheduledBodyChanges[i].m_scheduledBodyFlags.get();
				if( scehduledFlags & hknpBodyManager::TO_BE_DELETED )
				{
					continue;
				}

				hknpBodyId bodyId = m_bodyManager.m_scheduledBodyChanges[i].m_bodyId;

				if( scehduledFlags & hknpBodyManager::REBUILD_MASS_PROPERTIES )
				{
					const hknpBody& body = getBody( bodyId );
					rebuildMotionMassProperties( body.m_motionId, REBUILD_NOW );
				}
			}
			HK_TIMER_END2( timerStream );
		}
	}

	// Fire pre collide signal
	if( m_signals.m_preCollide.hasSubscriptions() )
	{
		HK_TIMER_SPLIT_LIST2( timerStream, "PreCollideSignal" );
		m_signals.m_preCollide.fire( this );
	}

	m_simulationStage = SIMULATION_PRE_COLLIDE;
	m_stepLocalStreamAllocator = stepInput.m_stepLocalStreamAllocator;

	if( areConsistencyChecksEnabled() )
	{
		checkConsistency();
	}

	// Body, motion maintenance
	HK_TIMER_SPLIT_LIST2( timerStream, "AddRemoveBodies" );
	{
		m_bodyManager.deleteMarkedBodies();
		m_motionManager.deleteMarkedMotions();

		if( areConsistencyChecksEnabled() )
		{
			checkConsistency();
		}

		commitAddBodies();
	}

	m_solverInfo.setStepInfo(
		m_gravity, m_collisionTolerance.getReal(), stepInput.m_deltaTime,
		m_solverInfo.m_numSteps, m_solverInfo.m_numMicroSteps );

	// Create simulation context.
	// We need to new because of SPU access to this thing.
	HK_TIMER_SPLIT_LIST2( timerStream, "CreateSimulationContext" );
	{
		m_simulationContext = new hknpSimulationContext;
		m_simulationContext->init( this, stepInput, m_stepLocalStreamAllocator );
	}

	m_deactivationManager->activateMarkedIslands();

	m_simulationStage = SIMULATION_COLLIDE;

	HK_TIMER_END_LIST2( timerStream ); // "PreCollide"
}


void hknpWorld::generateCollideTasks( const hknpStepInput& stepInput, hkTaskGraph* taskGraph, hknpSolverData*& solverDataOut )
{
	HK_CHECK_FLUSH_DENORMALS();
	hknpWorldEx* self = (hknpWorldEx*)this;

#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	HK_TIMER_BEGIN2( timerStream, "Collide", HK_NULL );

	switch( m_simulationStage )
	{
	case SIMULATION_DONE:
		{
			self->_checkConsistencyOfActiveBodies();

			// Add the pre collide task
			m_preCollideTask->m_world = this;
			m_preCollideTask->m_stepInput = stepInput;
			taskGraph->addTasks( (hkTask**)&m_preCollideTask, 1 );
		}
		break;

	case SIMULATION_COLLIDE:
		{
			// Call the simulation
			m_simulationContext->m_taskGraph = taskGraph;
			hknpSolverData* solverData;
			m_simulation->collide( *m_simulationContext, solverData );
			if( taskGraph->getNumTasks() == 0 )
			{
				// Add the post collide task
				HK_ASSERT( 0x62719db0, solverData );
				solverData->m_simulationContext = m_simulationContext;

				m_postCollideTask->m_world = this;
				m_postCollideTask->m_solverData = solverData;
				taskGraph->addTasks( (hkTask**)&m_postCollideTask, 1 );

				m_simulationStage = SIMULATION_POST_COLLIDE;
			}
		}
		break;

	case SIMULATION_POST_COLLIDE:
		{
			// Return the solver data
			HK_ASSERT( 0x62719db1, m_postCollideTask->m_solverData );
			solverDataOut = m_postCollideTask->m_solverData;
		}
		break;

	default:
		{
			HK_ASSERT2( 0x331415da, 0, "Collide cannot be executed in the current simulation state" );
		}
	}

	HK_TIMER_NAMED_END2( timerStream, "Collide" );
}


void hknpWorldEx::postCollide( hknpSolverData* solverData )
{
	if( m_simulationStage != SIMULATION_POST_COLLIDE )
	{
		HK_ASSERT2(0x331415da, 0, "PostCollide cannot be executed in the current simulation state");
		return;
	}

#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	HK_TIMER_BEGIN2( timerStream, "PostCollide", HK_NULL );

	hknpSimulationThreadContext* threadContext = m_simulationContext->getThreadContext();

	// Dispatch commands
	{
		HK_TIMER_BEGIN2( timerStream, "DispatchCommands", HK_NULL );

		m_eventDispatcher->beginDispatch( solverData, threadContext );
		m_internalPhysicsCommandDispatcher->beginDispatch( solverData, threadContext );

		m_simulationContext->dispatchPostCollideCommands( threadContext->m_world->m_commandDispatcher, m_eventDispatcher );

		m_internalPhysicsCommandDispatcher->endDispatch();
		m_eventDispatcher->endDispatch();

		HK_TIMER_END2( timerStream );
	}

	// Setup command writers on CPU thread contexts (SPUs don't have them enabled)
	{
		for( int i = 0; i < m_simulationContext->getNumCpuThreads(); i++ )
		{
			hknpSimulationThreadContext* otherTl = m_simulationContext->getThreadContext(i);
			otherTl->beginCommands(i);
		}
	}

	// Fire post collide signal
	if( m_signals.m_postCollide.hasSubscriptions() )
	{
		HK_TIMER_BEGIN2( timerStream, "PostCollideSignal", HK_NULL );
		m_signals.m_postCollide.fire( this, solverData );
		HK_TIMER_END2( timerStream );
	}

	HK_TIMER_NAMED_END2( timerStream, "PostCollide" );
}


void hknpWorld::stepCollide(
	const hknpStepInput& stepInput, hkTaskQueue* taskQueue, hknpSolverData*& solverDataOut )
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	HK_TIMER_BEGIN2( timerStream, "Physics", HK_NULL );

	solverDataOut = HK_NULL;
	while(1)
	{
		hkDefaultTaskGraph taskGraph;
		generateCollideTasks( stepInput, &taskGraph, solverDataOut );

		const int numTasks = taskGraph.getNumTasks();
		if( numTasks == 0 )
		{
			// Nothing more to do
			break;
		}

		const hkDefaultTaskGraph::TaskInfo& taskInfo = taskGraph.m_taskInfos[0];
		if( numTasks == 1 && taskInfo.m_multiplicity == 1
			HK_ON_PLATFORM_HAS_SPU( && taskInfo.m_task->getElf() == HK_INVALID_ELF ) )
		{
			// If we have just one CPU task with multiplicity 1, process it on this thread
			HK_TIMER_BEGIN2( timerStream, "Collide", HK_NULL );
			taskGraph.getTask( hkDefaultTaskGraph::TaskId(0) )->process();
			HK_TIMER_END2( timerStream );
		}
		else
		{
			// Add our tasks to the queue, where they will be available for other threads to process
			HK_ASSERT2( 0x4a47fb0d, taskQueue != HK_NULL, "No task queue provided" );
			hkTaskQueue::GraphId graphId = taskQueue->addGraph( &taskGraph, 0 );

			// Process them on this thread too, blocking until all our tasks have finished processing
			HK_TIMER_BEGIN2( timerStream, "Collide", HK_NULL );
			taskQueue->processGraph( graphId, hkTaskQueue::WAIT_UNTIL_ALL_TASKS_FINISHED );
			HK_TIMER_END2( timerStream );

			// Remove our processed tasks from the queue
			taskQueue->removeGraph( graphId );
		}
	};

	HK_TIMER_NAMED_END2( timerStream, "Physics" );
}


void hknpWorldEx::preSolve( hknpSolverData* solverData )
{
	if( m_simulationStage != SIMULATION_POST_COLLIDE )
	{
		HK_ASSERT2(0x331415da, 0, "PreSolve cannot be executed in the current simulation state");
		return;
	}

	HK_TIMER_BEGIN( "PreSolve", HK_NULL );

	if( m_signals.m_preSolve.hasSubscriptions() )
	{
		HK_TIME_CODE_BLOCK( "PreSolveSignal", this );
		m_signals.m_preSolve.fire( this, solverData );
	}

	for( int i = 0; i < m_simulationContext->getNumCpuThreads(); i++ )
	{
		hknpSimulationThreadContext* otherTl = m_simulationContext->getThreadContext(i);
		otherTl->endCommands(i);
	}

	HK_TIMER_NAMED_END("PreSolve");
}


void hknpWorld::generateSolveTasks( hknpSolverData*& solverData, hkTaskGraph* taskGraph )
{
	HK_CHECK_FLUSH_DENORMALS();
	hknpWorldEx* self = (hknpWorldEx*)this;

#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	HK_TIMER_BEGIN2( timerStream, "Solve", HK_NULL );

	switch( m_simulationStage )
	{
	case SIMULATION_POST_COLLIDE:
		{
			//
			// Single threaded setup phase
			//

			self->_checkConsistencyOfActiveBodies();
			self->preSolve( solverData );
			self->_checkConsistencyOfActiveBodies();

			m_simulationStage = SIMULATION_PRE_SOLVE;
		}
		// fall through

	case SIMULATION_PRE_SOLVE:
	case SIMULATION_SOLVE:
		{
			//
			// Multi threaded phase(s)
			//

			// Check for new inactive caches
			m_deactivationManager->clearAndTrackIslandForDeactivatingCaches();

			hknpSimulationContext* simulationContext = solverData->m_simulationContext;
			hknpSimulationThreadContext* threadContext = simulationContext->getThreadContext();

			// Dispatch all collide events. This is done here so that collide events get a chance to modify the data going into the solver
			
			{
				HK_TIMER_BEGIN2( timerStream, "DispatchCommands", HK_NULL);

				m_eventDispatcher->beginDispatch( solverData, threadContext );
				m_internalPhysicsCommandDispatcher->beginDispatch( solverData, threadContext );

				m_simulationContext->dispatchPostCollideCommands( threadContext->m_world->m_commandDispatcher, m_eventDispatcher );

				m_internalPhysicsCommandDispatcher->endDispatch();
				m_eventDispatcher->endDispatch();

				HK_TIMER_END2( timerStream );
			}

			// Call the simulation
			simulationContext->m_taskGraph = taskGraph;
			m_simulation->solve( *simulationContext, solverData );
			if( taskGraph->getNumTasks() == 0 )
			{
				// Add the post solve task
				m_postSolveTask->m_world = this;
				m_postSolveTask->m_solverData = solverData;
				taskGraph->addTasks( (hkTask**)&m_postSolveTask, 1 );

				m_simulationStage = SIMULATION_POST_SOLVE;
			}
		}
		break;

	case SIMULATION_POST_SOLVE:
		{
			solverData = HK_NULL;
			m_simulationStage = SIMULATION_DONE;
			self->_checkConsistencyOfActiveBodies();
		}
		break;

	default:
		{
			HK_ASSERT2( 0x331415da, 0, "Solve cannot be executed in the current simulation state" );
		}
		break;
	}

	HK_TIMER_NAMED_END2( timerStream, "Solve" );
}


void hknpWorldEx::postSolve( hknpSolverData* solverData )
{
	if( m_simulationStage != SIMULATION_POST_SOLVE )
	{
		HK_ASSERT2(0x331415da, 0, "PostSolve cannot be executed in the current simulation state");
		return;
	}

	HK_TIMER_BEGIN("PostSolve", HK_NULL);

	hknpSimulationContext* simulationContext = solverData->m_simulationContext;
	hknpSimulationThreadContext* threadContext = simulationContext->getThreadContext();

	{
		HK_TIME_CODE_BLOCK("clearAllScheduledBodyChanges", this);
		m_bodyManager.clearAllScheduledBodyChanges();
	}

	// Execute commands
	{
		HK_TIME_CODE_BLOCK("DispatchCommands", this);
		simulationContext->dispatchCommands( m_commandDispatcher );
		m_eventDispatcher->flushRemainingEvents();
	}

	m_solverInfo.m_stepSolveCount++;

	// Fire post simulation signal
	if( m_signals.m_postSolve.hasSubscriptions() )
	{
		HK_TIMER_BEGIN("PostSolveSignal", HK_NULL);
		m_signals.m_postSolve.fire( this );
		HK_TIMER_END();
	}

	// Shut down simulation context
	{
		solverData->clear(threadContext->m_tempAllocator);
		delete solverData;

		simulationContext->clear();
		delete simulationContext;
		solverData = HK_NULL;
	}

	m_stepLocalStreamAllocator->freeAllRemainingAllocations();

	HK_TIMER_NAMED_END("PostSolve");
}


void hknpWorld::stepSolve( hknpSolverData*& solverData, hkTaskQueue* taskQueue )
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	HK_TIMER_BEGIN2( timerStream, "Physics", HK_NULL );

	while(1)
	{
		hkDefaultTaskGraph taskGraph;
		generateSolveTasks( solverData, &taskGraph );

		const int numTasks = taskGraph.getNumTasks();
		if( numTasks == 0 )
		{
			// Nothing more to do
			break;
		}

		const hkDefaultTaskGraph::TaskInfo& taskInfo = taskGraph.m_taskInfos[0];
		if( numTasks == 1 && taskInfo.m_multiplicity == 1
			HK_ON_PLATFORM_HAS_SPU( && taskInfo.m_task->getElf() == HK_INVALID_ELF ) )
		{
			// If we have just one CPU task with multiplicity 1, process it on this thread
			HK_TIMER_BEGIN2( timerStream, "Solve", HK_NULL );
			taskGraph.getTask( hkDefaultTaskGraph::TaskId(0) )->process();
			HK_TIMER_END2( timerStream );
		}
		else
		{
			// Add our tasks to the queue, where they will be available for other threads to process
			HK_ASSERT2( 0x4a47fb0e, taskQueue != HK_NULL, "No task queue provided" );
			hkTaskQueue::GraphId graphId = taskQueue->addGraph( &taskGraph, 0 );

			// Process them on this thread too, blocking until all our tasks have finished processing
			HK_TIMER_BEGIN2( timerStream, "Solve", HK_NULL );
			taskQueue->processGraph( graphId, hkTaskQueue::WAIT_UNTIL_ALL_TASKS_FINISHED );
			HK_TIMER_END2( timerStream );

			// Remove our processed tasks from the queue
			taskQueue->removeGraph( graphId );
		}
	};

	HK_TIMER_NAMED_END2( timerStream, "Physics" );
}


void hknpWorld::shiftWorld( hkVector4Parameter shift )
{
	hknpWorldShiftUtil::shiftWorld( this, shift );
}

void hknpWorld::shiftBroadPhase( hkVector4Parameter requestedCenterPos, hkVector4& effectiveCenterPosOut, hkArray<hknpBodyId>* bodiesOutsideTheBroadPhase )
{
	hknpWorldShiftUtil::shiftBroadPhase( this, requestedCenterPos, effectiveCenterPosOut, bodiesOutsideTheBroadPhase );
}

void hknpWorldEx::onMaterialModifiedSignal( hknpMaterialId materialId )
{
	checkNotInSimulation();
	m_dirtyMaterials.set(materialId.value());
}

void hknpWorldEx::onMaterialAddedSignal( hknpMaterialId materialId )
{
	checkNotInSimulation(SIMULATION_POST_COLLIDE);

	// Resize dirty materials bitfield if necessary
	const int capacity = m_materialLibrary->getCapacity();
	if (m_dirtyMaterials.getSize() != capacity)
	{
		m_dirtyMaterials.setSizeAndFill(0, capacity, 0);
	}
}

#if defined(HK_DEBUG)

void hknpWorldEx::onMaterialRemovedSignal( hknpMaterialId materialId )
{
	checkNotInSimulation();

	// Check if any bodies are using the materials
	
	for( hknpBodyIterator it = getBodyIterator(); it.isValid(); it.next() )
	{
		HK_ASSERT2( 0x431cb5a2, it.getBody().m_materialId != materialId, "Destroying a material that is in use by a body" );
	}
}

void hknpWorldEx::onMotionPropertiesAddedSignal( hknpMotionPropertiesId motionPropertiesId )
{
	checkNotInSimulation(SIMULATION_POST_COLLIDE);
}

void hknpWorldEx::onMotionPropertiesRemovedSignal( hknpMotionPropertiesId motionPropertiesId )
{
	checkNotInSimulation();

	// Check if any motions are using the motion properties
	for( hknpMotionIterator it(m_motionManager); it.isValid(); it.next() )
	{
		HK_ASSERT2( 0x431cb5a3, it.getMotion().m_motionPropertiesId != motionPropertiesId,
			"Removing a motion properties that is in use by a motion" );
	}
}

#endif // !HK_DEBUG

void hknpWorldEx::onQualityModifiedSignal( hknpBodyQualityId qualityId )
{
	checkNotInSimulation();
	m_dirtyQualities.set( qualityId.value() );
}

void hknpWorld::setAsDebugInstance() const
{
	g_hknpWorldDebugInstance = this;
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
