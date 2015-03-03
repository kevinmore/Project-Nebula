/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_WORLD_CINFO_H
#define HKNP_WORLD_CINFO_H

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Collide/BroadPhase/hknpBroadPhaseConfig.h>
#include <Physics/Physics/Collide/Filter/hknpCollisionFilter.h>
#include <Physics/Physics/Collide/Shape/TagCodec/hknpShapeTagCodec.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterialLibrary.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionPropertiesLibrary.h>
#include <Physics/Physics/Dynamics/Body/hknpBodyQualityLibrary.h>


/// Construction info for a hknpWorld.
struct hknpWorldCinfo
{
	//+version(1)

	public:

		/// Simulation type.
		enum SimulationType
		{
			SIMULATION_TYPE_SINGLE_THREADED,	///< Single threaded simulation.
			SIMULATION_TYPE_MULTI_THREADED		///< Multi threaded simulation.
		};

		/// Predefined settings to specify the speed/stiffness balance of the solver.
		/// The solver is an iterative solver and similar to a "successive over relaxation" (SOR) method. That means
		/// that each constraint will not be solved perfectly, but rather solved like a very stiff spring, which gives
		/// the solver a "soft" feeling. So the goal of the solver is to make the constraint to appear hard.
		/// There are 3 parameters to control the behavior of the solver:
		///  - m_solverIterations: defines how often the solver iterates over all constraints. The higher the number,
		///    the stiffer the constraints, however you need to spend more CPU. Values between 2 and 16 are reasonable,
		///    higher values could lead to instable behavior (in this case you can achieve the same results by
		///    decreasing the step time).
		///  - m_tau: defines how much of the current constraint error is solved every solver iteration. High values
		///    (0.8 .. 1.2 ) make the constraints harder, however they get more unstable as well. Smaller values
		///    (0.3 .. 0.6) give much softer and smoother behavior, however you have to set the number of iterations
		///    pretty high to achieve a stiff behavior.
		///  - m_damping: defines how the current bodies velocity is taken into account. If you set damping to 0, you
		///    get an explicit solver, setting it to m_tau makes the solver semi-implicit. A good choice is to set it
		///    to tau.
		///  This enum list allows to define some reasonable defaults: NITERS refers to the number of iterations,
		///  SOFT/MEDIUM/HARD refers to the value of tau and damp.
		enum SolverType
		{
			SOLVER_TYPE_INVALID,

			SOLVER_TYPE_2ITERS_SOFT,	///< "Softest" solver settings.
			SOLVER_TYPE_2ITERS_MEDIUM,
			SOLVER_TYPE_2ITERS_HARD,
			SOLVER_TYPE_4ITERS_SOFT,
			SOLVER_TYPE_4ITERS_MEDIUM,	///< Default solver settings.
			SOLVER_TYPE_4ITERS_HARD,
			SOLVER_TYPE_8ITERS_SOFT,
			SOLVER_TYPE_8ITERS_MEDIUM,
			SOLVER_TYPE_8ITERS_HARD,	///< "Hardest" solver settings.

			SOLVER_TYPE_MAX
		};

		/// Defines the behavior of the engine in regards to bodies leaving the broad phase.
		enum LeavingBroadPhaseBehavior
		{
			/// Body continues to be simulated but at a higher performance cost.
			ON_LEAVING_BROAD_PHASE_DO_NOTHING,

			/// Body is automatically removed from the world.
			ON_LEAVING_BROAD_PHASE_REMOVE_BODY,

			/// Body is frozen, i.e. its velocity is set to zero and its motion properties to hknpMotionPropertiesId::FROZEN.
			ON_LEAVING_BROAD_PHASE_FREEZE_BODY,
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpWorldCinfo );
		HK_DECLARE_REFLECTION();

		/// Constructor. Sets default values.
		hknpWorldCinfo();

		/// Serialization constructor.
		hknpWorldCinfo( hkFinishLoadedObjectFlag flag );

		/// Set m_broadPhaseAabb to a cube centered at the origin, with the given extents.
		/// That means an object can travel half this distance away from the origin in all axes.
		void setBroadPhaseSize( hkReal side );

		/// Set m_solverTau, m_solverDamp and m_solverIterations using predefined values.
		/// Each solver type provides a different speed/stiffness balance for the simulation.
		/// The default type is SOLVER_TYPE_4ITERS_MEDIUM, which is fast and moderately soft.
		void setSolverType( SolverType type );

	public:

		//
		// Resources
		//

		/// The number of bodies that can fit in the body buffer.
		/// This includes special body 0 which is managed by the engine.
		/// Note that the body buffer can be resized after world construction by relocating it.
		hkInt32 m_bodyBufferCapacity;	//+default(4096)

		/// Optional user managed body storage. Size must be >= m_bodyBufferCapacity * sizeof(hknpBody).
		/// If set to NULL, a buffer will be allocated and managed automatically.
		hknpBody* m_userBodyBuffer;		//+default(0) //+nosave

		/// The number of motions that can fit in the motion buffer.
		/// This includes special motion 0 which is used by all static bodies.
		/// Note that the motion buffer can be resized after world construction by relocating it.
		hkInt32 m_motionBufferCapacity;	//+default(4096)

		/// Optional user managed motion storage. Size must be >= m_motionBufferCapacity * sizeof(hknpMotion).
		/// If set to NULL, a buffer will be allocated and managed automatically.
		hknpMotion* m_userMotionBuffer;	//+default(0) //+nosave

		/// Optional user owned material library.
		/// If set to NULL, a default library will be created automatically.
		hkRefPtr<hknpMaterialLibrary> m_materialLibrary; //+default(0)

		/// Optional user owned motion properties library.
		/// If set to NULL, a default library will be created automatically.
		hkRefPtr<hknpMotionPropertiesLibrary> m_motionPropertiesLibrary; //+default(0)

		/// Optional user owned body quality library.
		/// If set to NULL, a default library will be created automatically.
		hkRefPtr<hknpBodyQualityLibrary> m_qualityLibrary; //+default(0)

		/// Block stream allocator used for persistent simulation caches.
		/// Note that on PS3, only the fixed block stream allocator is supported
		/// Typical maximum memory usage:
		///		100K for the master thread
		///		50K per thread
		///		1K per body
		hkBlockStreamAllocator* m_persistentStreamAllocator; //+nosave

		//
		// Broad phase
		//

		/// The AABB of the broad phase.
		/// There is a significant performance cost for bodies that leave this volume.
		/// Defaults to a cube of side 1000, centered at the origin.
		hkAabb m_broadPhaseAabb;

		/// Defines the behavior of the engine in regards to bodies outside the broad phase AABB.
		/// Default behavior is to freeze those bodies.
		hkEnum<LeavingBroadPhaseBehavior,hkUint8> m_leavingBroadPhaseBehavior; //+default(hknpWorldCinfo::ON_LEAVING_BROAD_PHASE_FREEZE_BODY)

		/// Defines how the layers are setup in the broad phase.
		/// If set to NULL, a default broad phase configuration will be created (hknpDefaultBroadPhaseConfig).
		hkRefPtr<hknpBroadPhaseConfig> m_broadPhaseConfig; //+default(0)

		//
		// Collision detection
		//

		/// The collision filter used during simulation.
		/// Defaults to HK_NULL.
		hkRefPtr<hknpCollisionFilter> m_collisionFilter; //+default(0)

		/// The collision filter used during collision queries.
		/// Defaults to HK_NULL.
		hkRefPtr<hknpCollisionFilter> m_collisionQueryFilter; //+default(0)

		/// The shape tag codec to use for encoding/decoding a shape tag.
		/// Defaults to HK_NULL (and thus to the hknpNullShapeTagCodec).
		hkRefPtr<hknpShapeTagCodec> m_shapeTagCodec; //+default(0)

		/// A distance below which contact manifolds are always generated between any pair of bodies.
		/// Every body's AABB and narrow phase collision tolerance is expanded dynamically by the engine in order to
		/// create speculative contact manifolds. This value places a lower bound on that expansion so that any bodies
		/// closer than this distance will always generate manifolds. Higher values will result in more manifolds being
		/// created, against more distant bodies, which globally reduces the possibility of tunneling but at added cost.
		/// Defaults to 0.05f.
		/// Note if you are using m_solverDamp>1.0f, your solver might bounce object so that they end up outside the
		///		collision tolerance after one step. In this case restitution will not be applied. So if you are using
		///		a hard solver, set this m_collisionTolerance to higher values like 0.1f.
		hkReal m_collisionTolerance; //+default(0.05f)

		/// The accuracy of the collision detection relative to object scale.
		/// Large values (>1cm) results in less 'jittery' collision manifold changes but in larger artifacts should a manifold be updated.
		/// Small values (<1mm) can result in high frequency manifold updates with very little artifacts should an update happen.
		
		hkReal m_relativeCollisionAccuracy; //+default(0.05f)

		//
		// Integration
		//

		/// The global acceleration due to gravity, applied to all dynamic motions.
		/// This can be scaled per motion using hknpMotionProperties::m_gravityFactor.
		/// Defaults to (0,-9.81,0).
		hkVector4 m_gravity; //+default(0.0f,-9.81f,0.0f,0.0f)

		//
		// Solving
		//

		/// Number of constraint solver iterations during each step.
		/// Predefined solver settings can be automatically set using the setSolverType() method.
		hkInt32 m_solverIterations; //+default(4)

		/// Parameter of the constraint solver.
		/// Predefined solver settings can be automatically set using the setSolverType() method.
		hkReal m_solverTau; //+default(0.6f)

		/// Parameter of the constraint solver.
		/// Predefined solver settings can be automatically set using the setSolverType() method.
		hkReal m_solverDamp; //+default(1.0f)

		/// This is the number of Gauss-Seidel steps the solver performs during each solver step.
		/// Basically this allows for virtually increasing the number of solver steps without adding extra instability.
		/// However the micro-steps are not as powerful as the main solver steps.
		/// So by default this value should be 1. Only if you wish to increase the number of solver steps to be higher
		/// than 8 and you experience stability problems, you can increase this value and leave the number of solver
		/// steps to 4 or 8.
		hkInt32 m_solverMicrosteps; //+default(1)

		/// The preferred physics time step. This value is used to tune some internal constants.
		/// When in doubt always choose the highest timestep (==lowest frequency).
		/// Defaults to 1/30 (30Hz).
		hkReal m_defaultSolverTimestep; //+default(1.0f/30.0f)

		/// A speed threshold to limit high quality solving for any contacts using the
		/// BodyQuality::HIGH_QUALITY_SOLVER flag.
		/// Defaults to 1.0.
		
		hkReal m_maxApproachSpeedForHighQualitySolver; //+default(1.0f)

		/// If set, the solver will use a dynamic scheduler instead of the default static one.
		/// Defaults to false.
		hkBool m_enableSolverDynamicScheduling;	//+default(false)

		//
		// Deactivation
		//

		/// Set to false if you want deactivation to be disabled.
		/// Defaults to true.
		hkBool m_enableDeactivation; //+default(true)

		/// If the number of motions in a deactivated island is higher than this value it will do special allocations
		/// that uses slightly more space but prevents CPU spikes in the garbage collection of deactivated islands.
		hkInt32 m_largeIslandSize; //+default(100)

		//
		// Simulation
		//

		/// The simulation type. If the engine is compiled single threaded, this defaults to
		/// SIMULATION_TYPE_SINGLE_THREADED, otherwise it defaults to SIMULATION_TYPE_MULTI_THREADED.
		hkEnum<SimulationType,hkUint8> m_simulationType; //+default(hknpWorldCinfo::SIMULATION_TYPE_MULTI_THREADED)

		/// The number of cells to be used for the space splitter when simulation type is SIMULATION_TYPE_MULTI_THREADED.
		/// Must be a multiple of 4. Defaults to 16.
		hkInt32 m_numSplitterCells;	//+default(16)

		/// Some events may cancel each other out within a single frame: event merging functionality detects such cases
		/// before events are dispatched and provides a solution, at a very small runtime cost. For example, a trigger
		/// volume's "entered" and "left" event will be replaced by a single "data updated" event if this is set to true.
		hkBool m_mergeEventsBeforeDispatch;	 //+default(true)

		//
		// Advanced
		//

		/// The unit scale, which is applied to various distance based constants within the engine.
		/// The engine is tuned to work in meter units, which is appropriate for bodies sized in the range of 10cm to 10m.
		/// If all of your bodies deviate from this range significantly you can change this parameter.
		/// For example if your average body size is 1cm, you can set this to 0.01. Defaults to 1.0.
		hkReal m_unitScale; //+default(1.0f)
};


#endif // HKNP_WORLD_CINFO_H

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
