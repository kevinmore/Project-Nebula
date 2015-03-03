/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_WORLD_H
#define HKNP_WORLD_H

#include <Common/Base/Types/hkSignalSlots.h>
#include <Common/Base/Types/Geometry/IntSpaceUtil/hkIntSpaceUtil.h>
#include <Common/Base/Thread/Task/hkTaskQueue.h>

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Collide/Shape/TagCodec/Null/hknpNullShapeTagCodec.h>
#include <Physics/Physics/Dynamics/Body/hknpBody.h>
#include <Physics/Physics/Dynamics/Body/hknpBodyManager.h>
#include <Physics/Physics/Dynamics/Body/hknpBodyQualityLibrary.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionPropertiesLibrary.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionManager.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterialLibrary.h>
#include <Physics/Physics/Dynamics/World/hknpStepInput.h>
#include <Physics/Physics/Dynamics/World/hknpWorldCinfo.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventType.h>
#include <Physics/Physics/Dynamics/World/hknpWorldSignals.h>
#include <Physics/Physics/Dynamics/World/ShapeManager/hknpShapeManager.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverVelocity.h>

class hkPrimaryCommandDispatcher;
class hkSecondaryCommandDispatcher;
class hknpActionManager;
class hknpApiCommand;
class hknpBroadPhaseDispatcher;
class hknpCollisionCacheManager;
class hknpCollisionDispatcher;
class hknpInternalCommandProcessor;
class hknpConstraint;
class hknpConstraintAtomSolver;
class hknpContactSolver;
class hknpDeactiveCdCacheFilter;
class hknpEventDispatcher;
struct hknpEventSignal;
class hknpSolverData;
class hknpDefaultModifierSet;
class hkRefCountedProperties;
class hknpSimulationContext;
class hknpMotionWeldManager;
class hknpMotionConstraintManager;
class hknpCollisionQueryDispatcherBase;
struct hkTaskGraph;


/// A world is responsible for *storing* all data required by a simulation.
/// The actual simulation is performed by \a hknpSimulation.
class hknpWorld : public hkReferencedObject
{
	public:

		/// Flags to control body addition behavior.
		/// The default behavior is to add bodies as active at the start of the next step.
		enum AdditionFlagsEnum
		{
			DO_NOT_ADD_BODY		= 1<<0, ///< Do not add the body. Overrides all other flags.
			ADD_BODY_NOW		= 1<<1, ///< Add the body immediately instead of waiting until the next step.
			START_DEACTIVATED	= 1<<2, ///< Add the body as inactive, if possible.
		};
		typedef hkFlags<AdditionFlagsEnum, hkUint8> AdditionFlags;

		/// Pivot location about which to rotate a body, used in e.g. setBodyOrientation().
		enum PivotLocation
		{
			PIVOT_BODY_POSITION,	///< Rotate around the body's position.
			PIVOT_CENTER_OF_MASS	///< Rotate around the motion's center of mass.
		};

		/// Controls how to update mass and inertia when bodies are attached or detached (e.g. attachBodies()).
		enum UpdateMassPropertiesMode
		{
			REBUILD_MASS_PROPERTIES,	///< Rebuild the mass properties.
			UPDATE_MASS_PROPERTIES,		///< Incrementally update the mass properties (can lose precision).
			KEEP_MASS_PROPERTIES		///< Keep the existing mass properties.
		};

		/// Controls whether to rebuild collision caches when some functions are called.
		enum RebuildCachesMode
		{
			REBUILD_COLLISION_CACHES,	///< Rebuild all relevant collision caches.
			KEEP_COLLISION_CACHES,		///< Keep existing collision caches. For expert use only.
		};

		/// Controls when to rebuild a motion's mass properties.
		enum RebuildMassPropertiesMode
		{
			REBUILD_NOW,
			REBUILD_DELAYED,
		};

		/// A set of stages that the world goes through during a single simulation step.
		enum SimulationStage
		{
			SIMULATION_DONE			= 1<<0,
			SIMULATION_PRE_COLLIDE	= 1<<1,
			SIMULATION_COLLIDE		= 1<<2,
			SIMULATION_POST_COLLIDE	= 1<<3,
			SIMULATION_PRE_SOLVE	= 1<<4,
			SIMULATION_SOLVE		= 1<<5,
			SIMULATION_POST_SOLVE	= 1<<6,
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor.
		hknpWorld( const hknpWorldCinfo& cinfo );

		/// Destructor.
		virtual ~hknpWorld();


		/// Set the global gravity vector.
		HK_FORCE_INLINE void setGravity( hkVector4Parameter gravity );

		/// Get the global gravity vector.
		HK_FORCE_INLINE const hkVector4& getGravity() const;

		/// Set the global collision filter.
		HK_FORCE_INLINE void setCollisionFilter( hknpCollisionFilter* collisionFilter );

		/// Get the global collision filter.
		HK_FORCE_INLINE hknpCollisionFilter* getCollisionFilter() const;

		/// Set the global shape tag codec.
		HK_FORCE_INLINE void setShapeTagCodec( hknpShapeTagCodec* shapeTagCodec );

		/// Get the global shape tag codec.
		HK_FORCE_INLINE const hknpShapeTagCodec* getShapeTagCodec() const;

		/// Get access to the internal time step parameters.
		HK_FORCE_INLINE const hknpSolverInfo& getSolverInfo() const;

		/// Build a construction info structure describing the world configuration.
		/// This does not include any bodies, constraints, etc that may have been added.
		void getCinfo( hknpWorldCinfo& info ) const;


		// -------------------------------------------------------------------------------------------------------------
		// Stepping
		// -------------------------------------------------------------------------------------------------------------

		/// Advance the current collision detection step and populate the given task graph with any generated tasks.
		/// If any tasks are generated, they must be processed by the caller then this method must then be called again.
		/// When no more tasks are generated, solverDataOut will contain the collision detection results which should be
		/// passed to stepSolve().
		void generateCollideTasks( const hknpStepInput& stepInput, hkTaskGraph* taskGraph, hknpSolverData*& solverDataOut );

		/// Advance the current solving step and populate the given task graph with any generated tasks.
		/// If any tasks are generated, they must be processed by the caller then this method must then be called again.
		/// When solving is finished, solverData will have been consumed and set to NULL.
		/// For single threaded simulations, taskGraph can be set to NULL.
		void generateSolveTasks( hknpSolverData*& solverData, hkTaskGraph* taskGraph );

		/// Helper method to generate and process all collision detection tasks using a single call.
		/// Any generated tasks will be added to the given task queue, where they will be processed by the calling thread
		/// and by any thread pool that might be processing that queue.
		/// For single threaded simulations, taskQueue can be set to NULL.
		void stepCollide( const hknpStepInput& stepInfo, hkTaskQueue* taskQueue, hknpSolverData*& solverDataOut );

		/// Helper method to generate and process all solving tasks using a single call.
		/// Any generated tasks will be added to the given task queue, where they will be processed by the calling thread
		/// and by any thread pool that might be processing that queue.
		/// For single threaded simulations, taskQueue can be set to NULL.
		void stepSolve( hknpSolverData*& solverData, hkTaskQueue* taskQueue );

		/// Get the number of times that the world has been stepped since it was created.
		/// Specifically, how many times the solver has been stepped.
		HK_FORCE_INLINE hkUint32 getStepCount();


		// -------------------------------------------------------------------------------------------------------------
		// Libraries
		// -------------------------------------------------------------------------------------------------------------

		/// Get read only access to the material library.
		HK_FORCE_INLINE const hknpMaterialLibrary* getMaterialLibrary() const;

		/// Get read-write access to the material library.
		HK_FORCE_INLINE hknpMaterialLibrary* accessMaterialLibrary();

		/// Get read only access to the motion properties library.
		HK_FORCE_INLINE const hknpMotionPropertiesLibrary* getMotionPropertiesLibrary() const;

		/// Get read-write access to the motion properties library.
		HK_FORCE_INLINE hknpMotionPropertiesLibrary* accessMotionPropertiesLibrary();

		/// Get read only access to the body quality library.
		HK_FORCE_INLINE const hknpBodyQualityLibrary* getBodyQualityLibrary() const;

		/// Get read-write access to the body quality library.
		HK_FORCE_INLINE hknpBodyQualityLibrary* accessBodyQualityLibrary();


		// -------------------------------------------------------------------------------------------------------------
		// Body management
		// -------------------------------------------------------------------------------------------------------------

		/// Allocate an uninitialized body for later use. This can be used as a parameter in hknpBodyCinfo.
		hknpBodyId reserveBodyId();

		/// Create and add a static body.
		/// This uses the preset static motion (bodyCinfo.m_motionId is ignored).
		HK_FORCE_INLINE hknpBodyId createStaticBody( const hknpBodyCinfo& bodyCinfo, AdditionFlags additionFlags = 0 );

		/// Create and add a dynamic body.
		/// This creates a new motion for the body (bodyCinfo.m_motionId is ignored).
		HK_FORCE_INLINE hknpBodyId createDynamicBody( const hknpBodyCinfo& bodyCinfo, const hknpMotionCinfo& motionCinfo,
			AdditionFlags additionFlags = 0 );

		/// Helper method to create and add a set of attached bodies and a dynamic motion at once.
		/// All bodyCinfo.m_motionId fields are ignored. The motion's first attached body ID is returned.
		/// Use hknpMotionCinfo::initialize() to fill the motionCinfo with appropriate values for the whole set of
		/// attached bodies.
		hknpBodyId createAttachedBodies( const hknpBodyCinfo* bodyCinfos, int numBodyCinfos,
			const hknpMotionCinfo& motionCinfo, AdditionFlags additionFlags = 0 );

		/// Create and add a body which uses an existing motion, specified by bodyCinfo.m_motionId.
		hknpBodyId createBody( const hknpBodyCinfo& bodyCinfo, AdditionFlags additionFlags = 0 );

		/// Add a set of bodies to the world, which had the DO_NOT_ADD_BODY flag set on creation or have been removed.
		void addBodies( const hknpBodyId* ids, int numIds, AdditionFlags additionFlags = 0 );

		/// Immediately add any bodies that are waiting to be added to the world, instead of waiting for the next
		/// simulation step.
		void commitAddBodies();

		/// Removes a batch of bodies from the world.
		/// The bodies remain valid and can be re-added to world at any later time.
		/// activationMode specifies whether to activate any bodies overlapping the ones being removed.
		/// NOTE: If you set activationMode to KEEP_DEACTIVATED, you MUST make sure that no stale collision caches
		/// will end up in the engine. You can do this by either:
		///   - making sure the object has no collision caches in the first place.
		///   - activate the entire area by calling activateBodiesInAabb().
		void removeBodies( const hknpBodyId* bodyIds, int numBodyIds,
			hknpActivationMode::Enum activationMode = hknpActivationMode::ACTIVATE );

		/// Destroy a batch of bodies. If the bodies are still added to the world, they will be removed.
		/// Their IDs will be freed during the next collision detection step, as well as those of any orphaned motions.
		/// See the removeBodies() comments for a description of activationMode.
		void destroyBodies( const hknpBodyId* bodyIds, int numBodyIds,
			hknpActivationMode::Enum activationMode = hknpActivationMode::ACTIVATE );

		/// Immediately free the IDs of any destroyed bodies and motions, instead of waiting for the next
		/// simulation step. Advanced use.
		void freeDestroyedBodiesAndMotions();

		/// Return the maximum total number of bodies (special body 0 + dynamic + static + marked for deletion).
		HK_FORCE_INLINE hkUint32 getBodyCapacity() const;

		/// Get the number of allocated bodies (special body 0 + dynamic + static).
		HK_FORCE_INLINE hkUint32 getNumBodies() const;

		/// Returns the maximum number of static bodies that can be created.
		/// Note: static and dynamic bodies share some resources. Therefore you CANNOT create a total of
		/// getNumFreeDynamicBodies() + getNumFreeStaticBodies() bodies.
		HK_FORCE_INLINE hkUint32 getNumFreeStaticBodies();

		/// Returns the maximum number of dynamic bodies that can be created.
		/// Note: static and dynamic bodies share some resources. Therefore you CANNOT create a total of
		/// getNumFreeDynamicBodies() + getNumFreeStaticBodies() bodies.
		HK_FORCE_INLINE hkUint32 getNumFreeDynamicBodies();

		/// Returns non-zero if the body is valid (allocated and not destroyed),
		/// independent on whether it has been added to the world or not.
		HK_FORCE_INLINE hkBool32 isBodyValid( hknpBodyId bodyId ) const;

		/// Returns non-zero if the body is valid and added to the world.
		HK_FORCE_INLINE hkBool32 isBodyAdded( hknpBodyId bodyId ) const;

		/// Get read-only access to a body. Use hknpWorld::setBodyXxx() methods to update a body.
		HK_FORCE_INLINE const hknpBody& getBody( hknpBodyId bodyId ) const;

		/// Get read-only access to a body. In debug builds this asserts that the body is actually added to the world.
		HK_FORCE_INLINE const hknpBody& getSimulatedBody( hknpBodyId bodyId ) const;

		/// Get read-only access to a body. Use hknpWorld::setBodyXxx() methods to update a body.
		HK_FORCE_INLINE const hknpBody& getBodyUnchecked( hknpBodyId bodyId ) const;

		/// Get an iterator for all valid bodies (allocated and not marked for deletion).
		/// Use hknpBody::isAddedToWorld() on the results if you are only interested in bodies that are added to the world.
		HK_FORCE_INLINE hknpBodyIterator getBodyIterator() const;

		/// Get read-write access to a body. Advanced use only.
		/// To modify bodies, use hknpWorld::setBodyXxx() methods instead where possible.
		HK_FORCE_INLINE hknpBody& accessBody( hknpBodyId bodyId );

		/// Get the expanded broad phase AABB of a body.
		void getBodyAabb( hknpBodyId bodyId, hkAabb& aabbOut ) const;

		/// Build a body construction info describing a body's current state.
		HK_FORCE_INLINE void getBodyCinfo( hknpBodyId bodyId, hknpBodyCinfo& bodyCinfoOut ) const;


		// -------------------------------------------------------------------------------------------------------------
		// Body transform
		// -------------------------------------------------------------------------------------------------------------

		/// Get read-only access to the transform of a body.
		HK_FORCE_INLINE const hkTransform& getBodyTransform( hknpBodyId bodyId ) const;

		/// Estimate a body's transform at a relative time, based on the body's current motion.
		void predictBodyTransform( hknpBodyId bodyId, hkReal deltaTime, hkTransform* transformOut ) const;

		/// Set both the position and orientation of a body. Any attached bodies will also be transformed.
		/// - \a activationBehavior defines what should happen to any bodies overlapping this body.
		void setBodyTransform( hknpBodyId bodyId, const hkTransform& transform,
			hknpActivationBehavior::Enum activationBehavior = hknpActivationBehavior::ACTIVATE );

		/// Set the position of a body. Any attached bodies will also be transformed.
		/// - \a activationBehavior defines what should happen to any bodies overlapping this body.
		void setBodyPosition( hknpBodyId bodyId, hkVector4Parameter position,
			hknpActivationBehavior::Enum activationBehavior = hknpActivationBehavior::ACTIVATE );

		/// Set the orientation of a body. Any attached bodies will also be transformed.
		/// - \a pivot defines whether the orientation is centered on the body position or the center of mass.
		/// - \a activationBehavior defines what should happen to any bodies overlapping this body.
		void setBodyOrientation( hknpBodyId bodyId, const hkQuaternion& orientation,
			PivotLocation pivot = PIVOT_BODY_POSITION,
			hknpActivationBehavior::Enum activationBehavior = hknpActivationBehavior::ACTIVATE );


		// -------------------------------------------------------------------------------------------------------------
		// Body velocity
		// -------------------------------------------------------------------------------------------------------------

		/// Get both the linear and angular velocity of a body (in world space).
		void getBodyVelocity( hknpBodyId bodyId, hkVector4& linearVelocity, hkVector4& angularVelocity ) const;

		/// Get the linear velocity of a body at its center of mass (in world space).
		const hkVector4& getBodyLinearVelocity( hknpBodyId bodyId ) const;

		/// Get the angular velocity of a body (in world space).
		void getBodyAngularVelocity( hknpBodyId bodyId, hkVector4& angularVelocity ) const;

		/// Set both the linear and angular velocities of a body (in world space).
		void setBodyVelocity( hknpBodyId bodyId, hkVector4Parameter linearVelocity, hkVector4Parameter angularVelocity );

		/// Set the linear velocity of a body at its center of mass (in world space).
		void setBodyLinearVelocity( hknpBodyId bodyId, hkVector4Parameter linearVelocity );

		/// Set the angular velocity of a body (in world space).
		void setBodyAngularVelocity( hknpBodyId bodyId, hkVector4Parameter angularVelocity );

		/// Set a velocity on a body at a specified position (in world space).
		void setBodyVelocityAt( hknpBodyId bodyId, hkVector4Parameter velocity, hkVector4Parameter position );

		/// Apply a linear impulse to a body at its center of mass.
		/// This causes an immediate change in linear velocity.
		void applyBodyLinearImpulse( hknpBodyId bodyId, hkVector4Parameter linearImpulse );

		/// Apply an angular impulse to a body at its center of mass.
		/// This causes an immediate change in angular velocity.
		void applyBodyAngularImpulse( hknpBodyId bodyId, hkVector4Parameter angularImpulse );

		/// Apply an impulse to a body at a specified position (both in world space).
		/// This causes an immediate change in both linear and angular velocity.
		void applyBodyImpulseAt( hknpBodyId bodyId, hkVector4Parameter impulse, hkVector4Parameter position );

		/// Set a linear and angular velocity such that the body will reach the target position and orientation
		/// in the specified time. This is commonly used to animate keyframed bodies.
		void applyHardKeyFrame( hknpBodyId bodyId,
			hkVector4Parameter targetPosition, const hkQuaternion& targetOrientation, hkReal deltaTime );


		// -------------------------------------------------------------------------------------------------------------
		// Body state
		// -------------------------------------------------------------------------------------------------------------

		/// Attach a set of bodies to a target body.
		/// The attached bodies inherit the motion of the target body.
		void attachBodies( hknpBodyId targetBodyId, const hknpBodyId* bodyIds, int numIds,
			UpdateMassPropertiesMode updateMassProperties = KEEP_MASS_PROPERTIES );

		/// Detach a set of bodies from any other bodies they may be attached to.
		/// \a updateMassProperties defines what should happen with the remaining compound mass properties.
		/// The density of the new motion will be the density of the shape scaled by the densityFactor of the compound.
		/// If the compound is a keyframed object, the new body will get the hknpMotionPropertiesId::DYNAMIC
		/// motion properties.
		void detachBodies( const hknpBodyId* bodyIds, int numIds,
			UpdateMassPropertiesMode updateMassProperties = KEEP_MASS_PROPERTIES );

		/// Set the shape of a body.
		/// Note: even though a hknpShape is a hkReferencedObject, no reference counting will be performed.
		void setBodyShape( hknpBodyId bodyId, const hknpShape* shape );

		/// Set the collision filter info of a body.
		void setBodyCollisionFilterInfo( hknpBodyId bodyId, hkUint32 collisionFilterInfo,
			RebuildCachesMode cacheBehavior = REBUILD_COLLISION_CACHES );

		/// Set the material of a body.
		/// By default this will reset all collision caches, to propagate cached values to the caches (like friction,
		/// restitution and max impulse).
		void setBodyMaterial( hknpBodyId bodyId, hknpMaterialId materialId,
			RebuildCachesMode cacheBehavior = REBUILD_COLLISION_CACHES );

		/// Set the quality of a body.
		/// By default this will reset all collision caches to propagate cached values to the caches.
		void setBodyQuality( hknpBodyId bodyId, hknpBodyQualityId qualityId,
			RebuildCachesMode cacheBehavior = REBUILD_COLLISION_CACHES );

		/// Set the increased collision shell of a body. See hknpBodyCinfo::m_collisionLookAheadDistance for details.
		/// If distance is <= 0.0f, the distance will not be modified.
		void setBodyCollisionLookAheadDistance( hknpBodyId bodyId, hkReal distance,
			hkVector4Parameter tempExpansionVelocity = hkVector4::getZero() );

		/// Set the motion properties of a body's motion.
		void setBodyMotionProperties( hknpBodyId bodyId, hknpMotionPropertiesId motionPropertiesId );

		/// Sets the center of mass of a body's motion (in world space).
		void setBodyCenterOfMass( hknpBodyId bodyId, hkVector4Parameter centerOfMass );

		/// Set the mass of a dynamic body's motion.
		/// Use a negative value to set density, or zero to set infinite mass (keyframed).
		void setBodyMass( hknpBodyId bodyId, hkReal massOrNegativeDensity,
			RebuildCachesMode cacheBehavior = REBUILD_COLLISION_CACHES );

		/// Set the motion of a body.
		/// Does not change the mass properties of the motion. Call rebuildMotionMassProperties() afterward if desired.
		void setBodyMotion( hknpBodyId bodyId, hknpMotionId motionId,
			RebuildCachesMode cacheBehavior = REBUILD_COLLISION_CACHES );

		/// Set a body as static.
		/// Calls setBodyMotion() with motion Id = hknpMotionId::STATIC.
		HK_FORCE_INLINE void setBodyStatic( hknpBodyId bodyId,
			RebuildCachesMode cacheBehavior = REBUILD_COLLISION_CACHES );

		/// Set infinite mass on a dynamic body.
		/// Calls setBodyMass() with massOrNegativeDensity = 0.0.
		HK_FORCE_INLINE void setBodyKeyframed( hknpBodyId bodyId,
			RebuildCachesMode cacheBehavior = REBUILD_COLLISION_CACHES );

		/// Enable some flags on a body.
		/// Rebuilds collision caches if the flags have changed.
		HK_FORCE_INLINE void enableBodyFlags( hknpBodyId bodyId, hknpBody::Flags flagsToEnable );

		/// Disable some flags on a body.
		/// Rebuilds collision caches if the flags have changed.
		HK_FORCE_INLINE void disableBodyFlags( hknpBodyId bodyId, hknpBody::Flags flagsToDisable );

		/// Enable some SPU flags on a body.
		/// Rebuilds collision caches if the flags have changed.
		HK_FORCE_INLINE void enableBodySpuFlags( hknpBodyId bodyId, hknpBody::SpuFlags flagsToEnable );

		/// Disable some SPU flags on a body.
		/// Rebuilds collision caches if the flags have changed.
		HK_FORCE_INLINE void disableBodySpuFlags( hknpBodyId bodyId, hknpBody::SpuFlags flagsToDisable );


		// -------------------------------------------------------------------------------------------------------------
		// Body properties
		// -------------------------------------------------------------------------------------------------------------

		/// Set a property on a body.
		/// The property type must be consistent for all uses of the same key.
		template< typename T >
		HK_FORCE_INLINE void setBodyProperty( hknpBodyId bodyId, hknpPropertyKey key, const T& value );

		/// Get a property from a body.
		/// Returns HK_NULL if the property is not set for the body.
		template< typename T >
		HK_FORCE_INLINE T* getBodyProperty( hknpBodyId bodyId, hknpPropertyKey key ) const;

		/// Clear a property from a body, if present.
		HK_FORCE_INLINE void clearBodyProperty( hknpBodyId bodyId, hknpPropertyKey key );

		/// Clear a property from all bodies, if present.
		HK_FORCE_INLINE void clearPropertyFromAllBodies( hknpPropertyKey key );

		/// Clear all properties from a body.
		HK_FORCE_INLINE void clearAllPropertiesFromBody( hknpBodyId bodyId );

		/// Clear all properties from all bodies.
		HK_FORCE_INLINE void clearAllPropertiesFromAllBodies();


		// -------------------------------------------------------------------------------------------------------------
		// Motions
		// -------------------------------------------------------------------------------------------------------------

		/// Create a motion.
		hknpMotionId createMotion( const hknpMotionCinfo& motionCinfo );

		/// Destroy a batch of motions. The IDs will be freed during the next collision detection step.
		/// In debug builds this asserts that the motions are not used by any body.
		void destroyMotions( hknpMotionId* ids, int numIds );

		/// Get read-only access to a motion.
		HK_FORCE_INLINE const hknpMotion& getMotion( hknpMotionId motionId ) const;

		/// Build a motion construction info describing a motion's current state.
		HK_FORCE_INLINE void getMotionCinfo( hknpMotionId motionId, hknpMotionCinfo& motionCinfoOut ) const;

		/// Get write access to a motion. This performs safety checks.
		/// Try to use body functions like e.g. setBodyVelocity() instead of accessing motions directly.
		HK_FORCE_INLINE hknpMotion& accessMotion( hknpMotionId motionId );

		/// Get write access to a motion. No safety checks will be performed. Use at your own risk.
		HK_FORCE_INLINE hknpMotion& accessMotionUnchecked( hknpMotionId motionId );

		/// Rebuild the mass properties of a motion based on the bodies currently attached to the motion.
		/// This updates the attached bodies as well as the motion.
		void rebuildMotionMassProperties( hknpMotionId motionId,
			RebuildMassPropertiesMode rebuildMode = REBUILD_NOW );

		/// Redo body integration starting with previous frame by a \a fraction of the time step.
		/// - fraction = 0.0 : previous frame
		/// - fraction = 1.0 : current frame
		void reintegrateBody( hknpBodyId bodyId, hkReal fraction );	


		// -------------------------------------------------------------------------------------------------------------
		// Deactivation
		// -------------------------------------------------------------------------------------------------------------

		/// Returns whether deactivation is enabled.
		HK_FORCE_INLINE hkBool32 isDeactivationEnabled() const;

		/// Enables/disables deactivation of a body
		/// Note: This sets the deactivation of the motion linked to the body, hence it also affects all bodies sharing the same motion.
		/// If a body is set to be never deactivate, all bodies whose broad phase AABB overlap with this body will also not deactivate.
		HK_FORCE_INLINE void setBodyDeactivationEnabled( hknpBodyId bodyId, bool enableDeactivation );

		/// Get all active body IDs.
		/// This returns a cached array, so is much faster than iterating over all created bodies.
		HK_FORCE_INLINE const hkArray<hknpBodyId>& getActiveBodies();

		/// Activates a body during the next simulation step.
		/// Note: This also activates any other bodies in the given body's deactivation group.
		void activateBody( hknpBodyId bodyId );

		/// Activate a region which intersects a given world AABB.
		/// Alternatively you could also call queryAabb() and call activateBody() or deactivateBody() on the results,
		/// this method however is faster for activating groups of objects.
		/// Note: The actual activation will happen during the next simulation step.
		void activateBodiesInAabb( const hkAabb& aabb );

		/// Activate a region which intersects a given world AABB (integer space).
		/// This is useful if you want to activate a region which overlaps with an existing body.
		/// Note: The actual activation will happen during the next simulation step.
		void activateBodiesInAabb( const hkAabb16& aabb );

		/// 'Links' two bodies' deactivations.
		/// This enforces that either both bodies are active or both are inactive.
		/// This method keeps track of the number of times it is called for each pair.
		void linkBodyDeactivation( hknpBodyId bodyIdA, hknpBodyId bodyIdB );

		/// Undoes a previous call to linkBodyDeactivation().
		/// If linkBodyDeactivation() was called N times on the two bodies, unlinkBodyDeactivation() must also
		/// be called N times for the bodies to actually get unlinked.
		void unlinkBodyDeactivation( hknpBodyId bodyIdA, hknpBodyId bodyIdB );

		/// Set a filter controlling what happens to deactivated caches.
		void setDeactiveCdCacheFilter( hknpDeactiveCdCacheFilter* policy );


		// -------------------------------------------------------------------------------------------------------------
		// Constraints
		// -------------------------------------------------------------------------------------------------------------

		/// Add a constraint to the world.
		void addConstraint( hknpConstraint* constraint,
			hknpActivationMode::Enum activationMode = hknpActivationMode::ACTIVATE );

		/// Remove a constraint from the world.
		void removeConstraint( hknpConstraint* constraint );

		/// Disable a constraint.
		void disableConstraint( hknpConstraint* constraint );

		/// Enable a disabled constraint.
		void enableConstraint( hknpConstraint* constraint, hknpActivationMode::Enum activationMode );

		/// Check if a constraint has been added to the world.
		HK_FORCE_INLINE hkBool32 isConstraintAdded( const hknpConstraint* constraint ) const;

		/// Find any constraints attached to a given body.
		/// Note: This is slow, as it must examine every constraint in the world.
		HK_FORCE_INLINE void findBodyConstraints( hknpBodyId bodyId,
			hkArray<const hknpConstraint*>& constraintsOut ) const;


		// -------------------------------------------------------------------------------------------------------------
		// Actions (may be removed in a future version)
		// -------------------------------------------------------------------------------------------------------------

		/// Add an action to the world.
		void addAction( hknpAction* action, hknpActivationMode::Enum activationMode = hknpActivationMode::ACTIVATE );

		/// Remove an action from the world.
		void removeAction( hknpAction* action, hknpActivationMode::Enum activationMode = hknpActivationMode::ACTIVATE );


		// -------------------------------------------------------------------------------------------------------------
		// Cache management
		// -------------------------------------------------------------------------------------------------------------

		/// Flag all collision caches as dirty for this body.
		/// E.g. you need to call this function if the shape of this body has substantially changed
		/// (meaning that the collision caches are no longer valid).
		void rebuildBodyCollisionCaches( hknpBodyId bodyId );

		/// Rebuild the caches for a list of body pairs.
		void rebuildBodyPairCollisionCaches( hknpBodyIdPair* pairs, int numPairs );

		/// Flag all collision caches as dirty for all dynamic bodies touching the AABB.
		/// E.g. you need to call this function if you modify parts of a static landscape mesh and dynamic bodies
		/// might collide with the landscape.
		void rebuildDynamicBodyCollisionCachesInAabb( const hkAabb& aabb );

		/// Force garbage collection of all inactive collision cache streams.
		void garbageCollectInactiveCacheStreams();

		/// Clear ALL caches in the engine.
		void deleteAllCaches();


		// -------------------------------------------------------------------------------------------------------------
		// Modifiers and Events
		// -------------------------------------------------------------------------------------------------------------

		/// Get the modifier manager (which is needed to register your own modifiers).
		/// Note that for SPU you need to register those modifier in the SPU ELF as well.
		HK_FORCE_INLINE hknpModifierManager* getModifierManager();

		/// Get the event dispatcher.
		HK_FORCE_INLINE hknpEventDispatcher* getEventDispatcher() const;

		/// Get a signal for one event type, for just one body.
		/// Use this to register your event handling functions.
		hknpEventSignal& getEventSignal( hknpEventType::Enum eventType, hknpBodyId id );

		/// Get a signal for one event type, for all bodies.
		/// Use this to register your event handling functions.
		hknpEventSignal& getEventSignal( hknpEventType::Enum eventType );


		// -------------------------------------------------------------------------------------------------------------
		// Static queries
		// -------------------------------------------------------------------------------------------------------------

		/// Cast a ray into the world and enumerate all hits.
		/// Use your collector of choice to return the wanted hits.
		/// This method supports filtering.
		/// If no filter has been specified in \a query, the default world collision filter will be used.
		void castRay( const hknpRayCastQuery& query, hknpCollisionQueryCollector* collector ) const;

		/// Cast a shape into the world and enumerate all hits.
		/// Use your collector of choice to return the wanted hits.
		/// This method supports filtering.
		/// If no filter has been specified in \a query, the default world collision filter will be used.
		/// The query shape's world orientation is defined by \a queryShapeOrientation, its world space position by the
		/// starting position/origin of the \a query's underlying hkcdRay.
		void castShape( const hknpShapeCastQuery& query, const hkRotation& queryShapeOrientation,
			hknpCollisionQueryCollector* collector ) const;

		/// Get the closest points between a shape and all objects in the world.
		/// Use your collector of choice to return the wanted hits.
		/// This method supports filtering.
		/// If no filter has been specified in \a query, the default world collision filter will be used.
		/// The query shape is specified in \a query.
		/// The query shape's orientation and position (world space) are defined by \a queryShapeTransform.
		void getClosestPoints( const hknpClosestPointsQuery& query, const hkTransform& queryShapeTransform,
			hknpCollisionQueryCollector* collector ) const;

		/// Find all bodies intersecting the input AABB at the broad phase level.
		/// Note: no filtering is performed.
		void queryAabb( const hkAabb& aabb, hkArray<hknpBodyId>& hits ) const;

		/// Find all shapes of all bodies intersecting the input AABB.
		/// Use your collector of choice to return the wanted hits.
		/// This method supports filtering.
		/// If no filter has been specified in \a query, the default world collision filter will be used.
		/// The AABB is specified in \a query.
		void queryAabb( const hknpAabbQuery& query, hknpCollisionQueryCollector* collector ) const;


		// -------------------------------------------------------------------------------------------------------------
		// Helper methods
		// -------------------------------------------------------------------------------------------------------------

		/// Shift all objects in the world by a certain amount.
		/// This does also shift the broad phase, so the broad phase will no longer be centered. Use shiftBroadPhase()
		/// to recenter the broad phase.
		void shiftWorld( hkVector4Parameter shift );

		/// Shifts the broad phase to have it's center as close as possible to the requested position.
		void shiftBroadPhase( hkVector4Parameter requestedCenterPos, hkVector4& effectiveCenterPosOut,
			hkArray<hknpBodyId>* bodiesOutsideTheBroadPhase = HK_NULL );

		/// Optimize some internal spatial data structures after adding lots of static bodies.
		/// This should not be necessary, but can speed up the broad phase significantly, especially if you added bodies
		/// in a spatially ordered form. If you added the bodies using roughly random positions, you will automatically
		/// get a good broad phase layout.
		void optimizeBroadPhase();


		// -------------------------------------------------------------------------------------------------------------
		// Debug utilities
		// -------------------------------------------------------------------------------------------------------------

		/// Assert that the world is not currently being stepped.
		HK_FORCE_INLINE void checkNotInSimulation( int allowedStates = 0 ) const;

		/// Enable or disable automatic consistency checks. Only works in debug builds.
		HK_FORCE_INLINE void setConsistencyChecksEnabled( bool areEnabled );

		/// Returns whether automatic consistency checks are enabled.
		/// Note: in a release build this will always return false, so that the compiler can remove all checking code.
		HK_FORCE_INLINE hkBool32 areConsistencyChecksEnabled() const;

		/// Check the consistency of the world. Only works in debug builds.
		void checkConsistency();

		/// Check the consistency of a body. Only works in debug builds.
		void checkBodyConsistency( hknpWorld* world, hknpBodyId bodyId ) const;

		/// Set a trace dispatcher, used to dispatch addTrace() commands in debug builds.
		hkSecondaryCommandDispatcher* setTraceDispatcher( hkSecondaryCommandDispatcher* dispatcher );

		/// Sets this world to be the global 'debug' instance.
		/// This is used by MSVC autoexp.dat macros to expand body/motion/etc. IDs in the debugger.
		void setAsDebugInstance() const;


	protected:

		// -------------------------------------------------------------------------------------------------------------
		// Internal helpers
		// -------------------------------------------------------------------------------------------------------------

		/// Add a body to a motion.
		/// Does not change the motion's mass properties.
		void addBodyToMotion( hknpBodyId bodyId, hknpMotionId motionId, const hkQuaternion* worldQbody = HK_NULL );

		/// Remove a body from its motion.
		/// Does not change the motion's mass properties.
		void removeBodyFromMotion( hknpBody* body );

		/// Update a set of attached bodies if the motion has its velocity or position modified.
		void synchronizeBodiesFromMotion( hknpBodyId firstBodyId,
			hkSimdRealParameter movedDistance, hkSimdRealParameter movedAngle );

		/// Internal method used by setBodyTransform(), setBodyPosition() and setBodyOrientation().
		void updateMotionAndAttachedBodiesAfterModifyingTransform(
			hknpBodyId bodyId, const hkQuaternion* orientation, PivotLocation rotationPivot,
			hknpActivationBehavior::Enum activationBehavior );

		/// Register a body at an active list.
		void registerBodyAtActiveList( hknpBodyId bodyId, hknpMotionId motionId );

		/// Unregister a body at an active list.
		/// If multiple bodies are attached to a motion, this will unregister all bodies.
		void unregisterBodyAtActiveList( hknpBody* body );

#ifndef HK_DEBUG
		HK_FORCE_INLINE void addTrace( const hknpApiCommand& command ) {}
#else
		/// Function used by the engine to record hknpWorld function calls.
		void addTrace( const hknpApiCommand& command );
#endif

	public:

		// -------------------------------------------------------------------------------------------------------------
		// Members
		// -------------------------------------------------------------------------------------------------------------

		hknpBodyManager				m_bodyManager;
		hknpMotionManager			m_motionManager;
		hknpModifierManager*		m_modifierManager;	
		hknpActionManager*			m_actionManager;	// work in progress

		//
		// Memory management
		//

		/// Block stream allocator for persistent caches.
		hkRefPtr<hkBlockStreamAllocator> m_persistentStreamAllocator;

		/// Block stream allocator for step local objects, only valid during a simulation step.
		hkBlockStreamAllocator* m_stepLocalStreamAllocator;

		//
		// Collision detection
		//

		hknpBroadPhase* m_broadPhase;		///< The broad phase implementation.
		hkIntSpaceUtil m_intSpaceUtil;		///< Helper class used for AABB quantization (to/from hkAabb16).

		hkSimdReal m_collisionTolerance;	///< Minimum contact distance for narrow phase collision detection.

		/// Storage of collision caches.
		hknpCollisionCacheManager* m_collisionCacheManager;

		/// Responsible for creating collision caches for overlapping broad phase pairs.
		hknpCollisionDispatcher* m_collisionDispatcher;

		/// Responsible for dispatching user queries between shape pairs.
		hknpCollisionQueryDispatcherBase* m_collisionQueryDispatcher;

		//
		// Solving
		//

		hknpContactSolver*				m_contactSolver;
		hknpConstraintAtomSolver*		m_constraintAtomSolver;

		hknpSolverInfo					m_solverInfo;			///< Solver parameters.
		hkArray<hknpSolverVelocity>		m_solverVelocities;		///< Buffer used by the solver.
		hkArray<hknpSolverSumVelocity>	m_solverSumVelocities;	///< Buffer used by the solver.
		hkBool32						m_enableSolverDynamicScheduling;	///< See hknpWorldCinfo::m_enableSolverDynamicScheduling

		//
		// Deactivation
		//

		/// This value is set if the engine is performing deactivation.
		hkBool32 m_deactivationEnabled;

		hknpDeactivationManager* m_deactivationManager;

		/// policies sorted by type
		hkRefPtr<hknpDeactiveCdCacheFilter> m_deactiveCdCacheFilter;

		//
		// Simulation
		//

		hknpSimulation* m_simulation;			///< The simulation implementation.
		hknpSpaceSplitter* m_spaceSplitter;		///< The space splitter implementation.

		/// Used internally in the collide and solve steps.
		hknpSimulationContext* m_simulationContext;

		/// Internal flag to check if users call functions within the simulation which could lead to a crash.
		hkEnum<SimulationStage,hkUint32> m_simulationStage;

		//
		// Signals and commands
		//

		/// A set of signals used by the world.
		hknpWorldSignals m_signals;

		/// The command dispatcher, set by the world constructor.
		/// Register or replace your sub dispatchers to your liking.
		hkPrimaryCommandDispatcher* m_commandDispatcher;

		/// If set, any calls to hknpWorld methods will be duplicated here.
		hkRefPtr<hkSecondaryCommandDispatcher> m_traceDispatcher;

		//
		// Other
		//

		/// A helper array to keep references to user objects, which are removed during destruction of the world.
		/// This is a convenience for users, it is not used by the engine.
		hkArray< hkRefPtr<const hkReferencedObject> > m_userReferencedObjects;

	protected:

		hkVector4 m_gravity;	///< Global acceleration applied to all dynamic bodies.

		/// A set of built-in modifiers.
		hkRefPtr<hkReferencedObject> m_defaultModifierSet;	

		// Libraries
		hkRefPtr<hknpMaterialLibrary> m_materialLibrary;
		hkRefPtr<hknpMotionPropertiesLibrary> m_motionPropertiesLibrary;
		hkRefPtr<hknpBodyQualityLibrary> m_qualityLibrary;

		/// A bit field to mark any dirty materials.
		/// A material will be marked as dirty if it has been modified since the last call to preColide().
		hkBitField m_dirtyMaterials;

		/// A bit field to mark any dirty body qualities.
		/// A body quality will be marked as dirty if it has been modified since the last call to preColide().
		hkBitField m_dirtyQualities;

		hkRefPtr<hknpShapeTagCodec> m_shapeTagCodec;	///< The codec used for encoding/decoding shape tags.
		hknpNullShapeTagCodec m_nullShapeTagCodec;		///< A default NULL codec.

		/// Event dispatcher.
		hkRefPtr<hknpEventDispatcher> m_eventDispatcher;

		/// Dispatcher for internal commands.
		hkRefPtr<hknpInternalCommandProcessor> m_internalPhysicsCommandDispatcher;

		/// Flag to enable slow consistency checks, only works in debug builds.
		hkBool m_consistencyChecksEnabled;

		/// The shape manager handles mutation events from modified shapes.
		hknpShapeManager m_shapeManager;
		friend class hknpShapeManager;

		// Some tasks used during every simulation step
		class hknpPreCollideTask* m_preCollideTask;
		class hknpPostCollideTask* m_postCollideTask;
		class hknpPostSolveTask* m_postSolveTask;

		//
		// Cached construction info, needed for getCinfo()
		//

		/// The simulation type
		hkEnum<hknpWorldCinfo::SimulationType, hkUint8> m_simulationType;

		/// Events merging setup
		hkBool m_mergeEventsBeforeDispatch;

		/// The behavior of the engine in regards to bodies leaving the broad phase.
		hkEnum<hknpWorldCinfo::LeavingBroadPhaseBehavior, hkUint8> m_leavingBroadPhaseBehavior;
};

#include <Physics/Physics/Dynamics/World/hknpWorld.inl>


#endif // HKNP_WORLD_H

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
