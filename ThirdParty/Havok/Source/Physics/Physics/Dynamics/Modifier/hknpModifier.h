/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MODIFIER_H
#define HKNP_MODIFIER_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Collide/hknpCollideSharedData.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>
#include <Physics/Physics/Dynamics/Body/hknpBodyQuality.h>

struct hknpConvexConvexCollisionCache;
struct hknpManifold;
class hknpMaterial;
struct hknpSolverInfo;
struct hknpCdBody;
struct hknpCdBodyBase;
class hknpModifierSharedData;
struct hknpContactSolverTemp;
class hknpConstraint;
struct hknpAabbQuery;
struct hknpShapeQueryInfo;
class hknpSolverVelocity;
class hknpSolverSumVelocity;


/// A modifier is a function to capture and alter information within the physics engine.
///
/// Modifiers are similar to shaders in a graphics engine, they:
///   - run multi threaded,
///   - run in a non deterministic order (although the results are deterministic),
///   - on PlayStation(R)3 run on the SPU.
/// They are registered globally at the hknpModifierManager.
/// They are called if a body has a bit set in m_flags which matches at least one bit used when registering this hknpModifier.
/// Note : it is recommended to use the HK_OVERRIDE specifier when implementing the callback functions in the derived class.
class hknpModifier
{
	public:

		/// An identifier for a specific function in the modifier, used by getEnabledFunctions().
		enum FunctionType
		{
			FUNCTION_MANIFOLD_CREATED_OR_DESTROYED,
			FUNCTION_MANIFOLD_PROCESS,
			FUNCTION_POST_CONTACT_JACOBIAN_SETUP,
			FUNCTION_POST_CONTACT_SOLVE,
			FUNCTION_POST_CONTACT_IMPULSE_CLIPPED,
			FUNCTION_POST_CONSTRAINT_EXPORT,
			FUNCTION_POST_COMPOSITE_QUERY_AABB,
			FUNCTION_MODIFY_MOTION_GRAVITY,
			FUNCTION_MAX
		};

		/// Information available to the Manifold created callback.
		struct ManifoldCreatedCallbackInput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, ManifoldCreatedCallbackInput );

			/// Collision cache in local memory.
			HK_PAD_ON_SPU( hknpManifoldCollisionCache* ) m_collisionCache;

			/// Collision cache in main memory (if applicable).
			// It is the location where the CC will be stored after sync for post-processing (e.g. events)
			HK_PAD_ON_SPU( hknpManifoldCollisionCache* ) m_collisionCacheInMainMemory;

			/// Pointer to the manifold (not available for signed distance field collisions).
			HK_PAD_ON_SPU( hknpManifold* ) m_manifold;
		};

		/// Information available to solver callbacks.
		struct SolverCallbackInput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, SolverCallbackInput );

			/// Contact Jacobian in local memory.
			HK_PAD_ON_SPU( const hknpMxContactJacobian* ) m_contactJacobian;

			/// Contact Jacobian in main memory (if applicable).
			HK_PAD_ON_SPU( const hknpMxContactJacobian* ) m_contactJacobianInMainMemory;

			/// Collision cache. On PS3 SPU, this points to the local copy of the cache (as opposed to the one pointed
			/// by the jacobian, which is on PPU).
			HK_PAD_ON_SPU( const hknpManifoldCollisionCache*) m_collisionCache;

			/// Index of the relevant manifold in the contact Jacobian.
			HK_PAD_ON_SPU( int ) m_manifoldIndex;

			/// Whether any of the impulses for the manifold were clipped by the solver.
			HK_PAD_ON_SPU( hkBool32 ) m_wasImpulseClipped;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpModifier );

		/// Destructor.
		
		
		HK_ON_CPU( virtual ~hknpModifier() {} )
		HK_ON_SPU( protected: ~hknpModifier() {} public: )

#if !defined ( HK_PLATFORM_SPU )

		/// For performance reason a modifier can specify which set of functions it wants to be called on.
		/// An implementation should just a bitField which has the corresponding bits set: (1<<FunctionType)
		virtual int getEnabledFunctions() = 0;

#endif

		//
		// Potential functions to be called
		//

		/// Called when a new contact manifold is created. This is the only place where you can allocate properties in
		/// the collision cache.
		virtual void manifoldCreatedCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
			ManifoldCreatedCallbackInput* HK_RESTRICT input )
		{
		}

		/// Called when a contact manifold is processed.
		/// Note: if any of the two bodies has welding enabled, this callback is fired AFTER welding.
		/// In this case shapeA or shapeB points to the top level shape and not to the triangle shape.
		virtual void manifoldProcessCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
			hknpManifold* HK_RESTRICT manifold)
		{
		}

		/// Called when a contact manifold is destroyed.
		virtual void manifoldDestroyedCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
			hknpManifoldCollisionCache* HK_RESTRICT cache, hknpCdCacheDestructReason::Enum reason)
		{
		}

		/// Called after Jacobians are created for a contact manifold.
		virtual void postContactJacobianSetup(
			const hknpSimulationThreadContext& tl,
			const hknpSolverInfo& solverInfo,
			const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
			const hknpManifoldCollisionCache* cache, const hknpManifold* manifold,
			hknpMxContactJacobian* HK_RESTRICT mxJac, int mxJacIdx )
		{
		}

		/// Called after a contact Jacobian has been solved.
		/// frictionFactor is a value between 0 and 1 describing how much friction force has been applied relative
		/// to the theoretical friction force for infinite friction.
		///   0: The bodies slid against other.
		///   1: The bodies did not slide.
		virtual void postContactSolve(
			const hknpSimulationThreadContext& tl, const SolverCallbackInput& input,
			hkVector4Parameter contactImpulses, hkReal frictionFactor )
		{
		}

		/// Called after a contact Jacobian has been solved and its impulse has been clipped due to it's impulse limit
		/// being exceeding.
		virtual void postContactImpulseClipped(
			const hknpSimulationThreadContext& tl, const SolverCallbackInput& input,
			const hkReal& sumContactImpulseUnclipped, const hkReal& sumContactImpulseClipped )
		{
		}

		/// Called when a constraint has had it's Runtime updated.
		/// When the constraint is immediate, instance pointer is NK_NULL, immId is valid, and runtime a stack pointer.
		virtual void postConstraintExport(
			const hknpSimulationThreadContext& tl, hknpConstraint* constraint,
			hknpImmediateConstraintId immId, hkUint8 constraintAtomFlags, void* runtime )
		{
		}

		/// Called after a query-aabb call on a composite shape (during collide).
		/// The modifier can add and remove keys from the array. Returns the new number of keys.
		virtual int postCompositeQueryAabb(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpAabbQuery& aabbQuery, const hknpShapeQueryInfo& queryShapeInfo, const hknpShapeQueryInfo& targetShapeInfo,
			hknpShapeKey* keys, int numKeys, int maxNumKeys )
		{
			return numKeys;
		}

		/// This is a motion modifier.
		/// It allows modifying the gravity applied to the solver velocities of a motion during solver sub-steps.
		virtual void modifyMotionGravity(
			const hknpSimulationThreadContext& tl, const hknpMotionId motionId, hkVector4& gravity )
		{
		}
};


/// A "modifier" used for welding.
class hknpWeldingModifier
{
	public:

		struct WeldingInfo
		{
			hknpBodyQuality::Flags m_qualityFlags;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpWeldingModifier );

		/// Destructor.
		
		
		HK_ON_CPU( virtual ~hknpWeldingModifier() {} )
		HK_ON_SPU( protected: ~hknpWeldingModifier() {} public: )

		/// Called after one convex shape A has collided with all leaf shapes of mesh shape B.
		virtual void postMeshCollideCallback(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const WeldingInfo& wInfo,
			const hknpCdBody& cdBodyA, const hknpCdBodyBase& cdBodyB,
			hknpManifold* HK_RESTRICT manifolds, int numManifolds ) =0 ;
};


#endif	// HKNP_MODIFIER_H

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
