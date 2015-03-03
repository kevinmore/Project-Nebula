/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_TYPES_H
#define HKNP_TYPES_H

#include <Common/Base/Types/hkHandle.h>
#include <Common/Base/Container/BlockStream/hkBlockStream.h>

#include <Physics/Physics/hknpConfig.h>
#include <Physics/Physics/Dynamics/Body/hknpBodyId.h>
#include <Physics/Physics/Dynamics/Body/hknpBodyQualityId.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionId.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionPropertiesId.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterialId.h>


//
// Forward declarations
//

class hknpWorld;
class hknpSimulation;
class hknpBroadPhase;
class hknpSimulationThreadContext;
struct hknpStepInput;
struct hknpSolverInfo;
class hknpDeactivationManager;
class hknpDeactivatedIsland;
struct hknpDeactivationState;
struct hknpDeactivationThreadData;
class hknpModifierManager;
class hknpModifierSharedData;
class hkThreadLocalBlockStreamAllocator;
class hknpAction;
class hknpSpaceSplitter;
struct hknpEvent;
struct hknpEventHandlerInput;
class hknpBody;
class hknpBodyQuality;
class hknpMaterial;
class hknpMotion;
struct hknpMotionCinfo;
class hknpShape;
class hknpTriangleShape;
struct hknpCdBody;
struct hknpCdBodyBase;
class hkcdVertex;
struct hknpManifoldCollisionCache;
struct hknpManifold;
class hknpConstraint;
class hknpCdCacheRange : public hkBlockStreamBase::Range {};
class hknpCdCacheWriter;


//
// Shape related types
//

/// Identifier for a leaf shape within a composite shape.
typedef hkUint32 hknpShapeKey;
#define HKNP_INVALID_SHAPE_KEY 0xffffffff

/// Macro for the number of unused bits to the right of a specific shape key, defined by its length via NUM_SHAPE_KEY_BITS.
/// Ranges from 0 (shape key uses all available bits) to 32 (shape key uses none at all).
#define HKNP_NUM_UNUSED_SHAPE_KEY_BITS(NUM_SHAPE_KEY_BITS) (sizeof(hknpShapeKey) * 8 - NUM_SHAPE_KEY_BITS)

/// Shape tag. Can be used to control filters and materials in composite shapes.
typedef hkUint16 hknpShapeTag;
#define HKNP_INVALID_SHAPE_TAG 0xffff

/// Shape tag codec info. Stored per composite shape, and available for the codec to store information to help decoding.
typedef hkUint32 hknpShapeTagCodecInfo;
#define HKNP_INVALID_SHAPE_TAG_CODEC_INFO 0xFFFFFFFF

/// Identifier for a shape vertex (usually stored in hkcdVertex).
typedef hkUint16 hknpVertexId;

/// A full shape key hierarchy that can span more than one shape.
/// The full key is stored in \a m_key, the number of used bits (of all bits in \a m_key) is stored in \a m_size.
/// \a m_key is getting filled 'from the left', i.e. the topmost shape's key will be the leftmost key in \a m_key.
/// All unused bits to the right of the actual key (path) get set to '1'.

struct hknpShapeKeyPath
{
	public:

		/// Constructor.
		HK_FORCE_INLINE hknpShapeKeyPath();

		/// Constructor.
		/// Initializes shape key path from a given \a sourceKey of \a bitSize.
		/// Internally calls setFromKey(). See there for more information on the parameters.
		/// This constructor leaves the 'unused' bits untouched. If \a sourceKey does not have them
		/// set to 1, make sure to call setUnusedBits() as well.
		HK_FORCE_INLINE hknpShapeKeyPath(hknpShapeKey sourceKey, int bitSize = sizeof(hknpShapeKey)*8);

		/// Resets the shape key path for re-use.
		HK_FORCE_INLINE	void reset();

		/// Permanently append a new \a subkey of length \a sizeInBits to the current key path.
		HK_FORCE_INLINE	void appendSubKey(hknpShapeKey subkey, int sizeInBits);

		/// Virtually append a new \a subkey of length \a sizeInBits and return the full key path.
		/// This method will *not* change the stored \a m_key.
		HK_FORCE_INLINE	hknpShapeKey makeKey(hknpShapeKey subkey, int sizeInBits) const;

		/// Returns a valid shape key path or HKNP_INVALID_SHAPE_KEY.
		HK_FORCE_INLINE	hknpShapeKey getKey() const;

		/// Initialize from a given \a sourceKey of \a bitSize.
		/// This method assumes all unused bits in \a sourceKey to be set to 1. If this is not the case, make
		/// sure to call setUnusedBits() as well.
		/// If \a bitSize is omitted, the path's size will be set such that the path 'is full' and no further
		/// sub keys can be appended.
		HK_FORCE_INLINE	void setFromKey(hknpShapeKey sourceKey, int bitSize = sizeof(hknpShapeKey)*8);

		/// Sets all 'unused' bits (i.e. the bits to the right of the stored key) to 1.
		HK_FORCE_INLINE	void setUnusedBits();

		/// Returns the current size (in bits) of the stored shape key path.
		HK_FORCE_INLINE	int getKeySize() const;

	protected:

		/// The current key path, concatenated from all sub keys that have been added using appendSubKey().
		// Note: Do *NOT* pad this value on SPU as we need to allocate a large block of shape key paths in the
		//       hknpCompositeCompositeCollisionDetector. This additional required stack space would outweigh the
		//       small code size increase caused by not padding this value!
		hknpShapeKey m_key;

		/// The key path's current size in bits.
		// Note: Do *NOT* pad this value on SPU as we need to allocate a large block of shape key paths in the
		//       hknpCompositeCompositeCollisionDetector. This additional required stack space would outweigh the
		//       small code size increase caused by not padding this value!
		int m_size;
};


//
// Body related types
//

/// A bit field for storing various body and material flags, including a set of enabled modifiers.
typedef hkUint32 hknpBodyFlags;

/// A bit field for storing modifier flags.

typedef hkUint32 hknpModifierFlags;


//
// Constraint related types
//

/// Identifier for immediate constraints.
HK_DECLARE_HANDLE( hknpImmediateConstraintId, hkUint16, 0xffff );

template <int> struct hknpContactJacobian;
typedef hknpContactJacobian<HKNP_NUM_MX_JACOBIANS> hknpMxContactJacobian;


//
// Property related types
//

/// A key for generic shape and body properties.
/// Values above 0xf000 are reserved for Havok use:
///  - Values in range [0xf000, 0xf1ff] are reserved for Havok Physics.
///  - Values in range [0xf200, 0xf2ff] are reserved for Havok Destruction.

typedef hkUint16 hknpPropertyKey;

/// Property keys used with hknpBody.
struct hknpBodyPropertyKeys
{
	enum
	{
		DEBUG_DISPLAY_COLOR	= 0xf000,
		CHARACTER_PROXY		= 0xf001
	};
};

/// Property keys used with hknpShape.
struct hknpShapePropertyKeys
{
	enum
	{
		MASS_PROPERTIES 	= 0xf100,
		MATERIAL_PALETTE	= 0xf101,
		DESTRUCTION_INFO	= 0xf200,
		NAME				= 0xf201
	};
};


//
// Simulation related types
//

/// Identifier used by the broad phase.
typedef hkUint32 hknpBroadPhaseId;
#define HKNP_INVALID_BROAD_PHASE_ID 0xffffffff

/// Identifier for a broad phase layer.
typedef hkUint8 hknpBroadPhaseLayerIndex;

/// Measurement of the worst case movement of an object.
/// This allows for certain collision detection optimizations.
typedef hkUint16 hknpLinearTimType;

/// To improve multi-threading, all bodies are sorted into groups called cells. See hknpSpaceSplitter for details.
typedef hkUint8 hknpCellIndex;
enum { HKNP_INVALID_CELL_IDX = 0xff };

/// Internal body identifier used by the constraint solver.
HK_DECLARE_HANDLE( hknpSolverId, hkUint32, HK_INT32_MAX );	

/// Identifies a deactivated group of bodies.
HK_DECLARE_HANDLE( hknpIslandId, hkUint16, 0xffff );

/// Input for various functions
struct hknpActivationMode
{
	enum Enum
	{
		KEEP_DEACTIVATED,	///< No activation or deactivation
		ACTIVATE,			///< Activate all relevant objects
	};
};

/// Input for various functions

struct hknpActivationBehavior
{
	enum Enum
	{
		KEEP_DEACTIVATED,		///< No activation or deactivation
		ACTIVATE,				///< Activate all objects touched
		ACTIVATE_NEW_OVERLAPS	///< Activate all new overlaps
	};
};

/// Collision cache type.
struct hknpCollisionCacheType
{
	enum Enum
	{
		CONVEX_CONVEX,						///< hknpConvexConvexCollisionCache / hknpConvexConvexManifoldCollisionCache
		SET_SHAPE_KEY_A,					///< hknpSetShapeKeyACollisionCache
		COMPOSITE_BASE,						///< hknpCompositeCollisionCache
		CONVEX_COMPOSITE = COMPOSITE_BASE,	///< hknpConvexCompositeCollisionCache
		COMPOSITE_COMPOSITE,				///< hknpCompositeCompositeCollisionCache
		DISTANCE_FIELD,						///<
		USER_0,								///< User type
		USER_1,								///< User type
		NUM_TYPES
	};
};

/// A hint as to why a manifold cache was destroyed.
struct hknpCdCacheDestructReason
{
	enum Enum
	{
		AABBS_DONT_OVERLAP,		///< The AABBs of the shapes no longer overlap.
		OBJECTS_TOO_FAR,		///< The distance of the shapes is too large (used by the heightfield collision detector).
		BODY_IS_INVALID,		///< Bodies are deleted. If a body gets deleted, the caches will be cleaned up the next collide step.
		SHAPE_TAG_CHANGED,		///< A convex body colliding with a heightfield is hitting a different section with a different shape tag.
		CACHE_DISABLED,			///< Cache is disabled.
		CACHE_DEACTIVATED,		///< Cache deactivates and the hknpDeactiveCdCacheFilter deletes this cache.
		CACHE_DELETED_BY_USER,	///< Cache is deleted because hknpBody has the TEMP_REBUILD_COLLISION_CACHES flag set.
		BODY_MOTION_CHANGED		///< Body changed from dynamic to static via hknpWorld::setBodyMotion().
	};
};

/// Jacobian grid types for solving.
struct hknpJacobianGridType
{
	enum Enum
	{
		JOINT_CONSTRAINT,
		MOVING_CONTACT,
		FIXED_CONTACT,
		NUM_TYPES
	};
};

/// Constraint solver identifiers.
struct hknpConstraintSolverType
{
	enum Enum
	{
		CONTACT_CONSTRAINT_SOLVER,		///< hknpContactSolver
		ATOM_CONSTRAINT_SOLVER,			///< hknpConstraintAtomSolver
		USER_1,
		USER_2,
		USER_3,
		USER_4,
		USER_5,
		USER_6,
		NUM_TYPES,
	};
};

/// If you set the hkCommand.m_filterBits to hknpCommandDispatchType::AS_SOON_AS_POSSIBLE, the
/// command will be executed as soon as the engine enters a single threaded section,
/// e.g. at the end of the collide step, else the command will be dispatched at the end
/// of post simulation.
struct hknpCommandDispatchType
{
	enum Enum
	{
		POST_SOLVE = 0,
		AS_SOON_AS_POSSIBLE = 1,
	};
};

/// Structure of data exported from the solver to the collision cache
struct hknpManifoldSolverInfo
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpManifoldSolverInfo );

	/// Flags used by some built in modifiers (instead of allocating collision cache properties).
	enum Flags
	{
		WAS_TRIGGER_VOLUME_ENTERED				= 1<<0,	///< Flag used by hknpTriggerVolumeModifier
		WAS_TRIGGER_VOLUME_PENETRATING			= 1<<1,	///< Flag used by hknpTriggerVolumeModifier
		RAISE_CONTACT_IMPULSE_CONTINUED_EVENTS	= 1<<2,	///< Flag used by hknpContactImpulseEventCreator
		RAISE_CONTACT_IMPULSE_FINISHED_EVENTS	= 1<<3,	///< Flag used by hknpContactImpulseEventCreator

		// User flags
		USER_FLAG_0	= 1<<4,
		USER_FLAG_1	= 1<<5,
		USER_FLAG_2	= 1<<6,
		USER_FLAG_3	= 1<<7
	};

	/// The magnitude of the impulses applied to each contact point.
	hkHalf m_impulses[4];

	/// The contact Jacobian.
	/// This is only valid after collision detection and before the step local allocator is deleted at the end of the
	/// simulation step. It is NOT set to NULL at the end of the simulation step, so will point to deallocated memory.
	hknpMxContactJacobian* m_contactJacobian;

	/// Internal data used by friction calculations.
	hkHalf m_frictionRhsMultiplier;

	/// Index of the manifold data within m_contactJacobian.
	hkUint8 m_manifoldIndex;

	/// Flags.
	mutable hkFlags<Flags, hkUint8> m_flags;
};


//
// Misc helpers
//

/// This global variable is used to dereference the objects referred by an ID (e.g. bodies, motion, etc.)
/// when debugging. A set of autoexp.dat rules exploiting this variable is available for Visual Studio.
/// It is mostly a convenience and it is NOT intended to be used in production code.
/// The variable definition is in hknpWorld, and you can use the makeDebugInstance() function to assign
/// a given hknpWorld instance to be the debug world.
extern const hknpWorld* g_hknpWorldDebugInstance;

#if !defined( __HAVOK_PARSER__ )
#	define HKNP_PADDING_BYTES( M, B ) hkUint8 M[B];
#else
#	define HKNP_PADDING_BYTES( M, B )
#endif

#if !defined(HK_DEBUG) && HK_NUM_SIMD_REGISTERS <=  8
#	define HKNP_ON_FEW_VREG(code) code
#	define HKNP_ON_MANY_VREG(code)
#else
#	define HKNP_ON_FEW_VREG(code)
#	define HKNP_ON_MANY_VREG(code) code
#endif

enum { HKNP_SPU_DMA_GROUP_STALL = 14 };


#include <Physics/Physics/hknpTypes.inl>

#endif // HKNP_TYPES_H

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
