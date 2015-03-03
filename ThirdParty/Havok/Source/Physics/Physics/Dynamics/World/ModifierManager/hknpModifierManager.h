/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MODIFIER_MANAGER_H
#define HKNP_MODIFIER_MANAGER_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>
#include <Physics/Physics/Dynamics/Body/hknpBody.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionCinfo.h>
#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>

struct hknpManifold;
struct hknpSolverInfo;
class hknpCollisionDetector;
class hknpConstraintSolver;


/// A collection of modifiers.
/// This implementation sorts modifiers by function type to improve searching matching modifiers.
class hknpModifierManager
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpModifierManager );

		/// Internal constants
		enum
		{
			MAX_MODIFIERS_PER_FUNCTION = 8,		///< Maximum number of modifiers registered per function
			NUM_BODY_FLAGS = 32
		};

		/// A modifier with a set of flags that enable it
		struct ModifierEntry
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpModifierManager::ModifierEntry );

			hknpModifierFlags m_enablingFlags;
			hknpModifier* m_modifier;
		};

		/// A list of modifiers registered for one modifier function
		struct ModifierEntries
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpModifierManager::ModifierEntries );
			ModifierEntries();

			HK_PAD_ON_SPU(hknpModifierFlags) m_allEnablingFlags;	///< union of m_entries[].m_enablingFlags
			HK_PAD_ON_SPU(int) m_numModifiers;
			ModifierEntry m_entries[MAX_MODIFIERS_PER_FUNCTION];
		};

		/// Where to insert a new modifier
		enum Priority
		{
			PRIORITY_HIGHER,	///< insert at the beginning of the modifier list.
			PRIORITY_LOWER,		///< insert at the end of the modifier list (default).
		};

	public:

		/// Constructor
		hknpModifierManager();

		/// Register a modifier with the given body/material flag(s).
#if !defined ( HK_PLATFORM_SPU )
		void addModifier( hknpModifierFlags enablingFlags, hknpModifier* modifier, Priority priority = PRIORITY_LOWER );
#endif

		/// Internal function to register a modifier (but public on SPU).
		HK_ON_CPU( protected: )
#if !defined ( HK_PLATFORM_SPU )
		HK_FORCE_INLINE
#endif
		void addModifier( hknpModifierFlags enablingFlags, hknpModifier* modifier, hkUint32 enabledFunctions, Priority priority = PRIORITY_LOWER );

		HK_ON_CPU( public: )

		/// Remove a modifier.
		void removeModifier( hknpModifier* modifier );

		/// Check if any modifier is registered for the given function and flags.
		HK_FORCE_INLINE hkBool32 isFunctionRegistered( hknpModifier::FunctionType function, hknpModifierFlags filter = 0xffffffff ) const;

		/// Utility function to calculate the OR'ed body flags from hknpBodies, hknpMaterials, and global overrides.
		HK_FORCE_INLINE hknpBodyFlags getCombinedBodyFlags( const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB ) const;

		/// Utility function to calculate the OR'ed body flags from both hknpBodies, BODY hknpMaterials, and global overrides.
		HK_FORCE_INLINE hknpBodyFlags getCombinedBodyRootFlags( const hknpMaterial* materials, const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB );

		/// Register a solver.
		void setSolver( hknpConstraintSolverType::Enum solverType, hknpConstraintSolver* solver );

		/// Register a collision detector.
		void setCollisionDetector( hknpCollisionCacheType::Enum collisionType, hknpCollisionDetector* detector );

		/// Set the global collision filter.
		void setCollisionFilter( hknpCollisionFilter* collisionFilter );

		/// Set the global collision query filter.
		void setCollisionQueryFilter( hknpCollisionFilter* collisionQueryFilter );

		/// Get the global collision filter.
		HK_FORCE_INLINE hknpCollisionFilter* getCollisionFilter() const;

		/// Get the global collision query filter.
		HK_FORCE_INLINE hknpCollisionFilter* getCollisionQueryFilter() const;

		/// Set global body flags, which are applied to all bodies.
		/// This is OR'ed with any existing global body flags.
		void incrementGlobalBodyFlags( hknpBodyFlags flags );

		/// Unset global body flags.
		/// Individual flags are cleared only if they are decremented as many times as they have been incremented.
		void decrementGlobalBodyFlags( hknpBodyFlags flags );

		/// Get the global body flags.
		HK_FORCE_INLINE hknpBodyFlags getGlobalBodyFlags() const;

		void checkConsistency();

		HK_FORCE_INLINE void fireWeldingModifier(
			const hknpSimulationThreadContext & tl, const hknpModifierSharedData& sharedData,
			hknpBodyQuality::Flags flags, const hknpCdBody& cdBodyA, hknpCdBody* cdBodyB,
			hknpManifold* manifolds, int numManifolds );

	public:

		/// All registered modifiers for each function.
		ModifierEntries m_modifiersPerFunction[hknpModifier::FUNCTION_MAX];

		/// Modifier used for neighbor welding.
		/// This is not using the normal modifiers as it is triggered by hknpBodyQuality and not materials.
		hknpWeldingModifier* m_neighborWeldingModifier;

		/// Modifier used for motion welding.
		/// This is not using the normal modifiers as it is triggered by hknpBodyQuality and not materials.
		hknpWeldingModifier* m_motionWeldingModifier;

		/// Modifier used for triangle welding.
		/// This is not using the normal modifiers as it is triggered by hknpBodyQuality and not materials.
		hknpWeldingModifier* m_triangleWeldingModifier;

		HK_PAD_ON_SPU(hknpConstraintSolver*) m_constraintSolvers[hknpConstraintSolverType::NUM_TYPES];

		HK_PAD_ON_SPU(hknpCollisionDetector*) m_collisionDetectors[hknpCollisionCacheType::NUM_TYPES];

	protected:

#if defined(HK_PLATFORM_SPU)
		hkPadSpu<hknpCollisionFilter*> m_collisionFilter;

		// If there is no filter explicitly set in a collision query, this filter will be used instead.
		// Defaults to the 'AlwaysHit' filter.
		hkPadSpu<hknpCollisionFilter*> m_collisionQueryFilter;
#else
		hkRefPtr<hknpCollisionFilter> m_collisionFilter;

		// If there is no filter explicitly set in a collision query, this filter will be used instead.
		// Defaults to the 'AlwaysHit' filter.
		hkRefPtr<hknpCollisionFilter> m_collisionQueryFilter;
#endif

	HK_ON_SPU( public: )

		/// Global body flags, applied to all getCombinedBodyFlags() calls.
		HK_PAD_ON_SPU(hknpBodyFlags) m_globalBodyFlags;

		/// Per-flag counters to keep track of when to remove global body flags.
		hkUint8 m_globalBodyFlagCounts[NUM_BODY_FLAGS];
};

#include <Physics/Physics/Dynamics/World/ModifierManager/hknpModifierManager.inl>


// Helper macro
#define HKNP_FIRE_MODIFIER( MGR, MODIFIER_FUNCTION_TYPE, MODIFIER_TYPES, FUNC )					\
{																								\
	const hknpModifierManager::ModifierEntries& modifiersPerFunction = MGR->m_modifiersPerFunction[ MODIFIER_FUNCTION_TYPE ];	\
	HK_ASSERT( 0xf023dedf, modifiersPerFunction.m_numModifiers );								\
	int uniqueIterator = 0;																		\
	do {																						\
		if( MODIFIER_TYPES & modifiersPerFunction.m_entries[uniqueIterator].m_enablingFlags )	\
		{																						\
			hknpModifier* modifier = modifiersPerFunction.m_entries[uniqueIterator].m_modifier;	\
			FUNC;																				\
		}																						\
	} while( ++uniqueIterator < modifiersPerFunction.m_numModifiers );							\
}


#endif // HKNP_MODIFIER_MANAGER_H

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
