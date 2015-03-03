/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONSTRAINT_H
#define HKNP_CONSTRAINT_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Constraint/Data/hkpConstraintData.h>

class hknpConstraintCinfo;
struct hkpConstraintAtom;
struct hkpConstraintRuntime;


/// A constraint between two bodies, using a sharable hkpConstraintData to describe the constraint behavior.
class hknpConstraint : public hkReferencedObject
{
	public:

		// Constraint flags.
		enum FlagBits
		{
			NO_FLAGS								= 0,
			IS_EXPORTABLE							= 1 << 0,	///< Constraint has it's applied impulses exported after solving.
			IS_IMMEDIATE							= 1 << 1,	///< Constraint is immediate.
			IS_DISABLED								= 1 << 2,	///< Constraint is disabled.
			// Destruction runtime flags
			IS_DESTRUCTION_INTERNAL					= 1 << 3,	///< Constraint is internal, created by the Destruction runtime.
			AUTO_REMOVE_ON_DESTRUCTION_RESET		= 1 << 4,	///< Destruction runtime specific. Constraint will be automatically removed when the breakable is reset.
			AUTO_REMOVE_ON_DESTRUCTION				= 1 << 5,	///< Destruction runtime specific. Constraint will be automatically removed on break-off (i.e. not reattached to the broken parts).
			// Modifier flags
			RAISE_CONSTRAINT_FORCE_EVENTS			= 1 << 6,	///< Constraint will raise an event reporting the force every frame.
			RAISE_CONSTRAINT_FORCE_EXCEEDED_EVENTS	= 1 << 7,	///< Constraint will raise an event when its limit is broken
		};
		typedef hkFlags<FlagBits, hkUint8> Flags;

		enum
		{
			IMMEDIATE_RUNTIME_ON_STACK			= 0x1,
			IMMEDIATE_MAX_SOLVER_RESULT_COUNT	= 21,	///< Maximum number of solver results supported for exportable immediate constraints.
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Empty constructor. An initialization function must to be called after construction.
		HK_FORCE_INLINE hknpConstraint();

		/// Construction from a ConstraintCinfo. Calls the default initialization function.
		HK_FORCE_INLINE hknpConstraint( const hknpConstraintCinfo& constraintInfo );

		/// Destructor.
		~hknpConstraint();

		/// Initialize the constraint. This is the default initialization function.
		void init( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, FlagBits flags );

		/// Initialize the constraint as exportable.
		void initExportable( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, int additionalRuntimeSize = 0 );

		/// Initialize the constraint as immediate, affect only one physics step.
		/// An immediate constraint is only used to create the needed internal structures for solving.
		/// After adding it using addImmediateConstraint, it's pointer is never used again in the engine,
		/// It can be safely created as a stack variable, in contrast to normal (permanent) constraints.
		/// Immediate constraints do not support atom constraints that require a runtime.
		void initImmediate( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data );

		/// Initialize the constraint as an immediate instance that is exportable.
		/// The immediate Id provided is used to identify the constraint in hknpModifier::postConstraintExport().
		/// the function will receive memory allocated on the stack as a runtime, and used for exporting impulses, and not for permanent solver state.
		void initImmediateExportable( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, hknpImmediateConstraintId immediateId );

		HK_FORCE_INLINE hkBool32 isEnabled()	const	{ return ( m_flags.noneIsSet(IS_DISABLED) ); }
		HK_FORCE_INLINE hkBool32 isDisabled()	const	{ return ( m_flags.anyIsSet(IS_DISABLED) ); }
		HK_FORCE_INLINE hkBool32 isImmediate()	const	{ return ( m_flags.anyIsSet(IS_IMMEDIATE) ); }
		HK_FORCE_INLINE hkBool32 isPermanent()	const	{ return !( isImmediate() ); }
		HK_FORCE_INLINE hkBool32 isExportable() const	{ return ( m_flags.anyIsSet(IS_EXPORTABLE) ); }
		HK_FORCE_INLINE hkBool32 hasRuntime()	const	{ return ( m_runtimeSize > 0 ); }

#if !defined( HK_PLATFORM_SPU )

		/// Returns true if the constrained bodies are active.
		HK_FORCE_INLINE bool isActive( hknpWorld* world ) const;

		/// Returns true and set the space splitter link index if the constraint bodies are active.
		HK_FORCE_INLINE bool isActive( hknpWorld* world, int& linkIndex ) const;

		/// Returns the space splitter link index of an active constraint.
		/// Note: will assert if the constraint is inactive.
		HK_FORCE_INLINE int getLinkIndex( hknpWorld* world ) const;

		/// Checks that the two bodies have the same active state.
		static HK_FORCE_INLINE void checkActivationConsistency( const hknpBody& bodyA, const hknpBody& bodyB );

#endif

	private:

		HK_FORCE_INLINE void _initCommon	( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, hknpImmediateConstraintId immId );
		HK_FORCE_INLINE void _initPermanent	( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, FlagBits flags, int additionalRuntimeSize );
		HK_FORCE_INLINE void _initImmediate	( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, hknpImmediateConstraintId immId );

	public:

		/// The ID of the first body.
		hknpBodyId m_bodyIdA;

		/// The ID of the second body.
		hknpBodyId m_bodyIdB;

		/// Pointer to the constraint data.
		/// This data may be used by multiple constraints.
		hkRefPtr<hkpConstraintData> m_data;

		/// Optional pointer to store runtime information.
		/// This is typically needed for friction or if you want to export the forces from the solver.
		hkpConstraintRuntime* m_runtime;

		/// Size in bytes of the runtime data.
		hkUint16 m_runtimeSize;

		/// Cached information about atoms.
		hkUint16 m_sizeOfAtoms;

		/// Cached pointer to the atoms.
		const hkpConstraintAtom* m_atoms;

		/// Cached information about the constraint.
		hkUint16 m_sizeOfSchemas;
		hkUint8 m_numSolverResults;
		hkUint8 m_numSolverElemTemps;

		/// ID for immediate constraints.
		hknpImmediateConstraintId m_immediateId;

		/// Flags.
		Flags m_flags;

		/// Cached type information
		hkEnum<hkpConstraintData::ConstraintType, hkUint8> m_type;

		/// User data. Not used by the engine.
		mutable hkUlong m_userData;
};

#include <Physics/Physics/Dynamics/Constraint/hknpConstraint.inl>

#endif // HKNP_CONSTRAINT_H

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
