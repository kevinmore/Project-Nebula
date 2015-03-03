/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKP_JACOBIAN_HEADER_SCHEMA_H
#define HKP_JACOBIAN_HEADER_SCHEMA_H

#include <Physics/ConstraintSolver/Jacobian/hkpJacobianSchema.h>

#include <Physics/ConstraintSolver/Jacobian/hkpJacobianElement.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Physics/ConstraintSolver/Solve/hkpSolverResults.h>

class hkpSolverResults;
class hkpConstraintInstance;


///
class hkpJacobianHeaderSchema
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpJacobianHeaderSchema );

		HK_FORCE_INLINE void initHeader( hkpConstraintInstance* instance, const int bodyAIndex, const int bodyBIndex, HK_CPU_PTR(hkpSolverResults*) sr, const int solverResultStriding );

		HK_FORCE_INLINE hkpVelocityAccumulator* getBodyA   ( hkpVelocityAccumulator* accumulatorBuffer ) const { return accumulatorBuffer + m_bodyAIndex; }
		HK_FORCE_INLINE hkpVelocityAccumulator* getBodyB   ( hkpVelocityAccumulator* accumulatorBuffer ) const { return accumulatorBuffer + m_bodyBIndex; }

		HK_FORCE_INLINE hkpVelocityAccumulator* getBodyA   ( hkUint16* interIndices, hkpVelocityAccumulator* accumulatorBuffer ) const { return accumulatorBuffer + interIndices[m_bodyAIndex]; }
		HK_FORCE_INLINE hkpVelocityAccumulator* getBodyB   ( hkUint16* interIndices, hkpVelocityAccumulator* accumulatorBuffer ) const { return accumulatorBuffer + interIndices[m_bodyBIndex]; }

		HK_FORCE_INLINE const hkpVelocityAccumulator* getBodyA   ( const hkpVelocityAccumulator* accumulatorBuffer ) const { return accumulatorBuffer + m_bodyAIndex; }
		HK_FORCE_INLINE const hkpVelocityAccumulator* getBodyB   ( const hkpVelocityAccumulator* accumulatorBuffer ) const { return accumulatorBuffer + m_bodyBIndex; }

		HK_FORCE_INLINE const hkpVelocityAccumulator* getBodyA   ( hkUint16* interIndices, const hkpVelocityAccumulator* accumulatorBuffer ) const { return accumulatorBuffer + interIndices[m_bodyAIndex]; }
		HK_FORCE_INLINE const hkpVelocityAccumulator* getBodyB   ( hkUint16* interIndices, const hkpVelocityAccumulator* accumulatorBuffer ) const { return accumulatorBuffer + interIndices[m_bodyBIndex]; }

	private:

		HK_ALIGN_REAL(hkUint8 m_padding0); // type stored here .. or.

	public:

		hkUint8 m_solverResultStriding;

	private:

		hkUint16 m_padding1; // .. or type stored here

#if defined(HK_REAL_IS_DOUBLE)
		hkUint32 m_padding2;
#endif

	public:

		hkUint16 m_bodyAIndex; // points to the local accumulator, or could be the number of
		hkUint16 m_bodyBIndex;

	public:

		hkpSolverResults*		m_solverResultInMainMemory;
		hkpConstraintInstance*	m_constraintInstance;	// this pointer is only set and used of the hkConstraintSolver library is used with the hkDynamics library
};

void hkpJacobianHeaderSchema::initHeader( hkpConstraintInstance* instance, const int bodyAIndex, const int bodyBIndex, HK_CPU_PTR(hkpSolverResults*) sr, int solverResultStriding )
{
	HK_SET_SCHEMA_TYPE(this, SCHEMA_TYPE_HEADER); // set the beginning init function when storing type in dedicated padding space

	m_bodyAIndex    = hkUint16(bodyAIndex);
	m_bodyBIndex    = hkUint16(bodyBIndex);

	m_solverResultInMainMemory = sr;
	m_constraintInstance = instance;

	// !! Note: initialize solverResultsStrinding after assigning type (assigning type modifies the first 4/8 bytes)
	HK_ASSERT2(0xad789ddd, solverResultStriding < 0xf00, "Solver results striding to large.");
	HK_ASSERT2(0xad7866dd, (solverResultStriding & (sizeof(hkReal)-1)) == 0, "Solver results striding not a multiple of native size of hkReal!");
	m_solverResultStriding = hkUint8(solverResultStriding);
}


///
class hkpJacobianGotoSchema : public hkpJacobianSchemaInfo::Goto
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpJacobianGotoSchema );

		HK_FORCE_INLINE void initGoto( hkpJacobianSchema* destination )
		{
			HK_SET_SCHEMA_TYPE(this, SCHEMA_TYPE_GOTO); // set the beginning init function when storing type in dedicated padding space
			m_offset = hkGetByteOffset( this, destination );
		}

		HK_FORCE_INLINE void initOffset( hkLong offset )
		{
			HK_SET_SCHEMA_TYPE(this, SCHEMA_TYPE_GOTO); // set the beginning init function when storing type in dedicated padding space
			m_offset = offset;
		}

	public:

		HK_ALIGN_REAL(hkUint32 m_padding0); // type goes in here
		hkUint32 m_padding1;
		hkLong m_offset;
#if HK_POINTER_SIZE == 4
		hkUint32 m_padding2;
#endif
};


#endif // HKP_JACOBIAN_HEADER_SCHEMA_H

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
