/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/Data/hkpConstraintData.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolverSetup.h>
#include <Physics/Physics/Dynamics/Constraint/hknpConstraintCinfo.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>


HK_FORCE_INLINE hknpConstraint::hknpConstraint() : m_userData(0)
{
}

HK_FORCE_INLINE hknpConstraint::hknpConstraint( const hknpConstraintCinfo& constraintInfo )
	: m_userData(0)
{
	init(	constraintInfo.m_bodyA,
			constraintInfo.m_bodyB,
			constraintInfo.m_constraintData,
			NO_FLAGS );
}

HK_FORCE_INLINE void hknpConstraint::_initCommon( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, hknpImmediateConstraintId immId )
{
	m_data = data;
	m_bodyIdA = bodyIdA;
	m_bodyIdB = bodyIdB;
	m_immediateId = immId;

	hkpConstraintData::ConstraintInfo constraintInfo;
	data->getConstraintInfo( constraintInfo );

	m_atoms = constraintInfo.m_atoms;
	m_type = (hkpConstraintData::ConstraintType)data->getType();
	m_sizeOfAtoms   = hkUint16(constraintInfo.m_sizeOfAllAtoms);
	m_sizeOfSchemas = hkUint16(constraintInfo.m_sizeOfSchemas + constraintInfo.m_extraSchemaSize + sizeof(hknpJacobianHeaderSchema) - hkpJacobianSchemaInfo::Header::Sizeof);
	m_numSolverElemTemps = hkUint8(constraintInfo.m_numSolverElemTemps);
}


HK_FORCE_INLINE void hknpConstraint::_initPermanent( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, FlagBits flags, int additionalRuntimeSize )
{
	_initCommon( bodyIdA, bodyIdB, data, hknpImmediateConstraintId::invalid() );
	m_flags.setAll( static_cast<hkUint8>(flags) );

	hkpConstraintData::RuntimeInfo runtimeInfo;
	data->getRuntimeInfo( (flags & IS_EXPORTABLE) != 0, runtimeInfo );

	// Note: If we need to support more, we can use m_pad as a multiplier for m_numSolverElemTemps and m_numSolverResults.
	HK_ASSERT2(0x21503793, hkUint16(hkUint8(runtimeInfo.m_numSolverResults)) == runtimeInfo.m_numSolverResults, "A maximum of 255 solver results is allowed" );

	m_numSolverResults = hkUint8(runtimeInfo.m_numSolverResults);
	m_runtime = HK_NULL;
	unsigned int runtimeSize = runtimeInfo.m_sizeOfExternalRuntime + additionalRuntimeSize;

	if ( runtimeSize )
	{
		HK_ASSERT2(0x696de7d, HK_NEXT_MULTIPLE_OF(4, runtimeSize) == runtimeSize, "Constraint Runtime must have size aligned to 4." );
		m_runtime = (hkpConstraintRuntime*)hkAllocateChunk<char>(runtimeSize, HK_MEMORY_CLASS_PHYSICS );
		hkString::memSet4( m_runtime, 0, runtimeSize/4 );
	}

	m_runtimeSize = hkUint16(runtimeSize);
}

HK_FORCE_INLINE void hknpConstraint::_initImmediate( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, hknpImmediateConstraintId immId )
{
	_initCommon( bodyIdA, bodyIdB, data, immId );

#ifdef HK_DEBUG
	hkpConstraintData::RuntimeInfo runtimeInfo;
	data->getRuntimeInfo( false, runtimeInfo );
	HK_ASSERT2(0x7ba3ecab, runtimeInfo.m_sizeOfExternalRuntime == 0, "Immediate instances are not supported for constraints that don't work without a runtime" );
#endif
}


#if !defined( HK_PLATFORM_SPU )

HK_FORCE_INLINE bool hknpConstraint::isActive( hknpWorld* world ) const
{
	const int cellA = world->m_bodyManager.getCellIndex( m_bodyIdA );
	const int cellB = world->m_bodyManager.getCellIndex( m_bodyIdB );

	checkActivationConsistency( world->getBody(m_bodyIdA), world->getBody(m_bodyIdB) );

	// Only add the constraint if it is not inactive.
	// This particular condition assumes that at least one of the bodies is dynamic and active
	//	and in that case, at least one has a valid cell index.
	// The situation where the 2 bodies are dynamic and only one of the is active is not legal
	//	and should be prevented by the code managing states.
	return ( cellA != HKNP_INVALID_CELL_IDX || cellB != HKNP_INVALID_CELL_IDX );
}

HK_FORCE_INLINE bool hknpConstraint::isActive( hknpWorld* world, int& linkIndex ) const
{
	const int cellA = world->m_bodyManager.getCellIndex( m_bodyIdA );
	const int cellB = world->m_bodyManager.getCellIndex( m_bodyIdB );

	checkActivationConsistency( world->getBody(m_bodyIdA), world->getBody(m_bodyIdB) );

	if( cellA != HKNP_INVALID_CELL_IDX || cellB != HKNP_INVALID_CELL_IDX )
	{
		linkIndex = world->m_spaceSplitter->getLinkIdx( cellA, cellB );
		return true;
	}

	return false;
}

HK_FORCE_INLINE int hknpConstraint::getLinkIndex( hknpWorld* world ) const
{
	const int cellA = world->m_bodyManager.getCellIndex( m_bodyIdA );
	const int cellB = world->m_bodyManager.getCellIndex( m_bodyIdB );

	checkActivationConsistency( world->getBody(m_bodyIdA), world->getBody(m_bodyIdB) );
	HK_ASSERT(0xef1113a7, cellA != HKNP_INVALID_CELL_IDX || cellB != HKNP_INVALID_CELL_IDX);

	return world->m_spaceSplitter->getLinkIdx( cellA, cellB );
}

/*static*/ HK_FORCE_INLINE void hknpConstraint::checkActivationConsistency( const hknpBody& bodyA, const hknpBody& bodyB )
{
#ifdef HK_DEBUG
	if( bodyA.isDynamic() && bodyB.isDynamic() )
	{
		bool bothActivated = bodyA.isActive() && bodyB.isActive();
		bool bothDeactivated = !bodyA.isActive() && !bodyB.isActive();
		HK_ASSERT2( 0xef1113a7, bothActivated || bothDeactivated, "Both constrained bodies should have the same active state" );
	}
#endif
}

#endif

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
