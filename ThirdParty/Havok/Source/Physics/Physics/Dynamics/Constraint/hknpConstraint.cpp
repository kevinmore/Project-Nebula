/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Constraint/hknpConstraint.h>


void hknpConstraint::init( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, FlagBits flags )
{
	_initPermanent( bodyIdA, bodyIdB, data, flags, 0 );
}

void hknpConstraint::initExportable( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, int additionalRuntimeSize )
{
	_initPermanent( bodyIdA, bodyIdB, data, IS_EXPORTABLE, additionalRuntimeSize );
}

void hknpConstraint::initImmediate( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data )
{
	_initImmediate( bodyIdA, bodyIdB, data, hknpImmediateConstraintId::invalid() );
	m_flags = IS_IMMEDIATE;

	m_numSolverResults = 0;
	m_runtimeSize = 0;
	m_runtime = HK_NULL;
}

void hknpConstraint::initImmediateExportable( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, hknpImmediateConstraintId immId )
{
	_initImmediate( bodyIdA, bodyIdB, data, immId );
	m_flags = IS_EXPORTABLE | IS_IMMEDIATE;

	m_numSolverResults = 0;
	m_runtimeSize = 0;
	// Abuse the pointer to signal to the solver that we want a stack runtime for exporting.
	// This is of course not a valid address.
	m_runtime = (hkpConstraintRuntime*) IMMEDIATE_RUNTIME_ON_STACK;

#ifdef HK_DEBUG
	// Check the number of solver results.
	hkpConstraintData::RuntimeInfo runtimeInfo;
	data->getRuntimeInfo( true, runtimeInfo );
	HK_ASSERT2(0xcdd556f, runtimeInfo.m_numSolverResults <= IMMEDIATE_MAX_SOLVER_RESULT_COUNT, "Constraint has too many solver results to be exported as an immediate constraint." );
#endif
}

hknpConstraint::~hknpConstraint()
{
	if ( m_runtime > (void*) IMMEDIATE_RUNTIME_ON_STACK )
	{
		hkDeallocateChunk( (char*)m_runtime, m_runtimeSize, HK_MEMORY_CLASS_PHYSICS );
	}
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
