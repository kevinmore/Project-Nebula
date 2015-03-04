/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/DebugUtil/TraceStream/hkTraceStream.h>
#include <Common/Base/Config/hkConfigSolverLog.h>

#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>
#include <Physics2012/Internal/Solver/Atom/hkpBuildJacobianFromAtoms.h>

#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>

#include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>
#include <Physics2012/Dynamics/Constraint/Setup/hkpConstraintSolverSetup.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpSimpleContactConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpContactImpulseLimitBreachedListener.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>
#include <Physics2012/Dynamics/Motion/hkpMotion.h>
#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactPointEvent.h>

// needed to create a goto schema between schema buffers.
#include <Physics/ConstraintSolver/Jacobian/hkpJacobianHeaderSchema.h>
#include <Physics2012/Dynamics/Collide/ContactListener/hkpContactPointEvent.h>
#include <Physics/Constraint/Data/hkpConstraintInfo.h>

	// init the different buffers to taskHeader
int hkpConstraintSolverSetup::calcBufferOffsetsForSolve( const hkpSimulationIsland& island, char* buffer, int bufferSize, hkpBuildJacobianTaskHeader& taskHeader )
{
	taskHeader.m_buffer = buffer;
	int numBodies = island.getEntities().getSize();
	{
		taskHeader.m_accumulatorsBase = reinterpret_cast<hkpVelocityAccumulator*>(buffer);
			// fixed rigid body + end tag (16/32 byte aligned)
		int accumSize =  hkSizeOf(hkpVelocityAccumulator) + HK_REAL_ALIGNMENT + hkSizeOf(hkpVelocityAccumulator) * island.getEntities().getSize();
		buffer += accumSize; 
	}

	
	int extraTasksDueToMt = 0;
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
	{
		int minNumConstraintsPerTask = hkpBuildJacobianTask::MAX_NUM_ATOM_INFOS;
#		if defined(HK_PLATFORM_HAS_SPU)
		{
			 // max - one for fixed - optional one for the last slot (which cannot be used if the constraint requires two new accumulators, and there is just one free left)
			int minNumConstraintBecauseOfAccumOverflow = (hkpSolveConstraintBatchTask::MAX_NUM_ACCUMULATORS_PER_TASK-1-1)/2;
			minNumConstraintsPerTask = hkMath::min2( minNumConstraintsPerTask, minNumConstraintBecauseOfAccumOverflow );
		}
#		endif

		extraTasksDueToMt = (island.m_numConstraints/minNumConstraintsPerTask) +1;
		extraTasksDueToMt += HKP_MAX_NUMBER_OF_CHAIN_CONSTRAINTS_IN_ISLAND_WHEN_MULTITHREADING;
		extraTasksDueToMt += 3;  // one task is the ppu only constraints + for high-normal constraints switch + also another task -- for ppu-high-normal switch
		extraTasksDueToMt *= 2; // multiply the num by 2 -- in the worst case you have a full task followed by a one-constraint task in every batch
	}
#endif

	// schema base and size
	{
		HK_ASSERT2(0xad888ddd, !(hkUlong(buffer) & (HK_REAL_ALIGNMENT-1)), "The schema buffer start is not SIMD aligned!");
		taskHeader.m_schemasBase = reinterpret_cast<hkpJacobianSchema*>(buffer);

		// schemas + alignment padding for ppu/spu jobs + end schema
		int sizeForSchemas = island.m_constraintInfo.m_sizeOfSchemas + hkpJacobianSchemaInfo::End::Sizeof;

		// calculate padding for schemas
		sizeForSchemas += hkpJacobianSchemaInfo::End::Sizeof * extraTasksDueToMt;

		HK_ASSERT2(0xad786544, (sizeForSchemas & (HK_REAL_ALIGNMENT-1)) == 0, "Schemas must be SIMD aligned.");
		buffer += sizeForSchemas;
	}

	// now we continue 4 byte aligned
	{
		int sizeForElemTemp = island.m_constraintInfo.m_numSolverElemTemps * hkSizeOf(hkpSolverElemTemp) + 2 * hkSizeOf(hkpSolverElemTemp);
			// the 2 * sizeof(hkpSolverElemTemp) is needed due to micro mode double buffering technique
			// (actually always 3 results are written back)
		sizeForElemTemp += extraTasksDueToMt * (3*sizeof(hkReal));// (potential padding)
		sizeForElemTemp = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeForElemTemp);
		taskHeader.m_solverTempBase = reinterpret_cast<hkpSolverElemTemp*>(buffer);
		buffer += sizeForElemTemp;
	}
	
	int size = int(buffer - (char*)taskHeader.m_buffer);
	taskHeader.m_bufferSize = size;
	HK_ON_DEBUG( taskHeader.m_bufferCapacity = 0 );

	if ( taskHeader.m_buffer )
	{
		HK_ASSERT2( 0xf0434434, bufferSize >= size, "The buffer you used is too small" );
		hkpVelocityAccumulator* ra = taskHeader.m_accumulatorsBase;
		ra[0].setFixed();
		ra[1+numBodies].m_type = hkpVelocityAccumulator::HK_END;
	}
	return size;
}

int hkpConstraintSolverSetup::calcBufferSize( hkpSimulationIsland& island ) 
{
	hkpBuildJacobianTaskHeader taskHeader;
	return calcBufferOffsetsForSolve( island, HK_NULL, 0, taskHeader );
}

int HK_CALL hkpConstraintSolverSetup_calcBufferSize( const hkpSimulationIsland& island ) 
{
	hkpBuildJacobianTaskHeader taskHeader;
	return hkpConstraintSolverSetup::calcBufferOffsetsForSolve( island, HK_NULL, 0, taskHeader );
}

void HK_CALL hkpConstraintSolverSetup::_buildAccumulators( const hkStepInfo& info, hkpEntity*const* bodiesIn, int numEntities, hkpVelocityAccumulator* accumOut )
{
	// set the static rigid body
	accumOut[0].setFixed();

	for (int i = 0; i < numEntities; i++ )
	{
		hkUint32 newVal = (1+i)*sizeof(hkpVelocityAccumulator);
			// try to avoid making the cacheline dirty
		if ( newVal != bodiesIn[i]->m_solverData)
		{
			bodiesIn[i]->m_solverData = newVal;
		}
	}

	hkpVelocityAccumulator* end = hkRigidMotionUtilApplyForcesAndBuildAccumulators( info, (hkpMotion*const*)bodiesIn, numEntities, HK_OFFSET_OF(hkpEntity, m_motion), accumOut + 1  );
	end->m_type = hkpVelocityAccumulator::HK_END;
}



// this is the single-thread version of code.
void HK_CALL hkpConstraintSolverSetup::_buildJacobianElement( const hkConstraintInternal* c,
															 hkpConstraintQueryIn& in,
															 hkpVelocityAccumulator* baseAccum,
															 hkpConstraintQueryOut& out
										)
{
	hkMath::forcePrefetch<256>(c->getAtoms());
	if ( c->getAtomsSize() > 256 )
	{
		hkMath::forcePrefetch<256>( hkAddByteOffset(c->getAtoms(),256) );
		if ( c->getAtomsSize() > 512 )	{	hkMath::forcePrefetch<256>( hkAddByteOffset(c->getAtoms(),512) );}
	}

	hkpEntity* rA = c->m_entities[0];
	hkpEntity* rB = c->m_entities[1];

	hkpMotion* cA = rA->getMotion();
	hkpMotion* cB = rB->getMotion();
	in.m_bodyA = hkAddByteOffset(baseAccum, rA->m_solverData );
	in.m_bodyB = hkAddByteOffset(baseAccum, rB->m_solverData );

	in.m_transformA = &cA->getTransform();
	in.m_transformB = &cB->getTransform();

	in.m_constraintInstance = c->m_constraint;
	out.m_constraintRuntime  = c->m_runtime;

	{
		in.m_accumulatorAIndex = HK_ACCUMULATOR_OFFSET_TO_INDEX(rA->m_solverData);
		in.m_accumulatorBIndex = HK_ACCUMULATOR_OFFSET_TO_INDEX(rB->m_solverData);
		out.m_constraintRuntimeInMainMemory = out.m_constraintRuntime;
#	if defined (HK_PLATFORM_HAS_SPU)
		in.m_atomInMainMemory = c->getAtoms();
#	endif
	}


	HK_ASSERT( 0xf0140201, &rA->getCollidable()->getTransform() == in.m_transformA );
	HK_ASSERT( 0xf0140202, &rB->getCollidable()->getTransform() == in.m_transformB );

	HK_ON_DEBUG(hkpJacobianSchema* oldSchemas = out.m_jacobianSchemas);

	hkSolverBuildJacobianFromAtoms(	c->getAtoms(), c->getAtomsSize(), in, out );

#ifdef HK_DEBUG
	{
		// check for consistence within the out.m_jacobians variable
		//HK_ASSERT(0x5bd9c048,  info.m_sizeOfJacobians <= info.m_maxSizeOfJacobians );
		// HK_ASSERT(0x194320f2,  out.m_jacobians - el <= info.m_sizeOfJacobians * hkSizeOf(hkJacobianElement));
		// check m_maxSizeOfSchemas
		HK_ON_DEBUG( int currentSizeOfSchemas = hkGetByteOffsetInt(oldSchemas, out.m_jacobianSchemas.val() ));
		HK_ASSERT(0x2e11828a, currentSizeOfSchemas  <= c->m_sizeOfSchemas );
	}
#endif
}

extern void hkSimpleContactConstraintData_fireCallbacks(class hkpSimpleContactConstraintData* constraintData, const hkpConstraintQueryIn* in, hkpSimpleContactConstraintAtom* atom, hkpContactPointEvent::Type type );

///	Each constraints get queried: It produces a few jacobian elements, returns the number of jacobians generated
void HK_CALL hkpConstraintSolverSetup::_buildJacobianElements( hkpConstraintQueryIn& in,
										hkpEntity*const* bodies, int numBodies,
										hkpVelocityAccumulator* baseAccum,
										hkpJacobianSchema* schemaOut,
										hkpJacobianSchema* schemaOutEnd,
										hkpJacobianSchema* schemaOutB
										)
{
	hkpConstraintQueryOut out;
	out.m_jacobianSchemas = schemaOut; 

	hkpEntity*const* e = bodies;
	hkpEntity*const* eEnd = bodies + numBodies;
	
	hkInplaceArray<const hkConstraintInternal*, 256> criticalConstraints;

	for ( ; e < eEnd; e++ )
	{
		const hkConstraintInternal* c = (*e)->getConstraintMasters().begin();
		const hkConstraintInternal* cEnd = (*e)->getConstraintMasters().end();

		for ( ;c < cEnd; c++ )
		{
			if ( c->m_callbackRequest & ( hkpConstraintAtom::CALLBACK_REQUEST_SETUP_CALLBACK | hkpConstraintAtom::CALLBACK_REQUEST_NEW_CONTACT_POINT | hkpConstraintAtom::CALLBACK_REQUEST_CONTACT_POINT_CALLBACK ) )
			{
				in.m_constraintInstance = c->m_constraint;
				out.m_constraintRuntime = c->m_runtime;
				hkpEntity* eA = c->m_constraint->getEntityA();
				hkpEntity* eB = c->m_constraint->getEntityB();
				HK_ON_DEBUG_MULTI_THREADING( if ( !eA->isFixed() ) { eA->markForWrite(); } );
				HK_ON_DEBUG_MULTI_THREADING( if ( !eB->isFixed() ) { eB->markForWrite(); } );

				in.m_bodyA = hkAddByteOffset( baseAccum, eA->m_solverData );
				in.m_bodyB = hkAddByteOffset( baseAccum, eB->m_solverData );
				in.m_transformA = &eA->getCollidable()->getTransform();
				in.m_transformB = &eB->getCollidable()->getTransform();

				if ( c->m_callbackRequest & ( hkpConstraintAtom::CALLBACK_REQUEST_NEW_CONTACT_POINT | hkpConstraintAtom::CALLBACK_REQUEST_CONTACT_POINT_CALLBACK ) )
				{
					hkpSimpleContactConstraintData* cdata = static_cast<hkpSimpleContactConstraintData*>(c->m_constraint->getDataRw());
					hkpSimpleContactConstraintAtom* atom = cdata->m_atom;
					hkSimpleContactConstraintData_fireCallbacks( cdata, &in, atom, hkpContactPointEvent::TYPE_MANIFOLD );
				}
				HK_ON_DEBUG_MULTI_THREADING( if ( !eB->isFixed() ) { eB->unmarkForWrite(); } );
				HK_ON_DEBUG_MULTI_THREADING( if ( !eA->isFixed() ) { eA->unmarkForWrite(); } );

				if ( c->m_callbackRequest & hkpConstraintAtom::CALLBACK_REQUEST_SETUP_CALLBACK )
				{
					c->m_constraint->getDataRw()->buildJacobianCallback(in, out);
				}
			}


			if (c->m_priority >= hkpConstraintInstance::PRIORITY_TOI_HIGHER)
			{
				// Put this constraint on a list, which will be processed later
				criticalConstraints.pushBack(c);
				continue;
			}

			if ( out.m_jacobianSchemas >= schemaOutEnd )
			{
					 // a safety check 
				HK_ASSERT(0x77f6b793,  ( (hkUlong)(schemaOutEnd) & 0x1) == 0 );

				// Append a goto schema
				hkpJacobianGotoSchema* sc = reinterpret_cast<hkpJacobianGotoSchema*>(out.m_jacobianSchemas.val());
				sc->initGoto(schemaOutB);

				out.m_jacobianSchemas = schemaOutB;
				schemaOutEnd = reinterpret_cast<hkpJacobianSchema*>( ~(hkUlong)0 ); 
			}
			_buildJacobianElement( c, in, baseAccum, out );
		}
	}

	{
		for (int i = 0; i < criticalConstraints.getSize(); i++)
		{
			const hkConstraintInternal* c = criticalConstraints[i];

			if ( out.m_jacobianSchemas >= schemaOutEnd )
			{
					 // a safety check 
				HK_ASSERT(0x77f6b793,  ( (hkUlong)(schemaOutEnd) & 0x1) == 0 );

				// Append a goto schema
				hkpJacobianGotoSchema* sc = reinterpret_cast<hkpJacobianGotoSchema*>(out.m_jacobianSchemas.val());
				sc->initGoto(schemaOutB);
				
				out.m_jacobianSchemas = schemaOutB;
				schemaOutEnd = reinterpret_cast<hkpJacobianSchema*>( ~(hkUlong)0 ); 
			}
			_buildJacobianElement( c, in, baseAccum, out );
		}
	}
		// that's the CONSTRAINT_LIST_END flag.
	hkUint32* constraintListEndPtr = (hkUint32*)(out.m_jacobianSchemas.val());
	constraintListEndPtr[0] = 0;
}


	// integrates a list of bodies and returns the number of inactive frames
static HK_FORCE_INLINE int HK_CALL hkConstraintSolverSetup_integrate2( const struct hkpSolverInfo& si, const hkStepInfo& info, const hkpVelocityAccumulator* accumulators, hkpEntity*const* bodies, int numBodies )
{
#ifdef HK_DEBUG
	const hkpVelocityAccumulator* rm = accumulators + 1;
	for (int i = 0; i<numBodies; i++, rm++)	{	HK_ASSERT( 0xf0232343, hkAddByteOffsetConst( accumulators, bodies[i]->m_solverData) == rm );	}
#endif
	int numInactiveFrames = hkRigidMotionUtilApplyAccumulators(si, info, accumulators+1, (hkpMotion*const*)bodies, numBodies, HK_OFFSET_OF(hkpEntity, m_motion));
	hkpEntityAabbUtil::entityBatchRecalcAabb((*bodies)->getWorld()->getCollisionInput(), bodies, numBodies);
	return numInactiveFrames;
}

/// This function integrates the rigid bodies by using the data in the linear and angular velocity
/// field in the accumulators and not the sumLinearVelocity.
/// The sumLinearVelocities are typically set in the hkSolver::integrateVelocities, however if
/// you only call hkSolveStepJacobion, this sumLinearVelocities won't be used and you have to use this
/// function to integrate your rigid bodies
void HK_CALL hkpConstraintSolverSetup::oneStepIntegrate( const struct hkpSolverInfo& si, const hkStepInfo& info, const hkpVelocityAccumulator* accumulatorsBase, hkpEntity*const* entities, int numEntities )
{
	for (int i = 0; i<numEntities; i++)
	{
		hkpEntity* entity = entities[i];
		hkpMotion* motion = entity->getMotion();

		hkpVelocityAccumulator accu;
		{
			const hkpVelocityAccumulator* srcAccu = hkAddByteOffsetConst( accumulatorsBase, entity->m_solverData);
			accu = *srcAccu;
		}

		accu.getSumLinearVel()  = accu.m_linearVel;
		accu.getSumAngularVel() = accu.m_angularVel;

		// we have to call that function for one motion at a time because the order of the accumulators might not match the order of the entities (and thus motions)!
		const hkBool processDeactivation = false;
		hkUchar oldCounter = motion->m_deactivationIntegrateCounter;
		motion->m_deactivationIntegrateCounter = 0xff;	// disable deactivation for this frame
		hkRigidMotionUtilApplyAccumulators(si, info, &accu, &motion, 1, 0 );
		hkpEntityAabbUtil::entityBatchRecalcAabb(entity->getWorld()->getCollisionInput(), &entity, 1);
		motion->m_deactivationIntegrateCounter = oldCounter;
	}
}

HK_COMPILE_TIME_ASSERT( sizeof(hkpContactImpulseLimitBreachedListenerInfo) == sizeof(hkpImpulseLimitBreachedElem ));
int HK_CALL hkpConstraintSolverSetup::solve(
									const hkStepInfo& stepInfo, const hkpSolverInfo& solverInfo,
									hkpConstraintQueryIn& constraintQueryIn, hkpSimulationIsland& island,
									void* preallocatedBuffer, int preallocatedBufferSize, 
									hkpEntity*const * bodies, int numBodies )
{
	HK_INTERNAL_TIMER_BEGIN_LIST("solver","memory");
	int numInactiveFrames = 0;

	char* scratch2 = HK_NULL;
	int scratch2Size = 0;

	//
	//	values to be set
	//
	int sizeForAccumulators = hkSizeOf(hkpVelocityAccumulator) + HK_REAL_ALIGNMENT + hkSizeOf(hkpVelocityAccumulator) * numBodies; // fixed rigid body + end tag (16/32 byte aligned);
	int sizeForSchemas      = island.m_constraintInfo.m_sizeOfSchemas + hkpJacobianSchemaInfo::End::Sizeof;
	int sizeForElemTemp     = island.m_constraintInfo.m_numSolverElemTemps * hkSizeOf(hkpSolverElemTemp) + 2 * hkSizeOf(hkpSolverElemTemp);
	sizeForElemTemp = HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeForElemTemp );
	hkpJacobianSchema* schemasB = HK_NULL;

	int totalSize = sizeForAccumulators + sizeForSchemas + sizeForElemTemp;

	char* scratchpad;
	char* buffer;
	char* bufferEnd;
	if ( !preallocatedBuffer )
	{
		//
		//	Grep as much buffer memory as possible
		//
#ifndef HK_PLATFORM_SIM_SPU
		HK_ASSERT(0x660f3951, hkMemoryRouter::getInstance().stack().numExternalAllocations() == 0);
#endif
		scratchpad = hkMemSolverBufAlloc<char>( totalSize );
		buffer = scratchpad;
		bufferEnd = buffer + totalSize;
	}
	else
	{
		//
		//	Take preallocated buffer
		//
		scratchpad = HK_NULL;
		buffer = (char*)preallocatedBuffer;
		bufferEnd = buffer + preallocatedBufferSize;
		HK_ASSERT2( 0xf0323454, preallocatedBufferSize >= totalSize, "Your supplied buffer is too small" );
	}

	//
	//	Reserve memory for the DOF objects
	//
redoAll:
	hkpVelocityAccumulator* velAccums;
	{
		velAccums = reinterpret_cast<hkpVelocityAccumulator*>(buffer);
		buffer += sizeForAccumulators;
	}


	//
	//	Reserve memory for the schemas, just do it optimistically the first time round
	//
	hkpJacobianSchema*	schemas;
	hkpJacobianSchema*	schemasEnd;
	{
		HK_ASSERT2(0xad67bcd6, HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, hkUlong(buffer)) == hkUlong(buffer), "Schemas must be SIMD aligned.");
		schemas = reinterpret_cast<hkpJacobianSchema*>(buffer);
		buffer += sizeForSchemas;
		schemasEnd = reinterpret_cast<hkpJacobianSchema*>(buffer);
	}


redoElemTemp:

	//
	//  Reserve memory for temporary data
	//
	hkpSolverElemTemp*		elemTemp;
	{
		elemTemp = reinterpret_cast<hkpSolverElemTemp*>(buffer);
			// the 2 * sizeof(hkpSolverElemTemp) is needed due to micro mode double buffering technique
			// (actually always 3 results are written back)
		buffer += sizeForElemTemp;
	}
	
	

	if ( buffer <= bufferEnd )	// scratchpad ok,  
	{
			//
			//	Convert all rigid bodies to velocity accumulators
			//
		HK_INTERNAL_TIMER_SPLIT_LIST("BuildAccumulators");
		_buildAccumulators( stepInfo, bodies, numBodies, velAccums );

		{	//	clear a few arrays
			HK_ASSERT(0x2385ffbb, (hkUlong)(&elemTemp[island.m_constraintInfo.m_numSolverElemTemps].m_impulseApplied) - (hkUlong)(&elemTemp[0].m_impulseApplied) == (hkUlong)(island.m_constraintInfo.m_numSolverElemTemps*sizeof(hkReal)));
			hkString::memSet4(&elemTemp[0].m_impulseApplied, 0, island.m_constraintInfo.m_numSolverElemTemps * (sizeof(hkReal)>>2));
		}

			//
			//	Build jacobians
			//
		HK_INTERNAL_TIMER_SPLIT_LIST("BuildJacobians");
		_buildJacobianElements( constraintQueryIn, bodies, numBodies, velAccums, schemas, schemasEnd, schemasB );

		HK_INTERNAL_TIMER_SPLIT_LIST("solve");

		// solve constraints can't return a hkBool, so it returns an int instead {0|1}
		bool solved = hkSolveConstraints( solverInfo, schemas, velAccums, elemTemp );
		HK_MONITOR_ADD_VALUE( "NumJacobians", float(island.m_constraintInfo.m_numSolverResults), HK_MONITOR_TYPE_INT );
		HK_MONITOR_ADD_VALUE( "NumEntities", float(numBodies), HK_MONITOR_TYPE_INT );


		HK_INTERNAL_TIMER_SPLIT_LIST("WriteBack");
		if (solved )
		{
#			if ! defined (HK_PLATFORM_HAS_SPU)
			hkExportImpulsesAndRhs(solverInfo, elemTemp, schemas, velAccums );
#			else
			hkExportImpulsesAndRhs(solverInfo, elemTemp, schemas, velAccums, HK_NULL );
#			endif

			HK_INTERNAL_TIMER_SPLIT_LIST("Integrate");
			numInactiveFrames = hkConstraintSolverSetup_integrate2( solverInfo, stepInfo, velAccums, bodies, numBodies );

			// export limit breached 
			{
				hkpImpulseLimitBreachedHeader* h = (hkpImpulseLimitBreachedHeader*)(schemas);
				if ( h->m_numBreached )
				{
					hkpContactImpulseLimitBreachedListenerInfo* bi = (hkpContactImpulseLimitBreachedListenerInfo*)(&h->getElem(0));
					hkpWorldCallbackUtil::fireContactImpulseLimitBreached( island.m_world, bi, h->m_numBreached );
				}
			}
		}
		HK_INTERNAL_TIMER_END_LIST();

		if ( scratch2 )
		{
			hkMemoryRouter::getInstance().temp().blockFree( scratch2, scratch2Size );
		}

		if ( scratchpad )
		{
			hkMemSolverBufFree( scratchpad, totalSize );
		}

		flushSolverDebugOstream();

		return numInactiveFrames;
	}

	//
	//	redo all allocations by using secondary scratchpad
	//
	HK_ASSERT(0x692867c8,  scratch2 == HK_NULL );

	// check for vel accums fitting on scratch
	if ( (char*)schemas >= bufferEnd )
	{
		// if not, forget scratch1 area entirely
		scratch2Size = static_cast<int>( hkUlong(buffer - scratchpad) );
		scratch2 = (char*)hkMemoryRouter::getInstance().temp().blockAlloc( scratch2Size );

		buffer = scratch2;
		bufferEnd = buffer + scratch2Size;

		//!me wouldn't this be nice?
		HK_ASSERT2(0x2bf44da9,  buffer != HK_NULL, "out of memory, trying allocate space for solver on scratch2.  No simulation will happen" );

		goto redoAll;
	}

	int numSchemaBytesNotSplittable = island.m_constraintInfo.m_maxSizeOfSchema + hkpJacobianSchemaInfo::Goto::Sizeof;

	// split schemas 
	if ( (char*)schemasEnd >= bufferEnd )
	{
		// Goto schema must be added to the first buffer 
		int minUsedSizeOfSchemaInBufferA = static_cast<int>( (hkUlong)(bufferEnd - (char*)schemas) ) - numSchemaBytesNotSplittable; //64bit change, (was   char* - char* - int)
		int clipped_Sized = hkMath::max2( minUsedSizeOfSchemaInBufferA, 0);
		int schemaSize2   			 = island.m_constraintInfo.m_sizeOfSchemas + numSchemaBytesNotSplittable - clipped_Sized; // THIS IS NOT NEEDED: + island.m_constraintInfo.m_maxSizeOfSchemas

		scratch2Size = sizeForElemTemp + schemaSize2;
		scratch2 = (char*)hkMemoryRouter::getInstance().temp().blockAlloc( scratch2Size );

		if ( minUsedSizeOfSchemaInBufferA < 0)
		{
			// the size left on the scratchpad does not support a single schema, simply move the first schema to the newly allocated block
			buffer = scratch2;
			bufferEnd = buffer + scratch2Size;

			schemas    = reinterpret_cast<hkpJacobianSchema*>(buffer);
			buffer += island.m_constraintInfo.m_sizeOfSchemas + hkpJacobianSchemaInfo::End::Sizeof;
			schemasEnd    = reinterpret_cast<hkpJacobianSchema*>(buffer);
			schemasB = HK_NULL;
		}
		else
		{
			schemasEnd = reinterpret_cast<hkpJacobianSchema*>(bufferEnd - numSchemaBytesNotSplittable);

			buffer = scratch2;
			bufferEnd = buffer + scratch2Size;

			schemasB = reinterpret_cast<hkpJacobianSchema*>(buffer);
			buffer += schemaSize2;
		}
	}
	else
	{
		// only put elemTemp on scratch2
		int bufferSize = sizeForElemTemp;
		scratch2 = hkAllocateStack<char>( bufferSize );
		buffer = scratch2;
		bufferEnd = buffer + bufferSize;
	}

	goto redoElemTemp;
}




	// acquires the scratchpad and initializes a hkpConstraintSolverResources struct
void HK_CALL hkpConstraintSolverSetup::initializeSolverState( hkStepInfo& stepInfo, hkpSolverInfo& solverInfo, hkpConstraintQueryIn& constraintQueryIn, 
															 char* buffer, int bufferSize, const hkUint8* priorityClassMap, const hkReal* priorityClassRatios,
															 hkpConstraintSolverResources& s)

{
#ifdef HK_DEBUG
	hkReal totalRatios = 0.0f;
	for ( int i = 0; i < hkpConstraintSolverResources::NUM_PRIORITY_CLASSES; ++i )
	{
		totalRatios += priorityClassRatios[i];
	}
	HK_ASSERT2( 0x8e9a23eb, totalRatios <= 1.0f, "Ratio sum is too great" );
#endif // HK_DEBUG


	// Initialize SolverSate struct
	s.m_stepInfo             = &stepInfo;
	s.m_solverInfo           = &solverInfo;
	s.m_constraintQueryInput = &constraintQueryIn;
	hkMemUtil::memCpy( s.m_priorityClassMap, priorityClassMap, hkpConstraintInstance::NUM_PRIORITIES );

		  bufferSize -= bufferSize % 16;
	char* bufferEnd = buffer + bufferSize;


	const int ratioAccumulators = 10;	//hkSizeOf(hkpVelocityAccumulator),
	const int ratioAccumulatorsBackup = 2 + ratioAccumulators * sizeof(hkpConstraintSolverResources::VelocityAccumTransformBackup)/sizeof(hkpVelocityAccumulator);
	const int ratioSchema = 40;			//hkSizeOf(hkpJacobianSchema),
	const int ratioElemTemp = 10;		//hkSizeOf(hkpSolverElemTemp),
	const int sumSize = ratioAccumulators + ratioAccumulatorsBackup + ratioSchema + ratioElemTemp;


	{
		hkUlong b = hkUlong(buffer);

			// Memory
		b = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT,b);

		s.m_accumulators     = reinterpret_cast<hkpVelocityAccumulator*>(b);
		b = b + bufferSize * ratioAccumulators / sumSize;
		b = hkClearBits(b, 0xf);
		s.m_accumulatorsEnd  = reinterpret_cast<hkpVelocityAccumulator*>(b);	
		
		s.m_accumulatorsBackup     = reinterpret_cast<hkpConstraintSolverResources::VelocityAccumTransformBackup*>(b);
		b = b + bufferSize * ratioAccumulatorsBackup / sumSize;
		b = hkClearBits(b, 0xf);
		s.m_accumulatorsBackupEnd = reinterpret_cast<hkpConstraintSolverResources::VelocityAccumTransformBackup*>(b);

		const int bufferSizeWithoutEnds = bufferSize - ( 0x10 * hkpConstraintSolverResources::NUM_PRIORITY_CLASSES );

		for ( int i = 0; i < hkpConstraintSolverResources::NUM_PRIORITY_CLASSES; ++i )
		{
			s.m_schemas[i].m_begin  = reinterpret_cast<hkpJacobianSchema*>(b);
			const int schemaSize = (unsigned int)( ( bufferSizeWithoutEnds * ratioSchema * priorityClassRatios[i] ) / sumSize ) + 0x10;
			b = b + schemaSize;
			b = hkClearBits(b, 0xf);
			s.m_schemas[i].m_end    = reinterpret_cast<hkpJacobianSchema*>(b);
			s.m_schemas[i].m_lastProcessed = s.m_schemas[i].m_current = s.m_schemas[i].m_begin;
		}

	
	//	s.m_elemTemp[0].m_begin = reinterpret_cast<hkpSolverElemTemp*>(b);
	//	b = b + (unsigned int)(bufferSize * ratioElemTemp * schemaBuffersRatio / sumSize);
	//	b = hkClearBits(b, 0xf);
	//	HK_ASSERT(0xf0ff002a, b < hkUlong(bufferEnd));
	//	s.m_elemTemp[0].m_end   = reinterpret_cast<hkpSolverElemTemp*>(b);

	//	s.m_elemTemp[1].m_begin = reinterpret_cast<hkpSolverElemTemp*>(b);
	//	HK_ON_DEBUG( b = b + (unsigned int)(bufferSize * ratioElemTemp * (1.0f - schemaBuffersRatio) / sumSize ));
	//	HK_ON_DEBUG( b = hkClearBits(b, 0xf));
	//	HK_ASSERT(0xf0ff0021, b < hkUlong(bufferEnd));
	//	s.m_elemTemp[1].m_end   = reinterpret_cast<hkpSolverElemTemp*>(bufferEnd);

	//}
	//s.m_accumulatorsCurrent = s.m_accumulators;

	//s.m_schemas[0].m_lastProcessed = s.m_schemas[0].m_current = s.m_schemas[0].m_begin;
	//s.m_schemas[1].m_lastProcessed = s.m_schemas[1].m_current = s.m_schemas[1].m_begin;
	//s.m_elemTemp[0].m_lastProcessed = s.m_elemTemp[0].m_current = s.m_elemTemp[0].m_begin;
	//s.m_elemTemp[1].m_lastProcessed = s.m_elemTemp[1].m_current = s.m_elemTemp[1].m_begin;

	//// Postpone clearing solverTemp elements till schemas are built

	
		s.m_elemTemp         = reinterpret_cast<hkpSolverElemTemp*>(b);
		HK_ON_DEBUG( b = b + bufferSize * ratioElemTemp / sumSize );
		HK_ASSERT(0xf0ff0021, b <= hkUlong(bufferEnd));
		s.m_elemTempEnd      = reinterpret_cast<hkpSolverElemTemp*>(bufferEnd);
	}
	s.m_accumulatorsCurrent = s.m_accumulators;
	s.m_elemTempCurrent     = s.m_elemTemp;

	s.m_elemTempLastProcessed = s.m_elemTemp;
}

	// released the scratchpad
void HK_CALL hkpConstraintSolverSetup::shutdownSolver(hkpConstraintSolverResources& s)
{
}

	// builds accumulators
void HK_CALL hkpConstraintSolverSetup::internalAddAccumulators(hkpConstraintSolverResources& s, hkpEntity*const* entities, int numEntities)
{
	if (!numEntities)
	{
		return;
	}

	hkpVelocityAccumulator* mot = s.m_accumulatorsCurrent;

		// set the static rigid body
	if(s.m_accumulatorsCurrent == s.m_accumulators)
	{
		mot->setFixed();
		mot->convertToSolverType();
		s.m_accumulatorsCurrent = mot+1;
		hkpConstraintSolverResources::VelocityAccumTransformBackup* fixedBackup = s.m_accumulatorsBackup;
		fixedBackup->m_coreTworldRotation.setIdentity();
	}

	for (unsigned int i = 0; i < hkUint32(numEntities); i++ )
	{
		entities[i]->m_solverData = hkUint32(hkGetByteOffset(s.m_accumulators, s.m_accumulatorsCurrent )) + i * sizeof(hkpVelocityAccumulator);
	}

	hkpVelocityAccumulator* firstAccum = s.m_accumulatorsCurrent;
	hkpConstraintSolverResources::VelocityAccumTransformBackup* firstBackup = &s.m_accumulatorsBackup[entities[0]->m_solverData/sizeof(hkpVelocityAccumulator)];

	s.m_accumulatorsCurrent = hkRigidMotionUtilApplyForcesAndBuildAccumulators( *s.m_stepInfo, (hkpMotion*const*)entities, numEntities,
																				HK_OFFSET_OF(hkpEntity, m_motion),
																				firstAccum );
	// initialize the backup and set the velocity to zero
	for (unsigned int i = 0; i < hkUint32(numEntities); i++ )
	{
		firstBackup->m_coreTworldRotation  = firstAccum->getCoreFromWorldMatrix();

		firstAccum->convertToSolverType();
		firstAccum++;
		firstBackup++;
		HK_ASSERT2( 0xf03412de, firstBackup <= s.m_accumulatorsBackupEnd, "Internal velocity accumulator backup buffer error");
	}

	// do not update s.m_accumulatorsCurrent after this.
	s.m_accumulatorsCurrent->m_type = hkpVelocityAccumulator::HK_END;

	return;
}

//	Each constraint gets queried: It produces a few jacobian elements, returns the number of jacobians generated
void HK_CALL hkpConstraintSolverSetup::internalAddJacobianSchemas( 
										hkpConstraintSolverResources& s,
										hkpConstraintInstance** constraints, 	int numConstraints,
										hkArray<hkpConstraintSchemaInfo>& constraintStatus
										)
{
	hkpConstraintQueryIn& in = *s.m_constraintQueryInput;
	hkpConstraintQueryOut out;

	hkpConstraintInstance** constr = constraints;
	hkpConstraintInstance** cEnd   = constraints + numConstraints;


	for ( ;constr < cEnd; constr++ )
	{
		hkpConstraintInstance* instance = constr[0];
		int listId = s.m_priorityClassMap[instance->getPriority()];

		hkpEntity** entity = &instance->getInternal()->m_entities[0];
		hkpMotion* motion[2] = { entity[0]->getMotion(), entity[1]->getMotion() };

		hkpVelocityAccumulator* HK_RESTRICT accANonConst = hkAddByteOffset(s.m_accumulators, entity[0]->m_solverData );
		hkpVelocityAccumulator* HK_RESTRICT accBNonConst = hkAddByteOffset(s.m_accumulators, entity[1]->m_solverData );

		in.m_bodyA = accANonConst;
		in.m_bodyB = accBNonConst;

		HK_ASSERT( 0xf0ff4565, in.m_bodyA < s.m_accumulatorsCurrent );
		HK_ASSERT( 0xf0ff4566, in.m_bodyB < s.m_accumulatorsCurrent );

		// use our backup transform, as the solver destroyed it (it is used to store the sum velocities)
		{
			hkpConstraintSolverResources::VelocityAccumTransformBackup* HK_RESTRICT backupA = &s.m_accumulatorsBackup[entity[0]->m_solverData/sizeof(hkpVelocityAccumulator)];
			hkpConstraintSolverResources::VelocityAccumTransformBackup* HK_RESTRICT backupB = &s.m_accumulatorsBackup[entity[1]->m_solverData/sizeof(hkpVelocityAccumulator)];

			// Switching accumulators context to build Jacobians
			accANonConst->m_context = hkpVelocityAccumulator::ACCUMULATOR_CONTEXT_BUILD_JACOBIANS;
			accBNonConst->m_context = hkpVelocityAccumulator::ACCUMULATOR_CONTEXT_BUILD_JACOBIANS;

			accANonConst->setCoreFromWorldMatrix(backupA->m_coreTworldRotation);
			accBNonConst->setCoreFromWorldMatrix(backupB->m_coreTworldRotation);
		}

		const hkMotionState* HK_RESTRICT msA = static_cast<const hkpMotion*>(motion[0])->getMotionState();
		const hkMotionState* HK_RESTRICT msB = static_cast<const hkpMotion*>(motion[1])->getMotionState();

		in.m_transformA = &msA->getTransform();
		in.m_transformB = &msB->getTransform();

		in.m_constraintInstance = instance;
		out.m_constraintRuntime = instance->getRuntime();

		HK_ASSERT( 0xf0140201, &entity[0]->getCollidable()->getTransform() == in.m_transformA );
		HK_ASSERT( 0xf0140201, &entity[1]->getCollidable()->getTransform() == in.m_transformB );

		//in.m_constraintInstance = HK_NULL; // this cannot be zeroed for normal contact points
		{
			in.m_accumulatorAIndex = HK_ACCUMULATOR_OFFSET_TO_INDEX(entity[0]->m_solverData);
			in.m_accumulatorBIndex = HK_ACCUMULATOR_OFFSET_TO_INDEX(entity[1]->m_solverData);
			out.m_constraintRuntimeInMainMemory = (HK_CPU_PTR(void*))(instance->getRuntime());
#if defined (HK_PLATFORM_HAS_SPU)
			in.m_atomInMainMemory = instance->m_internal->getAtoms();
#endif
		}


		//add ConstraintSchemaInfo
		{
			hkpConstraintSchemaInfo& info = constraintStatus.expandOne();
			info.m_constraint     = instance;
			info.m_schema         = s.m_schemas[listId].m_current;
			hkReal allowedPenetrationDepthA = entity[0]->getCollidable()->m_allowedPenetrationDepth;
			hkReal allowedPenetrationDepthB = entity[1]->getCollidable()->m_allowedPenetrationDepth;
			info.m_allowedPenetrationDepth = hkMath::min2( allowedPenetrationDepthA, allowedPenetrationDepthB );

			out.m_jacobianSchemas = s.m_schemas[listId].m_current;
		}

		// fire contact callbacks
		if ( instance->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_CONTACT )
		{
			hkpSimpleContactConstraintData* cdata = static_cast<hkpSimpleContactConstraintData*>(instance->getDataRw());
			hkpSimpleContactConstraintAtom* atom = cdata->m_atom;
			hkSimpleContactConstraintData_fireCallbacks( cdata, &in, atom, hkpContactPointEvent::TYPE_EXPAND_MANIFOLD );
		}

		if ( instance->m_internal->m_callbackRequest & hkpConstraintAtom::CALLBACK_REQUEST_SETUP_CALLBACK)
		{
			instance->getDataRw()->buildJacobianCallback(in, out);
		}

		hkSolverBuildJacobianFromAtoms(	instance->m_internal->getAtoms(), instance->m_internal->getAtomsSize(), in, out );

		s.m_schemas[listId].m_current = out.m_jacobianSchemas;
		{
			
			//// Clearing of new temp elems
			//for (int ei = 0; ei < instance->m_internal->m_numSolverElemTemps; ei++ )
			//{
			//	s.m_elemTemp[listId].m_current[ei].m_impulseApplied = 0.0f;
			//}
			//
			//s.m_elemTemp[listId].m_current += instance->m_internal->m_numSolverElemTemps; 
			
			s.m_elemTempCurrent       += instance->m_internal->m_numSolverElemTemps; 
#	ifdef HK_DEBUG
			hkpConstraintData::ConstraintInfo cinfo;		instance->getData()->getConstraintInfo(cinfo);
			HK_ASSERT( 0xf032edfd, cinfo.m_numSolverElemTemps == instance->m_internal->m_numSolverElemTemps);
#	endif
		}

		{
			accANonConst->convertToSolverType();
			accBNonConst->convertToSolverType();	
		}
	}


	// that's the CONSTRAINT_LIST_END flag.
	for ( int i = 0; i < hkpConstraintSolverResources::NUM_PRIORITY_CLASSES; ++i )
	{
		*(reinterpret_cast<hkInt32*>(s.m_schemas[i].m_current)) = 0;
	}
	
	return;
}	

hkBool hkpConstraintSolverSetup::internalIsMemoryOkForNewAccumulators    (hkpConstraintSolverResources& s, hkpEntity**     entities,    int numEntities)
{
	int bytesLeft = int ( hkGetByteOffset(s.m_accumulatorsCurrent, s.m_accumulatorsEnd ) );

		// set the static rigid body
	if(s.m_accumulatorsCurrent == s.m_accumulators)
	{
		numEntities++;
	}
	bytesLeft -= 16; // end flag

	if ( numEntities * hkSizeOf(hkpVelocityAccumulator) > bytesLeft )
	{
		return false;
	}
	return true;

}

hkBool hkpConstraintSolverSetup::internalIsMemoryOkForNewJacobianSchemas(hkpConstraintSolverResources& s, hkpConstraintInstance** constraints, int numConstraints)
{
	hkpJacobianSchema*  schemasCurrent[hkpConstraintSolverResources::NUM_PRIORITY_CLASSES];
	for ( int i = 0; i < hkpConstraintSolverResources::NUM_PRIORITY_CLASSES; ++i )
	{
		schemasCurrent[i] = s.m_schemas[i].m_current;
	}
	
	//hkpSolverElemTemp*  elemTempCurrent[2] = { s.m_elemTemp[0].m_current, s.m_elemTemp[1].m_current };
	
	hkpSolverElemTemp*  elemTempCurrent  = s.m_elemTempCurrent;

	hkpConstraintInstance** constr = constraints;
	hkpConstraintInstance** cEnd   = constraints + numConstraints;


	for ( ;constr < cEnd; constr++ )
	{
		int whichType = s.m_priorityClassMap[constr[0]->getPriority()];
		hkConstraintInternal* ci = constr[0]->m_internal;
		
		//elemTempCurrent[ whichType ]  += ci->m_numSolverElemTemps;
		
		elemTempCurrent  += ci->m_numSolverElemTemps;
		schemasCurrent[ whichType ] = hkAddByteOffset(schemasCurrent[ whichType ], ci->m_sizeOfSchemas );
	}

	
	//if ( elemTempCurrent[0]   > s.m_elemTemp[0].m_end          ||
	//	 elemTempCurrent[1]   > s.m_elemTemp[1].m_end          ||
	

	hkBool b = ( elemTempCurrent <= s.m_elemTempEnd );
	for ( int i = 0; i < hkpConstraintSolverResources::NUM_PRIORITY_CLASSES; ++i )
	{
		 b = b && ( hkAddByteOffset( schemasCurrent[i], 4) <= s.m_schemas[i].m_end ); // extra space for the END_SCHEMA
	}
	
	return b;	
}

void HK_CALL hkpConstraintSolverSetup::subSolve(hkpConstraintSolverResources& s, SolverMode mode)
{
	if (mode == SOLVER_MODE_PROCESS_ALL)
	{
		for ( int i = 0; i < hkpConstraintSolverResources::NUM_PRIORITY_CLASSES; ++i )
		{
			s.m_schemas[i].m_lastProcessed = s.m_schemas[i].m_begin;
		}
		
		//s.m_elemTemp[0].m_lastProcessed = s.m_elemTemp[0].m_begin;
		//s.m_elemTemp[1].m_lastProcessed = s.m_elemTemp[1].m_begin;
	}

	// clear solver results for new elements
	{
		int tt = hkGetByteOffsetCpuPtr(s.m_elemTempLastProcessed, s.m_elemTempCurrent);
		HK_ASSERT(0x2385ffbc, tt % (hkUlong)sizeof(hkReal) == 0);
		hkString::memSet4(s.m_elemTempLastProcessed, 0, tt>>2);
		s.m_elemTempLastProcessed = s.m_elemTempCurrent;
	}

#ifdef HK_DEBUG
	{
		for ( hkpVelocityAccumulator* v = s.m_accumulators; v < s.m_accumulatorsCurrent; v++)
		{
			HK_ASSERT(0xf02de44, v->getSumLinearVel().allEqualZero<3>( hkSimdReal::fromFloat(0.00001f) ) );
			HK_ASSERT(0xf02de43, v->getSumAngularVel().allEqualZero<3>( hkSimdReal::fromFloat(0.00001f) ) );
		}
	}
#endif

	for ( int i = 0; i < hkpConstraintSolverResources::NUM_PRIORITY_CLASSES; ++i )
	{
		if ( s.m_schemas[i].m_lastProcessed != s.m_schemas[i].m_current )
		{
			
			//hkSolveStepJacobians(*s.m_solverInfo, s.m_schemas[i].m_lastProcessed, s.m_accumulators, s.m_elemTemp[i].m_lastProcessed);
			
			hkSolveStepJacobians(*s.m_solverInfo, s.m_schemas[i].m_lastProcessed, s.m_accumulators, s.m_elemTemp);
			// <todo.aa export impulses here and check for reached limits
		}

		s.m_schemas[i].m_lastProcessed = s.m_schemas[i].m_current;
	}
	
	
	//s.m_elemTemp[0].m_lastProcessed = s.m_elemTemp[0].m_current; //unused 
	//s.m_elemTemp[1].m_lastProcessed = s.m_elemTemp[1].m_current; //unused
}


// make sure that the sizeof hkpBuildJacobianTask is 2^x or just below

// This HAS to be TRUE for all pointer sizes!
HK_COMPILE_TIME_ASSERT( sizeof(hkpBuildJacobianTask) >= 0x0f80); // if this fails then adjust hkpBuildJacobianTask::MAX_NUM_ATOM_INFOS
HK_COMPILE_TIME_ASSERT( sizeof(hkpBuildJacobianTask) <= 0x1000); // if this fails then adjust hkpBuildJacobianTask::MAX_NUM_ATOM_INFOS

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
