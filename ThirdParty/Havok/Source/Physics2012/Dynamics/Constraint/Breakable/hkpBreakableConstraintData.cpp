/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintCallbackUtil.h>
#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>
#include <Physics2012/Internal/Solver/Atom/hkpBuildJacobianFromAtoms.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>


hkpBreakableConstraintData::hkpBreakableConstraintData( hkpConstraintData* constraintData )  
: m_constraintData(constraintData), 
  m_solverResultLimit(10),
  m_removeWhenBroken(false)
{
	m_revertBackVelocityOnBreak = false;
	m_constraintData->addReference();

	hkpConstraintData::RuntimeInfo info;
	m_constraintData->getRuntimeInfo( true, info );
	m_childRuntimeSize      = hkUint16(info.m_sizeOfExternalRuntime);
	m_childNumSolverResults = hkUint16(info.m_numSolverResults);

	m_atoms.m_bridgeAtom.init( this );
}

hkpBreakableConstraintData::hkpBreakableConstraintData(hkFinishLoadedObjectFlag f) : hkpConstraintData(f), m_atoms(f) 
{
	if( f.m_finishing )
	{
		m_atoms.m_bridgeAtom.init( this );
	}
}


hkpBreakableConstraintData::~hkpBreakableConstraintData()
{
	m_constraintData->removeReference();
}

void hkpBreakableConstraintData::buildJacobianCallback( const hkpConstraintQueryIn &in, const hkpConstraintQueryOut& out )
{
   // Determine if reaction forces from previous solution should cause a break
	hkpSolverResults* results = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() );
	int numResults = m_childNumSolverResults;
	HK_ASSERT2( 0xf06521fe, out.m_constraintRuntime , "The constraint which is wrapped by the hkpBreakableConstraintData does not support breaking" );

	Runtime* runtime = getRuntime( out.m_constraintRuntime );
	if ( !runtime->m_isBroken )
	{
		hkSimdReal sumReactionForce; sumReactionForce.setZero();
		for(int j = 0; j < numResults; j++)
		{ 
			hkSimdReal impulse; impulse.setFromFloat( results[j].m_impulseApplied );
			sumReactionForce.addMul(impulse, impulse);
		} 

		hkSimdReal solverResultLimit; solverResultLimit.setFromFloat(m_solverResultLimit);
		if(sumReactionForce >  solverResultLimit * solverResultLimit)
		{
			const hkSimdReal rootSumReactionForce = sumReactionForce.sqrt();
			setBroken( in.m_constraintInstance, true, rootSumReactionForce.getReal() );

			if ( this->m_revertBackVelocityOnBreak )
			{
				// revert back the velocities
				const hkSimdReal f = solverResultLimit / rootSumReactionForce;
				
				hkVector4 linA; linA.load<3,HK_IO_NATIVE_ALIGNED>(runtime->m_linearVelcityA);
				hkVector4 linB; linB.load<3,HK_IO_NATIVE_ALIGNED>(runtime->m_linearVelcityB);
				hkVector4 angA; angA.load<3,HK_IO_NATIVE_ALIGNED>(runtime->m_angularVelcityA);
				hkVector4 angB; angB.load<3,HK_IO_NATIVE_ALIGNED>(runtime->m_angularVelcityB);

				hkpVelocityAccumulator* bodyA = const_cast<hkpVelocityAccumulator*>(in.m_bodyA.val());
				hkpVelocityAccumulator* bodyB = const_cast<hkpVelocityAccumulator*>(in.m_bodyB.val());
				{ hkVector4& d = bodyA->m_linearVel; d.setInterpolate( linA, d, f ); }
				{ hkVector4& d = bodyB->m_linearVel; d.setInterpolate( linB, d, f ); }
				{ hkVector4& d = bodyA->m_angularVel; d.setInterpolate( angA, d, f ); }
				{ hkVector4& d = bodyB->m_angularVel; d.setInterpolate( angB, d, f ); }
			}
		} 
	}

	// zero results at the end so we can use recursive breakable constraints
	for(int j = 0; j < numResults; j++)
	{
		results[j].m_impulseApplied = hkReal(0);   
	}
}


void hkpBreakableConstraintData::setBroken (hkpConstraintInstance* instance, hkBool broken, hkReal currentForce )
{
	HK_ASSERT2(0x34322884, instance != HK_NULL, "Null instance pointer passed to setBroken");
	Runtime* runtime = getRuntime( instance->getRuntime() );
	HK_ASSERT2(0x34322883, runtime != HK_NULL, "Runtime must be non-null. This constraint may not be added to the world.");

	if ( broken == runtime->m_isBroken)
	{
		return;
	}

	runtime->m_isBroken = broken;

	if ( !broken )
	{
		instance->m_internal->m_callbackRequest |= hkpConstraintAtom::CALLBACK_REQUEST_SETUP_CALLBACK;
	}
	else
	{
		instance->m_internal->m_callbackRequest &= ~hkpConstraintAtom::CALLBACK_REQUEST_SETUP_CALLBACK;
	}

	{
		hkpSimulationIsland* island = static_cast<hkpSimulationIsland*>(instance->getOwner() ); // we are not on the spu, so this should work
		hkpWorld* world = island->getWorld();

		if ( broken )
		{
			hkpConstraintBrokenEvent be( world, instance, &hkpBreakableConstraintDataClass);
			be.m_actualImpulse = currentForce;
			be.m_impulseLimit  = m_solverResultLimit;
			if(instance->m_listeners.getSize())
			{
				hkpConstraintCallbackUtil::fireConstraintBroken( be );
			}
			hkpWorldCallbackUtil::fireConstraintBroken( world,be);
		}
		else
		{
			hkpConstraintRepairedEvent be( world, instance, &hkpBreakableConstraintDataClass);
			if(instance->m_listeners.getSize())
			{
				hkpConstraintCallbackUtil::fireConstraintRepaired( be );
			}
			hkpWorldCallbackUtil::fireConstraintRepaired( world, be);
		}
	}
}


void hkpBreakableConstraintData::buildJacobian( const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out )
{
   // Determine if reaction forces from previous solution should cause a break
	// 	hkpSolverResults* results = static_cast<hkpSolverResults*>( out.m_constraintRuntime.val() );
	// 	int numResults = m_childNumSolverResults;
	HK_ASSERT2( 0xf06521fe, out.m_constraintRuntime , "The constraint which is wrapped by the hkpBreakableConstraintData does not support breaking" );

	Runtime* runtime = getRuntime( out.m_constraintRuntime );

	if(!runtime->m_isBroken)
	{
		const hkpVelocityAccumulator* bodyA = in.m_bodyA;
		const hkpVelocityAccumulator* bodyB = in.m_bodyB;
		// Save the velocity accumulators
		bodyA->m_linearVel.store<3,HK_IO_NATIVE_ALIGNED>(runtime->m_linearVelcityA);
		bodyB->m_linearVel.store<3,HK_IO_NATIVE_ALIGNED>(runtime->m_linearVelcityB);
		bodyA->m_angularVel.store<3,HK_IO_NATIVE_ALIGNED>(runtime->m_angularVelcityA);
		bodyB->m_angularVel.store<3,HK_IO_NATIVE_ALIGNED>(runtime->m_angularVelcityB);
		
		hkpConstraintData::ConstraintInfo info;	m_constraintData->getConstraintInfo(info);
		hkSolverBuildJacobianFromAtoms(	info.m_atoms, info.m_sizeOfAllAtoms, in, out );
	}
	else
	{
		// insert a nop statement into the solver
		buildNopJacobian( in, out );
		if ( m_removeWhenBroken )
		{
			hkpConstraintInstance* constraint = in.m_constraintInstance.val();
			hkpWorld* world = constraint->getEntityA()->getWorld();
			world->removeConstraint(constraint);
		}
	}
}


void hkpBreakableConstraintData::buildNopJacobian( const hkpConstraintQueryIn& in, hkpConstraintQueryOut& out )
{
	in.beginConstraints( out, HK_NULL, sizeof(hkpSolverResults) );
	hkEndConstraints();
}



void hkpBreakableConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& info ) const
{
	m_constraintData->getConstraintInfo( info);
	info.m_atoms = const_cast<hkpConstraintAtom*>(m_atoms.getAtoms());
	info.m_sizeOfAllAtoms = m_atoms.getSizeOfAllAtoms();
}


void hkpBreakableConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const 
{
	infoOut.m_numSolverResults = m_childNumSolverResults;
	infoOut.m_sizeOfExternalRuntime = m_childRuntimeSize + sizeof(Runtime);
}

hkBool hkpBreakableConstraintData::isValid() const
{
	return m_constraintData->isValid();
}


int hkpBreakableConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE;
}

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
