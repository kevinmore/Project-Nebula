/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/Malleable/hkpMalleableConstraintData.h>
#include <Physics/ConstraintSolver/Solve/hkpSolve.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>
#include <Physics2012/Internal/Solver/Atom/hkpBuildJacobianFromAtoms.h>


hkpMalleableConstraintData::hkpMalleableConstraintData(hkpConstraintData* constraintData) 
: m_constraintData(constraintData),
  m_strength(hkReal(0.01f))
{
	m_constraintData->addReference();   
	m_atoms.m_bridgeAtom.init( this );
}
	

hkpMalleableConstraintData::hkpMalleableConstraintData(hkFinishLoadedObjectFlag f) : hkpConstraintData(f), m_atoms(f)
{
	if( f.m_finishing )
	{
		m_atoms.m_bridgeAtom.init( this );
	}
}

hkpMalleableConstraintData::~hkpMalleableConstraintData()
{
	m_constraintData->removeReference(); 	
}


void hkpMalleableConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& info ) const
{
	// this must be called first otherwise clear()
	// gets called inside and erases the effect of previous
	// additions
	m_constraintData->getConstraintInfo(info);

	info.m_atoms = const_cast<hkpConstraintAtom*>(m_atoms.getAtoms());
	info.m_sizeOfAllAtoms = m_atoms.getSizeOfAllAtoms();
}


void hkpMalleableConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const 
{
	m_constraintData->getRuntimeInfo( wantRuntime, infoOut );
}


void hkpMalleableConstraintData::buildJacobian( const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out )
{
	hkpConstraintQueryIn min = in;

	min.m_virtMassFactor = min.m_virtMassFactor * m_strength;

	hkpConstraintData::ConstraintInfo info; m_constraintData->getConstraintInfo(info);
	hkSolverBuildJacobianFromAtoms(	info.m_atoms, info.m_sizeOfAllAtoms, min, out );
}


void hkpMalleableConstraintData::setStrength(const hkReal tau)
{
	m_strength = tau;
}




hkReal hkpMalleableConstraintData::getStrength() const
{
	return m_strength;
}




hkBool hkpMalleableConstraintData::isValid() const
{
	return m_constraintData->isValid();
}


int hkpMalleableConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE;
}


hkpConstraintData* hkpMalleableConstraintData::getWrappedConstraintData()
{
	return m_constraintData;
}

const hkpConstraintData* hkpMalleableConstraintData::getWrappedConstraintData() const
{
	return m_constraintData;
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
