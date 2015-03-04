/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
 
#ifndef HKP_BUILD_JACOBIAN_FROM_ATOMS_H
#define HKP_BUILD_JACOBIAN_FROM_ATOMS_H

#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>
#include <Physics2012/Internal/Solver/Contact/hkpSimpleContactConstraintInfo.h>


void HK_CALL hkSolverBuildJacobianFromAtomsNotContact( const struct hkpConstraintAtom* atoms, int sizeOfAllAtoms, const class hkpConstraintQueryIn &in, class hkpConstraintQueryOut &out );


HK_FORCE_INLINE void HK_CALL hkSolverBuildJacobianFromAtoms( const struct hkpConstraintAtom* atoms, int sizeOfAllAtoms, const class hkpConstraintQueryIn &in, class hkpConstraintQueryOut &out )
{
	if (atoms->m_type == hkpConstraintAtom::TYPE_CONTACT )
	{
		struct hkpSimpleContactConstraintAtom* contactAtom = static_cast<hkpSimpleContactConstraintAtom*>( const_cast<hkpConstraintAtom*>(atoms) );
		const hkBool writeHeaderSchema = true;
		hkSimpleContactConstraintDataBuildJacobian( contactAtom, in, writeHeaderSchema, out );
	}
	else
	{
		hkSolverBuildJacobianFromAtomsNotContact( atoms, sizeOfAllAtoms, in, out );
	}
}


void HK_CALL hkpBeginConstraints( const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out, HK_CPU_PTR(hkpSolverResults*) sr, int solverResultStriding );

#endif // HKP_BUILD_JACOBIAN_FROM_ATOMS_H

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
