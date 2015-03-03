/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/PointToPlane/hkpPointToPlaneConstraintData.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpPointToPlaneConstraintData::hkpPointToPlaneConstraintData()
{
	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	m_atoms.m_lin.m_axisIndex = 0;
}

void hkpPointToPlaneConstraintData::setInWorldSpace(const hkTransform& bodyATransform,
														  const hkTransform& bodyBTransform,
														  const hkVector4& pivotW,
														  const hkVector4& planeNormalW)
{
	m_atoms.m_transforms.m_transformA.getTranslation().setTransformedInversePos(bodyATransform,pivotW);
	m_atoms.m_transforms.m_transformB.getTranslation().setTransformedInversePos(bodyBTransform,pivotW);

	// not used:
	m_atoms.m_transforms.m_transformA.getRotation().setIdentity();

	int planeNormalIndex = m_atoms.m_lin.m_axisIndex;
	hkVector4* baseB = &m_atoms.m_transforms.m_transformB.getColumn(0);
	baseB[planeNormalIndex].setRotatedInverseDir(bodyBTransform.getRotation(), planeNormalW);
	hkVector4Util::calculatePerpendicularVector( baseB[planeNormalIndex], baseB[(planeNormalIndex+1)%3] ); baseB[(planeNormalIndex+1)%3].normalize<3>();
	baseB[(planeNormalIndex+2)%3].setCross( baseB[planeNormalIndex], baseB[(planeNormalIndex+1)%3] );

	HK_ASSERT2(0xad7bbd76, isValid(), "Members of PointToPlane constraint inconsistent after World Space constructor..");
}

void hkpPointToPlaneConstraintData::setInBodySpace( const hkVector4& pivotA,const hkVector4& pivotB, const hkVector4& planeNormalB )
{
	m_atoms.m_transforms.m_transformA.getTranslation() = pivotA;
	m_atoms.m_transforms.m_transformB.getTranslation() = pivotB;

	// not used:
	m_atoms.m_transforms.m_transformA.getRotation().setIdentity();

	int planeNormalIndex = m_atoms.m_lin.m_axisIndex;
	hkVector4* baseB = &m_atoms.m_transforms.m_transformB.getColumn(0);
	baseB[planeNormalIndex] = planeNormalB;
	hkVector4Util::calculatePerpendicularVector( baseB[planeNormalIndex], baseB[(planeNormalIndex+1)%3] ); baseB[(planeNormalIndex+1)%3].normalize<3>();
	baseB[(planeNormalIndex+2)%3].setCross( baseB[planeNormalIndex], baseB[(planeNormalIndex+1)%3] );

	HK_ASSERT2(0xad7bbd78, isValid(), "Members of PointToPlane constraint inconsistent after Body Space constructor..");
}


void hkpPointToPlaneConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpPointToPlaneConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	if ( wantRuntime )
	{
		infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
		infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
	}
	else
	{
		infoOut.m_numSolverResults = 0;
		infoOut.m_sizeOfExternalRuntime = 0;
	}
}



hkBool hkpPointToPlaneConstraintData::isValid() const
{
	return m_atoms.m_transforms.m_transformA.getRotation().isOrthonormal() && m_atoms.m_transforms.m_transformB.getRotation().isOrthonormal();
}


int hkpPointToPlaneConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE;
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
