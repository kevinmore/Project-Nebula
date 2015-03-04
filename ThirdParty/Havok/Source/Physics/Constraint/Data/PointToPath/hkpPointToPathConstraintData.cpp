/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/PointToPath/hkpPointToPathConstraintData.h>
#include <Physics/Constraint/Data/PointToPath/hkpParametricCurve.h>

#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>
#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpPointToPathConstraintData::hkpPointToPathConstraintData()
: m_path(HK_NULL),
  m_maxFrictionForce(0),
  m_angularConstrainedDOF( OrientationConstraintType(CONSTRAIN_ORIENTATION_NONE) )
{
	m_transform_OS_KS[0].setIdentity();
	m_transform_OS_KS[1].setIdentity();

	m_atoms.m_bridgeAtom.init( this );
}

hkpPointToPathConstraintData::~hkpPointToPathConstraintData()
{
	if (m_path)
	{
		m_path->removeReference();
	}
}

void hkpPointToPathConstraintData::setPath(hkpParametricCurve* path)
{
	if (path)
	{
		path->addReference();
	}
	if (m_path)
	{
		m_path->removeReference();
	}
	m_path = path;
}

void hkpPointToPathConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& info ) const
{
	info.m_atoms = const_cast<hkpConstraintAtom*>(m_atoms.getAtoms());
	info.m_sizeOfAllAtoms = m_atoms.getSizeOfAllAtoms();
	info.clear();
	info.addHeader();

	// 2 linear
	info.addMultiple<hkpJacobianSchemaInfo::Bilateral1D>(2);

	// 3 angular
	info.addMultiple<hkpJacobianSchemaInfo::Bilateral1D>(3);

	// 1 for limit. 1 for friction
	info.add<hkpJacobianSchemaInfo::LinearLimits1D>();
	info.add<hkpJacobianSchemaInfo::Friction1D>();
}


void hkpPointToPathConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	// we need runtime data
	infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
	infoOut.m_sizeOfExternalRuntime = sizeof( Runtime);
}


void hkpPointToPathConstraintData::calcPivot( const hkTransform& transformBodyA, hkVector4& pivotOut ) const
{
	pivotOut.setTransformedPos( transformBodyA, m_transform_OS_KS[0].getTranslation() );
}

void hkpPointToPathConstraintData::buildJacobian( const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out )
{
	hkpSolverResults* solverResults = &getRuntime( out.m_constraintRuntime )->m_solverResults[0];

	in.beginConstraints( out, solverResults, sizeof(hkpSolverResults) );

	// transform from object space to world space
	// Abbreviations used:
	// 	A attached object
	// 	R reference object
	// 	ws world space
	// 	os object space
	// 	ks constraint space
	// For Transforms:
	// ws_T_Rks means
	//   transforms from reference constraint space to world space


	hkTransform ws_T_Rks;	ws_T_Rks.setMul( *in.m_transformB, m_transform_OS_KS[1]);
	hkTransform ws_T_Aks;	ws_T_Aks.setMul( *in.m_transformA, m_transform_OS_KS[0]);

	const hkVector4& headingAws  = ws_T_Aks.getColumn<0>();
	const hkVector4& rightAws	 = ws_T_Aks.getColumn<1>();
	const hkVector4& upAws		 = ws_T_Aks.getColumn<2>();
	const hkVector4& pivotAws    = ws_T_Aks.getTranslation();

	Runtime* runtime = getRuntime( out.m_constraintRuntime );

	hkVector4 pathPointWs;
	{
		hkVector4 pathPoint; pathPoint._setTransformedInversePos( ws_T_Rks, pivotAws );
		runtime->m_parametricPosition = m_path->getNearestPoint( runtime->m_parametricPosition, pathPoint, pathPoint );
		pathPointWs._setTransformedPos( ws_T_Rks, pathPoint );
	}

	hkVector4 tangentRws;
	hkVector4 perpTangentRws;
	hkVector4 perpTangent2Rws;
	{
		hkVector4 tangent;
		m_path->getTangent( runtime->m_parametricPosition, tangent );

		tangentRws._setRotatedDir( ws_T_Rks.getRotation(), tangent );
		hkVector4Util::calculatePerpendicularVector( tangentRws, perpTangentRws );
		perpTangentRws.normalize<3>();
		perpTangent2Rws.setCross( tangentRws, perpTangentRws );
	}

	hkReal actualDist = m_path->getLengthFromStart( runtime->m_parametricPosition );

	// this is fairly hacky.  This needs to be set up so that tangent.dot( localizedParametricWS )
	// is equal to the distance along the path
	hkVector4 localizedParametricWS;	localizedParametricWS.setAddMul( pivotAws, tangentRws, hkSimdReal::fromFloat(-actualDist) );

	// handle friction (Note: must be the first constraint)
	if( m_maxFrictionForce > hkReal(0) )
	{
		hkp1dLinearFrictionInfo lfi;

		// set up the direction the friction works in
		lfi.m_constrainedDofW = tangentRws;

		// set distance from the "zero" reference point for the positional friction
		lfi.m_pivot = localizedParametricWS;
		lfi.m_maxFrictionForce = m_maxFrictionForce;
		lfi.m_lastSolverResults = &runtime->m_solverResults[SOLVER_RESULT_FRICTION];
		hk1dLinearFrictionBuildJacobian( lfi, in, out );
	}

	// constraint off the linear bits
	hkp1dLinearBilateralConstraintInfo bp;
	bp.m_pivotA = pivotAws;
	bp.m_pivotB = pathPointWs;
	bp.m_constrainedDofW = perpTangentRws;
	hk1dLinearBilateralConstraintBuildJacobian( bp, in, out );
	bp.m_constrainedDofW = perpTangent2Rws;
	hk1dLinearBilateralConstraintBuildJacobian( bp, in, out );

	// constraint off angular DOFs

	// at least one constrained dof
	if( m_angularConstrainedDOF > hkpPointToPathConstraintData::CONSTRAIN_ORIENTATION_NONE )
	{
		hkp1dAngularBilateralConstraintInfo bp2;
		bp2.m_zeroErrorAxisAinW = tangentRws;
		bp2.m_perpZeroErrorAxisBinW = upAws;
		bp2.m_constrainedDofW = rightAws;
		hk1dAngularBilateralConstraintBuildJacobian( bp2, in, out );

		bp2.m_perpZeroErrorAxisBinW = rightAws;
		bp2.m_constrainedDofW.setNeg<4>( upAws );
		hk1dAngularBilateralConstraintBuildJacobian( bp2, in, out );

		if( m_angularConstrainedDOF == hkpPointToPathConstraintData::CONSTRAIN_ORIENTATION_TO_PATH)
		{
			hkVector4 rightR;	m_path->getBinormal( runtime->m_parametricPosition, rightR );
			hkVector4 rightRws;	rightRws._setRotatedDir( ws_T_Rks.getRotation(), rightR );

			bp2.m_zeroErrorAxisAinW = upAws;
			bp2.m_perpZeroErrorAxisBinW.setNeg<4>( rightRws );
			bp2.m_constrainedDofW = headingAws;

			hk1dAngularBilateralConstraintBuildJacobian( bp2, in, out );
		}
	}

	// handle limits
	if( !m_path->isClosedLoop() )
	{
		hkp1dLinearLimitInfo bpLim;
		// get our curve-space position in real length
		bpLim.m_pivotA = pivotAws;
		bpLim.m_pivotB = localizedParametricWS;
		bpLim.m_min = m_path->getLengthFromStart( m_path->getStart() );
		bpLim.m_max = m_path->getLengthFromStart( m_path->getEnd() );
		bpLim.m_constrainedDofW = tangentRws;
		hk1dLinearLimitBuildJacobian( bpLim, in, out );
	}

	hkEndConstraints();

}


hkBool hkpPointToPathConstraintData::isValid() const
{
	// needs more checks.
	return m_path && (m_angularConstrainedDOF != CONSTRAIN_ORIENTATION_INVALID);
}


int hkpPointToPathConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_POINTTOPATH;
}


void hkpPointToPathConstraintData::setInWorldSpace(const hkTransform& ws_T_Aos, const hkTransform& ws_T_Ros, const hkVector4& pivotWs, hkpParametricCurve *path, const hkTransform& os_T_Rks )
{
	if (path)
	{
		path->addReference();
	}

	if (m_path)
	{
		m_path->removeReference();
	}
	m_path = path;

	m_transform_OS_KS[1] = os_T_Rks;

	// Get the matrix from the path
	hkTransform os_T_RAks;
	{
		hkTransform ws_T_Rks; ws_T_Rks.setMul( ws_T_Ros, os_T_Rks );
		hkReal parametricPosition = hkReal(0);
		hkVector4 pathPoint; pathPoint._setTransformedInversePos( ws_T_Rks, pivotWs );
		parametricPosition = m_path->getNearestPoint( parametricPosition, pathPoint, pathPoint );

		hkVector4 tangent;		m_path->getTangent( parametricPosition, tangent );
		hkVector4 rightR;		m_path->getBinormal( parametricPosition, rightR );
		hkVector4 upR; upR.setCross( tangent, rightR );

		hkTransform ks_T_Rks;
		ks_T_Rks.getRotation().setCols( tangent, rightR, upR );
		ks_T_Rks.setTranslation( pathPoint );
		HK_ASSERT( 0xf0ef5421, os_T_Rks.getRotation().isOrthonormal() );

		os_T_RAks.setMul( os_T_Rks, ks_T_Rks );
	}

	// transform our matrix into attached space, as we cannot change the m_transform_OS_KS[0]
	hkTransform os_T_Aks;
	{
		hkTransform ws_T_Rks; ws_T_Rks.setMul( ws_T_Ros, os_T_RAks);
		os_T_Aks.setMulInverseMul( ws_T_Aos, ws_T_Rks );
	}
	m_transform_OS_KS[0] = os_T_Aks;
}


void hkpPointToPathConstraintData::setInBodySpace(const hkVector4& pivotA, const hkVector4& pivotB, hkpParametricCurve *path)
{
	HK_ASSERT2( 0xf03421de, 0, "Warning: Contact Havok support if you want to use this function and if you are not a constraint expert" );
	if (path)
	{
		path->addReference();
	}
	if (m_path)
	{
		m_path->removeReference();
	}
	m_path = path;

	m_transform_OS_KS[0].setIdentity();
	m_transform_OS_KS[1].setIdentity();

	m_transform_OS_KS[0].setTranslation(pivotA);
	m_transform_OS_KS[1].setTranslation(pivotB);
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
