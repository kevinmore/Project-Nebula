/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

template<class VEL>
/*static*/ HK_FORCE_INLINE hkSimdReal hkp1Lin2AngJacobian::computeDiag( hkVector4Parameter jacLinA, hkVector4Parameter jacAngA, hkVector4Parameter jacAngB,
																		const VEL& velAccA, const VEL& velAccB )
{
#if !defined( HK_PLATFORM_SIM_SPU )
	HK_ON_DEBUG( hkReal l = hkMath::fabs( jacLinA.length<3>().getReal() - 1.0f); )
	HK_ASSERT2(0x75d662fb, l < 0.01f , "To call getDiag, the length of the linear part must be 1.0f" );
#endif

	hkVector4 ang0; ang0.setMul( jacAngA, jacAngA );
	hkVector4 ang1; ang1.setMul( jacAngB, jacAngB );

	ang0.mul( velAccA->m_invMasses );
	ang1.mul( velAccB->m_invMasses );
	ang0.add( ang1 );

	hkVector4 sm; sm.setAdd(velAccA->m_invMasses, velAccB->m_invMasses);
	// Make sure we never get zero as a result.
	sm.setMax(sm, hkVector4::getConstant<HK_QUADREAL_EPS_SQRD>() );
	ang0.setW( sm );
	hkSimdReal dot = ang0.horizontalAdd<4>();
	return dot;
}

HK_FORCE_INLINE hkSimdReal hkp1Lin2AngJacobian::getDiag( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB ) const
{
	return computeDiag( m_linear0, m_angular[0], m_angular[1], &mA, &mB );
}

/*
hkReal hkp1Lin2AngJacobian::getDiagAny( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB ) const
{
	const hkp1Lin2AngJacobian& jac = *this;

	hkVector4 lin; lin.setMul( jac.m_linear0, jac.m_linear0 );
	hkVector4 ang0; ang0.setMul( jac.m_angular[0], jac.m_angular[0] );
	hkVector4 ang1; ang1.setMul( jac.m_angular[1], jac.m_angular[1] );

	hkVector4 lin0; lin0.setMul( mA.m_invMasses(3), lin );
	hkVector4 lin1; lin1.setMul( mB.m_invMasses(3), lin );

	ang0.mul( mA.m_invMasses );
	ang1.mul( mB.m_invMasses );

	lin0.add( lin1 );
	ang0.add( ang1 );

	ang0.add( lin0 );
	hkReal x = HK_REAL_EPSILON;
	return ang0.horizontalAdd<3>() + x;
}
*/

HK_FORCE_INLINE hkSimdReal hkp2AngJacobian::getAngularDiag( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB ) const
{
	const hkp2AngJacobian& jac = *this;
	hkVector4 ang0; ang0.setMul( jac.m_angular[0], jac.m_angular[0] );
	hkVector4 ang1; ang1.setMul( jac.m_angular[1], jac.m_angular[1] );

	ang0.mul( mA.m_invMasses );
	ang1.mul( mB.m_invMasses );
	ang0.add( ang1 );
	const hkSimdReal d = ang0.horizontalAdd<3>();
#	if defined(HK_REAL_IS_DOUBLE)
	hkSimdReal x; x.setMax(d, hkSimdReal_EpsSqrd);
	return x;
#	else
	return d + hkSimdReal_Eps;
#	endif
}


/// get the non diag element of the 2*2 inv mass matrix in the case that jacA and jacB share exactly the same rigid bodies
/// which is get J dot ((M-1)*JacB)
/// <todo: think of a better name and do all platforms
HK_FORCE_INLINE hkSimdReal hkp2AngJacobian::getNonDiagOptimized( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, const hkp2AngJacobian& jacB ) const
{
	const hkp2AngJacobian& jacA = *this;

	hkVector4 ang0; ang0.setMul( jacA.m_angular[0], jacB.m_angular[0] );
	hkVector4 ang1; ang1.setMul( jacA.m_angular[1], jacB.m_angular[1] );

	ang0.mul( mA.m_invMasses );
	ang1.mul( mB.m_invMasses );
	ang0.add( ang1 );
	hkSimdReal d = ang0.horizontalAdd<3>();
	return d;
}

/// get the non diag element in the case that jacA and jacB share exactly the same rigid bodies
HK_FORCE_INLINE hkSimdReal hkp2AngJacobian::getNonDiagSameObjects( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, const hkp2AngJacobian& jacB ) const
{
	return getNonDiagOptimized( mA, mB,jacB);
}


/// get the non diag element in the case that jacA and jacB share exactly the same rigid bodies
HK_FORCE_INLINE hkSimdReal hkp2AngJacobian::getNonDiagDifferentObjects_With2ndBodyFromFirstObject( const hkpVelocityAccumulator& mA, const hkp2AngJacobian& jacB ) const
{
	const hkp2AngJacobian& jacA = *this;

	hkVector4 ang0; ang0.setMul( jacA.m_angular[1], jacB.m_angular[0] );
	ang0.mul( mA.m_invMasses );
	hkSimdReal d = ang0.horizontalAdd<3>();
	return d;
}


/// get the non diag element of the 2*2 inv mass matrix in the case that jacA and jacB share exactly the same rigid bodies
/// which is get J dot ((M-1)*JacB)
HK_FORCE_INLINE hkSimdReal hkp1Lin2AngJacobian::getNonDiag( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, const hkp1Lin2AngJacobian& jacB ) const
{
	const hkp1Lin2AngJacobian& jacA = *this;

	hkVector4 lin; lin.setMul( jacA.m_linear0, jacB.m_linear0 );

	hkVector4 ang0; ang0.setMul( jacA.m_angular[0], jacB.m_angular[0] );
	hkVector4 ang1; ang1.setMul( jacA.m_angular[1], jacB.m_angular[1] );

	hkVector4 mA3; mA3.setBroadcast<3>(mA.m_invMasses);
	hkVector4 mB3; mB3.setBroadcast<3>(mB.m_invMasses);
	hkVector4 lin0; lin0.setMul( mA3, lin );
	hkVector4 lin1; lin1.setMul( mB3, lin );

	ang0.mul( mA.m_invMasses );
	ang1.mul( mB.m_invMasses );

	lin0.add( lin1 );
	ang0.add( ang1 );

	ang0.add( lin0 );

	hkSimdReal d = ang0.horizontalAdd<3>();
	return d;
}

HK_FORCE_INLINE hkSimdReal hkp1Lin2AngJacobian::getNonDiagSameObjects( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, const hkp1Lin2AngJacobian& jacB ) const
{
	return getNonDiag( mA, mB,jacB);
}

HK_FORCE_INLINE hkSimdReal hkp1Lin2AngJacobian::getNonDiagDifferentObjects( const hkpVelocityAccumulator& mA, const hkp1Lin2AngJacobian& jacB ) const
{
	const hkp1Lin2AngJacobian& jacA = *this;

	hkVector4 lin; lin.setMul( jacA.m_linear0, jacB.m_linear0 );

	hkVector4 ang0; ang0.setMul( jacA.m_angular[0], jacB.m_angular[0] );

	hkVector4 mA3; mA3.setBroadcast<3>( mA.m_invMasses );
	hkVector4 lin0; lin0.setMul( mA3, lin );

	ang0.mul( mA.m_invMasses );

	ang0.add( lin0 );

	hkSimdReal d = ang0.horizontalAdd<3>();
	return d;
}

/// get the non diag element in the case that jacA and jacB share exactly the same rigid bodies
HK_FORCE_INLINE hkSimdReal hkp1Lin2AngJacobian::getNonDiagDifferentObjects_With2ndBodyFromFirstObject( const hkpVelocityAccumulator& mA, const hkp1Lin2AngJacobian& jacB ) const
{
	const hkp1Lin2AngJacobian& jacA = *this;

	hkVector4 lin; lin.setMul( jacA.m_linear0, jacB.m_linear0 );

	hkVector4 ang0; ang0.setMul( jacA.m_angular[1], jacB.m_angular[0] );

	hkVector4 mA3; mA3.setBroadcast<3>( mA.m_invMasses );
	hkVector4 lin0; lin0.setMul( mA3, lin );

	ang0.mul( mA.m_invMasses );
	ang0.sub( lin0 );

	hkSimdReal d = ang0.horizontalAdd<3>();
	return d;
}

HK_FORCE_INLINE hkSimdReal hkp2Lin2AngJacobian::getDiag( const hkpVelocityAccumulator& mA, const hkpVelocityAccumulator& mB, hkSimdRealParameter leverageRatio ) const
{
	const hkp2Lin2AngJacobian& jac = *this;

	hkVector4 ang0; ang0.setMul( jac.m_angular[0], jac.m_angular[0] );
	hkVector4 ang1; ang1.setMul( jac.m_angular[1], jac.m_angular[1] );

	ang0.mul( mA.m_invMasses );
	ang1.mul( mB.m_invMasses );
	const hkSimdReal leverageRatio2 = leverageRatio * leverageRatio;
	ang0.add(ang1);
#if defined(HK_REAL_IS_DOUBLE)
	hkSimdReal sum = mA.m_invMasses.getComponent<3>() + mB.m_invMasses.getComponent<3>() * leverageRatio2;
	hkSimdReal x; x.setMax(sum, hkSimdReal::getConstant<HK_QUADREAL_EPS_SQRD>());
#else
	hkSimdReal x = mA.m_invMasses.getComponent<3>() + mB.m_invMasses.getComponent<3>() * leverageRatio2 + hkSimdReal::getConstant<HK_QUADREAL_EPS>();
#endif
	hkSimdReal dot = ang0.horizontalAdd<3>(); // ?
	return dot + x;
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
